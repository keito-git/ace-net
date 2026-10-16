"""
RAG (TF-IDF retrieval-augmented) baseline for emotion classification.

For each input, retrieve the top-k most similar sentences from a fixed context
corpus (the same corpus used to build the ACE-Net dictionary) by TF-IDF cosine
similarity, and prepend them to the input before DeBERTa fine-tuning:
  "Context: {ctx1} {ctx2} {ctx3} Text: {original_text}"  (truncated to max_length).

Fair comparison: same backbone, hyperparameters, seeds, and multi-label
threshold as the DeBERTa baseline. The retrieval index is built from the corpus
only (no train-data leakage).
"""
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoTokenizer,
    DebertaV2Config,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.baselines.deberta_baseline import (
    BACKBONE,
    BaselineConfig,
    _build_compute_metrics,
)
from src.data.base_loader import BaseEmotionLoader, EmotionDataset
from src.eval.metrics import compute_all_metrics, save_metrics_json
from src.models.deberta_classifier import DeBERTaEmotionClassifier
from src.utils.env_log import log_environment
from src.utils.seed import set_seed


@dataclass
class RAGConfig:
    """RAG-specific configuration."""
    top_k: int = 3
    context_sep: str = " "
    prefix: str = "Context: "
    infix: str = " Text: "


def build_tfidf_retriever(corpus: List[str]) -> Tuple[TfidfVectorizer, np.ndarray]:
    """Build a TF-IDF index from the corpus; returns (vectorizer, corpus_matrix)."""
    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )
    corpus_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, corpus_matrix


def retrieve_top_k(
    query: str,
    vectorizer: TfidfVectorizer,
    corpus_matrix: np.ndarray,
    corpus: List[str],
    k: int,
) -> List[str]:
    """Retrieve the top-k corpus sentences by TF-IDF cosine similarity."""
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, corpus_matrix).flatten()
    top_idx = np.argsort(sims)[::-1][:k]
    return [corpus[i] for i in top_idx]


def augment_texts_with_context(
    records: List[Dict[str, Any]],
    vectorizer: TfidfVectorizer,
    corpus_matrix: np.ndarray,
    corpus: List[str],
    rag_cfg: RAGConfig,
    split_name: str = "?",
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    For each record, retrieve top-k context sentences and prepend them to the
    text (the original 'label' is preserved).
    """
    augmented = []
    for i, rec in enumerate(records):
        original_text = rec["text"]
        retrieved = retrieve_top_k(
            original_text, vectorizer, corpus_matrix, corpus, rag_cfg.top_k
        )
        ctx_block = rag_cfg.context_sep.join(retrieved)
        new_text = rag_cfg.prefix + ctx_block + rag_cfg.infix + original_text
        augmented.append({**rec, "text": new_text, "_original_text": original_text})

        if verbose and i < 3:
            print(f"  [{split_name}] Example {i}: {new_text[:100]}...")
    return augmented


def _tokenize_records(
    records: List[Dict[str, Any]],
    tokenizer,
    max_length: int,
    is_multilabel: bool,
    n_labels: int,
) -> EmotionDataset:
    """Tokenize records (same logic as the DeBERTa baseline tokenizer)."""
    texts = [r["text"] for r in records]
    raw_labels = [r["label"] for r in records]
    encodings = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    if is_multilabel:
        label_tensor = torch.tensor(np.stack(raw_labels), dtype=torch.float32)
    else:
        label_tensor = torch.tensor(raw_labels, dtype=torch.long)
    return EmotionDataset(encodings, label_tensor)


def run_rag_baseline(
    loader: BaseEmotionLoader,
    cfg: BaselineConfig,
    rag_cfg: RAGConfig,
    corpus_path: str,
    seed: int,
) -> Dict[str, float]:
    """
    Train DeBERTa-v3-base on TF-IDF-retrieved context-augmented inputs.

    cfg must match the DeBERTa baseline for a fair comparison; the only
    difference is the input text. Returns the test-set metrics.
    """
    set_seed(seed)

    run_dir = os.path.join(cfg.output_dir, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    log_environment(
        seed=seed,
        dataset_name=loader.__class__.__name__,
        output_dir=run_dir,
        extra={
            "method": "RAG_TFIDF",
            "corpus_path": corpus_path,
            "top_k": rag_cfg.top_k,
        },
    )

    print(f"  Loading corpus from: {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"  Corpus size: {len(corpus)} sentences")

    # Build the TF-IDF index from the corpus only (avoid train-data leakage).
    print("  Building TF-IDF index...")
    vectorizer, corpus_matrix = build_tfidf_retriever(corpus)
    print(f"  Index built. Vocab size: {len(vectorizer.vocabulary_)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_records, val_records, test_records = loader.load_splits()
    train_aug = augment_texts_with_context(
        train_records, vectorizer, corpus_matrix, corpus, rag_cfg, "train"
    )
    val_aug = augment_texts_with_context(
        val_records, vectorizer, corpus_matrix, corpus, rag_cfg, "val"
    )
    test_aug = augment_texts_with_context(
        test_records, vectorizer, corpus_matrix, corpus, rag_cfg, "test"
    )
    print(f"  Augmented: train={len(train_aug)}, val={len(val_aug)}, test={len(test_aug)}")

    train_ds = _tokenize_records(
        train_aug, tokenizer, cfg.max_length, loader.is_multilabel, loader.n_labels
    )
    val_ds = _tokenize_records(
        val_aug, tokenizer, cfg.max_length, loader.is_multilabel, loader.n_labels
    )
    test_ds = _tokenize_records(
        test_aug, tokenizer, cfg.max_length, loader.is_multilabel, loader.n_labels
    )

    model_config = DebertaV2Config.from_pretrained(cfg.model_name)
    model_config.num_labels = loader.n_labels
    model_config.problem_type = (
        "multi_label_classification" if loader.is_multilabel
        else "single_label_classification"
    )
    model = DeBERTaEmotionClassifier.from_pretrained(
        cfg.model_name,
        config=model_config,
        ignore_mismatched_sizes=True,
    )

    use_fp16 = cfg.fp16 and torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.gradient_clip_norm,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=True,
        fp16=use_fp16,
        dataloader_num_workers=cfg.dataloader_num_workers,
        seed=seed,
        report_to="none",
        logging_steps=50,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_build_compute_metrics(
            loader.is_multilabel, cfg.multilabel_threshold
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )

    trainer.train()

    test_output = trainer.predict(test_ds)
    logits = test_output.predictions

    if loader.is_multilabel:
        probs = 1.0 / (1.0 + np.exp(-logits))
        y_pred = (probs >= cfg.multilabel_threshold).astype(int)
        y_true = np.stack([r["label"] for r in test_records])
    else:
        y_pred = np.argmax(logits, axis=-1)
        y_true = np.array([r["label"] for r in test_records])

    metrics = compute_all_metrics(y_true, y_pred, loader.label_names)
    metrics["seed"] = seed
    metrics["method"] = "RAG_TFIDF"
    metrics["top_k"] = rag_cfg.top_k

    save_metrics_json(metrics, os.path.join(run_dir, "test_metrics.json"))
    print(
        f"[seed={seed}] RAG macro_f1={metrics['macro_f1']:.4f}  "
        f"weighted_f1={metrics['weighted_f1']:.4f}"
    )
    return metrics
