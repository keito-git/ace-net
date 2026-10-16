"""
Self-training (pseudo-label) baseline for emotion classification.

For each seed: train a teacher DeBERTa on the labeled train set (same HPs as
the DeBERTa baseline), run the teacher on an unlabeled corpus and keep
high-confidence examples as pseudo-labels (any label sigmoid >= pseudo_threshold
for multi-label, or max softmax >= pseudo_threshold for multi-class), then
retrain a student on the augmented train set and evaluate it on the test set.

Fair comparison: same backbone, hyperparameters, seeds, and multi-label
threshold as the DeBERTa baseline; the teacher uses pseudo_threshold=0.7 for
pseudo-label selection. Total compute is roughly 2x the baseline per seed.
"""
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
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
    _tokenize_records,
)
from src.data.base_loader import BaseEmotionLoader, EmotionDataset
from src.eval.metrics import compute_all_metrics, save_metrics_json
from src.models.deberta_classifier import DeBERTaEmotionClassifier
from src.utils.env_log import log_environment
from src.utils.seed import set_seed


@dataclass
class SelfTrainingConfig:
    """Self-training-specific configuration."""
    pseudo_threshold: float = 0.7
    max_pseudo_samples: int = 3000
    min_pseudo_per_class: int = 1


def _run_teacher_inference(
    teacher: torch.nn.Module,
    tokenizer,
    corpus_texts: List[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
    is_multilabel: bool,
    n_labels: int,
    pseudo_threshold: float,
) -> List[Dict[str, Any]]:
    """
    Run the teacher on corpus texts and select high-confidence pseudo-labeled
    examples. Returns a list of records {'text': str, 'label': np.ndarray|int}.
    """
    teacher.eval()
    all_logits = []

    for start in range(0, len(corpus_texts), batch_size):
        batch = corpus_texts[start:start + batch_size]
        enc = tokenizer(
            batch,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = teacher(**enc)
            logits = out.logits.float().cpu().numpy()
        all_logits.append(logits)

    all_logits_np = np.concatenate(all_logits, axis=0)

    pseudo_records = []
    if is_multilabel:
        probs = 1.0 / (1.0 + np.exp(-all_logits_np))
        for i, (text, p) in enumerate(zip(corpus_texts, probs)):
            if np.any(p >= pseudo_threshold):
                pseudo_label = (p >= pseudo_threshold).astype(float)
                pseudo_records.append({"text": text, "label": pseudo_label})
    else:
        probs_softmax = np.exp(all_logits_np) / np.exp(all_logits_np).sum(axis=-1, keepdims=True)
        for i, (text, p) in enumerate(zip(corpus_texts, probs_softmax)):
            max_prob = float(p.max())
            if max_prob >= pseudo_threshold:
                pseudo_label = int(p.argmax())
                pseudo_records.append({"text": text, "label": pseudo_label})

    return pseudo_records


def _build_model(cfg: BaselineConfig, n_labels: int, is_multilabel: bool) -> torch.nn.Module:
    """Build a fresh DeBERTa model (for both teacher and student phases)."""
    model_config = DebertaV2Config.from_pretrained(cfg.model_name)
    model_config.num_labels = n_labels
    model_config.problem_type = (
        "multi_label_classification" if is_multilabel
        else "single_label_classification"
    )
    model = DeBERTaEmotionClassifier.from_pretrained(
        cfg.model_name,
        config=model_config,
        ignore_mismatched_sizes=True,
    )
    return model


def run_self_training_baseline(
    loader: BaseEmotionLoader,
    cfg: BaselineConfig,
    st_cfg: SelfTrainingConfig,
    corpus_path: str,
    seed: int,
) -> Dict[str, float]:
    """
    Train a teacher, pseudo-label the corpus, then train a student.

    cfg must be identical to the DeBERTa baseline for a fair comparison.
    corpus_path is an unlabeled corpus JSON (list of strings). Returns the
    student's test-set metrics.
    """
    set_seed(seed)

    run_dir = os.path.join(cfg.output_dir, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    log_environment(
        seed=seed,
        dataset_name=loader.__class__.__name__,
        output_dir=run_dir,
        extra={
            "method": "SelfTraining",
            "corpus_path": corpus_path,
            "pseudo_threshold": st_cfg.pseudo_threshold,
            "max_pseudo_samples": st_cfg.max_pseudo_samples,
        },
    )

    print(f"  Loading unlabeled corpus: {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_texts: List[str] = json.load(f)
    print(f"  Corpus size: {len(corpus_texts)} sentences")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_records, val_records, test_records = loader.load_splits()

    use_fp16 = cfg.fp16 and torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Phase 1: train teacher.
    print(f"\n  [Phase 1] Training teacher (seed={seed})...")
    teacher_dir = os.path.join(run_dir, "teacher")
    os.makedirs(teacher_dir, exist_ok=True)

    train_ds = _tokenize_records(
        train_records, tokenizer, cfg.max_length, loader.is_multilabel, loader.n_labels
    )
    val_ds = _tokenize_records(
        val_records, tokenizer, cfg.max_length, loader.is_multilabel, loader.n_labels
    )
    test_ds = _tokenize_records(
        test_records, tokenizer, cfg.max_length, loader.is_multilabel, loader.n_labels
    )

    teacher = _build_model(cfg, loader.n_labels, loader.is_multilabel)

    teacher_args = TrainingArguments(
        output_dir=teacher_dir,
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

    teacher_trainer = Trainer(
        model=teacher,
        args=teacher_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_build_compute_metrics(
            loader.is_multilabel, cfg.multilabel_threshold
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )
    teacher_trainer.train()
    teacher.to(device)

    # Phase 2: pseudo-label selection.
    print(f"\n  [Phase 2] Pseudo-labeling corpus with threshold={st_cfg.pseudo_threshold}...")
    pseudo_records = _run_teacher_inference(
        teacher, tokenizer, corpus_texts,
        cfg.max_length, cfg.batch_size, device,
        loader.is_multilabel, loader.n_labels, st_cfg.pseudo_threshold,
    )

    n_pseudo = len(pseudo_records)
    print(f"  Accepted pseudo-labeled examples: {n_pseudo} / {len(corpus_texts)}")

    if n_pseudo == 0:
        print("  WARNING: 0 pseudo-labeled examples accepted. "
              "Student training is identical to the teacher. "
              "Check pseudo_threshold; it may be too high for this corpus.")

    if n_pseudo > st_cfg.max_pseudo_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_pseudo, st_cfg.max_pseudo_samples, replace=False)
        pseudo_records = [pseudo_records[i] for i in sorted(idx)]
        print(f"  Capped to {len(pseudo_records)} pseudo-labeled examples.")

    pseudo_log = {
        "n_corpus": len(corpus_texts),
        "n_pseudo_accepted": n_pseudo,
        "n_pseudo_capped": len(pseudo_records),
        "pseudo_threshold": st_cfg.pseudo_threshold,
    }
    save_metrics_json(pseudo_log, os.path.join(run_dir, "pseudo_label_stats.json"))

    # Phase 3: train student on augmented data.
    print(f"\n  [Phase 3] Training student on {len(train_records)} labeled + {len(pseudo_records)} pseudo-labeled...")
    augmented_train = list(train_records) + list(pseudo_records)

    train_aug_ds = _tokenize_records(
        augmented_train, tokenizer, cfg.max_length, loader.is_multilabel, loader.n_labels
    )

    student = _build_model(cfg, loader.n_labels, loader.is_multilabel)
    student_dir = os.path.join(run_dir, "student")
    os.makedirs(student_dir, exist_ok=True)

    student_args = TrainingArguments(
        output_dir=student_dir,
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

    student_trainer = Trainer(
        model=student,
        args=student_args,
        train_dataset=train_aug_ds,
        eval_dataset=val_ds,
        compute_metrics=_build_compute_metrics(
            loader.is_multilabel, cfg.multilabel_threshold
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )
    student_trainer.train()

    test_output = student_trainer.predict(test_ds)
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
    metrics["method"] = "SelfTraining"
    metrics["n_pseudo_accepted"] = n_pseudo
    metrics["n_pseudo_used"] = len(pseudo_records)

    save_metrics_json(metrics, os.path.join(run_dir, "test_metrics.json"))
    print(
        f"[seed={seed}] SelfTraining macro_f1={metrics['macro_f1']:.4f}  "
        f"weighted_f1={metrics['weighted_f1']:.4f}  "
        f"n_pseudo={len(pseudo_records)}"
    )
    return metrics
