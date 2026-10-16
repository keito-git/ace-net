"""
DeBERTa-v3-base text-only supervised fine-tune (primary comparison baseline).

All hyperparameters live in BaselineConfig and are kept identical across every
baseline comparison to guarantee a fair comparison.

CUDA_VISIBLE_DEVICES must be set in os.environ before torch is imported; this
is handled by the training entry points.
"""
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    DebertaV2Config,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.data.base_loader import BaseEmotionLoader, EmotionDataset
from src.eval.metrics import compute_all_metrics, save_metrics_json
from src.models.deberta_classifier import DeBERTaEmotionClassifier
from src.utils.env_log import log_environment
from src.utils.seed import set_seed

BACKBONE: str = "microsoft/deberta-v3-base"


@dataclass
class BaselineConfig:
    """Hyperparameters shared across all baseline comparisons (fixed for fairness)."""
    model_name: str = BACKBONE
    max_length: int = 128
    batch_size: int = 32
    lr: float = 2e-5
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    fp16: bool = True
    dataloader_num_workers: int = 4
    early_stopping_patience: int = 3
    metric_for_best_model: str = "eval_macro_f1"
    multilabel_threshold: float = 0.5
    output_dir: str = "results/deberta_baseline"


def _build_compute_metrics(is_multilabel: bool, threshold: float = 0.5):
    """Build the compute_metrics callback for the HuggingFace Trainer."""
    def _multilabel(eval_pred):
        logits, labels = eval_pred
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= threshold).astype(int)
        return {
            "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        }

    def _multiclass(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        }

    return _multilabel if is_multilabel else _multiclass


def _tokenize_records(
    records: List[Dict[str, Any]],
    tokenizer,
    max_length: int,
    is_multilabel: bool,
    n_labels: int,
) -> EmotionDataset:
    """Tokenize records and return an EmotionDataset."""
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


def run_deberta_baseline(
    loader: BaseEmotionLoader,
    cfg: BaselineConfig,
    seed: int,
) -> Dict[str, float]:
    """
    Train and evaluate DeBERTa-v3-base on a single seed.

    os.environ["CUDA_VISIBLE_DEVICES"] must be set by the caller before this
    function runs. Returns the test-set metric dict.
    """
    set_seed(seed)

    run_dir = os.path.join(cfg.output_dir, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    log_environment(seed=seed, dataset_name=loader.__class__.__name__, output_dir=run_dir)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_records, val_records, test_records = loader.load_splits()
    train_ds = _tokenize_records(
        train_records, tokenizer, cfg.max_length, loader.is_multilabel, loader.n_labels
    )
    val_ds = _tokenize_records(
        val_records, tokenizer, cfg.max_length, loader.is_multilabel, loader.n_labels
    )
    test_ds = _tokenize_records(
        test_records, tokenizer, cfg.max_length, loader.is_multilabel, loader.n_labels
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

    save_metrics_json(metrics, os.path.join(run_dir, "test_metrics.json"))
    print(
        f"[seed={seed}] macro_f1={metrics['macro_f1']:.4f}  "
        f"weighted_f1={metrics['weighted_f1']:.4f}"
    )
    return metrics
