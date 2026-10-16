"""
Task type definitions for the multi-dataset evaluation harness.

TaskType controls the loss function, evaluation metrics, and label
representation used for each dataset.
"""
from enum import Enum


class TaskType(str, Enum):
    """Emotion task types with associated metric conventions."""

    CLASSIFICATION = "classification"
    """Single-label multi-class. Metric: Macro-F1, Weighted-F1."""

    MULTILABEL = "multilabel"
    """Multi-label binary per class. Metric: Macro-F1, Weighted-F1."""

    REGRESSION = "regression"
    """Continuous intensity/valence/arousal/dominance. Metric: Pearson r, MAE, RMSE."""

    CAUSE_EXTRACTION = "cause_extraction"
    """Binary span/utterance-level emotion cause. Metric: binary F1."""

    SARCASM = "sarcasm"
    """Binary sarcasm detection. Metric: binary F1."""


# Metric keys expected in results for each task type
TASK_PRIMARY_METRICS: dict[TaskType, list[str]] = {
    TaskType.CLASSIFICATION: ["macro_f1", "weighted_f1", "micro_f1"],
    TaskType.MULTILABEL: ["macro_f1", "weighted_f1", "micro_f1"],
    TaskType.REGRESSION: ["pearson_r", "mae", "rmse"],
    TaskType.CAUSE_EXTRACTION: ["f1_binary"],
    TaskType.SARCASM: ["f1_binary"],
}
