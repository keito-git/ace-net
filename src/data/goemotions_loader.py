"""
GoEmotions data loader.

Paper: Demszky et al. (2020). GoEmotions: A Dataset of Fine-Grained Emotions.
       ACL 2020. https://aclanthology.org/2020.acl-main.372/

HuggingFace path "go_emotions", config "simplified": 28 emotion labels
(27 emotions + neutral), multi-label. Splits: train=43,410 / val=5,426 /
test=5,427. Acquisition: datasets.load_dataset("go_emotions", "simplified").

The label count (28) is fixed from the dataset card and never inferred.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.data.base_loader import BaseEmotionLoader

# Fixed 28-label set in the canonical GoEmotions simplified order (do not reorder).
GOEMOTIONS_LABELS: List[str] = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]
assert len(GOEMOTIONS_LABELS) == 28, (
    f"GOEMOTIONS_LABELS must have exactly 28 entries, got {len(GOEMOTIONS_LABELS)}"
)

GOEMOTIONS_LABEL2ID: Dict[str, int] = {
    label: i for i, label in enumerate(GOEMOTIONS_LABELS)
}


class GoEmotionsLoader(BaseEmotionLoader):
    """Loader for GoEmotions (simplified, 28-class multi-label)."""

    def __init__(
        self,
        tokenizer=None,
        max_length: int = 128,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = cache_dir

    @property
    def label_names(self) -> List[str]:
        return GOEMOTIONS_LABELS

    @property
    def n_labels(self) -> int:
        return 28

    @property
    def is_multilabel(self) -> bool:
        return True

    def _to_multihot(self, label_indices: List[int]) -> np.ndarray:
        """Convert a list of label indices to a 28-dim multi-hot float32 vector."""
        vec = np.zeros(self.n_labels, dtype=np.float32)
        for idx in label_indices:
            if 0 <= idx < self.n_labels:
                vec[idx] = 1.0
        return vec

    def load_splits(
        self,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load train/validation/test from HuggingFace as multi-hot records."""
        from datasets import load_dataset

        ds = load_dataset(
            "go_emotions",
            "simplified",
            cache_dir=self.cache_dir,
        )

        def _convert(example: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "text": str(example["text"]),
                "label": self._to_multihot(example["labels"]),
            }

        train_records = [_convert(ex) for ex in ds["train"]]
        val_records = [_convert(ex) for ex in ds["validation"]]
        test_records = [_convert(ex) for ex in ds["test"]]

        return train_records, val_records, test_records
