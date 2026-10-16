"""
SemEval-2018 Task 1 (Subtask E-c, English) data loader.

Paper: Mohammad et al. (2018). SemEval-2018 Task 1: Affect in Tweets.
       SemEval-2018. https://aclanthology.org/S18-1001/
Task page: https://competitions.codalab.org/competitions/17751

HuggingFace path "sem_eval_2018_task_1", config "subtask5.english":
multi-label English emotion classification (trust_remote_code=True may be
required depending on the HF version). Splits: train=6,838 / val=886 /
test=3,259. The 11 emotion labels are fixed from the task definition.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.data.base_loader import BaseEmotionLoader

# Fixed 11-label set from SemEval-2018 Task 1 E-c (canonical task-paper order).
SEMEVAL_LABELS: List[str] = [
    "anger", "anticipation", "disgust", "fear", "joy",
    "love", "optimism", "pessimism", "sadness", "surprise", "trust",
]
assert len(SEMEVAL_LABELS) == 11, (
    f"SEMEVAL_LABELS must have exactly 11 entries, got {len(SEMEVAL_LABELS)}"
)

SEMEVAL_LABEL2ID: Dict[str, int] = {label: i for i, label in enumerate(SEMEVAL_LABELS)}


class SemEval2018Loader(BaseEmotionLoader):
    """Loader for SemEval-2018 Task 1 E-c (11-class multi-label, English tweets)."""

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
        return SEMEVAL_LABELS

    @property
    def n_labels(self) -> int:
        return 11

    @property
    def is_multilabel(self) -> bool:
        return True

    def _example_to_multihot(self, ex: Dict[str, Any]) -> np.ndarray:
        """Convert a HuggingFace example (one boolean column per label) to multi-hot."""
        vec = np.array(
            [float(ex.get(label, 0)) for label in SEMEVAL_LABELS],
            dtype=np.float32,
        )
        return vec

    def load_splits(
        self,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load train/validation/test from HuggingFace as multi-hot records."""
        from datasets import load_dataset

        ds = load_dataset(
            "sem_eval_2018_task_1",
            "subtask5.english",
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        def _convert(example: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "text": str(example.get("Tweet", "")),
                "label": self._example_to_multihot(example),
            }

        train_records = [_convert(ex) for ex in ds["train"]]
        val_records = [_convert(ex) for ex in ds["validation"]]
        test_records = [_convert(ex) for ex in ds["test"]]

        return train_records, val_records, test_records
