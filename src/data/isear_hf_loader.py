"""
ISEAR HuggingFace loader (dalopeza98/isear-cleaned-dataset).

Uses a public HF mirror of ISEAR so that a local CSV download is not required.
The HF integer labels are remapped onto the ISEAR_LABELS ordering used by the
CSV loader so that results from both loaders are directly comparable.

HF ClassLabel order:
  ['anger', 'disgust', 'fear', 'guilt', 'joy', 'sadness', 'shame']
Remapping (HF int -> ISEAR_LABEL2ID int):
  0 anger -> 2, 1 disgust -> 4, 2 fear -> 1, 3 guilt -> 6,
  4 joy -> 0, 5 sadness -> 3, 6 shame -> 5

HF provides train=6027 and test=1507 (no validation). Validation is carved
from train with a fixed split_seed=42 and val_ratio=0.10 (keep seed fixed).
"""
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.data.base_loader import BaseEmotionLoader
from src.data.isear_loader import ISEAR_LABELS, ISEAR_LABEL2ID
from src.data.task_type import TaskType

# HF integer index -> ISEAR emotion name, from the dataset ClassLabel order.
_HF_INT_TO_EMOTION: Dict[int, str] = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "guilt",
    4: "joy",
    5: "sadness",
    6: "shame",
}
# HF integer -> canonical ISEAR integer (ISEAR_LABEL2ID order).
_HF_INT_TO_ISEAR_INT: Dict[int, int] = {
    hf_idx: ISEAR_LABEL2ID[emo]
    for hf_idx, emo in _HF_INT_TO_EMOTION.items()
}
assert len(_HF_INT_TO_ISEAR_INT) == 7, "Must have exactly 7 mappings"
assert set(_HF_INT_TO_ISEAR_INT.values()) == set(range(7)), \
    "ISEAR integer target must cover all 7 classes without gaps"

HF_ISEAR_DATASET = "dalopeza98/isear-cleaned-dataset"


class ISEARHFLoader(BaseEmotionLoader):
    """
    ISEAR loader via HuggingFace (no local CSV required).

    Label integers are remapped to match ISEAR_LABELS ordering so results are
    directly comparable to ISEARLoader (CSV version).

    Args:
        cache_dir: HuggingFace cache directory
        val_ratio: fraction of train set used as validation (default 0.10)
        split_seed: fixed seed for the train/val carve-out (keep 42)
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        val_ratio: float = 0.10,
        split_seed: int = 42,
    ) -> None:
        self.cache_dir = cache_dir
        self.val_ratio = val_ratio
        self.split_seed = split_seed

    @property
    def label_names(self) -> List[str]:
        return list(ISEAR_LABELS)

    @property
    def n_labels(self) -> int:
        return 7

    @property
    def is_multilabel(self) -> bool:
        return False

    @property
    def task_type(self) -> TaskType:
        return TaskType.CLASSIFICATION

    def load_splits(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Load ISEAR from HuggingFace and return (train, val, test) record lists.

        Val is carved from the HF train split with split_seed=42 and
        val_ratio=0.10; HF integer labels are remapped to ISEAR_LABELS order.
        Each record is {'text': str, 'label': int (0-6, ISEAR_LABELS order)}.
        """
        from datasets import load_dataset

        ds = load_dataset(
            HF_ISEAR_DATASET,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )

        def _to_records(split_name: str) -> List[Dict[str, Any]]:
            records = []
            for ex in ds[split_name]:
                hf_int = int(ex["label"])
                isear_int = _HF_INT_TO_ISEAR_INT[hf_int]
                text = str(ex["text"]).strip()
                if not text:
                    continue
                records.append({"text": text, "label": isear_int})
            return records

        all_train = _to_records("train")
        test_records = _to_records("test")

        rng = np.random.default_rng(self.split_seed)
        n = len(all_train)
        perm = rng.permutation(n)
        n_val = max(1, math.floor(n * self.val_ratio))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        train_records = [all_train[i] for i in train_idx]
        val_records = [all_train[i] for i in val_idx]

        return train_records, val_records, test_records

    def verify_label_mapping(self, n_samples: int = 5) -> None:
        """Print n_samples examples per label for manual verification of the remapping."""
        from datasets import load_dataset

        ds = load_dataset(
            HF_ISEAR_DATASET,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        seen: Dict[int, List[str]] = {}
        for ex in ds["train"]:
            hf_int = int(ex["label"])
            isear_int = _HF_INT_TO_ISEAR_INT[hf_int]
            emo = _HF_INT_TO_EMOTION[hf_int]
            if isear_int not in seen:
                seen[isear_int] = []
            if len(seen[isear_int]) < n_samples:
                seen[isear_int].append(f"[HF={hf_int}/{emo}] {ex['text'][:70]}")
            if all(len(v) >= n_samples for v in seen.values()) and len(seen) == 7:
                break

        print("=== ISEARHFLoader label mapping verification ===")
        for isear_int in sorted(seen.keys()):
            emo_name = ISEAR_LABELS[isear_int]
            print(f"\nISEAR int={isear_int} ({emo_name}):")
            for t in seen[isear_int]:
                print(f"  {t}")
        print("=== End verification ===")
