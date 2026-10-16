"""
ISEAR (International Survey on Emotion Antecedents and Reactions) CSV loader.

Paper: Scherer and Wallbott (1994). Evidence for universality and cultural
       variation of differential emotion response patterning. Journal of
       Personality and Social Psychology.
Distribution: University of Geneva CISA Research.
  URL: https://www.unige.ch/cisa/research/materials-and-online-research/research-material/

7 emotion classes (joy, fear, anger, sadness, disgust, shame, guilt),
roughly 7,665 English self-report sentences.

ISEAR has no official split; this loader uses a fixed 80/10/10 random split
seeded at 42 for reproducibility. Set local_csv_path to the ISEAR CSV file.
"""
import os
import csv
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.data.base_loader import BaseEmotionLoader
from src.data.task_type import TaskType

ISEAR_LABELS: List[str] = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]
assert len(ISEAR_LABELS) == 7

ISEAR_LABEL2ID: Dict[str, int] = {l: i for i, l in enumerate(ISEAR_LABELS)}
_LABEL_NORMALIZE = {l.upper(): l for l in ISEAR_LABELS}
_LABEL_NORMALIZE.update({l: l for l in ISEAR_LABELS})


class ISEARLoader(BaseEmotionLoader):
    """
    Loader for ISEAR (7-class, no official split).

    Args:
        local_csv_path: path to the ISEAR CSV file
        val_ratio: fraction of data for validation (default 0.1)
        test_ratio: fraction of data for test (default 0.1)
        split_seed: fixed seed for the reproducible split (keep 42)
    """

    def __init__(
        self,
        local_csv_path: str,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        split_seed: int = 42,
        tokenizer=None,
        max_length: int = 128,
    ) -> None:
        if not os.path.exists(local_csv_path):
            raise FileNotFoundError(
                f"ISEAR CSV not found: {local_csv_path}\n"
                f"Download from: https://www.unige.ch/cisa/research/materials-and-online-research/research-material/"
            )
        self.local_csv_path = local_csv_path
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_seed = split_seed
        self.tokenizer = tokenizer
        self.max_length = max_length

    @property
    def label_names(self) -> List[str]:
        return ISEAR_LABELS

    @property
    def n_labels(self) -> int:
        return 7

    @property
    def is_multilabel(self) -> bool:
        return False

    @property
    def task_type(self) -> TaskType:
        return TaskType.CLASSIFICATION

    def _load_all(self) -> List[Dict[str, Any]]:
        """Load all records from the CSV, trying common column-name conventions."""
        records = []
        with open(self.local_csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            text_col = next((h for h in headers if h.upper() in ("SIT", "SITUATION", "TEXT", "STATEMENT")), None)
            emo_col = next((h for h in headers if h.upper() in ("EMOT", "EMOTION", "LABEL")), None)
            if text_col is None or emo_col is None:
                raise ValueError(
                    f"Cannot identify columns in {self.local_csv_path}.\n"
                    f"Found headers: {headers}\n"
                    f"Expected text column (SIT/SITUATION/TEXT) and emotion column (EMOT/EMOTION/LABEL)."
                )
            for row in reader:
                text = str(row[text_col]).strip()
                raw_emo = str(row[emo_col]).strip()
                emo = _LABEL_NORMALIZE.get(raw_emo, _LABEL_NORMALIZE.get(raw_emo.lower(), None))
                if emo is None or not text:
                    continue
                records.append({"text": text, "label": ISEAR_LABEL2ID[emo]})
        return records

    def load_splits(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split into train/val/test with a fixed seed (do not re-seed externally)."""
        all_records = self._load_all()
        rng = np.random.default_rng(self.split_seed)
        indices = rng.permutation(len(all_records))
        n = len(indices)
        n_test = max(1, int(n * self.test_ratio))
        n_val = max(1, int(n * self.val_ratio))
        test_idx = indices[:n_test]
        val_idx = indices[n_test:n_test + n_val]
        train_idx = indices[n_test + n_val:]
        return (
            [all_records[i] for i in train_idx],
            [all_records[i] for i in val_idx],
            [all_records[i] for i in test_idx],
        )
