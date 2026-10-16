"""
Abstract base class for all emotion dataset loaders.

Concrete loaders must:
  1. Define label_names, n_labels, is_multilabel as properties
  2. Implement load_splits() returning (train, val, test) record lists
  3. Hard-code n_labels from the dataset definition (never infer from data)

Record format:
  {'text': str, 'label': int (single-label) or np.ndarray (multi-label multi-hot)}
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    """
    Generic PyTorch Dataset wrapper for emotion classification.

    Args:
        encodings: dict of tokenizer outputs; each value a tensor (N, seq_len)
        labels: tensor (N,) for single-label or (N, C) for multi-label
    """

    def __init__(self, encodings: Dict[str, torch.Tensor], labels: torch.Tensor) -> None:
        self.encodings = encodings
        self.labels = labels
        assert all(
            v.shape[0] == labels.shape[0] for v in encodings.values()
        ), "Encoding and label batch sizes must match"

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class BaseEmotionLoader(ABC):
    """Abstract loader interface for emotion datasets."""

    @property
    @abstractmethod
    def label_names(self) -> List[str]:
        """Ordered list of label name strings (from the dataset definition)."""
        ...

    @property
    @abstractmethod
    def n_labels(self) -> int:
        """Number of classes (fixed by the dataset definition)."""
        ...

    @property
    @abstractmethod
    def is_multilabel(self) -> bool:
        """True for multi-label (BCE loss), False for single-label (CE loss)."""
        ...

    @abstractmethod
    def load_splits(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Load train / validation / test splits.

        Returns (train_records, val_records, test_records); each record is
        {'text': str, 'label': int | np.ndarray}.
        """
        ...
