"""
Reproducible seed management for the 5-seed protocol.

Canonical seeds: [42, 123, 456, 789, 1024]
All experiments use these seeds in this order.
"""
import os
import random

import numpy as np

CANONICAL_SEEDS: list[int] = [42, 123, 456, 789, 1024]


def set_seed(seed: int) -> None:
    """
    Fix all random sources for reproducibility.
    Call BEFORE any stochastic operation (model init, data shuffle, etc.)
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
