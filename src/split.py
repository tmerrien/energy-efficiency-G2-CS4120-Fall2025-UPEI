from __future__ import annotations
from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

def train_val_test_split_indices(
        n_samples: int,
        y_for_stratify: np.ndarray | None = None,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    Split indices into train, validation, and test sets.

    :param n_samples: Total number of samples in the dataset.
    :param y_for_stratify: Optional labels to preserve class distribution across splits.
    :param test_size: Proportion of the dataset to include in the test split.
    :param val_size: Proportion of the dataset to include in the validation split.
    :param random_seed: Random seed for reproducibility.
    :return: Three NumPy arrays containing indices for train, validation, and test sets.
    """
    idx = np.arange(n_samples)

    # train+val vs. test
    idx_trainval, idx_test = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_seed,
        stratify=y_for_stratify if y_for_stratify is not None else None,
    )

    # train vs. val (relative val size)
    rel_val = val_size / (1.0 - test_size)
    stratify_trainval = None
    if y_for_stratify is not None:
        stratify_trainval = y_for_stratify[idx_trainval]

    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=rel_val,
        random_state=random_seed,
        stratify=stratify_trainval,
    )

    return idx_train, idx_val, idx_test