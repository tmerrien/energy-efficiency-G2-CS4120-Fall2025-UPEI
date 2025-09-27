from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

# --------- Regression target ---------

def get_regression_target(
        df: pd.DataFrame,
        target_col: str = "heating_load",
) -> np.ndarray:
    """

    Extract a continuous regression target from the dataset

    :param df: Dataframe containing the Energy Efficiency dataset
    :param target_col: Name of the column to use as the regression target
    :return: NumPy array of continuous target values
    :raises KeyError: if target_col is not in the DataFrame
    :raises ValueError: if target_col contains missing values
    """
    if target_col not in df.columns:
        raise KeyError(f"{target_col} not in DataFrame")

    y = df[target_col].to_numpy(dtype=float)

    if np.isnan(y).any():
        raise ValueError(f"{target_col} contains NaN values")

    return y

# --------- Classification labels (High/Low via threshold) ---------

@dataclass(frozen=True)
class ClasLabelInfo:
    threshold: float
    positive_label: int = 1
    negative_label: int = 0
    positive_name: str = "High"
    negative_name: str = "Low"

def make_classification_labels_from_hl(
        df: pd.DataFrame,
        target_col: str = "heating_load",
        threshold: Optional[float] = None,
) -> Tuple[np.ndarray, ClasLabelInfo]:
    """

    Create binary classification labels from continuous Heating Load.

    :param df: DataFrame containing the Energy Efficiency dataset.
    :param target_col: Column name of the continuous target to binarize.
    :param threshold: Value used to split High vs Low; if None, the median is used.
    :return: Tuple of (labels as NumPy array, ClassLabelInfo with threshold metadata).
    """
    y = get_regression_target(df, target_col=target_col)

    thr = float(np.median(y)) if threshold is None else float(threshold)
    y_cls = (y >= thr).astype(int)

    info = ClasLabelInfo(threshold=thr)
    return y_cls, info

def class_distribution(y_cls: np.ndarray) -> Dict[int, int]:
    """

    Compute class distribution across splits.

    :param y_cls: NumPy array of continuous target values.
    :return: Dictionary mapping class label to number of instances.
    """
    unique, counts = np.unique(y_cls, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts)}