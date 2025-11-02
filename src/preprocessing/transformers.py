"""
Preprocessing utilities for feature engineering and data transformation.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import CATEGORICAL_FEATURES, TARGET_COL


def sanity_check_data(df: pd.DataFrame) -> None:
    """Confirm no missing values and print basic stats."""
    print("\n=== Data Sanity Check ===")
    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"\nTarget ({TARGET_COL}) stats:")
    print(df[TARGET_COL].describe())


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list, list]:
    """
    Identify numeric and categorical features.
    Returns features DataFrame (without targets) and feature lists.
    """
    target_cols = ["heating_load", "cooling_load"]
    X = df.drop(columns=target_cols, errors="ignore")

    categorical_features = CATEGORICAL_FEATURES
    numeric_features = [col for col in X.columns if col not in categorical_features]

    return X, numeric_features, categorical_features


def preprocess_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric_features: list,
    categorical_features: list,
    scale_numeric: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler | None, OneHotEncoder]:
    """
    Preprocess features consistently across splits.
    Fit on train, transform val/test to avoid leakage.
    """
    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(X_train[categorical_features])

    X_train_cat = encoder.transform(X_train[categorical_features])
    X_val_cat = encoder.transform(X_val[categorical_features])
    X_test_cat = encoder.transform(X_test[categorical_features])

    # Standardize numeric features (optional, for linear models)
    scaler = None
    if scale_numeric:
        scaler = StandardScaler()
        X_train_num = scaler.fit_transform(X_train[numeric_features])
        X_val_num = scaler.transform(X_val[numeric_features])
        X_test_num = scaler.transform(X_test[numeric_features])
    else:
        X_train_num = X_train[numeric_features].values
        X_val_num = X_val[numeric_features].values
        X_test_num = X_test[numeric_features].values

    # Concatenate numeric + categorical
    X_train_processed = np.hstack([X_train_num, X_train_cat])
    X_val_processed = np.hstack([X_val_num, X_val_cat])
    X_test_processed = np.hstack([X_test_num, X_test_cat])

    return X_train_processed, X_val_processed, X_test_processed, scaler, encoder
