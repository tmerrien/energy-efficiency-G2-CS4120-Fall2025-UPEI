"""
Shared data preparation utilities to avoid code duplication across training scripts.
"""

import numpy as np

from ..config import RANDOM_SEED, TARGET_COL, TEST_SIZE, VAL_SIZE
from ..data.loader import load_energy_efficiency
from ..data.splitter import train_val_test_split_indices
from ..preprocessing.transformers import prepare_features, sanity_check_data


def load_and_split_data(verbose=True):
    """
    Load dataset and create stratified train/val/test splits.
    
    This function contains the common data preparation logic used across
    all training scripts to avoid duplication.
    
    Args:
        verbose: Whether to print progress messages
    
    Returns:
        Tuple containing:
        - X_train, X_val, X_test: Feature dataframes
        - y_cls_train, y_cls_val, y_cls_test: Classification labels
        - y_reg_train, y_reg_val, y_reg_test: Regression targets
        - threshold: Classification threshold (train median)
        - numeric_features: List of numeric feature names
        - categorical_features: List of categorical feature names
    """
    if verbose:
        print("\nLoading data...")
    
    df = load_energy_efficiency()
    
    if verbose:
        sanity_check_data(df)
        print("\nPreparing features...")
    
    X, numeric_features, categorical_features = prepare_features(df)
    
    if verbose:
        print(f"Numeric features: {numeric_features}")
        print(f"Categorical features: {categorical_features}")
        print("\nCreating train/val/test splits...")
    
    y_regression = df[TARGET_COL].values
    
    # First split to compute threshold on train set only
    idx_train, idx_val, idx_test = train_val_test_split_indices(
        n_samples=len(df),
        y_for_stratify=None,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_seed=RANDOM_SEED,
    )
    
    train_hl = df.iloc[idx_train][TARGET_COL].values
    threshold = float(np.median(train_hl))
    
    if verbose:
        print(f"Classification threshold (train median): {threshold:.4f}")
    
    # Re-split with stratification on classification labels
    y_classification = (df[TARGET_COL].values >= threshold).astype(int)
    idx_train, idx_val, idx_test = train_val_test_split_indices(
        n_samples=len(df),
        y_for_stratify=y_classification,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_seed=RANDOM_SEED,
    )
    
    if verbose:
        print(f"\nSplit sizes - Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")
        print(f"Train class dist: {np.bincount(y_classification[idx_train])}")
        print(f"Val class dist: {np.bincount(y_classification[idx_val])}")
        print(f"Test class dist: {np.bincount(y_classification[idx_test])}")
    
    # Prepare data splits
    X_train, X_val, X_test = X.iloc[idx_train], X.iloc[idx_val], X.iloc[idx_test]
    y_cls_train, y_cls_val, y_cls_test = (
        y_classification[idx_train],
        y_classification[idx_val],
        y_classification[idx_test],
    )
    y_reg_train, y_reg_val, y_reg_test = (
        y_regression[idx_train],
        y_regression[idx_val],
        y_regression[idx_test],
    )
    
    return (
        X_train, X_val, X_test,
        y_cls_train, y_cls_val, y_cls_test,
        y_reg_train, y_reg_val, y_reg_test,
        threshold,
        numeric_features,
        categorical_features,
    )
