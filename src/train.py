"""
Main training orchestrator for baseline models.

High-level workflow:
1. Load and prepare data
2. Create train/val/test splits with stratification
3. Run classification pipeline
4. Run regression pipeline
5. Save all results
"""

import mlflow
import numpy as np

from .classification_pipeline import run_classification_pipeline
from .config import RANDOM_SEED, TARGET_COL, TEST_SIZE, VAL_SIZE
from .data import load_energy_efficiency
from .preprocessing import prepare_features, sanity_check_data
from .regression_pipeline import run_regression_pipeline
from .results import save_results
from .split import train_val_test_split_indices


def main():
    """Main training orchestrator."""
    print("BASELINE MODELS TRAINING PIPELINE")

    np.random.seed(RANDOM_SEED)

    # Step 1-2: Load and prepare data
    print("\n[1/11] Loading data...")
    df = load_energy_efficiency()
    sanity_check_data(df)

    print("\n[2/11] Preparing features...")
    X, numeric_features, categorical_features = prepare_features(df)
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    # Step 3: Create stratified splits
    print("\n[3/11] Creating train/val/test splits...")
    y_regression = df[TARGET_COL].values

    # Compute threshold on train set only
    idx_train, idx_val, idx_test = train_val_test_split_indices(
        n_samples=len(df),
        y_for_stratify=None,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_seed=RANDOM_SEED,
    )

    train_hl = df.iloc[idx_train][TARGET_COL].values
    threshold = float(np.median(train_hl))
    print(f"Classification threshold (train median): {threshold:.4f}")

    # Re-split with stratification
    y_classification = (df[TARGET_COL].values >= threshold).astype(int)
    idx_train, idx_val, idx_test = train_val_test_split_indices(
        n_samples=len(df),
        y_for_stratify=y_classification,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_seed=RANDOM_SEED,
    )

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

    # Configure MLflow
    mlflow.set_experiment("energy-efficiency-baselines")

    # Step 4-7: Classification pipeline
    classification_results = run_classification_pipeline(
        X_train,
        X_val,
        X_test,
        y_cls_train,
        y_cls_val,
        y_cls_test,
        numeric_features,
        categorical_features,
        threshold,
    )

    # Step 8-11: Regression pipeline
    regression_results = run_regression_pipeline(
        X_train,
        X_val,
        X_test,
        y_reg_train,
        y_reg_val,
        y_reg_test,
        numeric_features,
        categorical_features,
    )

    # Save all results
    save_results(
        classification_results,
        regression_results,
        threshold,
        numeric_features,
        categorical_features,
    )

    print("\nTRAINING COMPLETE")


if __name__ == "__main__":
    main()
