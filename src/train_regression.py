"""
Regression model training script.

Trains both baseline models and neural networks for the regression task.
"""

import mlflow
import numpy as np

from .config import RANDOM_SEED, TARGET_COL, TEST_SIZE, VAL_SIZE
from .data.loader import load_energy_efficiency
from .data.splitter import train_val_test_split_indices
from .pipelines.regression import run_regression_pipeline
from .pipelines.nn_regression import run_nn_regression_pipeline
from .preprocessing.transformers import prepare_features, sanity_check_data


def main():
    """Regression training orchestrator."""
    print("=" * 80)
    print("REGRESSION MODELS TRAINING PIPELINE")
    print("=" * 80)

    np.random.seed(RANDOM_SEED)

    # Load and prepare data
    print("\n[1/5] Loading data...")
    df = load_energy_efficiency()
    sanity_check_data(df)

    print("\n[2/5] Preparing features...")
    X, numeric_features, categorical_features = prepare_features(df)
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    # Create stratified splits
    print("\n[3/5] Creating train/val/test splits...")
    y_regression = df[TARGET_COL].values

    # Compute threshold on train set for stratification
    idx_train, idx_val, idx_test = train_val_test_split_indices(
        n_samples=len(df),
        y_for_stratify=None,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_seed=RANDOM_SEED,
    )

    train_hl = df.iloc[idx_train][TARGET_COL].values
    threshold = float(np.median(train_hl))

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

    # Prepare data splits
    X_train, X_val, X_test = X.iloc[idx_train], X.iloc[idx_val], X.iloc[idx_test]
    y_reg_train, y_reg_val, y_reg_test = (
        y_regression[idx_train],
        y_regression[idx_val],
        y_regression[idx_test],
    )

    # Configure MLflow
    mlflow.set_experiment("energy-efficiency-regression")

    # Train baseline models
    print("\n[4/5] Training baseline regression models...")
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

    # Train neural network
    print("\n[5/5] Training neural network regressor...")
    nn_result = run_nn_regression_pipeline(
        X_train,
        X_val,
        X_test,
        y_reg_train,
        y_reg_val,
        y_reg_test,
        numeric_features,
        categorical_features,
    )
    regression_results.append(nn_result)

    # Save results
    from pathlib import Path
    import pandas as pd
    
    OUTPUT_DIR = Path("outputs")
    METRICS_DIR = OUTPUT_DIR / "metrics"
    METRICS_DIR.mkdir(exist_ok=True, parents=True)
    
    reg_table = pd.DataFrame(
        [
            {
                "Model": r["model_name"],
                "Val_MAE": f"{r['val_mae']:.4f}",
                "Val_RMSE": f"{r['val_rmse']:.4f}",
                "Test_MAE": f"{r['test_mae']:.4f}",
                "Test_RMSE": f"{r['test_rmse']:.4f}",
            }
            for r in regression_results
        ]
    )
    reg_table_path = METRICS_DIR / "regression_results.csv"
    reg_table.to_csv(reg_table_path, index=False)
    
    print("\n" + "=" * 80)
    print("REGRESSION TRAINING COMPLETE")
    print("=" * 80)
    print("\nResults:")
    print(reg_table.to_string(index=False))


if __name__ == "__main__":
    main()
