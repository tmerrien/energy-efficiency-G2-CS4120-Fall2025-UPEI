"""
Utilities for saving training results and metrics.
"""

import json

import pandas as pd

from ..config import METRICS_DIR, RANDOM_SEED, TARGET_COL, TEST_SIZE, TRAIN_SIZE, VAL_SIZE


def save_results(
    classification_results: list,
    regression_results: list,
    threshold: float,
    numeric_features: list,
    categorical_features: list,
):
    """Save all training results to disk."""
    print("\nSAVING RESULTS SUMMARY")

    # Classification results table
    clf_table = pd.DataFrame(
        [
            {
                "Model": r["model_name"],
                "Val_Accuracy": f"{r['val_accuracy']:.4f}",
                "Val_F1": f"{r['val_f1']:.4f}",
                "Test_Accuracy": f"{r['test_accuracy']:.4f}",
                "Test_F1": f"{r['test_f1']:.4f}",
            }
            for r in classification_results
        ]
    )
    clf_table_path = METRICS_DIR / "classification_results.csv"
    clf_table.to_csv(clf_table_path, index=False)
    print("\nClassification Results:")
    print(clf_table.to_string(index=False))
    print(f"Saved: {clf_table_path}")

    # Regression results table
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
    print("\nRegression Results:")
    print(reg_table.to_string(index=False))
    print(f"Saved: {reg_table_path}")

    # Save configuration
    config = {
        "random_seed": RANDOM_SEED,
        "train_size": float(TRAIN_SIZE),
        "val_size": float(VAL_SIZE),
        "test_size": float(TEST_SIZE),
        "classification_threshold": threshold,
        "target_column": TARGET_COL,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }
    config_path = METRICS_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfiguration saved: {config_path}")
