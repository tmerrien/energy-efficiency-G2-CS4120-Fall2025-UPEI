"""
Main training script for baseline models.

This script orchestrates the ML workflow:
1. Load data and prepare features
2. Create reproducible train/val/test splits
3. Train classification and regression baselines
4. Generate evaluation metrics and visualizations
5. Save results for reporting
"""

import json

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .config import (
    METRICS_DIR,
    RANDOM_SEED,
    TARGET_COL,
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
)
from .data import load_energy_efficiency
from .models import train_classification_model, train_regression_model
from .preprocessing import prepare_features, preprocess_data, sanity_check_data
from .split import train_val_test_split_indices
from .visualization import (
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_residuals,
    plot_target_distribution,
)


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("BASELINE MODELS TRAINING PIPELINE")
    print("=" * 80)

    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)

    # Step 1: Load and validate data
    print("\n[1/11] Loading data...")
    df = load_energy_efficiency()
    sanity_check_data(df)

    # Step 2: Prepare features
    print("\n[2/11] Preparing features...")
    X, numeric_features, categorical_features = prepare_features(df)
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    # Step 3: Create splits with stratification
    print("\n[3/11] Creating train/val/test splits...")
    y_regression = df[TARGET_COL].values

    # First split to compute threshold on train only
    idx_train, idx_val, idx_test = train_val_test_split_indices(
        n_samples=len(df),
        y_for_stratify=None,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_seed=RANDOM_SEED,
    )

    # Compute classification threshold from training set only
    train_hl = df.iloc[idx_train][TARGET_COL].values
    threshold = float(np.median(train_hl))
    print(f"Classification threshold (train median): {threshold:.4f}")

    # Create classification labels and re-split with stratification
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

    # Split data and targets
    X_train, X_val, X_test = X.iloc[idx_train], X.iloc[idx_val], X.iloc[idx_test]
    y_cls_train = y_classification[idx_train]
    y_cls_val = y_classification[idx_val]
    y_cls_test = y_classification[idx_test]
    y_reg_train = y_regression[idx_train]
    y_reg_val = y_regression[idx_val]
    y_reg_test = y_regression[idx_test]

    # Configure MLflow
    mlflow.set_experiment("energy-efficiency-baselines")

    # =========================================================================
    # CLASSIFICATION TASK
    # =========================================================================
    print("\n" + "=" * 80)
    print("CLASSIFICATION TASK: High/Low Heating Load")
    print("=" * 80)

    classification_results = []

    # Logistic Regression (scaled features)
    print("\n[4/11] Training Logistic Regression...")
    X_train_lr, X_val_lr, X_test_lr, _, _ = preprocess_data(
        X_train, X_val, X_test, numeric_features, categorical_features, scale_numeric=True
    )
    lr = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
    result_lr = train_classification_model(
        X_train_lr,
        y_cls_train,
        X_val_lr,
        y_cls_val,
        X_test_lr,
        y_cls_test,
        "Logistic_Regression",
        lr,
        threshold,
    )
    classification_results.append(result_lr)

    # Decision Tree (no scaling needed)
    print("\n[5/11] Training Decision Tree Classifier...")
    X_train_dt, X_val_dt, X_test_dt, _, _ = preprocess_data(
        X_train, X_val, X_test, numeric_features, categorical_features, scale_numeric=False
    )
    dt_clf = DecisionTreeClassifier(random_state=RANDOM_SEED, max_depth=10, min_samples_split=10)
    result_dt_clf = train_classification_model(
        X_train_dt,
        y_cls_train,
        X_val_dt,
        y_cls_val,
        X_test_dt,
        y_cls_test,
        "Decision_Tree",
        dt_clf,
        threshold,
    )
    classification_results.append(result_dt_clf)

    # Select best classification model
    best_clf = max(classification_results, key=lambda x: x["val_f1"])
    print(
        f"\nBest Classification Model: {best_clf['model_name']} (Val F1: {best_clf['val_f1']:.4f})"
    )

    # Generate classification visualizations
    print("\n[6/11] Generating confusion matrix...")
    cm_path = plot_confusion_matrix(
        best_clf["y_test"], best_clf["y_test_pred"], best_clf["model_name"]
    )
    print(f"Saved: {cm_path}")

    print("\n[7/11] Generating target distribution plot...")
    td_path = plot_target_distribution(
        best_clf["y_test"], best_clf["y_test_pred"], best_clf["model_name"]
    )
    print(f"Saved: {td_path}")

    # =========================================================================
    # REGRESSION TASK
    # =========================================================================
    print("\n" + "=" * 80)
    print("REGRESSION TASK: Heating Load (Continuous)")
    print("=" * 80)

    regression_results = []

    # Linear Regression (scaled features)
    print("\n[8/11] Training Linear Regression...")
    X_train_linreg, X_val_linreg, X_test_linreg, _, _ = preprocess_data(
        X_train, X_val, X_test, numeric_features, categorical_features, scale_numeric=True
    )
    linreg = LinearRegression()
    result_linreg = train_regression_model(
        X_train_linreg,
        y_reg_train,
        X_val_linreg,
        y_reg_val,
        X_test_linreg,
        y_reg_test,
        "Linear_Regression",
        linreg,
    )
    regression_results.append(result_linreg)

    # Decision Tree Regressor (no scaling needed)
    print("\n[9/11] Training Decision Tree Regressor...")
    X_train_dt_reg, X_val_dt_reg, X_test_dt_reg, _, _ = preprocess_data(
        X_train, X_val, X_test, numeric_features, categorical_features, scale_numeric=False
    )
    dt_reg = DecisionTreeRegressor(random_state=RANDOM_SEED, max_depth=10, min_samples_split=10)
    result_dt_reg = train_regression_model(
        X_train_dt_reg,
        y_reg_train,
        X_val_dt_reg,
        y_reg_val,
        X_test_dt_reg,
        y_reg_test,
        "Decision_Tree_Regressor",
        dt_reg,
    )
    regression_results.append(result_dt_reg)

    # Select best regression model
    best_reg = min(regression_results, key=lambda x: x["val_rmse"])
    print(
        f"\nBest Regression Model: {best_reg['model_name']} (Val RMSE: {best_reg['val_rmse']:.4f})"
    )

    # Generate regression visualizations
    print("\n[10/11] Generating residuals plot...")
    residuals_path = plot_residuals(
        best_reg["y_test"], best_reg["y_test_pred"], best_reg["model_name"]
    )
    print(f"Saved: {residuals_path}")

    print("\n[11/11] Generating correlation heatmap...")
    heatmap_path = plot_correlation_heatmap(
        X_train[numeric_features].values, numeric_features, best_reg["model_name"]
    )
    print(f"Saved: {heatmap_path}")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("SAVING RESULTS SUMMARY")
    print("=" * 80)

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

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
