"""
Main training script for baseline models.

This script implements a clean ML workflow:
1. Load data and confirm features/targets
2. Create reproducible train/val/test splits
3. Preprocess consistently (avoid leakage)
4. Train baseline models
5. Track everything with MLflow
6. Generate evaluation metrics and plots
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .data import load_energy_efficiency
from .split import train_val_test_split_indices

# Configuration
RANDOM_SEED = 42
TRAIN_SIZE = 0.6
VAL_SIZE = 0.2
TEST_SIZE = 0.2
TARGET_COL = "heating_load"

# Create output directories
OUTPUT_DIR = Path("outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR = OUTPUT_DIR / "models"

for dir_path in [OUTPUT_DIR, PLOTS_DIR, METRICS_DIR, MODELS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)


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
    # Remove target columns
    target_cols = ["heating_load", "cooling_load"]
    X = df.drop(columns=target_cols, errors="ignore")

    # Identify categorical vs numeric features
    # Based on UCI docs: x6 (orientation) and x8 (glazing_area_distribution) are categorical
    categorical_features = ["x6", "x8"]
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
    # One-hot encode categorical features (always needed)
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


def train_classification_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    model,
    threshold: float,
) -> Dict:
    """Train classification model and return metrics."""
    with mlflow.start_run(run_name=f"classification_{model_name}_seed{RANDOM_SEED}"):
        # Log parameters
        mlflow.log_param("task", "classification")
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("classification_threshold", threshold)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_param("test_size", len(X_test))

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        # Metrics
        val_acc = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average="macro")
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average="macro")

        # Log metrics
        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("val_f1", val_f1)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1", test_f1)

        # Log class distribution
        train_dist = np.bincount(y_train)
        val_dist = np.bincount(y_val)
        test_dist = np.bincount(y_test)
        mlflow.log_param("train_class_dist", f"Low:{train_dist[0]}, High:{train_dist[1]}")
        mlflow.log_param("val_class_dist", f"Low:{val_dist[0]}, High:{val_dist[1]}")
        mlflow.log_param("test_class_dist", f"Low:{test_dist[0]}, High:{test_dist[1]}")

        print(f"\n{model_name} Results:")
        print(f"  Val  - Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
        print(f"  Test - Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")

        return {
            "model_name": model_name,
            "val_accuracy": val_acc,
            "val_f1": val_f1,
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "y_test": y_test,
            "y_test_pred": y_test_pred,
        }


def train_regression_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    model,
) -> Dict:
    """Train regression model and return metrics."""
    with mlflow.start_run(run_name=f"regression_{model_name}_seed{RANDOM_SEED}"):
        # Log parameters
        mlflow.log_param("task", "regression")
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_param("test_size", len(X_test))

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        # Metrics
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # Log metrics
        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_rmse", test_rmse)

        print(f"\n{model_name} Results:")
        print(f"  Val  - MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")
        print(f"  Test - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")

        return {
            "model_name": model_name,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "y_test": y_test,
            "y_test_pred": y_test_pred,
        }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> str:
    """Generate and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Low", "High"],
        yticklabels=["Low", "High"],
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    plot_path = PLOTS_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(plot_path)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> str:
    """Generate and save residuals plot."""
    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidth=0.5)
    ax1.axhline(y=0, color="r", linestyle="--", linewidth=2)
    ax1.set_xlabel("Predicted Heating Load")
    ax1.set_ylabel("Residuals")
    ax1.set_title(f"Residuals vs Predicted - {model_name}")
    ax1.grid(alpha=0.3)

    # Residuals histogram
    ax2.hist(residuals, bins=30, edgecolor="k", alpha=0.7)
    ax2.set_xlabel("Residuals")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Residuals Distribution")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = PLOTS_DIR / f"residuals_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(plot_path)


def plot_target_distribution(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> str:

    df = pd.DataFrame({
        "Label": np.concatenate([y_true, y_pred]),
        "Type": ["True"] * len(y_true) + ["Predicted"] * len(y_pred)
    })

    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x="Label", hue="Type")

    plt.title(f"Target Distribution {model_name}")
    plt.xlabel("Class")
    plt.ylabel("Count")

    plt.tight_layout()
    plot_path = PLOTS_DIR / f"target_distribution_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(plot_path)

def plot_correlation_heatmap(x_train: np.ndarray,numeric_features: list, model_name: str) -> str:
    df_num = pd.DataFrame(x_train, columns=numeric_features)
    corr = df_num.corr()

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True, cbar_kws={"shrink": 0.8})
    plt.title(f"Correlation Heatmap {model_name}")

    plt.tight_layout()
    plot_path = PLOTS_DIR / f"correlation_heatmap_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(plot_path)

def main():
    """Main training pipeline."""
    print("=" * 80)
    print("BASELINE MODELS TRAINING PIPELINE")
    print("=" * 80)

    # Set random seed
    np.random.seed(RANDOM_SEED)

    # 1. Load data and sanity check
    print("\n[1/11] Loading data...")
    df = load_energy_efficiency()
    sanity_check_data(df)

    # 2. Prepare features
    print("\n[2/11] Preparing features...")
    X, numeric_features, categorical_features = prepare_features(df)
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    # 3. Create classification labels from training set ONLY
    print("\n[3/11] Creating train/val/test splits...")
    # First, get initial splits to compute threshold on train only
    y_regression = df[TARGET_COL].values
    idx_train, idx_val, idx_test = train_val_test_split_indices(
        n_samples=len(df),
        y_for_stratify=None,  # Will stratify after computing threshold
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_seed=RANDOM_SEED,
    )

    # Compute threshold on TRAIN ONLY
    train_hl = df.iloc[idx_train][TARGET_COL].values
    threshold = float(np.median(train_hl))
    print(f"Classification threshold (train median): {threshold:.4f}")

    # Create classification labels using train-only threshold
    y_classification = (df[TARGET_COL].values >= threshold).astype(int)

    # Re-split with stratification on classification labels
    idx_train, idx_val, idx_test = train_val_test_split_indices(
        n_samples=len(df),
        y_for_stratify=y_classification,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_seed=RANDOM_SEED,
    )

    # Check class balance
    print(f"\nSplit sizes - Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")
    print(f"Train class dist: {np.bincount(y_classification[idx_train])}")
    print(f"Val class dist: {np.bincount(y_classification[idx_val])}")
    print(f"Test class dist: {np.bincount(y_classification[idx_test])}")

    # Split data
    X_train, X_val, X_test = X.iloc[idx_train], X.iloc[idx_val], X.iloc[idx_test]
    y_cls_train = y_classification[idx_train]
    y_cls_val = y_classification[idx_val]
    y_cls_test = y_classification[idx_test]
    y_reg_train = y_regression[idx_train]
    y_reg_val = y_regression[idx_val]
    y_reg_test = y_regression[idx_test]

    # Set MLflow experiment
    mlflow.set_experiment("energy-efficiency-baselines")

    # =========================================================================
    # CLASSIFICATION TASK
    # =========================================================================
    print("\n" + "=" * 80)
    print("CLASSIFICATION TASK: High/Low Heating Load")
    print("=" * 80)

    classification_results = []

    # Logistic Regression (with scaled features)
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

    # Determine best classification model
    best_clf = max(classification_results, key=lambda x: x["val_f1"])
    print(
        f"\n✅ Best Classification Model: {best_clf['model_name']} (Val F1: {best_clf['val_f1']:.4f})"
    )

    # Generate confusion matrix for best model
    print("\n[6/11] Generating confusion matrix...")
    cm_path = plot_confusion_matrix(
        best_clf["y_test"], best_clf["y_test_pred"], best_clf["model_name"]
    )
    print(f"Saved: {cm_path}")

    # Generate target distribution plot for best model
    print("\n[7/11] Generating target distribution plot...")
    target_dist_path = plot_target_distribution(best_clf["y_test"], best_clf["y_test_pred"],
                                                best_clf["model_name"])
    print(f"Saved: {target_dist_path}")

    # =========================================================================
    # REGRESSION TASK
    # =========================================================================
    print("\n" + "=" * 80)
    print("REGRESSION TASK: Heating Load (Continuous)")
    print("=" * 80)

    regression_results = []

    # Linear Regression (with scaled features)
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

    # Determine best regression model
    best_reg = min(regression_results, key=lambda x: x["val_rmse"])
    print(
        f"\n✅ Best Regression Model: {best_reg['model_name']} (Val RMSE: {best_reg['val_rmse']:.4f})"
    )

    # Generate residuals plot for best model
    print("\n[10/11] Generating residuals plot...")
    residuals_path = plot_residuals(
        best_reg["y_test"], best_reg["y_test_pred"], best_reg["model_name"]
    )
    print(f"Saved: {residuals_path}")

    # Generate correlation heatmap plot for best model
    print("\n[11/11] Generating correlation heatmap...")
    correlation_heatmap_path = plot_correlation_heatmap(X_train,numeric_features,best_reg["model_name"])
    print(f"Saved: {correlation_heatmap_path}")

    # =========================================================================
    # SAVE SUMMARY TABLES
    # =========================================================================
    print("\n" + "=" * 80)
    print("SAVING RESULTS SUMMARY")
    print("=" * 80)

    # Classification table
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
    print("\n📊 Classification Results:")
    print(clf_table.to_string(index=False))
    print(f"Saved: {clf_table_path}")

    # Regression table
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
    print("\n📊 Regression Results:")
    print(reg_table.to_string(index=False))
    print(f"Saved: {reg_table_path}")

    # Save configuration for reproducibility
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
    print(f"\n⚙️  Configuration saved: {config_path}")

    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
