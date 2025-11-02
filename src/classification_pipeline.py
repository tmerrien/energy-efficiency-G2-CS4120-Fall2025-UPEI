"""
Classification pipeline for High/Low Heating Load prediction.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_SEED
from .models import train_classification_model
from .preprocessing import preprocess_data
from .visualization import plot_confusion_matrix, plot_target_distribution


def run_classification_pipeline(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_cls_train,
    y_cls_val,
    y_cls_test,
    numeric_features: list,
    categorical_features: list,
    threshold: float,
) -> list:
    """
    Run complete classification pipeline.

    Returns:
        List of classification results dictionaries
    """
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

    # Select best model
    best_clf = max(classification_results, key=lambda x: x["val_f1"])
    print(
        f"\nBest Classification Model: {best_clf['model_name']} (Val F1: {best_clf['val_f1']:.4f})"
    )

    # Generate visualizations
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

    return classification_results
