"""
Model training utilities for classification and regression tasks.
"""

from typing import Dict

import mlflow
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

from .config import RANDOM_SEED


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
