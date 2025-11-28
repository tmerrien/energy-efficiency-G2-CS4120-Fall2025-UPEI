"""
Neural network classification pipeline for energy efficiency prediction.
"""

from __future__ import annotations

import mlflow
import numpy as np

from ..config import RANDOM_SEED
from ..evaluation.visualization import plot_learning_curves
from ..models.nn_classifier import create_mlp_classifier
from ..preprocessing.transformers import preprocess_data
from sklearn.metrics import accuracy_score, f1_score


def run_nn_classification_pipeline(
    X_train,
    X_val,
    X_test,
    y_cls_train: np.ndarray,
    y_cls_val: np.ndarray,
    y_cls_test: np.ndarray,
    numeric_features: list,
    categorical_features: list,
    threshold: float,
) -> dict:
    """
    Run the neural network classification pipeline.
    
    Args:
        X_train, X_val, X_test: Feature DataFrames for each split
        y_cls_train, y_cls_val, y_cls_test: Classification labels
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        threshold: Classification threshold used
    
    Returns:
        Dictionary containing model name, metrics, predictions, and learning curves
    """
    print("\n[NN Classification] Training Neural Network Classifier...")
    
    # Preprocess data with scaling (important for neural networks)
    X_train_processed, X_val_processed, X_test_processed, _, _ = preprocess_data(
        X_train, X_val, X_test, numeric_features, categorical_features, scale_numeric=True
    )
    
    # Create MLP classifier with tuned hyperparameters
    # Tuned via GridSearchCV: Best CV F1 = 0.9576
    model = create_mlp_classifier(
        hidden_layer_sizes=(64,),  # Tuned: single layer performs best
        learning_rate_init=0.01,    # Tuned: higher learning rate
        alpha=0.0001,               # Tuned: lower regularization
        batch_size=32,              # Tuned: 32 batch size
        max_iter=500,
        random_state=RANDOM_SEED,
        early_stopping=True,
        validation_fraction=0.1,
    )
    
    # Train with MLflow tracking
    with mlflow.start_run(run_name=f"nn_classification_seed{RANDOM_SEED}"):
        # Log parameters
        mlflow.log_param("task", "classification")
        mlflow.log_param("model_type", "neural_network")
        mlflow.log_param("architecture", "MLP")
        mlflow.log_param("hidden_layers", str(model.hidden_layer_sizes))
        mlflow.log_param("activation", model.activation)
        mlflow.log_param("solver", model.solver)
        mlflow.log_param("learning_rate_init", model.learning_rate_init)
        mlflow.log_param("alpha", model.alpha)
        mlflow.log_param("batch_size", model.batch_size)
        mlflow.log_param("max_iter", model.max_iter)
        mlflow.log_param("early_stopping", model.early_stopping)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("classification_threshold", threshold)
        mlflow.log_param("train_size", len(X_train_processed))
        mlflow.log_param("val_size", len(X_val_processed))
        mlflow.log_param("test_size", len(X_test_processed))
        
        # Train model
        model.fit(X_train_processed, y_cls_train, X_val=X_val_processed, y_val=y_cls_val)
        
        # Log number of iterations used
        mlflow.log_param("n_iter_actual", model.n_iter_)
        
        # Predict
        y_val_pred = model.predict(X_val_processed)
        y_test_pred = model.predict(X_test_processed)
        
        # Calculate metrics
        val_acc = accuracy_score(y_cls_val, y_val_pred)
        val_f1 = f1_score(y_cls_val, y_val_pred, average="macro")
        test_acc = accuracy_score(y_cls_test, y_test_pred)
        test_f1 = f1_score(y_cls_test, y_test_pred, average="macro")
        
        # Log metrics
        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("val_f1", val_f1)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1", test_f1)
        
        # Log class distribution
        train_dist = np.bincount(y_cls_train)
        val_dist = np.bincount(y_cls_val)
        test_dist = np.bincount(y_cls_test)
        mlflow.log_param("train_class_dist", f"Low:{train_dist[0]}, High:{train_dist[1]}")
        mlflow.log_param("val_class_dist", f"Low:{val_dist[0]}, High:{val_dist[1]}")
        mlflow.log_param("test_class_dist", f"Low:{test_dist[0]}, High:{test_dist[1]}")
        
        # Get learning curves
        learning_curves = model.get_learning_curves()
        
        # Generate learning curve plot
        print("[NN Classification] Generating learning curves plot...")
        plot_path = plot_learning_curves(
            learning_curves['train_loss'],
            learning_curves['val_loss'],
            "Neural_Network",
            "classification"
        )
        mlflow.log_artifact(plot_path)
        
        print("\nNeural Network Results:")
        print(f"  Val  - Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
        print(f"  Test - Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
        print(f"  Converged in {model.n_iter_} iterations")
        
        return {
            "model_name": "Neural_Network",
            "val_accuracy": val_acc,
            "val_f1": val_f1,
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "y_test": y_cls_test,
            "y_test_pred": y_test_pred,
            "learning_curves": learning_curves,
            "n_iterations": model.n_iter_,
        }
