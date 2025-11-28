"""
Neural network regression pipeline for energy efficiency prediction.
"""

from __future__ import annotations

import mlflow
import numpy as np

from ..config import RANDOM_SEED
from ..evaluation.visualization import plot_learning_curves
from ..models.nn_regressor import create_mlp_regressor
from ..preprocessing.transformers import preprocess_data
from sklearn.metrics import mean_absolute_error, mean_squared_error


def run_nn_regression_pipeline(
    X_train,
    X_val,
    X_test,
    y_reg_train: np.ndarray,
    y_reg_val: np.ndarray,
    y_reg_test: np.ndarray,
    numeric_features: list,
    categorical_features: list,
) -> dict:
    """
    Run the neural network regression pipeline.
    
    Args:
        X_train, X_val, X_test: Feature DataFrames for each split
        y_reg_train, y_reg_val, y_reg_test: Regression targets
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
    
    Returns:
        Dictionary containing model name, metrics, predictions, and learning curves
    """
    print("\n[NN Regression] Training Neural Network Regressor...")
    
    # Preprocess data with scaling (important for neural networks)
    X_train_processed, X_val_processed, X_test_processed, _, _ = preprocess_data(
        X_train, X_val, X_test, numeric_features, categorical_features, scale_numeric=True
    )
    
    # Create MLP regressor with tuned hyperparameters
    # Tuned via GridSearchCV: Best CV MAE = 0.4850
    model = create_mlp_regressor(
        hidden_layer_sizes=(64, 32),  # Tuned: 2 layers optimal
        learning_rate_init=0.001,     # Tuned: lower learning rate
        alpha=0.0001,                 # Tuned: lower regularization
        batch_size=32,                # Tuned: 32 batch size
        max_iter=500,
        random_state=RANDOM_SEED,
        early_stopping=True,
        validation_fraction=0.1,
    )
    
    # Train with MLflow tracking
    with mlflow.start_run(run_name=f"nn_regression_seed{RANDOM_SEED}"):
        # Log parameters
        mlflow.log_param("task", "regression")
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
        mlflow.log_param("train_size", len(X_train_processed))
        mlflow.log_param("val_size", len(X_val_processed))
        mlflow.log_param("test_size", len(X_test_processed))
        
        # Train model
        model.fit(X_train_processed, y_reg_train, X_val=X_val_processed, y_val=y_reg_val)
        
        # Log number of iterations used
        mlflow.log_param("n_iter_actual", model.n_iter_)
        
        # Predict
        y_val_pred = model.predict(X_val_processed)
        y_test_pred = model.predict(X_test_processed)
        
        # Calculate metrics
        val_mae = mean_absolute_error(y_reg_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_reg_val, y_val_pred))
        test_mae = mean_absolute_error(y_reg_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_reg_test, y_test_pred))
        
        # Log metrics
        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_rmse", test_rmse)
        
        # Get learning curves
        learning_curves = model.get_learning_curves()
        
        # Generate learning curve plot
        print("[NN Regression] Generating learning curves plot...")
        plot_path = plot_learning_curves(
            learning_curves['train_loss'],
            learning_curves['val_loss'],
            "Neural_Network",
            "regression"
        )
        mlflow.log_artifact(plot_path)
        
        print("\nNeural Network Results:")
        print(f"  Val  - MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")
        print(f"  Test - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
        print(f"  Converged in {model.n_iter_} iterations")
        
        return {
            "model_name": "Neural_Network",
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "y_test": y_reg_test,
            "y_test_pred": y_test_pred,
            "learning_curves": learning_curves,
            "n_iterations": model.n_iter_,
        }
