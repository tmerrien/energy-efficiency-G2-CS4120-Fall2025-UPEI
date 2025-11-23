"""
Neural network classifier for energy efficiency prediction.
"""

from __future__ import annotations

from typing import Dict, Tuple

from sklearn.neural_network import MLPClassifier


class MLPClassifierWithHistory(MLPClassifier):
    """
    MLP Classifier that tracks training history for learning curves.
    
    Extends sklearn's MLPClassifier to store loss history during training.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.train_loss_history = []
        self.val_loss_history = []
    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit the model and track loss history.
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        # Reset history
        self.train_loss_history = []
        self.val_loss_history = []
        
        # Fit the model
        super().fit(X, y)
        
        # Store training loss history
        self.train_loss_history = list(self.loss_curve_)
        
        # Compute validation loss if validation data provided
        if X_val is not None and y_val is not None:
            from sklearn.metrics import log_loss
            y_val_pred_proba = self.predict_proba(X_val)
            val_loss = log_loss(y_val, y_val_pred_proba)
            # Replicate val loss for each epoch (approximation)
            self.val_loss_history = [val_loss] * len(self.train_loss_history)
        
        return self
    
    def get_learning_curves(self) -> Dict[str, list]:
        """
        Get training and validation loss histories.
        
        Returns:
            Dictionary with 'train_loss' and 'val_loss' keys
        """
        return {
            'train_loss': self.train_loss_history,
            'val_loss': self.val_loss_history if self.val_loss_history else None
        }


def create_mlp_classifier(
    hidden_layer_sizes: Tuple[int, ...] = (100,),
    learning_rate_init: float = 0.001,
    alpha: float = 0.0001,
    batch_size: int = 32,
    max_iter: int = 500,
    random_state: int = 42,
    early_stopping: bool = True,
    validation_fraction: float = 0.1,
) -> MLPClassifierWithHistory:
    """
    Create an MLP classifier with sensible defaults for the energy efficiency dataset.
    
    Args:
        hidden_layer_sizes: Tuple of hidden layer sizes
        learning_rate_init: Initial learning rate
        alpha: L2 regularization parameter
        batch_size: Batch size for training
        max_iter: Maximum number of iterations
        random_state: Random seed for reproducibility
        early_stopping: Whether to use early stopping
        validation_fraction: Fraction of training data for validation in early stopping
    
    Returns:
        Configured MLPClassifierWithHistory instance
    """
    return MLPClassifierWithHistory(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        alpha=alpha,
        batch_size=batch_size,
        learning_rate='adaptive',
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        shuffle=True,
        random_state=random_state,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        n_iter_no_change=10,
        verbose=False,
    )
