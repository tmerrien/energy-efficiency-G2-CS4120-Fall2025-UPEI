"""
Visualization utilities for generating plots and figures.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from ..config import PLOTS_DIR


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
    """Generate and save target distribution plot."""
    df = pd.DataFrame(
        {
            "Label": np.concatenate([y_true, y_pred]),
            "Type": ["True"] * len(y_true) + ["Predicted"] * len(y_pred),
        }
    )

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


def plot_correlation_heatmap(x_train: np.ndarray, numeric_features: list, model_name: str) -> str:
    """Generate and save correlation heatmap."""
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


def plot_learning_curves(train_loss: list, val_loss: list, model_name: str, task: str) -> str:
    """
    Generate and save learning curve plot showing training and validation loss over epochs.
    
    Args:
        train_loss: Training loss values per epoch
        val_loss: Validation loss values per epoch
        model_name: Name of the model for the plot title
        task: Either 'classification' or 'regression'
    
    Returns:
        Path to the saved plot
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_loss) + 1)
    
    plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    if val_loss:
        plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Learning Curves - {model_name} ({task.capitalize()})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add final loss values as text
    final_train_loss = train_loss[-1]
    text_str = f'Final Training Loss: {final_train_loss:.4f}'
    if val_loss:
        final_val_loss = val_loss[-1]
        text_str += f'\nFinal Validation Loss: {final_val_loss:.4f}'
    
    plt.text(0.02, 0.98, text_str, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plot_path = PLOTS_DIR / f"learning_curves_{task}_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    return str(plot_path)
