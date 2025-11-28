"""
Main training orchestrator for baseline models.

High-level workflow:
1. Load and prepare data
2. Create train/val/test splits with stratification
3. Run classification pipeline
4. Run regression pipeline
5. Save all results
"""

import mlflow
import numpy as np

from src.config import RANDOM_SEED
from src.evaluation.results import save_results
from src.pipelines.classification import run_classification_pipeline
from src.pipelines.regression import run_regression_pipeline
from src.data.preparation import load_and_split_data


def main():
    """Main training orchestrator."""
    print("BASELINE MODELS TRAINING PIPELINE")

    np.random.seed(RANDOM_SEED)

    # Load and prepare data (shared function to avoid duplication)
    (X_train, X_val, X_test,
     y_cls_train, y_cls_val, y_cls_test,
     y_reg_train, y_reg_val, y_reg_test,
     threshold,
     numeric_features,
     categorical_features) = load_and_split_data(verbose=True)

    # Configure MLflow
    mlflow.set_experiment("energy-efficiency-baselines")

    # Step 4-7: Classification pipeline
    classification_results = run_classification_pipeline(
        X_train,
        X_val,
        X_test,
        y_cls_train,
        y_cls_val,
        y_cls_test,
        numeric_features,
        categorical_features,
        threshold,
    )

    # Step 8-11: Regression pipeline
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

    # Save all results
    save_results(
        classification_results,
        regression_results,
        threshold,
        numeric_features,
        categorical_features,
    )

    print("\nTRAINING COMPLETE")


if __name__ == "__main__":
    main()
