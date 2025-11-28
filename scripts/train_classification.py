"""
Classification model training script.

Trains both baseline models and neural networks for the classification task.
"""

import mlflow
import numpy as np

from src.config import RANDOM_SEED, TARGET_COL, TEST_SIZE, VAL_SIZE
from src.data.loader import load_energy_efficiency
from src.data.splitter import train_val_test_split_indices
from src.pipelines.classification import run_classification_pipeline
from src.pipelines.nn_classification import run_nn_classification_pipeline
from src.preprocessing.transformers import prepare_features, sanity_check_data


def main():
    """Classification training orchestrator."""
    print("=" * 80)
    print("CLASSIFICATION MODELS TRAINING PIPELINE")
    print("=" * 80)

    np.random.seed(RANDOM_SEED)

    # Load and prepare data
    print("\n[1/5] Loading data...")
    df = load_energy_efficiency()
    sanity_check_data(df)

    print("\n[2/5] Preparing features...")
    X, numeric_features, categorical_features = prepare_features(df)
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    # Create stratified splits
    print("\n[3/5] Creating train/val/test splits...")

    # Compute threshold on train set only
    idx_train, idx_val, idx_test = train_val_test_split_indices(
        n_samples=len(df),
        y_for_stratify=None,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_seed=RANDOM_SEED,
    )

    train_hl = df.iloc[idx_train][TARGET_COL].values
    threshold = float(np.median(train_hl))
    print(f"Classification threshold (train median): {threshold:.4f}")

    # Re-split with stratification
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

    # Prepare data splits
    X_train, X_val, X_test = X.iloc[idx_train], X.iloc[idx_val], X.iloc[idx_test]
    y_cls_train, y_cls_val, y_cls_test = (
        y_classification[idx_train],
        y_classification[idx_val],
        y_classification[idx_test],
    )

    # Configure MLflow
    mlflow.set_experiment("energy-efficiency-classification")

    # Train baseline models
    print("\n[4/5] Training baseline classification models...")
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

    # Train neural network
    print("\n[5/5] Training neural network classifier...")
    nn_result = run_nn_classification_pipeline(
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
    classification_results.append(nn_result)

    # Save results
    from pathlib import Path
    import pandas as pd
    
    OUTPUT_DIR = Path("outputs")
    METRICS_DIR = OUTPUT_DIR / "metrics"
    METRICS_DIR.mkdir(exist_ok=True, parents=True)
    
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
    
    print("\n" + "=" * 80)
    print("CLASSIFICATION TRAINING COMPLETE")
    print("=" * 80)
    print("\nResults:")
    print(clf_table.to_string(index=False))


if __name__ == "__main__":
    main()
