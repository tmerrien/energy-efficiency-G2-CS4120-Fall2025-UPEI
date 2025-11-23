"""
Hyperparameter tuning script for neural network models.

Uses GridSearchCV to find optimal hyperparameters for both classification
and regression neural network models.
"""

import json
from pathlib import Path

import mlflow
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor

from .config import RANDOM_SEED, TARGET_COL, TEST_SIZE, VAL_SIZE
from .data.loader import load_energy_efficiency
from .data.splitter import train_val_test_split_indices
from .preprocessing.transformers import prepare_features, preprocess_data, sanity_check_data


def tune_classification_nn():
    """
    Tune hyperparameters for classification neural network.
    
    Returns:
        Dictionary with best parameters and best score
    """
    print("=" * 80)
    print("TUNING CLASSIFICATION NEURAL NETWORK")
    print("=" * 80)
    
    # Load and prepare data
    print("\n[1/3] Loading and preparing data...")
    df = load_energy_efficiency()
    sanity_check_data(df)
    
    X, numeric_features, categorical_features = prepare_features(df)
    y_regression = df[TARGET_COL].values
    
    # Create splits
    idx_train, idx_val, idx_test = train_val_test_split_indices(
        n_samples=len(df),
        y_for_stratify=None,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_seed=RANDOM_SEED,
    )
    
    train_hl = df.iloc[idx_train][TARGET_COL].values
    threshold = float(np.median(train_hl))
    
    y_classification = (df[TARGET_COL].values >= threshold).astype(int)
    idx_train, idx_val, idx_test = train_val_test_split_indices(
        n_samples=len(df),
        y_for_stratify=y_classification,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_seed=RANDOM_SEED,
    )
    
    X_train, X_val, X_test = X.iloc[idx_train], X.iloc[idx_val], X.iloc[idx_test]
    y_cls_train = y_classification[idx_train]
    
    # Preprocess (combine train+val for GridSearchCV)
    X_train_processed, X_val_processed, _, _, _ = preprocess_data(
        X_train, X_val, X_test, numeric_features, categorical_features, scale_numeric=True
    )
    
    # Combine train and val for GridSearchCV (it will do its own CV split)
    X_trainval = np.vstack([X_train_processed, X_val_processed])
    y_trainval = np.concatenate([y_cls_train, y_classification[idx_val]])
    
    # Define parameter grid
    param_grid = {
        'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
        'learning_rate_init': [0.001, 0.01],
        'alpha': [0.0001, 0.001, 0.01],
        'batch_size': [32, 64],
    }
    
    print(f"\n[2/3] Running GridSearchCV...")
    print(f"Parameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    print(f"\nTotal combinations: {np.prod([len(v) for v in param_grid.values()])}")
    
    # Create base model
    base_model = MLPClassifier(
        activation='relu',
        solver='adam',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=RANDOM_SEED,
        verbose=False,
    )
    
    # Run GridSearchCV
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=3,  # 3-fold cross-validation
        scoring='f1_macro',
        n_jobs=-1,  # Use all CPU cores
        verbose=2,
    )
    
    grid_search.fit(X_trainval, y_trainval)
    
    print(f"\n[3/3] Results:")
    print(f"Best score (CV F1): {grid_search.best_score_:.4f}")
    print(f"Best parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': {
            'mean_test_scores': grid_search.cv_results_['mean_test_score'].tolist(),
            'std_test_scores': grid_search.cv_results_['std_test_score'].tolist(),
            'params': [str(p) for p in grid_search.cv_results_['params']],
        }
    }


def tune_regression_nn():
    """
    Tune hyperparameters for regression neural network.
    
    Returns:
        Dictionary with best parameters and best score
    """
    print("\n" + "=" * 80)
    print("TUNING REGRESSION NEURAL NETWORK")
    print("=" * 80)
    
    # Load and prepare data
    print("\n[1/3] Loading and preparing data...")
    df = load_energy_efficiency()
    X, numeric_features, categorical_features = prepare_features(df)
    y_regression = df[TARGET_COL].values
    
    # Create splits
    idx_train, idx_val, idx_test = train_val_test_split_indices(
        n_samples=len(df),
        y_for_stratify=None,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_seed=RANDOM_SEED,
    )
    
    train_hl = df.iloc[idx_train][TARGET_COL].values
    threshold = float(np.median(train_hl))
    
    y_classification = (df[TARGET_COL].values >= threshold).astype(int)
    idx_train, idx_val, idx_test = train_val_test_split_indices(
        n_samples=len(df),
        y_for_stratify=y_classification,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_seed=RANDOM_SEED,
    )
    
    X_train, X_val, X_test = X.iloc[idx_train], X.iloc[idx_val], X.iloc[idx_test]
    y_reg_train = y_regression[idx_train]
    
    # Preprocess (combine train+val for GridSearchCV)
    X_train_processed, X_val_processed, _, _, _ = preprocess_data(
        X_train, X_val, X_test, numeric_features, categorical_features, scale_numeric=True
    )
    
    # Combine train and val for GridSearchCV
    X_trainval = np.vstack([X_train_processed, X_val_processed])
    y_trainval = np.concatenate([y_reg_train, y_regression[idx_val]])
    
    # Define parameter grid
    param_grid = {
        'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
        'learning_rate_init': [0.001, 0.01],
        'alpha': [0.0001, 0.001, 0.01],
        'batch_size': [32, 64],
    }
    
    print(f"\n[2/3] Running GridSearchCV...")
    print(f"Parameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    print(f"\nTotal combinations: {np.prod([len(v) for v in param_grid.values()])}")
    
    # Create base model
    base_model = MLPRegressor(
        activation='relu',
        solver='adam',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=RANDOM_SEED,
        verbose=False,
    )
    
    # Run GridSearchCV (use negative MAE for scoring)
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',  # Lower MAE is better
        n_jobs=-1,
        verbose=2,
    )
    
    grid_search.fit(X_trainval, y_trainval)
    
    print(f"\n[3/3] Results:")
    print(f"Best score (CV MAE): {-grid_search.best_score_:.4f}")
    print(f"Best parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': -grid_search.best_score_,  # Convert back to positive MAE
        'cv_results': {
            'mean_test_scores': [-s for s in grid_search.cv_results_['mean_test_score']],
            'std_test_scores': grid_search.cv_results_['std_test_score'].tolist(),
            'params': [str(p) for p in grid_search.cv_results_['params']],
        }
    }


def main():
    """Main hyperparameter tuning orchestrator."""
    np.random.seed(RANDOM_SEED)
    
    # Tune classification NN
    clf_results = tune_classification_nn()
    
    # Tune regression NN
    reg_results = tune_regression_nn()
    
    # Save results
    OUTPUT_DIR = Path("outputs")
    METRICS_DIR = OUTPUT_DIR / "metrics"
    METRICS_DIR.mkdir(exist_ok=True, parents=True)
    
    config = {
        'classification': {
            'best_params': clf_results['best_params'],
            'best_cv_score': clf_results['best_score'],
            'metric': 'f1_macro',
        },
        'regression': {
            'best_params': reg_results['best_params'],
            'best_cv_score': reg_results['best_score'],
            'metric': 'mae',
        },
        'search_space': {
            'hidden_layer_sizes': ['(64,)', '(128,)', '(64, 32)', '(128, 64)'],
            'learning_rate_init': [0.001, 0.01],
            'alpha': [0.0001, 0.001, 0.01],
            'batch_size': [32, 64],
        },
        'method': 'GridSearchCV with 3-fold cross-validation',
    }
    
    config_path = METRICS_DIR / "best_nn_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("=" * 80)
    print(f"\nBest configuration saved to: {config_path}")
    print("\nClassification Best Params:")
    for param, value in clf_results['best_params'].items():
        print(f"  {param}: {value}")
    print("\nRegression Best Params:")
    for param, value in reg_results['best_params'].items():
        print(f"  {param}: {value}")


if __name__ == "__main__":
    main()
