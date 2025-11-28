"""
Hyperparameter tuning script for neural network models.

Uses GridSearchCV to find optimal hyperparameters for both classification
and regression neural network models.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor

from src.config import RANDOM_SEED, TARGET_COL, TEST_SIZE, VAL_SIZE
from src.data.loader import load_energy_efficiency
from src.data.splitter import train_val_test_split_indices
from src.preprocessing.transformers import prepare_features, preprocess_data


# Define parameter grid (same for both tasks)
PARAM_GRID = {
    'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
    'learning_rate_init': [0.001, 0.01],
    'alpha': [0.0001, 0.001, 0.01],
    'batch_size': [32, 64],
}


def prepare_data():
    """Load and prepare data with stratified splits.
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_cls, y_reg, numeric_features, categorical_features)
    """
    df = load_energy_efficiency()
    X, numeric_features, categorical_features = prepare_features(df)
    y_regression = df[TARGET_COL].values
    
    # Create splits with stratification
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
    
    return (
        X_train, X_val, X_test,
        y_classification, y_regression,
        idx_train, idx_val,
        numeric_features, categorical_features
    )


def create_base_model(model_type='classifier'):
    """Create base MLP model with common parameters.
    
    Args:
        model_type: Either 'classifier' or 'regressor'
    
    Returns:
        MLPClassifier or MLPRegressor instance
    """
    ModelClass = MLPClassifier if model_type == 'classifier' else MLPRegressor
    return ModelClass(
        activation='relu',
        solver='adam',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=RANDOM_SEED,
        verbose=False,
    )


def run_grid_search(X_trainval, y_trainval, model_type='classifier'):
    """Run GridSearchCV for a given task.
    
    Args:
        X_trainval: Combined training + validation features
        y_trainval: Combined training + validation targets
        model_type: Either 'classifier' or 'regressor'
    
    Returns:
        GridSearchCV fitted object
    """
    print("\n[2/3] Running GridSearchCV...")
    print("Parameter grid:")
    for param, values in PARAM_GRID.items():
        print(f"  {param}: {values}")
    print(f"\nTotal combinations: {np.prod([len(v) for v in PARAM_GRID.values()])}")
    
    base_model = create_base_model(model_type)
    
    scoring = 'f1_macro' if model_type == 'classifier' else 'neg_mean_absolute_error'
    
    grid_search = GridSearchCV(
        base_model,
        PARAM_GRID,
        cv=3,
        scoring=scoring,
        n_jobs=-1,
        verbose=2,
    )
    
    grid_search.fit(X_trainval, y_trainval)
    return grid_search


def tune_classification_nn():
    """Tune hyperparameters for classification neural network."""
    print("=" * 80)
    print("TUNING CLASSIFICATION NEURAL NETWORK")
    print("=" * 80)
    
    print("\n[1/3] Loading and preparing data...")
    X_train, X_val, X_test, y_cls, y_reg, idx_train, idx_val, num_feat, cat_feat = prepare_data()
    
    # Preprocess and combine train+val
    X_train_proc, X_val_proc, _, _, _ = preprocess_data(
        X_train, X_val, X_test, num_feat, cat_feat, scale_numeric=True
    )
    X_trainval = np.vstack([X_train_proc, X_val_proc])
    y_trainval = np.concatenate([y_cls[idx_train], y_cls[idx_val]])
    
    # Run grid search
    grid_search = run_grid_search(X_trainval, y_trainval, model_type='classifier')
    
    # Print results
    print("\n[3/3] Results:")
    print(f"Best score (CV F1): {grid_search.best_score_:.4f}")
    print("Best parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
    }


def tune_regression_nn():
    """Tune hyperparameters for regression neural network."""
    print("\n" + "=" * 80)
    print("TUNING REGRESSION NEURAL NETWORK")
    print("=" * 80)
    
    print("\n[1/3] Loading and preparing data...")
    X_train, X_val, X_test, y_cls, y_reg, idx_train, idx_val, num_feat, cat_feat = prepare_data()
    
    # Preprocess and combine train+val
    X_train_proc, X_val_proc, _, _, _ = preprocess_data(
        X_train, X_val, X_test, num_feat, cat_feat, scale_numeric=True
    )
    X_trainval = np.vstack([X_train_proc, X_val_proc])
    y_trainval = np.concatenate([y_reg[idx_train], y_reg[idx_val]])
    
    # Run grid search
    grid_search = run_grid_search(X_trainval, y_trainval, model_type='regressor')
    
    # Print results
    print("\n[3/3] Results:")
    print(f"Best score (CV MAE): {-grid_search.best_score_:.4f}")
    print("Best parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': -grid_search.best_score_,  # Convert back to positive MAE
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
