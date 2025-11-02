"""
Regression pipeline for continuous Heating Load prediction.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from .config import RANDOM_SEED
from .models import train_regression_model
from .preprocessing import preprocess_data
from .visualization import plot_correlation_heatmap, plot_residuals


def run_regression_pipeline(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_reg_train,
    y_reg_val,
    y_reg_test,
    numeric_features: list,
    categorical_features: list,
) -> list:
    """
    Run complete regression pipeline.

    Returns:
        List of regression results dictionaries
    """
    print("\n" + "=" * 80)
    print("REGRESSION TASK: Heating Load (Continuous)")
    print("=" * 80)

    regression_results = []

    # Linear Regression (scaled features)
    print("\n[8/11] Training Linear Regression...")
    X_train_linreg, X_val_linreg, X_test_linreg, _, _ = preprocess_data(
        X_train, X_val, X_test, numeric_features, categorical_features, scale_numeric=True
    )
    linreg = LinearRegression()
    result_linreg = train_regression_model(
        X_train_linreg,
        y_reg_train,
        X_val_linreg,
        y_reg_val,
        X_test_linreg,
        y_reg_test,
        "Linear_Regression",
        linreg,
    )
    regression_results.append(result_linreg)

    # Decision Tree Regressor (no scaling needed)
    print("\n[9/11] Training Decision Tree Regressor...")
    X_train_dt_reg, X_val_dt_reg, X_test_dt_reg, _, _ = preprocess_data(
        X_train, X_val, X_test, numeric_features, categorical_features, scale_numeric=False
    )
    dt_reg = DecisionTreeRegressor(random_state=RANDOM_SEED, max_depth=10, min_samples_split=10)
    result_dt_reg = train_regression_model(
        X_train_dt_reg,
        y_reg_train,
        X_val_dt_reg,
        y_reg_val,
        X_test_dt_reg,
        y_reg_test,
        "Decision_Tree_Regressor",
        dt_reg,
    )
    regression_results.append(result_dt_reg)

    # Select best model
    best_reg = min(regression_results, key=lambda x: x["val_rmse"])
    print(
        f"\nBest Regression Model: {best_reg['model_name']} (Val RMSE: {best_reg['val_rmse']:.4f})"
    )

    # Generate visualizations
    print("\n[10/11] Generating residuals plot...")
    residuals_path = plot_residuals(
        best_reg["y_test"], best_reg["y_test_pred"], best_reg["model_name"]
    )
    print(f"Saved: {residuals_path}")

    print("\n[11/11] Generating correlation heatmap...")
    heatmap_path = plot_correlation_heatmap(
        X_train[numeric_features].values, numeric_features, best_reg["model_name"]
    )
    print(f"Saved: {heatmap_path}")

    return regression_results
