"""
Generate feature importance plot for Decision Tree Regressor.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor

from src.config import RANDOM_SEED, TARGET_COL, TEST_SIZE, VAL_SIZE
from src.data.loader import load_energy_efficiency
from src.data.splitter import train_val_test_split_indices
from src.preprocessing.transformers import prepare_features, preprocess_data


def main():
    print("=" * 80)
    print("GENERATING FEATURE IMPORTANCE PLOT")
    print("=" * 80)

    # Load data
    print("\n[1/4] Loading data...")
    df = load_energy_efficiency()
    X, numeric_features, categorical_features = prepare_features(df)
    y_regression = df[TARGET_COL].values

    # Create splits
    print("[2/4] Creating train/val/test splits...")
    idx_train, idx_val, idx_test = train_val_test_split_indices(
        n_samples=len(df),
        y_for_stratify=None,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_seed=RANDOM_SEED,
    )

    X_train = X.iloc[idx_train]
    X_val = X.iloc[idx_val]
    X_test = X.iloc[idx_test]
    y_train = y_regression[idx_train]
    y_val = y_regression[idx_val]
    y_test = y_regression[idx_test]

    # Preprocess (no scaling for Decision Tree)
    print("[3/4] Preprocessing features...")
    X_train_proc, X_val_proc, X_test_proc, scaler, encoder = preprocess_data(
        X_train, X_val, X_test, numeric_features, categorical_features, scale_numeric=False
    )

    # Construct feature names
    # Numeric features first, then one-hot encoded categorical
    feature_names = numeric_features.copy()
    for i, cat_feature in enumerate(categorical_features):
        categories = encoder.categories_[i]
        for category in categories:
            feature_names.append(f"{cat_feature}_{category}")

    # Train Decision Tree Regressor
    print("[4/4] Training Decision Tree and extracting feature importance...")
    dt_reg = DecisionTreeRegressor(
        random_state=RANDOM_SEED,
        max_depth=10,
        min_samples_split=10
    )
    dt_reg.fit(X_train_proc, y_train)

    # Get feature importances
    importances = dt_reg.feature_importances_

    # Create DataFrame for easier handling
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Print top features
    print("\nTop 10 Most Important Features:")
    print("-" * 50)
    for idx, row in importance_df.head(10).iterrows():
        print(f"{row['feature']:30s} {row['importance']:.4f}")

    # Create visualization
    print("\nGenerating feature importance plot...")

    # Plot top 10 features
    top_n = 10
    top_features = importance_df.head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), top_features['importance'], color='steelblue')
    plt.yticks(range(top_n), top_features['feature'])
    plt.xlabel('Feature Importance (Gini)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Top 10 Feature Importances - Decision Tree Regressor', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # Highest importance at top

    # Add value labels on bars
    for i, (idx, row) in enumerate(top_features.iterrows()):
        plt.text(row['importance'], i, f" {row['importance']:.4f}",
                va='center', fontsize=10)

    plt.tight_layout()

    # Save plot
    OUTPUT_DIR = Path("outputs/plots")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    plot_path = OUTPUT_DIR / "feature_importance_decision_tree_regressor.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_path}")

    # Save importance data as CSV
    METRICS_DIR = Path("outputs/metrics")
    METRICS_DIR.mkdir(exist_ok=True, parents=True)
    csv_path = METRICS_DIR / "feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Calculate cumulative importance
    cumsum = importance_df['importance'].cumsum()
    top3_pct = cumsum.iloc[2] * 100
    top5_pct = cumsum.iloc[4] * 100

    print(f"\nCumulative Importance:")
    print(f"  Top 3 features: {top3_pct:.1f}%")
    print(f"  Top 5 features: {top5_pct:.1f}%")

    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
