from __future__ import annotations
from collections import Counter
import numpy as np

from src.data import load_energy_efficiency
from src.tasks import (
    get_regression_target,
    make_classification_labels_from_hl
)
from src.split import train_val_test_split_indices

def main() -> None:
    df = load_energy_efficiency()

    # --- Regression summary (heating_load continuous) ---
    y_reg = get_regression_target(df, target_col="heating_load")
    print(f"[Regression] heating_load stats: min={y_reg.min():.3f}, max={y_reg.max():.3f}, mean={y_reg.mean():.3f}")

    # --- PREVIEW classification w/ global median
    y_cls_all, info_all = make_classification_labels_from_hl(df, target_col="heating_load", threshold=None)
    print(f"[Classification] global median threshold = {info_all.threshold:.3f}")
    print(f"Counts (0=Low, 1=High): {Counter(y_cls_all)}")

    # --- Leakage-safe version
    idx_train, idx_val, idx_test = train_val_test_split_indices(
        n_samples=len(df),
        y_for_stratify=y_cls_all,
        test_size=0.2,
        val_size=0.2,
    )

    train_thr= float(np.median(y_reg[idx_train]))
    print(f"[Leakage-safe] TRAIN-only threshold: {train_thr:.3f}")

    # Apply train threshold to all splits
    def apply_thr(arr: np.ndarray, thr: float) -> np.ndarray:
        return (arr >=thr).astype(int)

    y_train = apply_thr(y_reg[idx_train], train_thr)
    y_val = apply_thr(y_reg[idx_val], train_thr)
    y_test = apply_thr(y_reg[idx_test], train_thr)

    print("Train class counts:", Counter(y_train))
    print("Val class counts:", Counter(y_val))
    print("Test class counts:", Counter(y_test))

if __name__ == "__main__":
    main()