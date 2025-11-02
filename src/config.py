"""
Configuration constants for the training pipeline.
"""

from pathlib import Path

# Training configuration
RANDOM_SEED = 42
TRAIN_SIZE = 0.6
VAL_SIZE = 0.2
TEST_SIZE = 0.2
TARGET_COL = "heating_load"

# Feature configuration
CATEGORICAL_FEATURES = ["x6", "x8"]

# Output directories
OUTPUT_DIR = Path("outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR = OUTPUT_DIR / "models"

# Create output directories
for dir_path in [OUTPUT_DIR, PLOTS_DIR, METRICS_DIR, MODELS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)
