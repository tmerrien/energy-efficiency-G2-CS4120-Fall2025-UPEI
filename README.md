# CS-4120 Course Project – Energy Efficiency

This repository contains our course project for **CS-4120 (Fall 2025)**,  
School of Mathematical and Computational Sciences, UPEI.

**Professor:** Dr. Dania Tamayo-Vera  
**Team:** Tanguy Merrien & Alex Deboer

## Project Overview

We are analyzing the **Energy Efficiency dataset** (UCI ID 242) to tackle two machine learning tasks:

- **Classification:** Predict whether a building's heating load is *High* or *Low*
- **Regression:** Predict heating load as a continuous variable

Our approach will compare **classical ML baselines** with a **Neural Network (NN)**, with full reproducibility via **MLflow** and pinned dependencies.

📄 **[Project Proposal](docs/PROPOSAL.md)**  
📄 **[Midpoint Report](docs/MIDPOINT.md)**

## Quick Start

### Install Dependencies
```bash
# Production dependencies only
pip install -r requirements.txt

# Or, for development (includes testing and linting tools)
pip install -r requirements-dev.txt
```

### Run Preview Script
```bash
# From the project root directory
python scripts/preview_dataset.py
```

### Train Models
```bash
# Train all baseline models (classification + regression)
python scripts/train_all.py

# Or train specific tasks
python scripts/train_classification.py
python scripts/train_regression.py

# Tune neural network hyperparameters
python scripts/tune_hyperparameters.py
```

## CI/CD Pipeline

This project uses **GitHub Actions** for automated code quality checks and ML experiment tracking. Every pull request triggers:
- Code formatting and linting checks (Black, isort, Flake8)
- Automated test execution
- ML training with automated reports posted as PR comments (via CML)

📖 **[View CI/CD Documentation](.github/workflows/README.md)**

## Repository Structure

```
project/
├── README.md                      # This file
├── requirements.txt               # Production dependencies
├── requirements-dev.txt           # Development dependencies
├── pyproject.toml                 # Tool configurations (Black, isort, pytest)
├── .flake8                        # Linting configuration
├── .github/
│   └── workflows/                 # CI/CD pipeline definitions
│       ├── README.md              # CI/CD documentation
│       ├── ci.yml                 # Code quality checks
│       └── ml-train.yml           # ML training & reporting
├── scripts/                       # Executable entry points
│   ├── train_all.py               # Train all baseline models
│   ├── train_classification.py    # Train classification models only
│   ├── train_regression.py        # Train regression models only
│   ├── tune_hyperparameters.py    # Neural network hyperparameter tuning
│   └── preview_dataset.py         # Dataset preview tool
├── src/                           # Source code (library modules)
│   ├── config.py                  # Configuration constants
│   ├── data/                      # Data loading and splitting
│   │   ├── loader.py              # Dataset loading utilities
│   │   ├── splitter.py            # Train/val/test split logic
│   │   ├── targets.py             # Target extraction utilities
│   │   └── preparation.py         # Data preparation helpers
│   ├── preprocessing/             # Data preprocessing
│   │   └── transformers.py        # Feature transformers
│   ├── models/                    # Model definitions
│   │   ├── baselines.py           # Baseline model trainers
│   │   ├── nn_classifier.py       # Neural network classifier
│   │   └── nn_regressor.py        # Neural network regressor
│   ├── pipelines/                 # End-to-end pipelines
│   │   ├── classification.py      # Classification baseline pipeline
│   │   ├── regression.py          # Regression baseline pipeline
│   │   ├── nn_classification.py   # NN classification pipeline
│   │   └── nn_regression.py       # NN regression pipeline
│   └── evaluation/                # Results and visualization
│       ├── visualization.py       # Plot generation
│       └── results.py             # Results saving utilities
├── outputs/                       # Generated artifacts
│   ├── logs/                      # Training logs
│   ├── metrics/                   # Evaluation metrics (CSV, JSON)
│   ├── models/                    # Saved model files
│   └── plots/                     # Visualization plots (PNG)
├── mlruns/                        # MLflow experiment tracking
└── docs/                          # Documentation
    ├── PROPOSAL.md                # Project proposal
    └── MIDPOINT.md                # Midpoint report
```

## Dataset Information

**Energy Efficiency Dataset** (UCI ML Repository ID: 242)
- **Samples:** 768 buildings
- **Features:** 8 building characteristics
- **Targets:** Heating load (continuous) and Cooling load (continuous)
- **Focus:** Predicting heating load for energy efficiency analysis
