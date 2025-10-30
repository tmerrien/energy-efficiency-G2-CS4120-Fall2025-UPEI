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
python -m src.preview_targets
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
├── README.md                 # This file
├── requirements.txt          # Production dependencies
├── requirements-dev.txt      # Development dependencies
├── pyproject.toml            # Tool configurations (Black, isort, pytest)
├── .flake8                   # Linting configuration
├── .github/
│   └── workflows/            # CI/CD pipeline definitions
│       ├── README.md         # CI/CD documentation
│       ├── ci.yml            # Code quality checks
│       └── ml-train.yml      # ML training & reporting
├── src/                      # Source code
│   ├── data.py              # Dataset loading
│   ├── tasks.py             # Regression + classification helpers
│   ├── split.py             # Train/val/test split helpers
│   └── preview_targets.py   # Dataset stats for proposal
└── docs/
    ├── PROPOSAL.md           # Project proposal
    └── MIDPOINT.md           # Midpoint report
```

## Dataset Information

**Energy Efficiency Dataset** (UCI ML Repository ID: 242)
- **Samples:** 768 buildings
- **Features:** 8 building characteristics
- **Targets:** Heating load (continuous) and Cooling load (continuous)
- **Focus:** Predicting heating load for energy efficiency analysis
