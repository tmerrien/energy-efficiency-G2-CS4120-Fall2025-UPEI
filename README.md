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
pip install -r requirements.txt
```

### Run Preview Script
```bash
# From the project root directory
python -m src.preview_targets
```

## Repository Structure

```
project/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── src/                      # Source code
│   ├── data.py              # Dataset loading
│   ├── tasks.py             # Regression + classification helpers
│   ├── split.py             # Train/val/test split helpers
│   └── preview_targets.py   # Dataset stats for proposal
└── docs/
    └── PROPOSAL.md          # Draft proposal
```

## Dataset Information

**Energy Efficiency Dataset** (UCI ML Repository ID: 242)
- **Samples:** 768 buildings
- **Features:** 8 building characteristics
- **Targets:** Heating load (continuous) and Cooling load (continuous)
- **Focus:** Predicting heating load for energy efficiency analysis
