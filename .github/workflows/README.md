# GitHub Actions Workflows

This directory contains CI/CD workflows for the Energy Efficiency project.

## Workflows

### 1. CI - Code Quality & Tests (`ci.yml`)
**Triggers:** Pull requests and pushes to `main` and `midpoint` branches

**What it does:**
- Checks code formatting with Black
- Checks import sorting with isort
- Runs linting with Flake8
- Runs test suite (when available)
- Verifies data loading works

**Purpose:** Ensures code quality and catches errors early

### 2. ML Training & Evaluation (`ml-train.yml`)
**Triggers:** Pull requests that modify `src/`, `requirements.txt`, or the workflow itself

**What it does:**
- Sets up Python environment
- Installs dependencies
- Trains models (placeholder - update with your training script)
- Generates metrics and plots
- Posts automated report as PR comment using CML

**Purpose:** Provides automated feedback on model performance changes

## Local Development

### Install development dependencies
```bash
pip install -r requirements-dev.txt
```

### Run code quality checks locally
```bash
# Format code
black src/

# Sort imports
isort src/

# Lint code
flake8 src/

# Run tests
pytest tests/
```

## Configuration Files

- `.flake8` - Flake8 linter configuration
- `pyproject.toml` - Black, isort, and pytest configuration
- `requirements-dev.txt` - Development dependencies

## Next Steps

1. Create a training script (e.g., `src/train.py`) that:
   - Loads and splits data
   - Trains baseline models
   - Logs metrics to MLflow
   - Saves plots and metrics to `outputs/`

2. Update `ml-train.yml` to call your training script

3. Add tests in a `tests/` directory

4. (Optional) Add a deployment workflow for model serving
