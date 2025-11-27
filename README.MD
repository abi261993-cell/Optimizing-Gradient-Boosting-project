# Pure NumPy Gradient Boosting Machine (GBM)

This repository contains a from-scratch, pure-NumPy implementation of a Gradient Boosting Machine (GBM) for regression, including:
- A small CART-style regression tree base learner (NumPy-only)
- Iterative boosting using residual (negative gradient) fitting
- Grid-search cross-validation tuning
- Comparison with scikit-learn's GradientBoostingRegressor

## Files
- pure_numpy_gbm.py — main implementation and CLI.
- run_example.py — runs a quick demo with --quick.
- tests/test_quick_run.py — pytest smoke test.
- requirements.txt — Python dependencies.
- LICENSE — MIT license.

## Quick start
1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
