#!/usr/bin/env python3
"""
pure_numpy_gbm.py

Pure-NumPy implementation of a Gradient Boosting Machine (GBM) for regression.

Usage (example):
    python pure_numpy_gbm.py --quick

Author: (You)
License: MIT
"""
from _future_ import annotations
import time
import math
import argparse
import numpy as np
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle as skshuffle
import joblib
import warnings
warnings.filterwarnings("ignore")


def generate_synthetic_regression(n_samples=5000, n_features=20, random_state=42, noise_std=0.5):
    rng = np.random.RandomState(random_state)
    X = rng.normal(0, 1, size=(n_samples, n_features))

    idx_sin = np.arange(0, min(6, n_features))
    idx_poly = np.arange(6, min(12, n_features))
    idx_inter = np.arange(12, min(16, n_features))

    y = np.zeros(n_samples, dtype=float)
    freqs = rng.uniform(0.5, 3.0, size=idx_sin.shape)
    for k, fi in enumerate(idx_sin):
        a = freqs[k]
        y += np.sin(a * X[:, fi])

    for j in idx_poly:
        coeff = rng.uniform(-1.5, 1.5)
        y += coeff * (X[:, j] ** 2) + 0.5 * (X[:, j] ** 3)

    for t in range(0, len(idx_inter), 2):
        if t + 1 < len(idx_inter):
            i1, i2 = idx_inter[t], idx_inter[t + 1]
            y += 0.75 * X[:, i1] * X[:, i2]

    rem = np.setdiff1d(np.arange(n_features), np.concatenate([idx_sin, idx_poly, idx_inter]))
    if rem.size > 0:
        lin_coefs = rng.uniform(-0.5, 0.5, size=rem.size)
        y += X[:, rem].dot(lin_coefs)

    y = y - y.mean()
    y = y / (y.std() + 1e-12)
    y = y * 3.0
    y += rng.normal(0, noise_std, size=n_samples)

    return X, y


class NumPyRegressionTree:
    def _init_(self, max_depth=3, min_samples_split=10, n_thresholds=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_thresholds = max(2, n_thresholds)
        self.root = None

    class Node:
        _slots_ = ("feature", "threshold", "left", "right", "value", "is_leaf")

        def _init_(self, is_leaf=False, value=None, feature=None, threshold=None):
            self.is_leaf = is_leaf
            self.value = value
            self.feature = feature
            self.threshold = threshold
            self.left = None
            self.right = None

    def _mse(self, y):
        if y.size == 0:
            return 0.0
        mu = y.mean()
        return ((y - mu) ** 2).mean()

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None, 0

        base_mse = self._mse(y)
        best_feat, best_thr, best_gain = None, None, 0.0

        for f in range(n_features):
            col = X[:, f]
            quantiles = np.linspace(0.05, 0.95, self.n_thresholds)
            thresholds = np.unique(np.quantile(col, quantiles))
            for thr in thresholds:
                left_mask = col <= thr
                right_mask = ~left_mask
                if left_mask.sum() < self.min_samples_split or right_mask.sum() < self.min_samples_split:
                    continue
                left_y = y[left_mask]
                right_y = y[right_mask]
                mse_left = self._mse(left_y)
                mse_right = self._mse(right_y)
                w_mse = (left_y.size * mse_left + right_y.size * mse_right) / n_samples
                gain = base_mse - w_mse
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_feat = f
                    best_thr = thr

        return best_feat, best_thr, best_gain

    def _build_tree(self, X, y, depth):
        node_value = float(np.mean(y)) if y.size > 0 else 0.0
        if depth >= self.max_depth or X.shape[0] < self.min_samples_split or np.unique(y).size == 1:
            node = NumPyRegressionTree.Node(is_leaf=True, value=node_value)
            return node

        feat, thr, gain = self._best_split(X, y)
        if feat is None or gain <= 0.0:
            node = NumPyRegressionTree.Node(is_leaf=True, value=node_value)
            return node

        node = NumPyRegressionTree.Node(is_leaf=False, feature=feat, threshold=thr)
        mask_left = X[:, feat] <= thr
        X_left, y_left = X[mask_left], y[mask_left]
        X_right, y_right = X[~mask_left], y[~mask_left]

        node.left = self._build_tree(X_left, y_left, depth + 1)
        node.right = self._build_tree(X_right, y_right, depth + 1)
        return node

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.root = self._build_tree(X, y, depth=0)

    def _predict_row(self, x, node):
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_row(x, node.left)
        else:
            return self._predict_row(x, node.right)

    def predict(self, X):
        X = np.asarray(X)
        if self.root is None:
            raise RuntimeError("Tree not fitted.")
        preds = np.array([self._predict_row(row, self.root) for row in X], dtype=float)
        return preds


class NumPyGBM:
    def _init_(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=10, n_thresholds=10):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_thresholds = n_thresholds
        self.trees = []
        self.init_pred = 0.0

    def fit(self, X, y, verbose=False):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = X.shape[0]
        self.init_pred = float(np.mean(y))
        y_pred = np.full(n_samples, self.init_pred, dtype=float)
        self.trees = []

        for m in range(self.n_estimators):
            resid = y - y_pred
            tree = NumPyRegressionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, n_thresholds=self.n_thresholds)
            tree.fit(X, resid)
            update = tree.predict(X)
            y_pred = y_pred + self.learning_rate * update
            self.trees.append(tree)

            if verbose and ((m + 1) % max(1, self.n_estimators // 10) == 0 or m == 0):
                rmse = math.sqrt(mean_squared_error(y, y_pred))
                print(f"[GBM] Iter {m+1}/{self.n_estimators} - train RMSE: {rmse:.5f}")

    def predict(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]
        y_pred = np.full(n_samples, self.init_pred, dtype=float)
        for tree in self.trees:
            y_pred = y_pred + self.learning_rate * tree.predict(X)
        return y_pred


def grid_search_cv(X, y, param_grid, cv=3, random_state=42, verbose=False, max_train_samples=None):
    from itertools import product
    keys = list(param_grid.keys())
    combos = list(product(*(param_grid[k] for k in keys)))

    results = []
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    for combo in combos:
        params = dict(zip(keys, combo))
        val_scores = []
        start_time = time.time()
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            if max_train_samples is not None and X_train.shape[0] > max_train_samples:
                X_train, y_train = X_train[:max_train_samples], y_train[:max_train_samples]

            model = NumPyGBM(
                n_estimators=int(params.get('n_estimators', 100)),
                learning_rate=float(params.get('learning_rate', 0.1)),
                max_depth=int(params.get('max_depth', 3)),
                min_samples_split=int(params.get('min_samples_split', 10)),
                n_thresholds=int(params.get('n_thresholds', 10))
            )
            model.fit(X_train, y_train, verbose=False)
            y_pred = model.predict(X_val)
            rmse = math.sqrt(mean_squared_error(y_val, y_pred))
            val_scores.append(rmse)

        elapsed = time.time() - start_time
        mean_rmse = float(np.mean(val_scores))
        std_rmse = float(np.std(val_scores))
        result = {'params': params, 'mean_rmse': mean_rmse, 'std_rmse': std_rmse, 'time_sec': elapsed}
        results.append(result)
        if verbose:
            print(f"Grid {params} -> mean RMSE: {mean_rmse:.5f} (+/- {std_rmse:.5f}), time {elapsed:.2f}s")

    best = min(results, key=lambda r: r['mean_rmse'])
    return best['params'], results


def compare_with_sklearn(X_train, y_train, X_test, y_test, gbm_params, skl_params=None):
    if skl_params is None:
        skl_params = {}

    start = time.time()
    custom = NumPyGBM(n_estimators=gbm_params['n_estimators'],
                      learning_rate=gbm_params['learning_rate'],
                      max_depth=gbm_params.get('max_depth', 3),
                      min_samples_split=gbm_params.get('min_samples_split', 10),
                      n_thresholds=gbm_params.get('n_thresholds', 10))
    custom.fit(X_train, y_train, verbose=False)
    t_custom = time.time() - start
    y_pred_custom = custom.predict(X_test)
    rmse_custom = math.sqrt(mean_squared_error(y_test, y_pred_custom))

    start = time.time()
    skl = GradientBoostingRegressor(
        n_estimators=gbm_params['n_estimators'],
        learning_rate=gbm_params['learning_rate'],
        max_depth=gbm_params.get('max_depth', 3),
        random_state=42,
        **skl_params
    )
    skl.fit(X_train, y_train)
    t_skl = time.time() - start
    y_pred_skl = skl.predict(X_test)
    rmse_skl = math.sqrt(mean_squared_error(y_test, y_pred_skl))

    return {
        'custom': {'rmse': rmse_custom, 'time_sec': t_custom},
        'sklearn': {'rmse': rmse_skl, 'time_sec': t_skl},
        'custom_model': custom,
        'sklearn_model': skl
    }


def main(args=None):
    parser = argparse.ArgumentParser(description="Pure-NumPy GBM demo")
    parser.add_argument("--quick", action="store_true", help="Run a quick demo with small data (fast)")
    parser.add_argument("--save-models", action="store_true", help="Save models to disk")
    parsed = parser.parse_args(args)

    RANDOM_STATE = 42
    if parsed.quick:
        N_SAMPLES = 600
        N_FEATURES = 10
        NOISE_STD = 0.4
    else:
        N_SAMPLES = 2000
        N_FEATURES = 20
        NOISE_STD = 0.4

    print("Generating synthetic dataset...")
    X, y = generate_synthetic_regression(n_samples=N_SAMPLES, n_features=N_FEATURES,
                                         random_state=RANDOM_STATE, noise_std=NOISE_STD)

    X, y = skshuffle(X, y, random_state=RANDOM_STATE)
    split = int(0.8 * N_SAMPLES)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Dataset: {N_SAMPLES} samples, {N_FEATURES} features. Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    param_grid = {
        'n_estimators': [30, 60] if parsed.quick else [50, 100],
        'learning_rate': [0.1, 0.05],
        'max_depth': [2, 3],
        'min_samples_split': [8],
        'n_thresholds': [6] if parsed.quick else [8]
    }

    print("\nStarting grid search CV (this can take time)...")
    best_params, grid_results = grid_search_cv(X_train, y_train, param_grid, cv=3, verbose=False,
                                              max_train_samples=500 if parsed.quick else 1500)
    print("\nGrid search complete.")
    print(f"Best params (by CV mean RMSE): {best_params}")

    best_params_parsed = {
        'n_estimators': int(best_params['n_estimators']),
        'learning_rate': float(best_params['learning_rate']),
        'max_depth': int(best_params.get('max_depth', 3)),
        'min_samples_split': int(best_params.get('min_samples_split', 10)),
        'n_thresholds': int(best_params.get('n_thresholds', 10))
    }

    print("\nTraining final models and comparing to scikit-learn...")
    compare_results = compare_with_sklearn(X_train, y_train, X_test, y_test, best_params_parsed)

    print("\n=== Comparison Summary ===")
    print(f"Custom NumPy GBM   -> Test RMSE: {compare_results['custom']['rmse']:.5f}, Time: {compare_results['custom']['time_sec']:.2f}s")
    print(f"sklearn GBM        -> Test RMSE: {compare_results['sklearn']['rmse']:.5f}, Time: {compare_results['sklearn']['time_sec']:.2f}s")

    if parsed.save_models:
        joblib.dump(compare_results['custom_model'], 'custom_gbm_model.joblib')
        joblib.dump(compare_results['sklearn_model'], 'sklearn_gbm_model.joblib')
        joblib.dump(grid_results, 'grid_results.joblib')
        print("\nModels and grid search results saved to disk.")

    print("End.")


if _name_ == "_main_":
    main()
