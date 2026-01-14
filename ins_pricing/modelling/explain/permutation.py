from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .metrics import resolve_metric


def _compute_feature_importance(
    feat, X_data, y_arr, w_arr, predict_fn, metric_fn,
    baseline_score, higher_is_better, n_repeats, random_state, metric_name,
    return_scores, is_dataframe=True, feat_idx=None
):
    """Helper function to compute importance for a single feature (parallelizable)."""
    rng = np.random.default_rng(random_state)

    if is_dataframe:
        # Work on a copy for thread safety in parallel execution
        X_work = X_data.copy()
        orig_values = X_work[feat].to_numpy(copy=False).copy()
        scores = []
        for _ in range(n_repeats):
            X_work[feat] = rng.permutation(orig_values)
            pred = predict_fn(X_work)
            score = metric_fn(y_arr, pred, w_arr)
            scores.append(float(score))
    else:
        X_work = X_data.copy()
        orig_col = X_data[:, feat_idx].copy()
        scores = []
        for _ in range(n_repeats):
            X_work[:, feat_idx] = rng.permutation(orig_col)
            pred = predict_fn(X_work)
            score = metric_fn(y_arr, pred, w_arr)
            scores.append(float(score))

    scores_arr = np.asarray(scores, dtype=float)
    if higher_is_better:
        delta = baseline_score - scores_arr
    else:
        delta = scores_arr - baseline_score

    entry = {
        "feature": feat,
        "importance_mean": float(np.mean(delta)),
        "importance_std": float(np.std(delta)),
        "baseline_score": float(baseline_score),
        "permutation_score_mean": float(np.mean(scores_arr)),
        "metric": metric_name,
    }
    if return_scores:
        entry["permutation_scores"] = scores
    return entry


def _prepare_data(X, y, sample_weight, max_rows, rng):
    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        y_arr = y_arr.reshape(-1)

    w_arr = None
    if sample_weight is not None:
        w_arr = np.asarray(sample_weight).reshape(-1)
        if w_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("sample_weight length must match y.")

    if isinstance(X, pd.DataFrame):
        X_data = X
        if len(X_data) != len(y_arr):
            raise ValueError("X and y must have the same length.")
        if max_rows and len(X_data) > max_rows:
            idx = rng.choice(len(X_data), size=int(max_rows), replace=False)
            X_data = X_data.iloc[idx].copy()
            y_arr = y_arr[idx]
            if w_arr is not None:
                w_arr = w_arr[idx]
        return X_data, y_arr, w_arr

    X_np = np.asarray(X)
    if X_np.ndim != 2:
        raise ValueError("X must be 2d when not a DataFrame.")
    if X_np.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y must have the same length.")
    if max_rows and X_np.shape[0] > max_rows:
        idx = rng.choice(X_np.shape[0], size=int(max_rows), replace=False)
        X_np = X_np[idx]
        y_arr = y_arr[idx]
        if w_arr is not None:
            w_arr = w_arr[idx]
    return X_np, y_arr, w_arr


def permutation_importance(
    predict_fn: Callable,
    X,
    y,
    *,
    sample_weight=None,
    metric: str | Callable = "auto",
    task_type: Optional[str] = None,
    higher_is_better: Optional[bool] = None,
    n_repeats: int = 5,
    random_state: Optional[int] = None,
    max_rows: Optional[int] = 5000,
    features: Optional[Sequence[str]] = None,
    return_scores: bool = False,
    safe_copy: bool = False,
    n_jobs: Optional[int] = None,
) -> pd.DataFrame:
    """Permutation importance on tabular data.

    predict_fn should accept the same type as X (DataFrame or ndarray).
    Set safe_copy=True if predict_fn mutates its input.
    Set n_jobs to enable parallel processing across features (default: None = sequential).
    """
    rng = np.random.default_rng(random_state)
    n_repeats = max(1, int(n_repeats))

    X_data, y_arr, w_arr = _prepare_data(X, y, sample_weight, max_rows, rng)
    metric_fn, higher_is_better, metric_name = resolve_metric(
        metric, task_type=task_type, higher_is_better=higher_is_better
    )

    baseline_pred = predict_fn(X_data)
    baseline_score = metric_fn(y_arr, baseline_pred, w_arr)

    if isinstance(X_data, pd.DataFrame):
        feature_names = list(X_data.columns)
        if features is not None:
            feature_names = [f for f in features if f in X_data.columns]

        # Use parallel processing if n_jobs is specified
        if n_jobs is not None and n_jobs != 1:
            # Generate different random seeds for each feature to ensure reproducibility
            seeds = [random_state + i if random_state is not None else None
                     for i in range(len(feature_names))]
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_compute_feature_importance)(
                    feat, X_data, y_arr, w_arr, predict_fn, metric_fn,
                    baseline_score, higher_is_better, n_repeats, seed,
                    metric_name, return_scores, is_dataframe=True
                )
                for feat, seed in zip(feature_names, seeds)
            )
        else:
            # Sequential processing (original optimized version)
            X_perm = X_data if not safe_copy else X_data.copy()
            results = []
            for feat in feature_names:
                # Store original values directly without extra copy
                orig_values = X_perm[feat].to_numpy(copy=False)
                orig_copy = orig_values.copy()  # Only copy the column, not the entire DataFrame
                scores = []
                for _ in range(n_repeats):
                    X_perm[feat] = rng.permutation(orig_copy)
                    pred = predict_fn(X_perm)
                    score = metric_fn(y_arr, pred, w_arr)
                    scores.append(float(score))
                # Restore original column values
                X_perm[feat] = orig_copy

                scores_arr = np.asarray(scores, dtype=float)
                if higher_is_better:
                    delta = baseline_score - scores_arr
                else:
                    delta = scores_arr - baseline_score
                entry = {
                    "feature": feat,
                    "importance_mean": float(np.mean(delta)),
                    "importance_std": float(np.std(delta)),
                    "baseline_score": float(baseline_score),
                    "permutation_score_mean": float(np.mean(scores_arr)),
                    "metric": metric_name,
                }
                if return_scores:
                    entry["permutation_scores"] = scores
                results.append(entry)
    else:
        if features is not None:
            if len(features) != X_data.shape[1]:
                raise ValueError("features length must match X columns for ndarray input.")
            feature_names = list(features)
        else:
            feature_names = [f"x{i}" for i in range(X_data.shape[1])]

        X_base = np.asarray(X_data)

        # Use parallel processing if n_jobs is specified
        if n_jobs is not None and n_jobs != 1:
            seeds = [random_state + i if random_state is not None else None
                     for i in range(len(feature_names))]
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_compute_feature_importance)(
                    feat, X_base, y_arr, w_arr, predict_fn, metric_fn,
                    baseline_score, higher_is_better, n_repeats, seed,
                    metric_name, return_scores, is_dataframe=False, feat_idx=idx
                )
                for idx, (feat, seed) in enumerate(zip(feature_names, seeds))
            )
        else:
            # Sequential processing
            X_perm = X_base.copy()
            results = []
            for idx, feat in enumerate(feature_names):
                orig_col = X_base[:, idx].copy()
                scores = []
                for _ in range(n_repeats):
                    X_perm[:, idx] = rng.permutation(orig_col)
                    pred_input = X_perm.copy() if safe_copy else X_perm
                    pred = predict_fn(pred_input)
                    score = metric_fn(y_arr, pred, w_arr)
                    scores.append(float(score))
                X_perm[:, idx] = orig_col

                scores_arr = np.asarray(scores, dtype=float)
                if higher_is_better:
                    delta = baseline_score - scores_arr
                else:
                    delta = scores_arr - baseline_score
                entry = {
                    "feature": feat,
                    "importance_mean": float(np.mean(delta)),
                    "importance_std": float(np.std(delta)),
                    "baseline_score": float(baseline_score),
                    "permutation_score_mean": float(np.mean(scores_arr)),
                    "metric": metric_name,
                }
                if return_scores:
                    entry["permutation_scores"] = scores
                results.append(entry)

    df = pd.DataFrame(results)
    df = df.sort_values(by="importance_mean", ascending=False).reset_index(drop=True)
    return df
