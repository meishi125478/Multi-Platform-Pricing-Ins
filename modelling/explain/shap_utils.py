from __future__ import annotations

from typing import Callable, Optional
import warnings

import numpy as np
import pandas as pd


def _require_shap():
    try:
        import shap  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("SHAP is required. Install with `pip install shap`.") from exc
    return shap


def compute_shap_core(
    ctx,
    model_key: str,
    n_background: int,
    n_samples: int,
    on_train: bool,
    X_df: pd.DataFrame,
    prep_fn: Callable[[pd.DataFrame], np.ndarray],
    predict_fn: Callable[[np.ndarray], np.ndarray],
    cleanup_fn: Optional[Callable[[], None]] = None,
    use_parallel: bool = False,
    n_jobs: int = -1,
    batch_size: Optional[int] = None,
) -> dict:
    """Shared SHAP pipeline using KernelExplainer with lazy import.

    Args:
        ctx: Context object with model and data
        model_key: Model identifier
        n_background: Number of background samples for SHAP
        n_samples: Number of samples to explain
        on_train: Whether to use training data
        X_df: Input dataframe
        prep_fn: Function to prepare data for model
        predict_fn: Model prediction function
        cleanup_fn: Optional cleanup function
        use_parallel: Whether to use parallel computation (default: False)
        n_jobs: Number of parallel jobs (-1 for all cores, default: -1)
        batch_size: Batch size for processing (default: auto-computed)

    Returns:
        Dictionary with explainer, X_explain, shap_values, base_value

    Note:
        Setting use_parallel=True can speed up computation 2-8x on multi-core systems,
        but may increase memory usage. Recommended for n_samples > 100.
    """
    _ = on_train
    if model_key not in ctx.trainers or ctx.trainers[model_key].model is None:
        raise RuntimeError(f"Model {model_key} not trained.")
    if cleanup_fn:
        cleanup_fn()
    shap = _require_shap()
    bg_df = ctx._sample_rows(X_df, n_background)
    bg_mat = prep_fn(bg_df)
    explainer = shap.KernelExplainer(predict_fn, bg_mat)
    ex_df = ctx._sample_rows(X_df, n_samples)
    ex_mat = prep_fn(ex_df)
    nsample_eff = ctx._shap_nsamples(ex_mat)

    # Compute SHAP values (with optional parallelization)
    if use_parallel and n_samples > 50:
        shap_values = _compute_shap_parallel(
            explainer, ex_mat, nsample_eff, n_jobs, batch_size
        )
    else:
        shap_values = explainer.shap_values(ex_mat, nsamples=nsample_eff)

    bg_pred = predict_fn(bg_mat)
    base_value = float(np.asarray(bg_pred).mean())

    return {
        "explainer": explainer,
        "X_explain": ex_df,
        "shap_values": shap_values,
        "base_value": base_value,
    }


def _compute_shap_parallel(
    explainer,
    X: np.ndarray,
    nsamples: int,
    n_jobs: int = -1,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """Compute SHAP values in parallel using joblib.

    Args:
        explainer: SHAP KernelExplainer instance
        X: Input data array (n_samples, n_features)
        nsamples: Number of samples for SHAP kernel
        n_jobs: Number of parallel jobs (-1 for all cores)
        batch_size: Batch size (auto if None)

    Returns:
        SHAP values array

    Note:
        This function splits the data into batches and processes them in parallel.
        Performance gain depends on number of cores and batch size.
    """
    try:
        from joblib import Parallel, delayed
    except ImportError:
        warnings.warn(
            "joblib not available, falling back to sequential computation. "
            "Install joblib for parallel SHAP: pip install joblib"
        )
        return explainer.shap_values(X, nsamples=nsamples)

    n_samples = X.shape[0]

    # Auto-compute batch size if not provided
    if batch_size is None:
        # Heuristic: aim for ~4-8 batches per core
        import multiprocessing
        n_cores = multiprocessing.cpu_count() if n_jobs == -1 else abs(n_jobs)
        target_batches = n_cores * 6
        batch_size = max(1, n_samples // target_batches)

    # Split data into batches
    batches = []
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batches.append(X[i:end_idx])

    # Process batches in parallel
    def process_batch(batch):
        return explainer.shap_values(batch, nsamples=nsamples)

    try:
        shap_values_list = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(process_batch)(batch) for batch in batches
        )
    except Exception as e:
        warnings.warn(
            f"Parallel SHAP computation failed: {e}. "
            "Falling back to sequential computation."
        )
        return explainer.shap_values(X, nsamples=nsamples)

    # Concatenate results
    if isinstance(shap_values_list[0], list):
        # Multi-output case (e.g., multi-class classification)
        n_outputs = len(shap_values_list[0])
        shap_values = []
        for output_idx in range(n_outputs):
            output_values = np.concatenate(
                [batch_values[output_idx] for batch_values in shap_values_list],
                axis=0
            )
            shap_values.append(output_values)
    else:
        # Single output case
        shap_values = np.concatenate(shap_values_list, axis=0)

    return shap_values


def compute_shap_glm(
    ctx,
    n_background: int = 500,
    n_samples: int = 200,
    on_train: bool = True,
    use_parallel: bool = False,
    n_jobs: int = -1,
):
    """Compute SHAP values for GLM model.

    Args:
        ctx: Context object
        n_background: Number of background samples
        n_samples: Number of samples to explain
        on_train: Whether to use training data
        use_parallel: Enable parallel computation (faster for n_samples > 100)
        n_jobs: Number of parallel jobs (-1 for all cores)

    Returns:
        Dictionary with SHAP results
    """
    data = ctx.train_oht_scl_data if on_train else ctx.test_oht_scl_data
    design_all = ctx._build_glm_design(data)
    design_cols = list(design_all.columns)

    def predict_wrapper(x_np):
        x_df = pd.DataFrame(x_np, columns=design_cols)
        y_pred = ctx.glm_best.predict(x_df)
        return np.asarray(y_pred, dtype=np.float64).reshape(-1)

    return compute_shap_core(
        ctx,
        "glm",
        n_background,
        n_samples,
        on_train,
        X_df=design_all,
        prep_fn=lambda df: df.to_numpy(dtype=np.float64),
        predict_fn=predict_wrapper,
        use_parallel=use_parallel,
        n_jobs=n_jobs,
    )


def compute_shap_xgb(
    ctx,
    n_background: int = 500,
    n_samples: int = 200,
    on_train: bool = True,
    use_parallel: bool = False,
    n_jobs: int = -1,
):
    """Compute SHAP values for XGBoost model.

    Args:
        ctx: Context object
        n_background: Number of background samples
        n_samples: Number of samples to explain
        on_train: Whether to use training data
        use_parallel: Enable parallel computation (faster for n_samples > 100)
        n_jobs: Number of parallel jobs (-1 for all cores)

    Returns:
        Dictionary with SHAP results
    """
    data = ctx.train_data if on_train else ctx.test_data
    X_raw = data[ctx.factor_nmes]

    def predict_wrapper(x_mat):
        df_input = ctx._decode_ft_shap_matrix_to_df(x_mat)
        return ctx.xgb_best.predict(df_input)

    return compute_shap_core(
        ctx,
        "xgb",
        n_background,
        n_samples,
        on_train,
        X_df=X_raw,
        prep_fn=lambda df: ctx._build_ft_shap_matrix(df).astype(np.float64),
        predict_fn=predict_wrapper,
        use_parallel=use_parallel,
        n_jobs=n_jobs,
    )


def compute_shap_resn(
    ctx,
    n_background: int = 500,
    n_samples: int = 200,
    on_train: bool = True,
    use_parallel: bool = False,
    n_jobs: int = -1,
):
    """Compute SHAP values for ResNet model.

    Args:
        ctx: Context object
        n_background: Number of background samples
        n_samples: Number of samples to explain
        on_train: Whether to use training data
        use_parallel: Enable parallel computation (faster for n_samples > 100)
        n_jobs: Number of parallel jobs (-1 for all cores)

    Returns:
        Dictionary with SHAP results
    """
    data = ctx.train_oht_scl_data if on_train else ctx.test_oht_scl_data
    X = data[ctx.var_nmes]

    def cleanup():
        import torch

        ctx.resn_best.device = torch.device("cpu")
        ctx.resn_best.resnet.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return compute_shap_core(
        ctx,
        "resn",
        n_background,
        n_samples,
        on_train,
        X_df=X,
        prep_fn=lambda df: df.to_numpy(dtype=np.float64),
        predict_fn=lambda x: ctx._resn_predict_wrapper(x),
        cleanup_fn=cleanup,
        use_parallel=use_parallel,
        n_jobs=n_jobs,
    )


def compute_shap_ft(
    ctx,
    n_background: int = 500,
    n_samples: int = 200,
    on_train: bool = True,
    use_parallel: bool = False,
    n_jobs: int = -1,
):
    """Compute SHAP values for FT-Transformer model.

    Args:
        ctx: Context object
        n_background: Number of background samples
        n_samples: Number of samples to explain
        on_train: Whether to use training data
        use_parallel: Enable parallel computation (faster for n_samples > 100)
        n_jobs: Number of parallel jobs (-1 for all cores)

    Returns:
        Dictionary with SHAP results
    """
    if str(ctx.config.ft_role) != "model":
        raise RuntimeError(
            "FT is configured as embedding-only (ft_role != 'model'); FT SHAP is disabled."
        )
    data = ctx.train_data if on_train else ctx.test_data
    X_raw = data[ctx.factor_nmes]

    def cleanup():
        import torch

        ctx.ft_best.device = torch.device("cpu")
        ctx.ft_best.ft.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return compute_shap_core(
        ctx,
        "ft",
        n_background,
        n_samples,
        on_train,
        X_df=X_raw,
        prep_fn=lambda df: ctx._build_ft_shap_matrix(df).astype(np.float64),
        predict_fn=ctx._ft_shap_predict_wrapper,
        cleanup_fn=cleanup,
        use_parallel=use_parallel,
        n_jobs=n_jobs,
    )
