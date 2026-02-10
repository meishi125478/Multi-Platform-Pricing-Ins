from __future__ import annotations

import inspect
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import torch
import xgboost as xgb
from sklearn.metrics import log_loss

from ins_pricing.modelling.bayesopt.trainers.trainer_base import TrainerBase
from ins_pricing.utils import EPS, get_logger, log_print
from ins_pricing.utils.losses import regression_loss

_logger = get_logger("ins_pricing.trainer.xgb")


def _log(*args, **kwargs) -> None:
    log_print(_logger, *args, **kwargs)

_XGB_CUDA_CHECKED = False
_XGB_HAS_CUDA = False


def _is_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or ("cuda" in msg and "memory" in msg)


def _is_host_memory_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        isinstance(exc, MemoryError)
        or "unable to allocate" in msg
        or "arraymemoryerror" in msg
        or "bad_alloc" in msg
        or "std::bad_alloc" in msg
    )


def _has_categorical_columns(X: Any) -> bool:
    """Best-effort check for pandas categorical columns."""
    dtypes = getattr(X, "dtypes", None)
    if dtypes is not None:
        try:
            return any(str(dtype) == "category" for dtype in dtypes)
        except Exception:
            pass
    dtype = getattr(X, "dtype", None)
    if dtype is not None:
        return str(dtype) == "category"
    return False


class _XGBDMatrixWrapper:
    """Sklearn-like wrapper that uses xgb.train + (Quantile)DMatrix internally."""

    def __init__(
        self,
        params: Dict[str, Any],
        *,
        task_type: str,
        use_gpu: bool,
        chunk_size: Optional[int] = None,
        allow_cpu_fallback: bool = True,
    ) -> None:
        self.params = dict(params)
        self.task_type = task_type
        self.use_gpu = bool(use_gpu)
        self.chunk_size = self._coerce_chunk_size(chunk_size)
        self.allow_cpu_fallback = allow_cpu_fallback
        self._booster: Optional[xgb.Booster] = None
        self.best_iteration: Optional[int] = None
        self._chunk_early_stopping_warned = False
        self._chunk_mode_logged = False

    @staticmethod
    def _coerce_chunk_size(raw_chunk_size: Any) -> Optional[int]:
        if raw_chunk_size is None:
            return None
        try:
            chunk_size = int(raw_chunk_size)
        except (TypeError, ValueError):
            return None
        if chunk_size < 1:
            return None
        return chunk_size

    def set_params(self, **params: Any) -> "_XGBDMatrixWrapper":
        if "xgb_chunk_size" in params:
            self.chunk_size = self._coerce_chunk_size(params.pop("xgb_chunk_size"))
        elif "chunk_size" in params:
            self.chunk_size = self._coerce_chunk_size(params.pop("chunk_size"))
        self.params.update(params)
        return self

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        _ = deep
        params = dict(self.params)
        params["xgb_chunk_size"] = self.chunk_size
        return params

    def _select_dmatrix_class(self) -> Any:
        if hasattr(xgb, "QuantileDMatrix"):
            return xgb.QuantileDMatrix
        if self.use_gpu and hasattr(xgb, "DeviceQuantileDMatrix"):
            return xgb.DeviceQuantileDMatrix
        return xgb.DMatrix

    def _raise_incompatible_categorical_error(self, dmatrix_cls: Any, exc: Exception) -> None:
        cls_name = getattr(dmatrix_cls, "__name__", str(dmatrix_cls))
        raise TypeError(
            f"{cls_name} does not support native categorical handling in this "
            f"XGBoost build. Categorical columns are present, and this trainer "
            f"will not silently disable categorical support. "
            f"Reinstall a consistent XGBoost build (CPU/GPU variants must match) "
            f"or set xgb_use_dmatrix=false as a temporary workaround."
        ) from exc

    def _build_dmatrix(self, X, y=None, weight=None) -> xgb.DMatrix:
        if isinstance(X, (str, os.PathLike)):
            raise ValueError(
                "External-memory DMatrix is disabled; pass in-memory data instead."
            )
        if isinstance(X, xgb.DMatrix):
            raise ValueError(
                "DMatrix inputs are disabled; pass raw in-memory data instead."
            )
        dmatrix_cls = self._select_dmatrix_class()
        kwargs: Dict[str, Any] = {}
        if y is not None:
            kwargs["label"] = y
        if weight is not None:
            kwargs["weight"] = weight
        if bool(self.params.get("enable_categorical", False)):
            kwargs["enable_categorical"] = True
        try:
            return dmatrix_cls(X, **kwargs)
        except TypeError as exc:
            if "enable_categorical" in kwargs:
                if _has_categorical_columns(X):
                    self._raise_incompatible_categorical_error(dmatrix_cls, exc)
                kwargs.pop("enable_categorical", None)
            return dmatrix_cls(X, **kwargs)
        except Exception:
            if dmatrix_cls is not xgb.DMatrix:
                try:
                    return xgb.DMatrix(X, **kwargs)
                except TypeError as exc:
                    if "enable_categorical" in kwargs and _has_categorical_columns(X):
                        self._raise_incompatible_categorical_error(xgb.DMatrix, exc)
                    kwargs.pop("enable_categorical", None)
                    return xgb.DMatrix(X, **kwargs)
            raise

    def _resolve_train_params(self) -> Dict[str, Any]:
        params = dict(self.params)
        # enable_categorical is a DMatrix-side option; passing it to xgb.train
        # triggers "Parameters ... are not used" warnings.
        params.pop("enable_categorical", None)
        if not self.use_gpu:
            params["tree_method"] = "hist"
            params["predictor"] = "cpu_predictor"
            params.pop("gpu_id", None)
        return params

    @staticmethod
    def _slice_rows(data: Any, start: int, end: int) -> Any:
        if data is None:
            return None
        if hasattr(data, "iloc"):
            return data.iloc[start:end]
        return data[start:end]

    @staticmethod
    def _normalize_eval_weights(sample_weight_eval_set: Any) -> List[Any]:
        if sample_weight_eval_set is None:
            return []
        if isinstance(sample_weight_eval_set, (list, tuple)):
            return list(sample_weight_eval_set)
        return [sample_weight_eval_set]

    @staticmethod
    def _build_chunk_plan(
        *,
        total_rows: int,
        chunk_size: Optional[int],
        num_boost_round: int,
    ) -> List[Tuple[int, int, int]]:
        total_rows = int(total_rows)
        if total_rows < 1:
            return []
        total_rounds = max(1, int(num_boost_round))
        resolved_chunk_size = _XGBDMatrixWrapper._coerce_chunk_size(chunk_size)
        if resolved_chunk_size is None or resolved_chunk_size >= total_rows:
            return [(0, total_rows, total_rounds)]

        n_chunks = int(math.ceil(total_rows / float(resolved_chunk_size)))
        if n_chunks > total_rounds:
            n_chunks = total_rounds
            resolved_chunk_size = int(math.ceil(total_rows / float(n_chunks)))

        bounds: List[Tuple[int, int]] = []
        start = 0
        for _ in range(n_chunks):
            end = min(total_rows, start + resolved_chunk_size)
            bounds.append((start, end))
            start = end

        base_rounds = total_rounds // len(bounds)
        extra_rounds = total_rounds % len(bounds)

        plan: List[Tuple[int, int, int]] = []
        for idx, (chunk_start, chunk_end) in enumerate(bounds):
            rounds = base_rounds + (1 if idx < extra_rounds else 0)
            if rounds > 0 and chunk_end > chunk_start:
                plan.append((chunk_start, chunk_end, rounds))
        return plan

    def _build_eval_matrices(
        self,
        *,
        eval_set=None,
        sample_weight_eval_set=None,
    ) -> List[Tuple[xgb.DMatrix, str]]:
        evals: List[Tuple[xgb.DMatrix, str]] = []
        if not eval_set:
            return evals
        weights = self._normalize_eval_weights(sample_weight_eval_set)
        for idx, (X_val, y_val) in enumerate(eval_set):
            w_val = weights[idx] if idx < len(weights) else None
            dval = self._build_dmatrix(X_val, y_val, w_val)
            evals.append((dval, f"val{idx}"))
        return evals

    def _train_booster(
        self,
        X,
        y,
        *,
        sample_weight=None,
        eval_set=None,
        sample_weight_eval_set=None,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        params = self._resolve_train_params()
        num_boost_round = max(1, int(params.pop("n_estimators", 100)))
        evals = self._build_eval_matrices(
            eval_set=eval_set,
            sample_weight_eval_set=sample_weight_eval_set,
        )
        chunk_plan = self._build_chunk_plan(
            total_rows=len(X),
            chunk_size=self.chunk_size,
            num_boost_round=num_boost_round,
        )
        if not chunk_plan:
            raise ValueError("Training data is empty; cannot fit XGBoost model.")

        is_chunk_mode = len(chunk_plan) > 1
        if not is_chunk_mode:
            chunk_start, chunk_end, _ = chunk_plan[0]
            dtrain = self._build_dmatrix(
                self._slice_rows(X, chunk_start, chunk_end),
                self._slice_rows(y, chunk_start, chunk_end),
                self._slice_rows(sample_weight, chunk_start, chunk_end),
            )
            self._booster = xgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=evals,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose,
            )
            self.best_iteration = getattr(self._booster, "best_iteration", None)
            return

        if not self._chunk_mode_logged:
            _log(
                "[XGBoost] chunked training enabled: "
                f"chunk_size={self.chunk_size}, chunks={len(chunk_plan)}, "
                f"total_boost_rounds={num_boost_round}.",
                flush=True,
            )
            self._chunk_mode_logged = True
        if early_stopping_rounds is not None and not self._chunk_early_stopping_warned:
            _log(
                "[XGBoost] early_stopping_rounds is ignored when xgb_chunk_size is enabled.",
                flush=True,
            )
            self._chunk_early_stopping_warned = True

        booster: Optional[xgb.Booster] = None
        for chunk_start, chunk_end, chunk_rounds in chunk_plan:
            dtrain = self._build_dmatrix(
                self._slice_rows(X, chunk_start, chunk_end),
                self._slice_rows(y, chunk_start, chunk_end),
                self._slice_rows(sample_weight, chunk_start, chunk_end),
            )
            booster = xgb.train(
                params,
                dtrain,
                num_boost_round=chunk_rounds,
                evals=evals,
                early_stopping_rounds=None,
                verbose_eval=verbose,
                xgb_model=booster,
            )
        self._booster = booster
        self.best_iteration = None

    def fit(self, X, y, **fit_kwargs) -> "_XGBDMatrixWrapper":
        sample_weight = fit_kwargs.pop("sample_weight", None)
        eval_set = fit_kwargs.pop("eval_set", None)
        sample_weight_eval_set = fit_kwargs.pop("sample_weight_eval_set", None)
        early_stopping_rounds = fit_kwargs.pop("early_stopping_rounds", None)
        verbose = bool(fit_kwargs.pop("verbose", False))
        fit_kwargs.pop("eval_metric", None)
        try:
            self._train_booster(
                X,
                y,
                sample_weight=sample_weight,
                eval_set=eval_set,
                sample_weight_eval_set=sample_weight_eval_set,
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose,
            )
        except Exception as exc:
            if self.use_gpu and self.allow_cpu_fallback and _is_oom_error(exc):
                _log("[XGBoost] GPU OOM detected; retrying with CPU.", flush=True)
                self.use_gpu = False
                self._train_booster(
                    X,
                    y,
                    sample_weight=sample_weight,
                    eval_set=eval_set,
                    sample_weight_eval_set=sample_weight_eval_set,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=verbose,
                )
            else:
                raise
        return self

    def _resolve_iteration_range(self) -> Optional[Tuple[int, int]]:
        if self.best_iteration is None:
            return None
        return (0, int(self.best_iteration) + 1)

    def _predict_raw(self, X) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("Booster not trained.")
        dtest = self._build_dmatrix(X)
        iteration_range = self._resolve_iteration_range()
        if iteration_range is None:
            return self._booster.predict(dtest)
        try:
            return self._booster.predict(dtest, iteration_range=iteration_range)
        except TypeError:
            return self._booster.predict(dtest, ntree_limit=iteration_range[1])

    def predict(self, X, **_kwargs) -> np.ndarray:
        pred = self._predict_raw(X)
        if self.task_type == "classification":
            if pred.ndim == 1:
                return (pred > 0.5).astype(int)
            return np.argmax(pred, axis=1)
        return pred

    def predict_proba(self, X, **_kwargs) -> np.ndarray:
        pred = self._predict_raw(X)
        if pred.ndim == 1:
            return np.column_stack([1 - pred, pred])
        return pred

    def get_booster(self) -> Optional[xgb.Booster]:
        return self._booster


def _xgb_cuda_available() -> bool:
    # Best-effort check for XGBoost CUDA build; cached to avoid repeated checks.
    global _XGB_CUDA_CHECKED, _XGB_HAS_CUDA
    if _XGB_CUDA_CHECKED:
        return _XGB_HAS_CUDA
    _XGB_CUDA_CHECKED = True
    if not torch.cuda.is_available():
        _XGB_HAS_CUDA = False
        return False
    try:
        build_info = getattr(xgb, "build_info", None)
        if callable(build_info):
            info = build_info()
            for key in ("USE_CUDA", "use_cuda", "cuda"):
                if key in info:
                    val = info[key]
                    if isinstance(val, str):
                        _XGB_HAS_CUDA = val.strip().upper() in (
                            "ON", "YES", "TRUE", "1")
                    else:
                        _XGB_HAS_CUDA = bool(val)
                    return _XGB_HAS_CUDA
    except Exception:
        pass
    try:
        has_cuda = getattr(getattr(xgb, "core", None), "_has_cuda_support", None)
        if callable(has_cuda):
            _XGB_HAS_CUDA = bool(has_cuda())
            return _XGB_HAS_CUDA
    except Exception:
        pass
    _XGB_HAS_CUDA = False
    return False

class XGBTrainer(TrainerBase):
    def __init__(self, context: "BayesOptModel") -> None:
        super().__init__(context, 'Xgboost', 'Xgboost')
        self.model: Optional[xgb.XGBModel] = None
        self._xgb_use_gpu = False
        self._xgb_gpu_warned = False
        self._xgb_chunk_warned = False

    def _build_sklearn_estimator(self, params: Dict[str, Any]) -> xgb.XGBModel:
        if self.ctx.task_type == 'classification':
            return xgb.XGBClassifier(**params)
        return xgb.XGBRegressor(**params)

    def _build_estimator(self) -> xgb.XGBModel:
        use_gpu = bool(self.ctx.use_gpu and _xgb_cuda_available())
        self._xgb_use_gpu = use_gpu
        params = dict(
            objective=self.ctx.obj,
            random_state=self.ctx.rand_seed,
            subsample=0.9,
            tree_method='gpu_hist' if use_gpu else 'hist',
            enable_categorical=True,
            predictor='gpu_predictor' if use_gpu else 'cpu_predictor'
        )
        if self.ctx.use_gpu and not use_gpu and not self._xgb_gpu_warned:
            _log(
                "[XGBoost] CUDA requested but not available; falling back to CPU.",
                flush=True,
            )
            self._xgb_gpu_warned = True
        if use_gpu:
            gpu_id = self._resolve_gpu_id()
            params['gpu_id'] = gpu_id
            _log(f">>> XGBoost using GPU ID: {gpu_id}")
        eval_metric = self._resolve_eval_metric()
        if eval_metric is not None:
            params.setdefault("eval_metric", eval_metric)
        use_dmatrix = bool(getattr(self.config, "xgb_use_dmatrix", True))
        chunk_size = self._resolve_chunk_size()
        if chunk_size is not None and not use_dmatrix:
            if not self._xgb_chunk_warned:
                _log(
                    "[XGBoost] xgb_chunk_size is set while xgb_use_dmatrix=false; "
                    "forcing xgb_use_dmatrix=true.",
                    flush=True,
                )
                self._xgb_chunk_warned = True
            use_dmatrix = True
        if use_dmatrix:
            return _XGBDMatrixWrapper(
                params,
                task_type=self.ctx.task_type,
                use_gpu=use_gpu,
                chunk_size=chunk_size,
            )
        return self._build_sklearn_estimator(params)

    def _resolve_chunk_size(self) -> Optional[int]:
        raw = getattr(self.config, "xgb_chunk_size", None)
        if raw is None:
            return None
        try:
            chunk_size = int(raw)
        except (TypeError, ValueError):
            return None
        if chunk_size < 1:
            return None
        return chunk_size

    def _resolve_gpu_id(self) -> int:
        gpu_id = getattr(self.config, "xgb_gpu_id", None)
        if gpu_id is None:
            return 0
        try:
            return int(gpu_id)
        except (TypeError, ValueError):
            return 0

    def _maybe_cleanup_gpu(self) -> None:
        if not bool(getattr(self.config, "xgb_cleanup_per_fold", False)):
            return
        synchronize = bool(getattr(self.config, "xgb_cleanup_synchronize", False))
        self._clean_gpu(synchronize=synchronize)

    def _resolve_eval_metric(self) -> Optional[Any]:
        fit_params = self.ctx.fit_params or {}
        eval_metric = fit_params.get("eval_metric")
        if eval_metric is None:
            return "logloss" if self.ctx.task_type == 'classification' else "rmse"
        return eval_metric

    def _fit_supports_param(self, name: str) -> bool:
        try:
            fit = xgb.XGBClassifier.fit if self.ctx.task_type == 'classification' else xgb.XGBRegressor.fit
            return name in inspect.signature(fit).parameters
        except (TypeError, ValueError):
            return True

    def _resolve_early_stopping_rounds(self, n_estimators: int) -> int:
        n_estimators = max(1, int(n_estimators))
        base = max(5, n_estimators // 10)
        return min(50, base)

    def _build_fit_kwargs(self,
                          w_train,
                          X_val=None,
                          y_val=None,
                          w_val=None,
                          n_estimators: Optional[int] = None) -> Dict[str, Any]:
        supports_early = self._fit_supports_param("early_stopping_rounds")
        fit_kwargs = dict(self.ctx.fit_params or {})
        fit_kwargs.pop("sample_weight", None)
        fit_kwargs.pop("eval_metric", None)
        fit_kwargs["sample_weight"] = w_train

        if "eval_set" not in fit_kwargs and X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            if w_val is not None:
                fit_kwargs["sample_weight_eval_set"] = [w_val]

        if (
            supports_early
            and "early_stopping_rounds" not in fit_kwargs
            and "eval_set" in fit_kwargs
        ):
            rounds = self._resolve_early_stopping_rounds(n_estimators or 100)
            fit_kwargs["early_stopping_rounds"] = rounds
        if not supports_early:
            fit_kwargs.pop("early_stopping_rounds", None)

        fit_kwargs.setdefault("verbose", False)
        return fit_kwargs

    def _resolve_optuna_total_trials(self) -> int:
        raw = getattr(self, "_optuna_total_trials", None)
        if raw is None:
            return 100
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            return 100

    def _should_prune_slow_config(
        self,
        *,
        max_depth: int,
        n_estimators: int,
    ) -> bool:
        if max_depth < 20 or n_estimators < 300:
            return False
        # Avoid pruning away the full search budget when max_evals is tiny.
        return self._resolve_optuna_total_trials() > 2

    def ensemble_predict(self, k: int) -> None:
        if not self.best_params:
            raise RuntimeError("Run tune() first to obtain best XGB parameters.")
        k = max(2, int(k))
        X_all = self.ctx.train_data[self.ctx.factor_nmes]
        y_all = self.ctx.train_data[self.ctx.resp_nme].values
        w_all = self.ctx.train_data[self.ctx.weight_nme].values
        X_test = self.ctx.test_data[self.ctx.factor_nmes]
        n_samples = len(X_all)
        split_iter, _ = self._resolve_ensemble_splits(X_all, k=k)
        if split_iter is None:
            _log(
                f"[XGB Ensemble] unable to build CV split (n_samples={n_samples}); skip ensemble.",
                flush=True,
            )
            return
        preds_train_sum = np.zeros(n_samples, dtype=np.float64)
        preds_test_sum = np.zeros(len(X_test), dtype=np.float64)

        split_count = 0
        for train_idx, val_idx in split_iter:
            X_train = X_all.iloc[train_idx]
            y_train = y_all[train_idx]
            w_train = w_all[train_idx]
            X_val = X_all.iloc[val_idx]
            y_val = y_all[val_idx]
            w_val = w_all[val_idx]

            clf = self._build_estimator()
            clf.set_params(**self.best_params)
            fit_kwargs = self._build_fit_kwargs(
                w_train=w_train,
                X_val=X_val,
                y_val=y_val,
                w_val=w_val,
                n_estimators=self.best_params.get("n_estimators", 100),
            )
            clf.fit(X_train, y_train, **fit_kwargs)

            if self.ctx.task_type == 'classification':
                pred_train = clf.predict_proba(X_all)[:, 1]
                pred_test = clf.predict_proba(X_test)[:, 1]
            else:
                pred_train = clf.predict(X_all)
                pred_test = clf.predict(X_test)
            preds_train_sum += np.asarray(pred_train, dtype=np.float64)
            preds_test_sum += np.asarray(pred_test, dtype=np.float64)
            self._maybe_cleanup_gpu()
            split_count += 1

        if split_count < 1:
            _log(
                f"[XGB Ensemble] no CV splits generated; skip ensemble.",
                flush=True,
            )
            return
        preds_train = preds_train_sum / float(split_count)
        preds_test = preds_test_sum / float(split_count)
        self._cache_predictions("xgb", preds_train, preds_test)

    def cross_val(self, trial: optuna.trial.Trial) -> float:
        learning_rate = trial.suggest_float(
            'learning_rate', 1e-5, 1e-1, log=True)
        gamma = trial.suggest_float('gamma', 0, 10000)
        max_depth_max = max(
            3, int(getattr(self.config, "xgb_max_depth_max", 25)))
        n_estimators_max = max(
            10, int(getattr(self.config, "xgb_n_estimators_max", 500)))
        max_depth = trial.suggest_int('max_depth', 3, max_depth_max)
        n_estimators = trial.suggest_int(
            'n_estimators', 10, n_estimators_max, step=10)
        min_child_weight = trial.suggest_int(
            'min_child_weight', 100, 10000, step=100)
        reg_alpha = trial.suggest_float('reg_alpha', 1e-10, 1, log=True)
        reg_lambda = trial.suggest_float('reg_lambda', 1e-10, 1, log=True)
        if trial is not None:
            _log(
                f"[Optuna][Xgboost] trial_id={trial.number} max_depth={max_depth} "
                f"n_estimators={n_estimators}",
                flush=True,
            )
        if self._should_prune_slow_config(
            max_depth=max_depth,
            n_estimators=n_estimators,
        ):
            raise optuna.TrialPruned(
                "XGB config is likely too slow (max_depth>=20 & n_estimators>=300)")
        clf = self._build_estimator()
        params = {
            'learning_rate': learning_rate,
            'gamma': gamma,
            'max_depth': max_depth,
            'n_estimators': n_estimators,
            'min_child_weight': min_child_weight,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda
        }
        loss_name = getattr(self.ctx, "loss_name", "tweedie")
        tweedie_variance_power = None
        if self.ctx.task_type != 'classification':
            if loss_name == "tweedie":
                tweedie_variance_power = trial.suggest_float(
                    'tweedie_variance_power', 1, 2)
                params['tweedie_variance_power'] = tweedie_variance_power
            elif loss_name == "poisson":
                tweedie_variance_power = 1.0
            elif loss_name == "gamma":
                tweedie_variance_power = 2.0
        X_all = self.ctx.train_data[self.ctx.factor_nmes]
        y_all = self.ctx.train_data[self.ctx.resp_nme].values
        w_all = self.ctx.train_data[self.ctx.weight_nme].values

        losses: List[float] = []
        for train_idx, val_idx in self.ctx.cv.split(X_all):
            X_train = X_all.iloc[train_idx]
            y_train = y_all[train_idx]
            w_train = w_all[train_idx]
            X_val = X_all.iloc[val_idx]
            y_val = y_all[val_idx]
            w_val = w_all[val_idx]

            clf = self._build_estimator()
            clf.set_params(**params)
            fit_kwargs = self._build_fit_kwargs(
                w_train=w_train,
                X_val=X_val,
                y_val=y_val,
                w_val=w_val,
                n_estimators=n_estimators,
            )
            clf.fit(X_train, y_train, **fit_kwargs)

            if self.ctx.task_type == 'classification':
                y_pred = clf.predict_proba(X_val)[:, 1]
                y_pred = np.clip(y_pred, EPS, 1 - EPS)
                loss = log_loss(y_val, y_pred, sample_weight=w_val)
            else:
                y_pred = clf.predict(X_val)
                loss = regression_loss(
                    y_val,
                    y_pred,
                    w_val,
                    loss_name=loss_name,
                    tweedie_power=tweedie_variance_power,
                )
            losses.append(float(loss))
            self._maybe_cleanup_gpu()

        return float(np.mean(losses))

    def train(self) -> None:
        if not self.best_params:
            raise RuntimeError("Run tune() first to obtain best XGB parameters.")
        self.model = self._build_estimator()
        self.model.set_params(**self.best_params)
        use_refit = bool(getattr(self.ctx.config, "final_refit", True))
        predict_fn = None
        if self.ctx.task_type == 'classification':
            def _predict_proba(X, **_kwargs):
                return self.model.predict_proba(X)[:, 1]
            predict_fn = _predict_proba
        X_all = self.ctx.train_data[self.ctx.factor_nmes]
        y_all = self.ctx.train_data[self.ctx.resp_nme].values
        w_all = self.ctx.train_data[self.ctx.weight_nme].values

        split = self._resolve_train_val_indices(X_all)
        if split is not None:
            train_idx, val_idx = split
            X_train = X_all.iloc[train_idx]
            y_train = y_all[train_idx]
            w_train = w_all[train_idx]
            X_val = X_all.iloc[val_idx]
            y_val = y_all[val_idx]
            w_val = w_all[val_idx]
            fit_kwargs = self._build_fit_kwargs(
                w_train=w_train,
                X_val=X_val,
                y_val=y_val,
                w_val=w_val,
                n_estimators=self.best_params.get("n_estimators", 100),
            )
            self.model.fit(X_train, y_train, **fit_kwargs)
            best_iter = getattr(self.model, "best_iteration", None)
            if use_refit and best_iter is not None:
                refit_model = self._build_estimator()
                refit_params = dict(self.best_params)
                refit_params["n_estimators"] = int(best_iter) + 1
                refit_model.set_params(**refit_params)
                refit_kwargs = dict(self.ctx.fit_params or {})
                refit_kwargs.setdefault("sample_weight", w_all)
                refit_kwargs.pop("eval_set", None)
                refit_kwargs.pop("sample_weight_eval_set", None)
                refit_kwargs.pop("early_stopping_rounds", None)
                refit_kwargs.pop("eval_metric", None)
                refit_kwargs.setdefault("verbose", False)
                try:
                    refit_model.fit(X_all, y_all, **refit_kwargs)
                    self.model = refit_model
                except Exception as exc:
                    if _is_host_memory_error(exc):
                        _log(
                            "[XGBoost] final_refit failed due to host memory; "
                            "keeping early-stopped model. Consider setting "
                            "final_refit=false for large datasets.",
                            flush=True,
                        )
                    else:
                        raise
        else:
            fit_kwargs = dict(self.ctx.fit_params or {})
            fit_kwargs.setdefault("sample_weight", w_all)
            fit_kwargs.pop("eval_metric", None)
            self.model.fit(X_all, y_all, **fit_kwargs)

        self.ctx.model_label.append(self.label)
        self._predict_and_cache(
            self.model,
            pred_prefix='xgb',
            predict_fn=predict_fn
        )
        self.ctx.xgb_best = self.model
