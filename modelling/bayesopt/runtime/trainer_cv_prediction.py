from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ins_pricing.modelling.bayesopt.trainers.cv_utils import CVStrategyResolver
from ins_pricing.utils import GPUMemoryManager, get_logger, log_print

_logger = get_logger("ins_pricing.trainer")


def _log(*args, **kwargs) -> None:
    log_print(_logger, *args, **kwargs)


class TrainerCVPredictionMixin:
    def _classification_prediction_outputs(self) -> str:
        cfg = getattr(self.ctx, "config", None)
        raw = getattr(cfg, "classification_prediction_outputs", "score")
        mode = str(raw or "score").strip().lower()
        return mode if mode in {"score", "both"} else "score"

    def _classification_label_threshold(self) -> float:
        cfg = getattr(self.ctx, "config", None)
        raw = getattr(cfg, "classification_label_threshold", 0.5)
        try:
            threshold = float(raw)
        except (TypeError, ValueError):
            threshold = 0.5
        return min(1.0, max(0.0, threshold))

    def _derive_classification_labels(self, preds: Any) -> np.ndarray:
        arr = np.asarray(preds)
        threshold = self._classification_label_threshold()
        if arr.ndim <= 1:
            return (arr.reshape(-1) >= threshold).astype(int)
        if arr.ndim == 2:
            if arr.shape[1] == 1:
                return (arr[:, 0] >= threshold).astype(int)
            return np.argmax(arr, axis=1).astype(int)
        raise ValueError(f"Unexpected classification prediction shape: {arr.shape}")

    def _store_classification_label_predictions(
        self,
        pred_prefix: str,
        preds_train: Any,
        preds_test: Any,
    ) -> None:
        if str(getattr(self.ctx, "task_type", "")).lower() != "classification":
            return
        if self._classification_prediction_outputs() != "both":
            return
        label_col = f"pred_label_{pred_prefix}"
        train_labels = self._derive_classification_labels(preds_train)
        test_labels = self._derive_classification_labels(preds_test)
        self.ctx.train_data[label_col] = train_labels
        self.ctx.test_data[label_col] = test_labels
        self.ctx.train_data[f"w_{label_col}"] = (
            self.ctx.train_data[label_col] * self.ctx.train_data[self.ctx.weight_nme]
        )
        self.ctx.test_data[f"w_{label_col}"] = (
            self.ctx.test_data[label_col] * self.ctx.test_data[self.ctx.weight_nme]
        )

    def _clean_gpu(
        self,
        *,
        synchronize: bool = True,
        empty_cache: bool = True,
    ) -> None:
        """Clean up GPU memory using shared GPUMemoryManager."""
        GPUMemoryManager.clean(synchronize=synchronize, empty_cache=empty_cache)

    def _standardize_fold(self,
                          X_train: pd.DataFrame,
                          X_val: pd.DataFrame,
                          columns: Optional[List[str]] = None
                          ) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
        """Fit StandardScaler on the training fold and transform train/val features."""
        scaler = StandardScaler()
        cols = list(columns) if columns else list(X_train.columns)
        X_train_scaled = X_train.copy(deep=True)
        X_val_scaled = X_val.copy(deep=True)
        if cols:
            scaler.fit(X_train_scaled[cols])
            X_train_scaled[cols] = scaler.transform(X_train_scaled[cols])
            X_val_scaled[cols] = scaler.transform(X_val_scaled[cols])
        return X_train_scaled, X_val_scaled, scaler

    def _resolve_train_val_indices(
        self,
        X_all: pd.DataFrame,
        *,
        allow_default: bool = False,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Resolve train/validation split indices based on configured CV strategy."""
        val_ratio = float(self.ctx.prop_test) if self.ctx.prop_test is not None else 0.25
        if not (0.0 < val_ratio < 1.0):
            if not allow_default:
                return None
            val_ratio = 0.25
        if len(X_all) < 10:
            return None

        base_data = self.ctx.train_data.loc[X_all.index]
        resolver = CVStrategyResolver(self.ctx.config, base_data, self.ctx.rand_seed)
        (train_idx, val_idx), _ = resolver.create_train_val_splitter(X_all, val_ratio)
        return train_idx, val_idx

    def _resolve_time_sample_indices(
        self,
        X_all: pd.DataFrame,
        sample_limit: int,
    ) -> Optional[pd.Index]:
        """Get the most recent indices for time-based sampling."""
        if sample_limit <= 0:
            return None

        base_data = self.ctx.train_data.loc[X_all.index]
        resolver = CVStrategyResolver(self.ctx.config, base_data, self.ctx.rand_seed)
        if not resolver.is_time_strategy():
            return None

        order = resolver.get_time_ordered_indices(X_all)
        if len(order) == 0:
            return None

        if len(order) > sample_limit:
            order = order[-sample_limit:]

        return X_all.index[order]

    def _resolve_ensemble_splits(
        self,
        X_all: pd.DataFrame,
        *,
        k: int,
    ) -> Tuple[Optional[Iterable[Tuple[np.ndarray, np.ndarray]]], int]:
        """Resolve K-fold splits for ensemble training based on configured CV strategy."""
        base_data = self.ctx.train_data.loc[X_all.index]
        resolver = CVStrategyResolver(self.ctx.config, base_data, self.ctx.rand_seed)
        return resolver.create_kfold_splitter(X_all, k)

    def cross_val_generic(
            self,
            trial: optuna.trial.Trial,
            hyperparameter_space: Dict[str, Callable[[optuna.trial.Trial], Any]],
            data_provider: Callable[[], Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]],
            model_builder: Callable[[Dict[str, Any]], Any],
            metric_fn: Callable[[pd.Series, np.ndarray, Optional[pd.Series]], float],
            sample_limit: Optional[int] = None,
            preprocess_fn: Optional[Callable[[
                pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]] = None,
            fit_predict_fn: Optional[
                Callable[[Any, pd.DataFrame, pd.Series, Optional[pd.Series],
                          pd.DataFrame, pd.Series, Optional[pd.Series],
                          optuna.trial.Trial], np.ndarray]
            ] = None,
            cleanup_fn: Optional[Callable[[Any], None]] = None,
            splitter: Optional[Iterable[Tuple[np.ndarray, np.ndarray]]] = None) -> float:
        """Generic holdout/CV helper to reuse tuning workflows."""
        params: Optional[Dict[str, Any]] = None
        if self._distributed_forced_params is not None:
            params = self._distributed_forced_params
            self._distributed_forced_params = None
        else:
            if trial is None:
                raise RuntimeError(
                    "Missing Optuna trial for parameter sampling.")
            params = {name: sampler(trial)
                      for name, sampler in hyperparameter_space.items()}
            if self._should_use_distributed_optuna():
                self._distributed_prepare_trial(params)
        X_all, y_all, w_all = data_provider()
        ctx_config = getattr(self.ctx, "config", None)
        cfg_limit = getattr(ctx_config, "bo_sample_limit", None)
        if cfg_limit is not None:
            cfg_limit = int(cfg_limit)
            if cfg_limit > 0:
                sample_limit = cfg_limit if sample_limit is None else min(sample_limit, cfg_limit)
        if sample_limit is not None and len(X_all) > sample_limit:
            sampled_idx = self._resolve_time_sample_indices(X_all, int(sample_limit))
            if sampled_idx is None:
                sampled_idx = X_all.sample(
                    n=sample_limit,
                    random_state=self.ctx.rand_seed
                ).index
            X_all = X_all.loc[sampled_idx]
            y_all = y_all.loc[sampled_idx]
            w_all = w_all.loc[sampled_idx] if w_all is not None else None

        if splitter is None:
            val_ratio = float(self.ctx.prop_test) if self.ctx.prop_test is not None else 0.25
            if not (0.0 < val_ratio < 1.0):
                val_ratio = 0.25
            cv_splits = getattr(self.ctx.config, "cv_splits", None)
            if cv_splits is None:
                cv_splits = max(2, int(round(1 / val_ratio)))
            cv_splits = max(2, int(cv_splits))

            base_data = self.ctx.train_data.loc[X_all.index]
            resolver = CVStrategyResolver(self.ctx.config, base_data, self.ctx.rand_seed)
            split_iter, actual_splits = resolver.create_cv_splitter(X_all, y_all, cv_splits, val_ratio)
            if actual_splits < 2:
                raise ValueError("Not enough samples for cross-validation.")
        else:
            if hasattr(splitter, "split"):
                split_iter = splitter.split(X_all, y_all, groups=None)
            else:
                split_iter = splitter

        losses: List[float] = []
        for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
            X_train = X_all.iloc[train_idx]
            y_train = y_all.iloc[train_idx]
            X_val = X_all.iloc[val_idx]
            y_val = y_all.iloc[val_idx]
            w_train = w_all.iloc[train_idx] if w_all is not None else None
            w_val = w_all.iloc[val_idx] if w_all is not None else None

            if preprocess_fn:
                X_train, X_val = preprocess_fn(X_train, X_val)

            model = model_builder(params)
            try:
                if fit_predict_fn:
                    trial_for_fold = trial if fold_idx == 0 else None
                    y_pred = fit_predict_fn(
                        model, X_train, y_train, w_train,
                        X_val, y_val, w_val, trial_for_fold
                    )
                else:
                    fit_kwargs = {}
                    if w_train is not None:
                        fit_kwargs["sample_weight"] = w_train
                    model.fit(X_train, y_train, **fit_kwargs)
                    y_pred = model.predict(X_val)
                losses.append(metric_fn(y_val, y_pred, w_val))
            finally:
                if cleanup_fn:
                    cleanup_fn(model)
                self._clean_gpu()

        return float(np.mean(losses))

    def _predict_and_cache(self,
                           model,
                           pred_prefix: str,
                           use_oht: bool = False,
                           design_fn=None,
                           predict_kwargs_train: Optional[Dict[str, Any]] = None,
                           predict_kwargs_test: Optional[Dict[str, Any]] = None,
                           predict_fn: Optional[Callable[..., Any]] = None) -> None:
        if design_fn:
            X_train = design_fn(train=True)
            X_test = design_fn(train=False)
        elif use_oht:
            X_train = self.ctx.train_oht_scl_data[self.ctx.var_nmes]
            X_test = self.ctx.test_oht_scl_data[self.ctx.var_nmes]
        else:
            X_train = self.ctx.train_data[self.ctx.factor_nmes]
            X_test = self.ctx.test_data[self.ctx.factor_nmes]

        predictor = predict_fn or model.predict
        preds_train = predictor(X_train, **(predict_kwargs_train or {}))
        preds_test = predictor(X_test, **(predict_kwargs_test or {}))
        preds_train, preds_test = self._store_predictions(
            pred_prefix, preds_train, preds_test
        )
        self._maybe_cache_predictions(pred_prefix, preds_train, preds_test)

    def _normalize_prediction_arrays(
        self,
        pred_prefix: str,
        preds_train: Any,
        preds_test: Any,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        train_arr = np.asarray(preds_train)
        test_arr = np.asarray(preds_test)
        is_scalar = train_arr.ndim <= 1 or (
            train_arr.ndim == 2 and train_arr.shape[1] == 1
        )
        if is_scalar:
            return train_arr.reshape(-1), test_arr.reshape(-1), False
        if train_arr.ndim != 2:
            raise ValueError(
                f"Unexpected prediction shape for '{pred_prefix}': {train_arr.shape}"
            )
        if test_arr.ndim != 2 or test_arr.shape[1] != train_arr.shape[1]:
            raise ValueError(
                f"Train/test prediction dims mismatch for '{pred_prefix}': "
                f"{train_arr.shape} vs {test_arr.shape}"
            )
        return train_arr, test_arr, True

    def _store_predictions(self,
                           pred_prefix: str,
                           preds_train: Any,
                           preds_test: Any) -> Tuple[np.ndarray, np.ndarray]:
        train_arr, test_arr, is_vector = self._normalize_prediction_arrays(
            pred_prefix, preds_train, preds_test
        )
        if is_vector:
            self._assign_vector_predictions(pred_prefix, train_arr, test_arr)
            self._store_classification_label_predictions(
                pred_prefix, train_arr, test_arr
            )
            return train_arr, test_arr

        col_name = f'pred_{pred_prefix}'
        self.ctx.train_data[col_name] = train_arr
        self.ctx.test_data[col_name] = test_arr
        self.ctx.train_data[f'w_{col_name}'] = (
            self.ctx.train_data[col_name] *
            self.ctx.train_data[self.ctx.weight_nme]
        )
        self.ctx.test_data[f'w_{col_name}'] = (
            self.ctx.test_data[col_name] *
            self.ctx.test_data[self.ctx.weight_nme]
        )
        self._store_classification_label_predictions(
            pred_prefix, train_arr, test_arr
        )
        return train_arr, test_arr

    def _assign_vector_predictions(self,
                                   pred_prefix: str,
                                   preds_train: np.ndarray,
                                   preds_test: np.ndarray) -> None:
        col_names = [f'pred_{pred_prefix}_{j}' for j in range(preds_train.shape[1])]
        train_block = pd.DataFrame(
            preds_train, columns=col_names, index=self.ctx.train_data.index)
        test_block = pd.DataFrame(
            preds_test, columns=col_names, index=self.ctx.test_data.index)

        self.ctx.train_data = pd.concat(
            [self.ctx.train_data.drop(columns=col_names, errors='ignore'), train_block],
            axis=1
        )
        self.ctx.test_data = pd.concat(
            [self.ctx.test_data.drop(columns=col_names, errors='ignore'), test_block],
            axis=1
        )

    def _cache_predictions(self,
                           pred_prefix: str,
                           preds_train,
                           preds_test) -> None:
        preds_train, preds_test = self._store_predictions(
            pred_prefix, preds_train, preds_test
        )
        self._maybe_cache_predictions(pred_prefix, preds_train, preds_test)

    def _maybe_cache_predictions(self, pred_prefix: str, preds_train, preds_test) -> None:
        cfg = getattr(self.ctx, "config", None)
        if cfg is None or not bool(getattr(cfg, "cache_predictions", False)):
            return
        fmt = str(getattr(cfg, "prediction_cache_format", "parquet") or "parquet").lower()
        cache_dir = getattr(cfg, "prediction_cache_dir", None)
        if cache_dir:
            target_dir = Path(str(cache_dir))
            if not target_dir.is_absolute():
                target_dir = Path(self.output.result_dir) / target_dir
        else:
            target_dir = Path(self.output.result_dir) / "predictions"
        target_dir.mkdir(parents=True, exist_ok=True)

        def _build_frame(preds) -> pd.DataFrame:
            arr = np.asarray(preds)
            if arr.ndim <= 1:
                return pd.DataFrame({f"pred_{pred_prefix}": arr.reshape(-1)})
            cols = [f"pred_{pred_prefix}_{i}" for i in range(arr.shape[1])]
            return pd.DataFrame(arr, columns=cols)

        for split_label, preds in [("train", preds_train), ("test", preds_test)]:
            frame = _build_frame(preds)
            filename = f"{self.ctx.model_nme}_{pred_prefix}_{split_label}.{ 'csv' if fmt == 'csv' else 'parquet' }"
            path = target_dir / filename
            try:
                if fmt == "csv":
                    frame.to_csv(path, index=False)
                else:
                    frame.to_parquet(path, index=False)
            except Exception as exc:
                _log(
                    f"[PredictionCache] Failed to persist {path}: {exc}",
                    flush=True,
                )

    def _resolve_best_epoch(self,
                            history: Optional[Dict[str, List[float]]],
                            default_epochs: int) -> int:
        if not history:
            return max(1, int(default_epochs))
        vals = history.get("val") or []
        if not vals:
            return max(1, int(default_epochs))
        best_idx = int(np.nanargmin(vals))
        return max(1, best_idx + 1)

    def _fit_predict_cache(self,
                           model,
                           X_train,
                           y_train,
                           sample_weight,
                           pred_prefix: str,
                           use_oht: bool = False,
                           design_fn=None,
                           fit_kwargs: Optional[Dict[str, Any]] = None,
                           sample_weight_arg: Optional[str] = 'sample_weight',
                           predict_kwargs_train: Optional[Dict[str, Any]] = None,
                           predict_kwargs_test: Optional[Dict[str, Any]] = None,
                           predict_fn: Optional[Callable[..., Any]] = None,
                           record_label: bool = True) -> None:
        fit_kwargs = fit_kwargs.copy() if fit_kwargs else {}
        if sample_weight is not None and sample_weight_arg:
            fit_kwargs.setdefault(sample_weight_arg, sample_weight)
        model.fit(X_train, y_train, **fit_kwargs)
        if record_label:
            self.ctx.model_label.append(self.label)
        self._predict_and_cache(
            model,
            pred_prefix,
            use_oht=use_oht,
            design_fn=design_fn,
            predict_kwargs_train=predict_kwargs_train,
            predict_kwargs_test=predict_kwargs_test,
            predict_fn=predict_fn)


__all__ = ["TrainerCVPredictionMixin"]
