from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import torch
import xgboost as xgb
from sklearn.metrics import log_loss, mean_tweedie_deviance

from .trainer_base import TrainerBase
from ..utils import EPS

_XGB_CUDA_CHECKED = False
_XGB_HAS_CUDA = False

_XGB_CUDA_CHECKED = False
_XGB_HAS_CUDA = False


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
            print(
                "[XGBoost] CUDA requested but not available; falling back to CPU.",
                flush=True,
            )
            self._xgb_gpu_warned = True
        if use_gpu:
            params['gpu_id'] = 0
            print(f">>> XGBoost using GPU ID: 0 (Single GPU Mode)")
        eval_metric = self._resolve_eval_metric()
        if eval_metric is not None:
            params.setdefault("eval_metric", eval_metric)
        if self.ctx.task_type == 'classification':
            return xgb.XGBClassifier(**params)
        return xgb.XGBRegressor(**params)

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
            print(
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
            self._clean_gpu()
            split_count += 1

        if split_count < 1:
            print(
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
            print(
                f"[Optuna][Xgboost] trial_id={trial.number} max_depth={max_depth} "
                f"n_estimators={n_estimators}",
                flush=True,
            )
        if max_depth >= 20 and n_estimators >= 300:
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
        tweedie_variance_power = None
        if self.ctx.task_type != 'classification':
            if self.ctx.obj == 'reg:tweedie':
                tweedie_variance_power = trial.suggest_float(
                    'tweedie_variance_power', 1, 2)
                params['tweedie_variance_power'] = tweedie_variance_power
            elif self.ctx.obj == 'count:poisson':
                tweedie_variance_power = 1
            elif self.ctx.obj == 'reg:gamma':
                tweedie_variance_power = 2
            else:
                tweedie_variance_power = 1.5
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
                y_pred_safe = np.maximum(y_pred, EPS)
                loss = mean_tweedie_deviance(
                    y_val,
                    y_pred_safe,
                    sample_weight=w_val,
                    power=tweedie_variance_power,
                )
            losses.append(float(loss))
            self._clean_gpu()

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
                refit_model.fit(X_all, y_all, **refit_kwargs)
                self.model = refit_model
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


