from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import log_loss, mean_tweedie_deviance

from .trainer_base import TrainerBase
from ..utils import EPS

class GLMTrainer(TrainerBase):
    def __init__(self, context: "BayesOptModel") -> None:
        super().__init__(context, 'GLM', 'GLM')
        self.model = None

    def _select_family(self, tweedie_power: Optional[float] = None):
        if self.ctx.task_type == 'classification':
            return sm.families.Binomial()
        if self.ctx.obj == 'count:poisson':
            return sm.families.Poisson()
        if self.ctx.obj == 'reg:gamma':
            return sm.families.Gamma()
        power = tweedie_power if tweedie_power is not None else 1.5
        return sm.families.Tweedie(var_power=power, link=sm.families.links.log())

    def _prepare_design(self, data: pd.DataFrame) -> pd.DataFrame:
        # Add intercept to the statsmodels design matrix.
        X = data[self.ctx.var_nmes]
        return sm.add_constant(X, has_constant='add')

    def _metric_power(self, family, tweedie_power: Optional[float]) -> float:
        if isinstance(family, sm.families.Poisson):
            return 1.0
        if isinstance(family, sm.families.Gamma):
            return 2.0
        if isinstance(family, sm.families.Tweedie):
            return tweedie_power if tweedie_power is not None else getattr(family, 'var_power', 1.5)
        return 1.5

    def cross_val(self, trial: optuna.trial.Trial) -> float:
        param_space = {
            "alpha": lambda t: t.suggest_float('alpha', 1e-6, 1e2, log=True),
            "l1_ratio": lambda t: t.suggest_float('l1_ratio', 0.0, 1.0)
        }
        if self.ctx.task_type == 'regression' and self.ctx.obj == 'reg:tweedie':
            param_space["tweedie_power"] = lambda t: t.suggest_float(
                'tweedie_power', 1.0, 2.0)

        def data_provider():
            data = self.ctx.train_oht_data if self.ctx.train_oht_data is not None else self.ctx.train_oht_scl_data
            assert data is not None, "Preprocessed training data is missing."
            return data[self.ctx.var_nmes], data[self.ctx.resp_nme], data[self.ctx.weight_nme]

        def preprocess_fn(X_train, X_val):
            X_train_s, X_val_s, _ = self._standardize_fold(
                X_train, X_val, self.ctx.num_features)
            return self._prepare_design(X_train_s), self._prepare_design(X_val_s)

        metric_ctx: Dict[str, Any] = {}

        def model_builder(params):
            family = self._select_family(params.get("tweedie_power"))
            metric_ctx["family"] = family
            metric_ctx["tweedie_power"] = params.get("tweedie_power")
            return {
                "family": family,
                "alpha": params["alpha"],
                "l1_ratio": params["l1_ratio"],
                "tweedie_power": params.get("tweedie_power")
            }

        def fit_predict(model_cfg, X_train, y_train, w_train, X_val, y_val, w_val, _trial):
            glm = sm.GLM(y_train, X_train,
                         family=model_cfg["family"],
                         freq_weights=w_train)
            result = glm.fit_regularized(
                alpha=model_cfg["alpha"],
                L1_wt=model_cfg["l1_ratio"],
                maxiter=200
            )
            return result.predict(X_val)

        def metric_fn(y_true, y_pred, weight):
            if self.ctx.task_type == 'classification':
                y_pred_clipped = np.clip(y_pred, EPS, 1 - EPS)
                return log_loss(y_true, y_pred_clipped, sample_weight=weight)
            y_pred_safe = np.maximum(y_pred, EPS)
            return mean_tweedie_deviance(
                y_true,
                y_pred_safe,
                sample_weight=weight,
                power=self._metric_power(
                    metric_ctx.get("family"), metric_ctx.get("tweedie_power"))
            )

        return self.cross_val_generic(
            trial=trial,
            hyperparameter_space=param_space,
            data_provider=data_provider,
            model_builder=model_builder,
            metric_fn=metric_fn,
            preprocess_fn=preprocess_fn,
            fit_predict_fn=fit_predict
        )

    def train(self) -> None:
        if not self.best_params:
            raise RuntimeError("Run tune() first to obtain best GLM parameters.")
        tweedie_power = self.best_params.get('tweedie_power')
        family = self._select_family(tweedie_power)

        X_train = self._prepare_design(self.ctx.train_oht_scl_data)
        y_train = self.ctx.train_oht_scl_data[self.ctx.resp_nme]
        w_train = self.ctx.train_oht_scl_data[self.ctx.weight_nme]

        glm = sm.GLM(y_train, X_train, family=family,
                     freq_weights=w_train)
        self.model = glm.fit_regularized(
            alpha=self.best_params['alpha'],
            L1_wt=self.best_params['l1_ratio'],
            maxiter=300
        )

        self.ctx.glm_best = self.model
        self.ctx.model_label += [self.label]
        self._predict_and_cache(
            self.model,
            'glm',
            design_fn=lambda train: self._prepare_design(
                self.ctx.train_oht_scl_data if train else self.ctx.test_oht_scl_data
            )
        )

    def ensemble_predict(self, k: int) -> None:
        if not self.best_params:
            raise RuntimeError("Run tune() first to obtain best GLM parameters.")
        k = max(2, int(k))
        data = self.ctx.train_oht_scl_data
        if data is None:
            raise RuntimeError("Missing standardized data for GLM ensemble.")
        X_all = data[self.ctx.var_nmes]
        y_all = data[self.ctx.resp_nme]
        w_all = data[self.ctx.weight_nme]
        X_test = self.ctx.test_oht_scl_data
        if X_test is None:
            raise RuntimeError("Missing standardized test data for GLM ensemble.")

        n_samples = len(X_all)
        X_all_design = self._prepare_design(data)
        X_test_design = self._prepare_design(X_test)
        tweedie_power = self.best_params.get('tweedie_power')
        family = self._select_family(tweedie_power)

        split_iter, _ = self._resolve_ensemble_splits(X_all, k=k)
        if split_iter is None:
            print(
                f"[GLM Ensemble] unable to build CV split (n_samples={n_samples}); skip ensemble.",
                flush=True,
            )
            return
        preds_train_sum = np.zeros(n_samples, dtype=np.float64)
        preds_test_sum = np.zeros(len(X_test_design), dtype=np.float64)

        split_count = 0
        for train_idx, _val_idx in split_iter:
            X_train = X_all_design.iloc[train_idx]
            y_train = y_all.iloc[train_idx]
            w_train = w_all.iloc[train_idx]

            glm = sm.GLM(y_train, X_train, family=family, freq_weights=w_train)
            result = glm.fit_regularized(
                alpha=self.best_params['alpha'],
                L1_wt=self.best_params['l1_ratio'],
                maxiter=300
            )
            pred_train = result.predict(X_all_design)
            pred_test = result.predict(X_test_design)
            preds_train_sum += np.asarray(pred_train, dtype=np.float64)
            preds_test_sum += np.asarray(pred_test, dtype=np.float64)
            split_count += 1

        if split_count < 1:
            print(
                f"[GLM Ensemble] no CV splits generated; skip ensemble.",
                flush=True,
            )
            return
        preds_train = preds_train_sum / float(split_count)
        preds_test = preds_test_sum / float(split_count)
        self._cache_predictions("glm", preds_train, preds_test)


