from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit

from ins_pricing.modelling.bayesopt.trainers.trainer_base import TrainerBase
from ins_pricing.modelling.bayesopt.trainers.cv_utils import _OrderSplitter
from ins_pricing.modelling.bayesopt.models import FTTransformerSklearn
from ins_pricing.utils.losses import regression_loss
from ins_pricing.utils import get_logger, log_print

_logger = get_logger("ins_pricing.trainer.ft")


def _log(*args, **kwargs) -> None:
    log_print(_logger, *args, **kwargs)


class FTTrainer(TrainerBase):
    def __init__(self, context: "BayesOptModel") -> None:
        if context.task_type == 'classification':
            super().__init__(context, 'FTTransformerClassifier', 'FTTransformer')
        else:
            super().__init__(context, 'FTTransformer', 'FTTransformer')
        self.model: Optional[FTTransformerSklearn] = None
        dist_cfg = getattr(context.config, "distributed", context.config)
        self.enable_distributed_optuna = bool(
            getattr(dist_cfg, "use_ft_ddp", False) and context.use_gpu
        )
        self._cv_geo_warned = False
        self._param_probe_model: Optional[FTTransformerSklearn] = None

    def _dist_cfg(self):
        return getattr(self.ctx.config, "distributed", self.ctx.config)

    def _use_ft_data_parallel(self) -> bool:
        cfg = self._dist_cfg()
        return bool(getattr(cfg, "use_ft_data_parallel", False))

    def _use_ft_ddp(self) -> bool:
        cfg = self._dist_cfg()
        return bool(getattr(cfg, "use_ft_ddp", False))

    def _geo_feature_names(self) -> List[str]:
        geo_cfg = getattr(self.ctx.config, "geo_token", None)
        if geo_cfg is not None and hasattr(geo_cfg, "feature_nmes"):
            return list(getattr(geo_cfg, "feature_nmes") or [])
        return list(getattr(self.ctx.config, "geo_feature_nmes", []) or [])

    def _maybe_cleanup_gpu(self, model: Optional[FTTransformerSklearn]) -> None:
        if not bool(getattr(self.ctx.config, "ft_cleanup_per_fold", False)):
            return
        if model is not None:
            getattr(getattr(model, "ft", None), "to",
                    lambda *_args, **_kwargs: None)("cpu")
        synchronize = bool(getattr(self.ctx.config, "ft_cleanup_synchronize", False))
        self._clean_gpu(synchronize=synchronize)

    def _resolve_numeric_tokens(self) -> int:
        ft_cfg = getattr(self.ctx.config, "ft_transformer", None)
        if ft_cfg is not None and hasattr(ft_cfg, "num_numeric_tokens"):
            requested = getattr(ft_cfg, "num_numeric_tokens")
        else:
            requested = getattr(self.ctx.config, "ft_num_numeric_tokens", None)
        return FTTransformerSklearn.resolve_numeric_token_count(
            self.ctx.num_features,
            self.ctx.cate_list,
            requested,
        )

    def _create_ft_model(self, **overrides: Any) -> FTTransformerSklearn:
        kwargs: Dict[str, Any] = {
            "model_nme": self.ctx.model_nme,
            "num_cols": self.ctx.num_features,
            "cat_cols": self.ctx.cate_list,
            "task_type": self.ctx.task_type,
            "epochs": int(getattr(self.ctx, "epochs", 100)),
            "use_data_parallel": self._use_ft_data_parallel(),
            "use_ddp": self._use_ft_ddp(),
            "use_gpu": bool(getattr(self.ctx, "use_gpu", False)),
            "num_numeric_tokens": self._resolve_numeric_tokens(),
            "loss_name": getattr(self.ctx, "loss_name", "tweedie"),
            "distribution": getattr(self.ctx, "distribution", None),
        }
        kwargs.update(overrides)
        return FTTransformerSklearn(**kwargs)

    def _create_runtime_ft_model(self, **overrides: Any) -> FTTransformerSklearn:
        model = self._create_ft_model(**overrides)
        return self._apply_dataloader_overrides(model)

    def _resolve_adaptive_heads(self,
                                d_model: int,
                                requested_heads: Optional[int] = None) -> Tuple[int, bool]:
        d_model = int(d_model)
        if d_model <= 0:
            raise ValueError(f"Invalid d_model={d_model}, expected > 0.")

        default_heads = max(2, d_model // 16)
        base_heads = default_heads if requested_heads is None else int(
            requested_heads)
        base_heads = max(1, min(base_heads, d_model))

        if d_model % base_heads == 0:
            return base_heads, False

        for candidate in range(min(d_model, base_heads), 0, -1):
            if d_model % candidate == 0:
                return candidate, True
        return 1, True

    def _apply_adaptive_heads(
        self,
        params: Dict[str, Any],
        *,
        default_d_model: int,
        log_adjustment: bool = True,
    ) -> Dict[str, Any]:
        resolved = dict(params)
        d_model_value = resolved.get("d_model", default_d_model)
        adaptive_heads, heads_adjusted = self._resolve_adaptive_heads(
            d_model=d_model_value,
            requested_heads=resolved.get("n_heads"),
        )
        if log_adjustment and heads_adjusted:
            _log(
                f"[FTTrainer] Auto-adjusted n_heads from "
                f"{resolved.get('n_heads')} to {adaptive_heads} "
                f"(d_model={d_model_value})."
            )
        resolved["n_heads"] = adaptive_heads
        return resolved

    def _apply_model_params(
        self,
        model: FTTransformerSklearn,
        resolved_params: Dict[str, Any],
        *,
        context: str = "FTTransformer model params",
    ) -> Dict[str, Any]:
        model_params = self._filter_params_by_model_support(
            model,
            resolved_params,
            context=context,
        )
        if model_params:
            model.set_params(model_params)
        return model_params

    def _slice_geo_tokens(self, index: pd.Index) -> Optional[pd.DataFrame]:
        geo_all = self.ctx.train_geo_tokens
        if geo_all is None:
            return None
        try:
            return geo_all.loc[index]
        except Exception:
            return None

    def _build_geo_io_kwargs(
        self,
        *,
        geo_train: Optional[pd.DataFrame],
        geo_test: Optional[pd.DataFrame],
        return_embedding: bool = False,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        fit_kwargs: Dict[str, Any] = {}
        predict_kwargs_train: Optional[Dict[str, Any]] = None
        predict_kwargs_test: Optional[Dict[str, Any]] = None
        if geo_train is not None and geo_test is not None:
            fit_kwargs["geo_train"] = geo_train
            predict_kwargs_train = {"geo_tokens": geo_train}
            predict_kwargs_test = {"geo_tokens": geo_test}
        if return_embedding:
            predict_kwargs_train = dict(predict_kwargs_train or {})
            predict_kwargs_test = dict(predict_kwargs_test or {})
            predict_kwargs_train["return_embedding"] = True
            predict_kwargs_test["return_embedding"] = True
        return fit_kwargs, predict_kwargs_train, predict_kwargs_test

    def _resolve_geo_tokens_for_fold(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        if not self._geo_feature_names():
            return None, None
        geo_train = self._slice_geo_tokens(X_train.index)
        geo_val = self._slice_geo_tokens(X_val.index)
        if (geo_train is None or geo_val is None) and not self._cv_geo_warned:
            _log(
                "[FTTrainer] Geo tokens unavailable for CV split; continue without geo tokens.",
                flush=True,
            )
            self._cv_geo_warned = True
        return geo_train, geo_val

    def _sanitize_best_params(
        self,
        params: Dict[str, Any],
        *,
        context: str = "best_params",
    ) -> Dict[str, Any]:
        probe = self._param_probe_model
        if probe is None:
            probe = self._create_ft_model(
                use_data_parallel=False,
                use_ddp=False,
                use_gpu=False,
            )
            self._param_probe_model = probe
        return self._filter_params_by_model_support(
            probe,
            params,
            context=f"{context} (FT model params)",
        )

    def cross_val_unsupervised(self, trial: Optional[optuna.trial.Trial]) -> float:
        """Optuna objective A: minimize validation loss for masked reconstruction."""
        search_space = self._get_search_space_config("ft_unsupervised_search_space")
        param_space: Dict[str, Callable[[optuna.trial.Trial], Any]] = {}
        param_space = self._augment_param_space_with_search_space(
            model_key="ft_unsupervised",
            param_space=param_space,
            search_space=search_space,
        )

        params: Optional[Dict[str, Any]] = None
        if self._distributed_forced_params is not None:
            params = self._distributed_forced_params
            self._distributed_forced_params = None
        else:
            if trial is None:
                raise RuntimeError(
                    "Missing Optuna trial for parameter sampling.")
            params = {name: sampler(trial)
                      for name, sampler in param_space.items()}
            _log(
                f"[Optuna][{self.label}] trial_id={getattr(trial, 'number', 'unknown')} "
                f"sampled params={params}",
                flush=True,
            )
            if self._should_use_distributed_optuna():
                self._distributed_prepare_trial(params)

        X_all = self.ctx.train_data[self.ctx.factor_nmes]
        default_limit = min(1_000_000, int(len(X_all) / 2))
        effective_limit = self._resolve_effective_sample_limit(
            base_limit=default_limit if default_limit > 0 else None,
            n_rows=len(X_all),
        )
        if effective_limit is not None and len(X_all) > effective_limit:
            sampled_idx = self._resolve_time_sample_indices(
                X_all, effective_limit)
            if sampled_idx is None:
                X_all = X_all.sample(
                    n=effective_limit,
                    random_state=self.ctx.rand_seed,
                )
            else:
                X_all = X_all.loc[sampled_idx]

        split = self._resolve_train_val_indices(X_all, allow_default=True)
        if split is None:
            raise ValueError(
                "Unable to build train/val split for FT unsupervised CV.")
        train_idx, val_idx = split
        X_train = X_all.iloc[train_idx]
        X_val = X_all.iloc[val_idx]
        geo_train, geo_val = self._resolve_geo_tokens_for_fold(X_train, X_val)

        d_model = int(params.get("d_model", 64))
        n_layers = int(params.get("n_layers", 4))
        num_numeric_tokens = self._resolve_numeric_tokens()
        token_count = num_numeric_tokens + len(self.ctx.cate_list)
        if geo_train is not None:
            token_count += 1
        approx_units = d_model * n_layers * max(1, token_count)
        if approx_units > 12_000_000:
            raise optuna.TrialPruned(
                f"config exceeds safe memory budget (approx_units={approx_units})")

        adaptive_heads, _ = self._resolve_adaptive_heads(
            d_model=d_model,
            requested_heads=params.get("n_heads")
        )

        mask_prob_num = float(params.get("mask_prob_num", 0.15))
        mask_prob_cat = float(params.get("mask_prob_cat", 0.15))
        num_loss_weight = float(params.get("num_loss_weight", 1.0))
        cat_loss_weight = float(params.get("cat_loss_weight", 1.0))

        model_params = dict(params)
        model_params["n_heads"] = adaptive_heads
        for k in ("mask_prob_num", "mask_prob_cat", "num_loss_weight", "cat_loss_weight"):
            model_params.pop(k, None)

        model = self._create_runtime_ft_model(
            epochs=self.ctx.epochs,
            patience=5,
            weight_decay=float(params.get("weight_decay", 0.0)),
            num_numeric_tokens=num_numeric_tokens,
        )
        self._apply_model_params(
            model,
            model_params,
            context="FT unsupervised model params",
        )
        try:
            return float(model.fit_unsupervised(
                X_train,
                X_val=X_val,
                trial=trial,
                geo_train=geo_train,
                geo_val=geo_val,
                mask_prob_num=mask_prob_num,
                mask_prob_cat=mask_prob_cat,
                num_loss_weight=num_loss_weight,
                cat_loss_weight=cat_loss_weight
            ))
        finally:
            self._maybe_cleanup_gpu(model)

    def cross_val(self, trial: optuna.trial.Trial) -> float:
        # FT-Transformer CV also focuses on memory control:
        #   - Shrink search space to avoid oversized models.
        #   - Release GPU memory after each fold so the next trial can run.
        # Slightly shrink hyperparameter space to avoid oversized models.
        search_space = self._get_search_space_config("ft_search_space")
        param_space: Dict[str, Callable[[optuna.trial.Trial], Any]] = {}
        loss_name = getattr(self.ctx, "loss_name", "tweedie")
        geo_enabled = bool(
            self.ctx.geo_token_cols or self._geo_feature_names())
        param_space = self._augment_param_space_with_search_space(
            model_key="ft",
            param_space=param_space,
            search_space=search_space,
        )

        metric_ctx: Dict[str, Any] = {}

        def data_provider():
            data = self.ctx.train_data
            return data[self.ctx.factor_nmes], data[self.ctx.resp_nme], data[self.ctx.weight_nme]

        def model_builder(params):
            d_model = int(params.get("d_model", 64))
            n_layers = int(params.get("n_layers", 4))
            num_numeric_tokens = self._resolve_numeric_tokens()
            token_count = num_numeric_tokens + len(self.ctx.cate_list)
            if geo_enabled:
                token_count += 1
            approx_units = d_model * n_layers * max(1, token_count)
            if approx_units > 12_000_000:
                _log(
                    f"[FTTrainer] Trial pruned early: d_model={d_model}, n_layers={n_layers} -> approx_units={approx_units}")
                raise optuna.TrialPruned(
                    "config exceeds safe memory budget; prune before training")

            tw_power = params.get("tw_power")
            if self.ctx.task_type == 'regression':
                base_tw = self.ctx.default_tweedie_power()
                if loss_name == "tweedie":
                    tw_power = base_tw if tw_power is None else tw_power
                elif loss_name in ("poisson", "gamma"):
                    tw_power = base_tw
                else:
                    tw_power = None
            metric_ctx["tw_power"] = tw_power

            adaptive_heads, _ = self._resolve_adaptive_heads(
                d_model=d_model,
                requested_heads=params.get("n_heads")
            )

            model = self._create_runtime_ft_model(
                d_model=d_model,
                n_heads=adaptive_heads,
                n_layers=n_layers,
                dropout=float(params.get("dropout", 0.1)),
                epochs=self.ctx.epochs,
                tweedie_power=tw_power,
                learning_rate=float(params.get("learning_rate", 1e-3)),
                patience=5,
                weight_decay=float(params.get("weight_decay", 0.0)),
                num_numeric_tokens=num_numeric_tokens,
            )
            handled_keys = {
                "d_model",
                "n_heads",
                "n_layers",
                "dropout",
                "learning_rate",
                "weight_decay",
                "tw_power",
            }
            extra_params = {
                key: value for key, value in params.items()
                if key not in handled_keys
            }
            extra_params = self._filter_params_by_model_support(
                model,
                extra_params,
                context="FT CV model params",
            )
            if extra_params:
                model.set_params(extra_params)
            return model

        def fit_predict(model, X_train, y_train, w_train, X_val, y_val, w_val, trial_obj):
            geo_train, geo_val = (None, None)
            if geo_enabled:
                geo_train, geo_val = self._resolve_geo_tokens_for_fold(X_train, X_val)
            model.fit(
                X_train, y_train, w_train,
                X_val, y_val, w_val,
                trial=trial_obj,
                geo_train=geo_train,
                geo_val=geo_val
            )
            return model.predict(X_val, geo_tokens=geo_val)

        def metric_fn(y_true, y_pred, weight):
            if self.ctx.task_type == 'regression':
                return regression_loss(
                    y_true,
                    y_pred,
                    weight,
                    loss_name=loss_name,
                    tweedie_power=metric_ctx.get("tw_power", 1.5),
                )
            return log_loss(y_true, y_pred, sample_weight=weight)

        data_for_cap = data_provider()[0]
        max_rows_for_ft_bo = min(1000000, int(len(data_for_cap)/2))

        return self.cross_val_generic(
            trial=trial,
            hyperparameter_space=param_space,
            data_provider=data_provider,
            model_builder=model_builder,
            metric_fn=metric_fn,
            sample_limit=max_rows_for_ft_bo if len(
                data_for_cap) > max_rows_for_ft_bo > 0 else None,
            fit_predict_fn=fit_predict,
            cleanup_fn=lambda m: getattr(
                getattr(m, "ft", None), "to", lambda *_args, **_kwargs: None)("cpu")
        )

    def train(self) -> None:
        if not self.best_params:
            raise RuntimeError(
                "Run tune() first to obtain best FT-Transformer parameters.")
        resolved_params = self._apply_adaptive_heads(
            dict(self.best_params),
            default_d_model=64,
            log_adjustment=True,
        )
        geo_train_full = self.ctx.train_geo_tokens
        geo_test_full = self.ctx.test_geo_tokens

        use_refit = bool(getattr(self.ctx.config, "final_refit", True))
        refit_epochs = None
        X_all = self.ctx.train_data[self.ctx.factor_nmes]
        y_all = self.ctx.train_data[self.ctx.resp_nme]
        w_all = self.ctx.train_data[self.ctx.weight_nme]
        split = self._resolve_train_val_indices(X_all)
        if use_refit and split is not None:
            train_idx, val_idx = split
            tmp_model = self._create_runtime_ft_model(
                weight_decay=float(resolved_params.get("weight_decay", 0.0)),
            )
            self._apply_model_params(
                tmp_model,
                resolved_params,
                context="FT final refit params",
            )
            geo_train = None if geo_train_full is None else geo_train_full.iloc[train_idx]
            geo_val = None if geo_train_full is None else geo_train_full.iloc[val_idx]
            tmp_model.fit(
                X_all.iloc[train_idx],
                y_all.iloc[train_idx],
                w_all.iloc[train_idx],
                X_all.iloc[val_idx],
                y_all.iloc[val_idx],
                w_all.iloc[val_idx],
                trial=None,
                geo_train=geo_train,
                geo_val=geo_val,
            )
            refit_epochs = self._resolve_best_epoch(
                getattr(tmp_model, "training_history", None),
                default_epochs=int(self.ctx.epochs),
            )
            self._maybe_cleanup_gpu(tmp_model)

        self.model = self._create_runtime_ft_model(
            weight_decay=float(resolved_params.get("weight_decay", 0.0)),
        )
        if refit_epochs is not None:
            self.model.epochs = int(refit_epochs)
        self._apply_model_params(
            self.model,
            resolved_params,
            context="FT final train params",
        )
        self.best_params = self._sanitize_best_params(
            resolved_params,
            context="FT final train params",
        )
        loss_plot_path = self.output.plot_path(
            f'{self.ctx.model_nme}/loss/loss_{self.ctx.model_nme}_{self.model_name_prefix}.png')
        self.model.loss_curve_path = loss_plot_path
        geo_train = geo_train_full
        geo_test = geo_test_full
        fit_kwargs, predict_kwargs_train, predict_kwargs_test = self._build_geo_io_kwargs(
            geo_train=geo_train,
            geo_test=geo_test,
            return_embedding=False,
        )
        self._fit_predict_cache(
            self.model,
            self.ctx.train_data[self.ctx.factor_nmes],
            self.ctx.train_data[self.ctx.resp_nme],
            sample_weight=self.ctx.train_data[self.ctx.weight_nme],
            pred_prefix='ft',
            sample_weight_arg='w_train',
            fit_kwargs=fit_kwargs,
            predict_kwargs_train=predict_kwargs_train,
            predict_kwargs_test=predict_kwargs_test
        )
        self.ctx.ft_best = self.model

    def ensemble_predict(self, k: int) -> None:
        if not self.best_params:
            raise RuntimeError(
                "Run tune() first to obtain best FT-Transformer parameters.")
        k = max(2, int(k))
        X_all = self.ctx.train_data[self.ctx.factor_nmes]
        y_all = self.ctx.train_data[self.ctx.resp_nme]
        w_all = self.ctx.train_data[self.ctx.weight_nme]
        X_test = self.ctx.test_data[self.ctx.factor_nmes]
        n_samples = len(X_all)
        geo_train_full = self.ctx.train_geo_tokens
        geo_test_full = self.ctx.test_geo_tokens

        resolved_params = dict(self.best_params)
        default_d_model = getattr(self.model, "d_model", 64)
        resolved_params = self._apply_adaptive_heads(
            resolved_params,
            default_d_model=default_d_model,
            log_adjustment=False,
        )

        split_iter, _ = self._resolve_ensemble_splits(X_all, k=k)
        if split_iter is None:
            _log(
                f"[FT Ensemble] unable to build CV split (n_samples={n_samples}); skip ensemble.",
                flush=True,
            )
            return
        preds_train_sum = np.zeros(n_samples, dtype=np.float64)
        preds_test_sum = np.zeros(len(X_test), dtype=np.float64)

        split_count = 0
        for train_idx, val_idx in split_iter:
            model = self._create_runtime_ft_model(
                weight_decay=float(resolved_params.get("weight_decay", 0.0)),
            )
            self._apply_model_params(
                model,
                resolved_params,
                context="FT ensemble params",
            )

            geo_train = geo_val = None
            if geo_train_full is not None:
                geo_train = geo_train_full.iloc[train_idx]
                geo_val = geo_train_full.iloc[val_idx]

            model.fit(
                X_all.iloc[train_idx],
                y_all.iloc[train_idx],
                w_all.iloc[train_idx],
                X_all.iloc[val_idx],
                y_all.iloc[val_idx],
                w_all.iloc[val_idx],
                trial=None,
                geo_train=geo_train,
                geo_val=geo_val,
            )

            pred_train = model.predict(X_all, geo_tokens=geo_train_full)
            pred_test = model.predict(X_test, geo_tokens=geo_test_full)
            preds_train_sum += np.asarray(pred_train, dtype=np.float64)
            preds_test_sum += np.asarray(pred_test, dtype=np.float64)
            self._maybe_cleanup_gpu(model)
            split_count += 1

        if split_count < 1:
            _log(
                f"[FT Ensemble] no CV splits generated; skip ensemble.",
                flush=True,
            )
            return
        preds_train = preds_train_sum / float(split_count)
        preds_test = preds_test_sum / float(split_count)
        self._cache_predictions("ft", preds_train, preds_test)

    def _resolve_oof_splitter(self, n_samples: int):
        cfg = self.ctx.config
        raw_strategy = str(getattr(cfg, "ft_oof_strategy",
                           "auto") or "auto").strip().lower()
        base_strategy = str(
            getattr(cfg, "cv_strategy", "random") or "random").strip().lower()
        if raw_strategy == "auto":
            strategy = base_strategy
        else:
            strategy = raw_strategy

        oof_folds = getattr(cfg, "ft_oof_folds", None)
        if oof_folds is None:
            if strategy in {"random", "group", "grouped"}:
                val_ratio = float(
                    self.ctx.prop_test) if self.ctx.prop_test else 0.25
                if not (0.0 < val_ratio < 1.0):
                    val_ratio = 0.25
                oof_folds = max(2, int(round(1 / val_ratio)))
            else:
                oof_folds = 0
        oof_folds = int(oof_folds)

        if oof_folds < 2 or n_samples < oof_folds:
            return None, None, 0

        if strategy in {"group", "grouped"}:
            group_col = getattr(cfg, "cv_group_col", None)
            if not group_col:
                raise ValueError(
                    "cv_group_col is required for FT OOF group strategy.")
            if group_col not in self.ctx.train_data.columns:
                raise KeyError(
                    f"cv_group_col '{group_col}' not in train_data.")
            groups = self.ctx.train_data[group_col]
            splitter = GroupKFold(n_splits=oof_folds)
            return splitter, groups, oof_folds

        if strategy in {"time", "timeseries", "temporal"}:
            time_col = getattr(cfg, "cv_time_col", None)
            if not time_col:
                raise ValueError(
                    "cv_time_col is required for FT OOF time strategy.")
            if time_col not in self.ctx.train_data.columns:
                raise KeyError(f"cv_time_col '{time_col}' not in train_data.")
            ascending = bool(getattr(cfg, "cv_time_ascending", True))
            order_index = self.ctx.train_data[time_col].sort_values(
                ascending=ascending).index
            order = self.ctx.train_data.index.get_indexer(order_index)
            if n_samples <= oof_folds:
                return None, None, 0
            splitter = TimeSeriesSplit(n_splits=oof_folds)
            return _OrderSplitter(splitter, order), None, oof_folds

        shuffle = bool(getattr(cfg, "ft_oof_shuffle", True))
        splitter = KFold(
            n_splits=oof_folds,
            shuffle=shuffle,
            random_state=self.ctx.rand_seed if shuffle else None,
        )
        return splitter, None, oof_folds

    def _build_ft_feature_model(self, resolved_params: Dict[str, Any]) -> FTTransformerSklearn:
        model = self._create_runtime_ft_model()
        adjusted_params = self._apply_adaptive_heads(
            resolved_params,
            default_d_model=model.d_model,
            log_adjustment=True,
        )
        resolved_params.clear()
        resolved_params.update(adjusted_params)
        if resolved_params:
            self._apply_model_params(
                model,
                resolved_params,
                context="FT feature model params",
            )
        return model

    def _oof_predict_train(
        self,
        resolved_params: Dict[str, Any],
        *,
        feature_mode: str,
        geo_train_full: Optional[pd.DataFrame],
    ) -> Optional[np.ndarray]:
        X_all = self.ctx.train_data[self.ctx.factor_nmes]
        y_all = self.ctx.train_data[self.ctx.resp_nme]
        w_all = self.ctx.train_data[self.ctx.weight_nme]
        splitter, groups, oof_folds = self._resolve_oof_splitter(len(X_all))
        if splitter is None:
            return None

        preds_train = None
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_all, y_all, groups=groups), start=1):
            X_train = X_all.iloc[train_idx]
            y_train = y_all.iloc[train_idx]
            w_train = w_all.iloc[train_idx]
            X_val = X_all.iloc[val_idx]
            y_val = y_all.iloc[val_idx]
            w_val = w_all.iloc[val_idx]

            geo_train = geo_val = None
            if geo_train_full is not None:
                geo_train = geo_train_full.iloc[train_idx]
                geo_val = geo_train_full.iloc[val_idx]

            model = self._build_ft_feature_model(dict(resolved_params))
            model.fit(
                X_train,
                y_train,
                w_train=w_train,
                X_val=X_val,
                y_val=y_val,
                w_val=w_val,
                trial=None,
                geo_train=geo_train,
                geo_val=geo_val,
            )

            predict_kwargs = {}
            if geo_val is not None:
                predict_kwargs["geo_tokens"] = geo_val
            if feature_mode == "embedding":
                predict_kwargs["return_embedding"] = True
            fold_pred = model.predict(X_val, **predict_kwargs)
            fold_pred = np.asarray(fold_pred)
            if preds_train is None:
                preds_train = np.empty(
                    (len(X_all),) + fold_pred.shape[1:], dtype=fold_pred.dtype)
            preds_train[val_idx] = fold_pred

            self._maybe_cleanup_gpu(model)

        if preds_train is None:
            return None
        if oof_folds < 2:
            return None
        return preds_train

    def train_as_feature(self, pred_prefix: str = "ft_feat", feature_mode: str = "prediction") -> None:
        """Train FT-Transformer only to generate features (not recorded as final model)."""
        if not self.best_params:
            raise RuntimeError(
                "Run tune() first to obtain best FT-Transformer parameters.")
        resolved_params = dict(self.best_params)
        if feature_mode not in ("prediction", "embedding"):
            raise ValueError(
                f"Unsupported feature_mode='{feature_mode}', expected 'prediction' or 'embedding'.")
        geo_train = self.ctx.train_geo_tokens
        geo_test = self.ctx.test_geo_tokens
        fit_kwargs, predict_kwargs_train, predict_kwargs_test = self._build_geo_io_kwargs(
            geo_train=geo_train,
            geo_test=geo_test,
            return_embedding=(feature_mode == "embedding"),
        )

        oof_preds = self._oof_predict_train(
            resolved_params,
            feature_mode=feature_mode,
            geo_train_full=geo_train,
        )
        if oof_preds is not None:
            self.model = self._build_ft_feature_model(resolved_params)
            self.best_params = resolved_params
            self.model.fit(
                self.ctx.train_data[self.ctx.factor_nmes],
                self.ctx.train_data[self.ctx.resp_nme],
                w_train=self.ctx.train_data[self.ctx.weight_nme],
                X_val=None,
                y_val=None,
                w_val=None,
                trial=None,
                geo_train=geo_train,
                geo_val=None,
            )
            predict_kwargs = dict(predict_kwargs_test or {})
            preds_test = self.model.predict(
                self.ctx.test_data[self.ctx.factor_nmes],
                **predict_kwargs,
            )
            self._cache_predictions(pred_prefix, oof_preds, preds_test)
            return

        self.model = self._build_ft_feature_model(resolved_params)
        self.best_params = resolved_params
        self._fit_predict_cache(
            self.model,
            self.ctx.train_data[self.ctx.factor_nmes],
            self.ctx.train_data[self.ctx.resp_nme],
            sample_weight=self.ctx.train_data[self.ctx.weight_nme],
            pred_prefix=pred_prefix,
            sample_weight_arg='w_train',
            fit_kwargs=fit_kwargs,
            predict_kwargs_train=predict_kwargs_train,
            predict_kwargs_test=predict_kwargs_test,
            record_label=False,
        )

    def pretrain_unsupervised_as_feature(self,
                                         pred_prefix: str = "ft_uemb",
                                         params: Optional[Dict[str,
                                                               Any]] = None,
                                         mask_prob_num: float = 0.15,
                                         mask_prob_cat: float = 0.15,
                                         num_loss_weight: float = 1.0,
                                         cat_loss_weight: float = 1.0) -> None:
        """Self-supervised pretraining (masked reconstruction) and cache embeddings."""
        self.model = self._create_runtime_ft_model()
        resolved_params = dict(params or {})
        # Reuse supervised tuning structure params unless explicitly overridden.
        if not resolved_params and self.best_params:
            resolved_params = dict(self.best_params)

        # If params include masked reconstruction fields, they take precedence.
        mask_prob_num = float(resolved_params.pop(
            "mask_prob_num", mask_prob_num))
        mask_prob_cat = float(resolved_params.pop(
            "mask_prob_cat", mask_prob_cat))
        num_loss_weight = float(resolved_params.pop(
            "num_loss_weight", num_loss_weight))
        cat_loss_weight = float(resolved_params.pop(
            "cat_loss_weight", cat_loss_weight))

        resolved_params = self._apply_adaptive_heads(
            resolved_params,
            default_d_model=self.model.d_model,
            log_adjustment=True,
        )
        if resolved_params:
            self._apply_model_params(
                self.model,
                resolved_params,
                context="FT unsupervised feature params",
            )

        loss_plot_path = self.output.plot_path(
            f'{self.ctx.model_nme}/loss/loss_{self.ctx.model_nme}_FTTransformerUnsupervised.png')
        self.model.loss_curve_path = loss_plot_path

        # Build a simple holdout split for pretraining early stopping.
        X_all = self.ctx.train_data[self.ctx.factor_nmes]
        split = self._resolve_train_val_indices(X_all, allow_default=True)
        if split is None:
            raise ValueError(
                "Unable to build train/val split for FT unsupervised training.")
        train_idx, val_idx = split
        X_tr = X_all.iloc[train_idx]
        X_val = X_all.iloc[val_idx]

        geo_all = self.ctx.train_geo_tokens
        geo_test_full = self.ctx.test_geo_tokens
        geo_tr = geo_val = None
        if geo_all is not None:
            geo_tr = geo_all.loc[X_tr.index]
            geo_val = geo_all.loc[X_val.index]

        self.model.fit_unsupervised(
            X_tr,
            X_val=X_val,
            geo_train=geo_tr,
            geo_val=geo_val,
            mask_prob_num=mask_prob_num,
            mask_prob_cat=mask_prob_cat,
            num_loss_weight=num_loss_weight,
            cat_loss_weight=cat_loss_weight
        )

        geo_train_full = geo_all
        _, predict_kwargs_train, predict_kwargs_test = self._build_geo_io_kwargs(
            geo_train=geo_train_full,
            geo_test=geo_test_full,
            return_embedding=True,
        )
        if predict_kwargs_train is None or predict_kwargs_test is None:
            predict_kwargs_train = {"return_embedding": True}
            predict_kwargs_test = {"return_embedding": True}

        self._predict_and_cache(
            self.model,
            pred_prefix=pred_prefix,
            predict_kwargs_train=predict_kwargs_train,
            predict_kwargs_test=predict_kwargs_test
        )


# =============================================================================
