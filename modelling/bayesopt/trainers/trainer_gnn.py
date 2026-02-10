from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import torch
from sklearn.metrics import log_loss

from ins_pricing.modelling.bayesopt.checkpoints import rebuild_gnn_model_from_payload
from ins_pricing.modelling.bayesopt.trainers.trainer_base import TrainerBase
from ins_pricing.modelling.bayesopt.models import GraphNeuralNetSklearn
from ins_pricing.utils import EPS, get_logger, log_print
from ins_pricing.utils.losses import regression_loss
from ins_pricing.utils.torch_compat import torch_load

_logger = get_logger("ins_pricing.trainer.gnn")


def _log(*args, **kwargs) -> None:
    log_print(_logger, *args, **kwargs)

class GNNTrainer(TrainerBase):
    def __init__(self, context: "BayesOptModel") -> None:
        super().__init__(context, 'GNN', 'GNN')
        self.model: Optional[GraphNeuralNetSklearn] = None
        try:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        except (TypeError, ValueError):
            world_size = 1
        gpu_enabled = bool(getattr(context, "use_gpu", True))
        dist_cfg = getattr(context.config, "distributed", context.config)
        requested_ddp = bool(getattr(dist_cfg, "use_gnn_ddp", False))
        supports_ddp = bool(getattr(GraphNeuralNetSklearn, "SUPPORTS_MULTI_PROCESS_DDP", False))
        self._runtime_use_ddp = requested_ddp and supports_ddp and gpu_enabled
        if requested_ddp and not gpu_enabled:
            _log(
                "[GNNTrainer] use_gnn_ddp=true but use_gpu=false; forcing CPU single-process mode.",
                flush=True,
            )
        if requested_ddp and world_size > 1 and not supports_ddp:
            _log(
                "[GNNTrainer] use_gnn_ddp=true but GNN multi-process DDP is unsupported; "
                "falling back to single-process training on rank0.",
                flush=True,
            )
        self.enable_distributed_optuna = bool(self._runtime_use_ddp and world_size > 1)

    def _dist_cfg(self):
        return getattr(self.ctx.config, "distributed", self.ctx.config)

    def _geo_feature_names(self) -> List[str]:
        geo_cfg = getattr(self.ctx.config, "geo_token", None)
        if geo_cfg is not None and hasattr(geo_cfg, "feature_nmes"):
            return list(getattr(geo_cfg, "feature_nmes") or [])
        return list(getattr(self.ctx.config, "geo_feature_nmes", []) or [])

    def _resolve_gnn_setting(self, nested_attr: str, flat_attr: str, default: Any) -> Any:
        gnn_cfg = getattr(self.ctx.config, "gnn", None)
        if gnn_cfg is not None and hasattr(gnn_cfg, nested_attr):
            return getattr(gnn_cfg, nested_attr)
        return getattr(self.ctx.config, flat_attr, default)

    def _maybe_cleanup_gpu(self, model: Optional[GraphNeuralNetSklearn]) -> None:
        if not bool(getattr(self.ctx.config, "gnn_cleanup_per_fold", False)):
            return
        if model is not None:
            getattr(getattr(model, "gnn", None), "to",
                    lambda *_args, **_kwargs: None)("cpu")
        synchronize = bool(getattr(self.ctx.config, "gnn_cleanup_synchronize", False))
        self._clean_gpu(synchronize=synchronize)

    def _build_model(self, params: Optional[Dict[str, Any]] = None) -> GraphNeuralNetSklearn:
        params = params or {}
        dist_cfg = self._dist_cfg()
        base_tw_power = self.ctx.default_tweedie_power()
        loss_name = getattr(self.ctx, "loss_name", "tweedie")
        tw_power = params.get("tw_power")
        if self.ctx.task_type == "regression":
            if loss_name == "tweedie":
                tw_power = base_tw_power if tw_power is None else float(tw_power)
            elif loss_name in ("poisson", "gamma"):
                tw_power = base_tw_power
            else:
                tw_power = None
        model = GraphNeuralNetSklearn(
            model_nme=f"{self.ctx.model_nme}_gnn",
            input_dim=len(self.ctx.var_nmes),
            hidden_dim=int(params.get("hidden_dim", 64)),
            num_layers=int(params.get("num_layers", 2)),
            k_neighbors=int(params.get("k_neighbors", 10)),
            dropout=float(params.get("dropout", 0.1)),
            learning_rate=float(params.get("learning_rate", 1e-3)),
            epochs=int(params.get("epochs", self.ctx.epochs)),
            patience=int(params.get("patience", 5)),
            task_type=self.ctx.task_type,
            tweedie_power=tw_power,
            weight_decay=float(params.get("weight_decay", 0.0)),
            use_data_parallel=bool(getattr(dist_cfg, "use_gnn_data_parallel", False)),
            use_ddp=bool(self._runtime_use_ddp),
            use_gpu=self.ctx.use_gpu,
            use_approx_knn=bool(
                self._resolve_gnn_setting(
                    "use_approx_knn",
                    "gnn_use_approx_knn",
                    True,
                )
            ),
            approx_knn_threshold=int(
                self._resolve_gnn_setting(
                    "approx_knn_threshold",
                    "gnn_approx_knn_threshold",
                    50000,
                )
            ),
            graph_cache_path=self._resolve_gnn_setting(
                "graph_cache",
                "gnn_graph_cache",
                None,
            ),
            max_gpu_knn_nodes=self._resolve_gnn_setting(
                "max_gpu_knn_nodes",
                "gnn_max_gpu_knn_nodes",
                200000,
            ),
            knn_gpu_mem_ratio=float(
                self._resolve_gnn_setting(
                    "knn_gpu_mem_ratio",
                    "gnn_knn_gpu_mem_ratio",
                    0.9,
                )
            ),
            knn_gpu_mem_overhead=float(
                self._resolve_gnn_setting(
                    "knn_gpu_mem_overhead",
                    "gnn_knn_gpu_mem_overhead",
                    2.0,
                )
            ),
            max_fit_rows=self._resolve_gnn_setting(
                "max_fit_rows",
                "gnn_max_fit_rows",
                None,
            ),
            max_predict_rows=self._resolve_gnn_setting(
                "max_predict_rows",
                "gnn_max_predict_rows",
                None,
            ),
            predict_chunk_rows=self._resolve_gnn_setting(
                "predict_chunk_rows",
                "gnn_predict_chunk_rows",
                None,
            ),
            loss_name=loss_name,
            distribution=getattr(self.ctx, "distribution", None),
        )
        return self._apply_dataloader_overrides(model)

    def cross_val(self, trial: optuna.trial.Trial) -> float:
        base_tw_power = self.ctx.default_tweedie_power()
        loss_name = getattr(self.ctx, "loss_name", "tweedie")
        metric_ctx: Dict[str, Any] = {}

        def data_provider():
            data = self.ctx.train_oht_data if self.ctx.train_oht_data is not None else self.ctx.train_oht_scl_data
            assert data is not None, "Preprocessed training data is missing."
            return data[self.ctx.var_nmes], data[self.ctx.resp_nme], data[self.ctx.weight_nme]

        def model_builder(params: Dict[str, Any]):
            if loss_name == "tweedie":
                tw_power = params.get("tw_power", base_tw_power)
            elif loss_name in ("poisson", "gamma"):
                tw_power = base_tw_power
            else:
                tw_power = None
            metric_ctx["tw_power"] = tw_power
            if tw_power is None:
                params = dict(params)
                params.pop("tw_power", None)
            return self._build_model(params)

        def preprocess_fn(X_train, X_val):
            X_train_s, X_val_s, _ = self._standardize_fold(
                X_train, X_val, self.ctx.num_features)
            return X_train_s, X_val_s

        def fit_predict(model, X_train, y_train, w_train, X_val, y_val, w_val, trial_obj):
            model.fit(
                X_train,
                y_train,
                w_train=w_train,
                X_val=X_val,
                y_val=y_val,
                w_val=w_val,
                trial=trial_obj,
            )
            return model.predict(X_val)

        def metric_fn(y_true, y_pred, weight):
            if self.ctx.task_type == 'classification':
                y_pred_clipped = np.clip(y_pred, EPS, 1 - EPS)
                return log_loss(y_true, y_pred_clipped, sample_weight=weight)
            return regression_loss(
                y_true,
                y_pred,
                weight,
                loss_name=loss_name,
                tweedie_power=metric_ctx.get("tw_power", base_tw_power),
            )

        # Keep GNN BO lightweight: sample during CV, use full data for final training.
        X_cap = data_provider()[0]
        sample_limit = min(200000, len(X_cap)) if len(X_cap) > 200000 else None

        param_space: Dict[str, Callable[[optuna.trial.Trial], Any]] = {
            "learning_rate": lambda t: t.suggest_float('learning_rate', 1e-4, 5e-3, log=True),
            "hidden_dim": lambda t: t.suggest_int('hidden_dim', 16, 128, step=16),
            "num_layers": lambda t: t.suggest_int('num_layers', 1, 4),
            "k_neighbors": lambda t: t.suggest_int('k_neighbors', 5, 30),
            "dropout": lambda t: t.suggest_float('dropout', 0.0, 0.3),
            "weight_decay": lambda t: t.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        }
        if self.ctx.task_type == 'regression' and loss_name == 'tweedie':
            param_space["tw_power"] = lambda t: t.suggest_float(
                'tw_power', 1.0, 2.0)

        return self.cross_val_generic(
            trial=trial,
            hyperparameter_space=param_space,
            data_provider=data_provider,
            model_builder=model_builder,
            metric_fn=metric_fn,
            sample_limit=sample_limit,
            preprocess_fn=preprocess_fn,
            fit_predict_fn=fit_predict,
            cleanup_fn=lambda m: getattr(
                getattr(m, "gnn", None), "to", lambda *_args, **_kwargs: None)("cpu")
        )

    def train(self) -> None:
        if not self.best_params:
            raise RuntimeError("Run tune() first to obtain best GNN parameters.")

        data = self.ctx.train_oht_scl_data
        assert data is not None, "Preprocessed training data is missing."
        X_all = data[self.ctx.var_nmes]
        y_all = data[self.ctx.resp_nme]
        w_all = data[self.ctx.weight_nme]

        use_refit = bool(getattr(self.ctx.config, "final_refit", True))
        refit_epochs = None

        split = self._resolve_train_val_indices(X_all)
        if split is not None:
            train_idx, val_idx = split
            X_train = X_all.iloc[train_idx]
            y_train = y_all.iloc[train_idx]
            w_train = w_all.iloc[train_idx]
            X_val = X_all.iloc[val_idx]
            y_val = y_all.iloc[val_idx]
            w_val = w_all.iloc[val_idx]

            if use_refit:
                tmp_model = self._build_model(self.best_params)
                tmp_model.fit(
                    X_train,
                    y_train,
                    w_train=w_train,
                    X_val=X_val,
                    y_val=y_val,
                    w_val=w_val,
                    trial=None,
                )
                refit_epochs = int(getattr(tmp_model, "best_epoch", None) or self.ctx.epochs)
                self._maybe_cleanup_gpu(tmp_model)
            else:
                self.model = self._build_model(self.best_params)
                self.model.fit(
                    X_train,
                    y_train,
                    w_train=w_train,
                    X_val=X_val,
                    y_val=y_val,
                    w_val=w_val,
                    trial=None,
                )
        else:
            use_refit = False

        if use_refit:
            self.model = self._build_model(self.best_params)
            if refit_epochs is not None:
                self.model.epochs = int(refit_epochs)
            self.model.fit(
                X_all,
                y_all,
                w_train=w_all,
                X_val=None,
                y_val=None,
                w_val=None,
                trial=None,
            )
        elif self.model is None:
            self.model = self._build_model(self.best_params)
            self.model.fit(
                X_all,
                y_all,
                w_train=w_all,
                X_val=None,
                y_val=None,
                w_val=None,
                trial=None,
            )
        self.ctx.model_label.append(self.label)
        self._predict_and_cache(self.model, pred_prefix='gnn', use_oht=True)
        self.ctx.gnn_best = self.model

        # If geo_feature_nmes is set, refresh geo tokens for FT input.
        if self._geo_feature_names():
            self.prepare_geo_tokens(force=True)

    def ensemble_predict(self, k: int) -> None:
        if not self.best_params:
            raise RuntimeError("Run tune() first to obtain best GNN parameters.")
        data = self.ctx.train_oht_scl_data
        test_data = self.ctx.test_oht_scl_data
        if data is None or test_data is None:
            raise RuntimeError("Missing standardized data for GNN ensemble.")
        X_all = data[self.ctx.var_nmes]
        y_all = data[self.ctx.resp_nme]
        w_all = data[self.ctx.weight_nme]
        X_test = test_data[self.ctx.var_nmes]

        k = max(2, int(k))
        n_samples = len(X_all)
        split_iter, _ = self._resolve_ensemble_splits(X_all, k=k)
        if split_iter is None:
            _log(
                f"[GNN Ensemble] unable to build CV split (n_samples={n_samples}); skip ensemble.",
                flush=True,
            )
            return
        preds_train_sum = np.zeros(n_samples, dtype=np.float64)
        preds_test_sum = np.zeros(len(X_test), dtype=np.float64)

        split_count = 0
        for train_idx, val_idx in split_iter:
            model = self._build_model(self.best_params)
            model.fit(
                X_all.iloc[train_idx],
                y_all.iloc[train_idx],
                w_train=w_all.iloc[train_idx],
                X_val=X_all.iloc[val_idx],
                y_val=y_all.iloc[val_idx],
                w_val=w_all.iloc[val_idx],
                trial=None,
            )
            pred_train = model.predict(X_all)
            pred_test = model.predict(X_test)
            preds_train_sum += np.asarray(pred_train, dtype=np.float64)
            preds_test_sum += np.asarray(pred_test, dtype=np.float64)
            self._maybe_cleanup_gpu(model)
            split_count += 1

        if split_count < 1:
            _log(
                f"[GNN Ensemble] no CV splits generated; skip ensemble.",
                flush=True,
            )
            return
        preds_train = preds_train_sum / float(split_count)
        preds_test = preds_test_sum / float(split_count)
        self._cache_predictions("gnn", preds_train, preds_test)

    def prepare_geo_tokens(self, force: bool = False) -> None:
        """Train/update the GNN encoder for geo tokens and inject them into FT input."""
        geo_cols = self._geo_feature_names()
        if not geo_cols:
            return
        if (not force) and self.ctx.train_geo_tokens is not None and self.ctx.test_geo_tokens is not None:
            return

        result = self.ctx._build_geo_tokens()
        if result is None:
            return
        train_tokens, test_tokens, cols, geo_gnn = result
        self.ctx.train_geo_tokens = train_tokens
        self.ctx.test_geo_tokens = test_tokens
        self.ctx.geo_token_cols = cols
        self.ctx.geo_gnn_model = geo_gnn
        _log(f"[GeoToken][GNNTrainer] Generated {len(cols)} dims and injected into FT.", flush=True)

    def save(self) -> None:
        if self.model is None:
            _log(f"[save] Warning: No model to save for {self.label}")
            return
        path = self.output.model_path(self._get_model_filename())
        base_gnn = getattr(self.model, "_unwrap_gnn", lambda: None)()
        if base_gnn is not None:
            base_gnn = base_gnn.to("cpu")
        state = None if base_gnn is None else base_gnn.state_dict()
        payload = {
            "best_params": self.best_params,
            "state_dict": state,
            "preprocess_artifacts": self._export_preprocess_artifacts(),
        }
        torch.save(payload, path)

    def load(self) -> None:
        path = self.output.model_path(self._get_model_filename())
        if not os.path.exists(path):
            _log(f"[load] Warning: Model file not found: {path}")
            return
        payload = torch_load(path, map_location='cpu', weights_only=False)
        try:
            model, params, warning = rebuild_gnn_model_from_payload(
                payload=payload,
                model_builder=self._build_model,
                strict=True,
                allow_non_strict_fallback=True,
            )
        except ValueError as exc:
            raise ValueError(f"Invalid GNN checkpoint: {path}") from exc
        if warning:
            _log(f"[GNN load] Warning: State dict mismatch, loading with strict=False: {warning}")
        self.model = model
        self.best_params = dict(params) if isinstance(params, dict) else None
        self.ctx.gnn_best = self.model
