from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from ins_pricing.modelling.bayesopt.artifacts import (
    best_params_csv_path,
    load_best_params_csv,
)
from ins_pricing.modelling.bayesopt.config_preprocess import BayesOptConfig, DatasetPreprocessor, OutputManager, VersionManager
from ins_pricing.modelling.bayesopt.model_explain_mixin import BayesOptExplainMixin
from ins_pricing.modelling.bayesopt.model_plotting_mixin import BayesOptPlottingMixin
from ins_pricing.modelling.bayesopt.models import GraphNeuralNetSklearn
from ins_pricing.modelling.bayesopt.trainers import FTTrainer, GLMTrainer, GNNTrainer, ResNetTrainer, XGBTrainer
from ins_pricing.modelling.bayesopt.trainers.cv_utils import CVStrategyResolver
from ins_pricing.utils import EPS, DeviceManager, set_global_seed, get_logger, log_print
from ins_pricing.utils.io import IOUtils
from ins_pricing.utils.losses import (
    normalize_distribution_name,
    resolve_effective_loss_name,
    resolve_tweedie_power,
    resolve_xgb_objective,
)

_logger = get_logger("ins_pricing.modelling.bayesopt.core")


def _log(*args, **kwargs) -> None:
    log_print(_logger, *args, **kwargs)


class _ResolvedCVSplitter:
    """Adapter exposing a sklearn-like split() API via CVStrategyResolver."""

    def __init__(
        self,
        resolver: CVStrategyResolver,
        *,
        n_splits: int,
        val_ratio: float,
    ) -> None:
        self._resolver = resolver
        self._n_splits = max(2, int(n_splits))
        self._val_ratio = float(val_ratio)

    def split(self, X, y=None, groups=None):  # pylint: disable=unused-argument
        split_iter, _ = self._resolver.create_cv_splitter(
            X_all=X,
            y_all=y,
            n_splits=self._n_splits,
            val_ratio=self._val_ratio,
        )
        for tr_idx, val_idx in split_iter:
            yield tr_idx, val_idx

# BayesOpt orchestration and SHAP utilities
# =============================================================================
class BayesOptModel(BayesOptPlottingMixin, BayesOptExplainMixin):
    def __init__(self, train_data, test_data, config: BayesOptConfig):
        """Orchestrate BayesOpt training across multiple trainers.

        Args:
            train_data: Training DataFrame.
            test_data: Test DataFrame.
            config: BayesOptConfig instance with all configuration.

        Examples:
            config = BayesOptConfig(
                model_nme="my_model",
                resp_nme="target",
                weight_nme="weight",
                factor_nmes=["feat1", "feat2"]
            )
            model = BayesOptModel(train_df, test_df, config=config)
        """
        if not isinstance(config, BayesOptConfig):
            raise TypeError(
                f"config must be a BayesOptConfig instance, got {type(config).__name__}"
            )
        cfg = config
        self.config = cfg
        self.model_nme = cfg.model_nme
        self.task_type = cfg.task_type
        normalized_distribution = normalize_distribution_name(
            getattr(cfg, "distribution", None),
            self.task_type,
        )
        self.distribution = None if normalized_distribution == "auto" else normalized_distribution
        self.loss_name = resolve_effective_loss_name(
            getattr(cfg, "loss_name", None),
            task_type=self.task_type,
            model_name=self.model_nme,
            distribution=self.distribution,
        )
        if hasattr(self.config, "distribution"):
            self.config.distribution = self.distribution
        self.resp_nme = cfg.resp_nme
        self.weight_nme = cfg.weight_nme
        self.factor_nmes = cfg.factor_nmes
        self.binary_resp_nme = cfg.binary_resp_nme
        self.cate_list = list(cfg.cate_list or [])
        self.prop_test = cfg.prop_test
        self.epochs = cfg.epochs
        self.rand_seed = cfg.rand_seed if cfg.rand_seed is not None else np.random.randint(
            1, 10000)
        set_global_seed(int(self.rand_seed))
        self.use_gpu = bool(
            cfg.use_gpu
            and (torch.cuda.is_available() or DeviceManager.is_mps_available())
        )
        self.output_manager = OutputManager(
            cfg.output_dir or os.getcwd(), self.model_nme)

        bundle_path = self._resolve_preprocess_bundle_path()
        load_preprocess_bundle = bool(
            getattr(self.config, "load_preprocess_bundle", False)
        )
        save_preprocess_bundle = bool(
            getattr(self.config, "save_preprocess_bundle", False)
        )

        if load_preprocess_bundle:
            if bundle_path is None:
                raise ValueError(
                    "load_preprocess_bundle=True requires preprocess_bundle_path."
                )
            self._load_preprocess_bundle(bundle_path)
        else:
            if train_data is None or test_data is None:
                raise ValueError(
                    "train_data/test_data must be provided unless "
                    "load_preprocess_bundle=True."
                )
            preprocessor = DatasetPreprocessor(train_data, test_data, cfg).run()
            self.train_data = preprocessor.train_data
            self.test_data = preprocessor.test_data
            self.train_oht_data = preprocessor.train_oht_data
            self.test_oht_data = preprocessor.test_oht_data
            self.train_oht_scl_data = preprocessor.train_oht_scl_data
            self.test_oht_scl_data = preprocessor.test_oht_scl_data
            self.var_nmes = preprocessor.var_nmes
            self.num_features = preprocessor.num_features
            self.cat_categories_for_shap = preprocessor.cat_categories_for_shap
            self.numeric_scalers = preprocessor.numeric_scalers
            self.ohe_feature_names = list(
                getattr(preprocessor, "ohe_feature_names", []) or []
            )
            self.oht_sparse_csr = bool(
                getattr(preprocessor, "oht_sparse_csr", False)
            )
            if getattr(self.config, "save_preprocess", False):
                artifact_path = getattr(self.config, "preprocess_artifact_path", None)
                if artifact_path:
                    target = Path(str(artifact_path))
                    if not target.is_absolute():
                        target = Path(self.output_manager.result_dir) / target
                else:
                    target = Path(self.output_manager.result_path(
                        f"{self.model_nme}_preprocess.json"
                    ))
                preprocessor.save_artifacts(target)
            if save_preprocess_bundle and bundle_path is not None:
                self._save_preprocess_bundle(bundle_path)
        self.geo_token_cols: List[str] = []
        self.train_geo_tokens: Optional[pd.DataFrame] = None
        self.test_geo_tokens: Optional[pd.DataFrame] = None
        self.geo_gnn_model: Optional[GraphNeuralNetSklearn] = None
        self._add_region_effect()

        self.cv = self._build_cv_splitter()
        if self.task_type == 'classification':
            self.obj = 'binary:logistic'
        else:  # regression task
            self.obj = resolve_xgb_objective(self.loss_name)
        self.fit_params = {
            'sample_weight': self.train_data[self.weight_nme].values
        }
        self.model_label: List[str] = []
        self.optuna_storage = cfg.optuna_storage
        self.optuna_study_prefix = cfg.optuna_study_prefix or "bayesopt"

        # Keep trainers in a dict for unified access and easy extension.
        self.trainers: Dict[str, TrainerBase] = {
            'glm': GLMTrainer(self),
            'xgb': XGBTrainer(self),
            'resn': ResNetTrainer(self),
            'ft': FTTrainer(self),
            'gnn': GNNTrainer(self),
        }
        self._prepare_geo_tokens()
        self.version_manager = VersionManager(self.output_manager)

    def _resolve_preprocess_bundle_path(self) -> Optional[Path]:
        raw_path = getattr(self.config, "preprocess_bundle_path", None)
        if raw_path is None:
            return None
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = Path(self.output_manager.result_dir) / path
        return path

    def _save_preprocess_bundle(self, target: Path) -> None:
        payload = {
            "schema_version": 1,
            "model_nme": self.model_nme,
            "train_data": self.train_data,
            "test_data": self.test_data,
            "train_oht_data": self.train_oht_data,
            "test_oht_data": self.test_oht_data,
            "train_oht_scl_data": self.train_oht_scl_data,
            "test_oht_scl_data": self.test_oht_scl_data,
            "var_nmes": list(self.var_nmes),
            "num_features": list(self.num_features),
            "cat_categories_for_shap": dict(self.cat_categories_for_shap),
            "numeric_scalers": dict(self.numeric_scalers),
            "ohe_feature_names": list(getattr(self, "ohe_feature_names", []) or []),
            "oht_sparse_csr": bool(getattr(self, "oht_sparse_csr", False)),
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        _log(f"[PreprocessBundle] Saved: {target}", flush=True)

    def _load_preprocess_bundle(self, source: Path) -> None:
        if not source.exists():
            raise FileNotFoundError(
                f"preprocess bundle not found: {source}"
            )
        with source.open("rb") as fh:
            payload = pickle.load(fh)
        if not isinstance(payload, dict):
            raise TypeError(
                f"Invalid preprocess bundle payload type: {type(payload).__name__}"
            )
        required_keys = [
            "train_data",
            "test_data",
            "train_oht_scl_data",
            "test_oht_scl_data",
            "var_nmes",
            "num_features",
            "cat_categories_for_shap",
            "numeric_scalers",
        ]
        missing = [k for k in required_keys if k not in payload]
        if missing:
            raise KeyError(
                f"Preprocess bundle missing keys: {missing}"
            )
        self.train_data = payload["train_data"]
        self.test_data = payload["test_data"]
        self.train_oht_data = payload.get("train_oht_data")
        self.test_oht_data = payload.get("test_oht_data")
        self.train_oht_scl_data = payload["train_oht_scl_data"]
        self.test_oht_scl_data = payload["test_oht_scl_data"]
        self.var_nmes = list(payload["var_nmes"])
        self.num_features = list(payload["num_features"])
        self.cat_categories_for_shap = dict(payload["cat_categories_for_shap"])
        self.numeric_scalers = dict(payload["numeric_scalers"])
        self.ohe_feature_names = list(payload.get("ohe_feature_names", []) or [])
        self.oht_sparse_csr = bool(payload.get("oht_sparse_csr", False))
        _log(f"[PreprocessBundle] Loaded: {source}", flush=True)

    def _build_cv_splitter(self) -> _ResolvedCVSplitter:
        val_ratio = float(self.prop_test) if self.prop_test is not None else 0.25
        if not (0.0 < val_ratio < 1.0):
            val_ratio = 0.25
        cv_splits = getattr(self.config, "cv_splits", None)
        if cv_splits is None:
            cv_splits = max(2, int(round(1 / val_ratio)))
        resolver = CVStrategyResolver(
            self.config,
            self.train_data,
            self.rand_seed,
        )
        return _ResolvedCVSplitter(
            resolver,
            n_splits=int(cv_splits),
            val_ratio=val_ratio,
        )

    def default_tweedie_power(self, obj: Optional[str] = None) -> Optional[float]:
        if self.task_type == 'classification':
            return None
        loss_name = getattr(self, "loss_name", None)
        if loss_name:
            resolved = resolve_tweedie_power(str(loss_name), default=1.5)
            if resolved is not None:
                return resolved
        objective = obj or getattr(self, "obj", None)
        if objective == 'count:poisson':
            return 1.0
        if objective == 'reg:gamma':
            return 2.0
        return 1.5

    def _build_geo_tokens(self, params_override: Optional[Dict[str, Any]] = None):
        """Internal builder; allows trial overrides and returns None on failure."""
        geo_cfg = self.config.geo_token
        gnn_cfg = self.config.gnn
        geo_cols = list(geo_cfg.feature_nmes or [])
        if not geo_cols:
            return None

        available = [c for c in geo_cols if c in self.train_data.columns]
        if not available:
            return None

        # Preprocess text/numeric: fill numeric with median, label-encode text, map unknowns.
        proc_train = {}
        proc_test = {}
        for col in available:
            s_train = self.train_data[col]
            s_test = self.test_data[col]
            if pd.api.types.is_numeric_dtype(s_train):
                tr = pd.to_numeric(s_train, errors="coerce")
                te = pd.to_numeric(s_test, errors="coerce")
                med = np.nanmedian(tr)
                proc_train[col] = np.nan_to_num(tr, nan=med).astype(np.float32)
                proc_test[col] = np.nan_to_num(te, nan=med).astype(np.float32)
            else:
                cats = pd.Categorical(s_train.astype(str))
                tr_codes = cats.codes.astype(np.float32, copy=True)
                tr_codes[tr_codes < 0] = len(cats.categories)
                te_cats = pd.Categorical(
                    s_test.astype(str), categories=cats.categories)
                te_codes = te_cats.codes.astype(np.float32, copy=True)
                te_codes[te_codes < 0] = len(cats.categories)
                proc_train[col] = tr_codes
                proc_test[col] = te_codes

        train_geo_raw = pd.DataFrame(proc_train, index=self.train_data.index)
        test_geo_raw = pd.DataFrame(proc_test, index=self.test_data.index)

        scaler = StandardScaler()
        train_geo = pd.DataFrame(
            scaler.fit_transform(train_geo_raw),
            columns=available,
            index=self.train_data.index
        )
        test_geo = pd.DataFrame(
            scaler.transform(test_geo_raw),
            columns=available,
            index=self.test_data.index
        )

        tw_power = self.default_tweedie_power()

        cfg = params_override or {}
        try:
            geo_gnn = GraphNeuralNetSklearn(
                model_nme=f"{self.model_nme}_geo",
                input_dim=len(available),
                hidden_dim=cfg.get("geo_token_hidden_dim", geo_cfg.hidden_dim),
                num_layers=cfg.get("geo_token_layers", geo_cfg.layers),
                k_neighbors=cfg.get("geo_token_k_neighbors", geo_cfg.k_neighbors),
                dropout=cfg.get("geo_token_dropout", geo_cfg.dropout),
                learning_rate=cfg.get(
                    "geo_token_learning_rate",
                    geo_cfg.learning_rate,
                ),
                epochs=int(cfg.get("geo_token_epochs", geo_cfg.epochs)),
                patience=5,
                task_type=self.task_type,
                tweedie_power=tw_power,
                loss_name=self.loss_name,
                distribution=self.distribution,
                use_data_parallel=False,
                use_ddp=False,
                use_approx_knn=gnn_cfg.use_approx_knn,
                approx_knn_threshold=gnn_cfg.approx_knn_threshold,
                graph_cache_path=None,
                max_gpu_knn_nodes=gnn_cfg.max_gpu_knn_nodes,
                knn_gpu_mem_ratio=gnn_cfg.knn_gpu_mem_ratio,
                knn_gpu_mem_overhead=gnn_cfg.knn_gpu_mem_overhead,
            )
            geo_gnn.fit(
                train_geo,
                self.train_data[self.resp_nme],
                self.train_data[self.weight_nme]
            )
            train_embed = geo_gnn.encode(train_geo)
            test_embed = geo_gnn.encode(test_geo)
            cols = [f"geo_token_{i}" for i in range(train_embed.shape[1])]
            train_tokens = pd.DataFrame(
                train_embed, index=self.train_data.index, columns=cols)
            test_tokens = pd.DataFrame(
                test_embed, index=self.test_data.index, columns=cols)
            return train_tokens, test_tokens, cols, geo_gnn
        except Exception as exc:
            _log(f"[GeoToken] Generation failed: {exc}")
            return None

    def _prepare_geo_tokens(self) -> None:
        """Build and persist geo tokens with default config values."""
        gnn_trainer = self.trainers.get("gnn")
        if gnn_trainer is not None and hasattr(gnn_trainer, "prepare_geo_tokens"):
            try:
                gnn_trainer.prepare_geo_tokens(force=False)  # type: ignore[attr-defined]
                return
            except Exception as exc:
                _log(f"[GeoToken] GNNTrainer generation failed: {exc}")

        result = self._build_geo_tokens()
        if result is None:
            return
        train_tokens, test_tokens, cols, geo_gnn = result
        self.train_geo_tokens = train_tokens
        self.test_geo_tokens = test_tokens
        self.geo_token_cols = cols
        self.geo_gnn_model = geo_gnn
        _log(f"[GeoToken] Generated {len(cols)}-dim geo tokens; injecting into FT.")

    def _add_region_effect(self) -> None:
        """Partial pooling over province/city to create a smoothed region_effect feature."""
        region_cfg = self.config.region
        prov_col = region_cfg.province_col
        city_col = region_cfg.city_col
        if not prov_col or not city_col:
            return
        for col in [prov_col, city_col]:
            if col not in self.train_data.columns:
                _log(f"[RegionEffect] Missing column {col}; skipped.")
                return

        def safe_mean(df: pd.DataFrame) -> float:
            w = df[self.weight_nme]
            y = df[self.resp_nme]
            denom = max(float(w.sum()), EPS)
            return float((y * w).sum() / denom)

        global_mean = safe_mean(self.train_data)
        alpha = max(float(region_cfg.effect_alpha), 0.0)

        w_all = self.train_data[self.weight_nme]
        y_all = self.train_data[self.resp_nme]
        yw_all = y_all * w_all

        prov_sumw = w_all.groupby(self.train_data[prov_col]).sum()
        prov_sumyw = yw_all.groupby(self.train_data[prov_col]).sum()
        prov_mean = (prov_sumyw / prov_sumw.clip(lower=EPS)).astype(float)
        prov_mean = prov_mean.fillna(global_mean)

        city_sumw = self.train_data.groupby([prov_col, city_col])[
            self.weight_nme].sum()
        city_sumyw = yw_all.groupby(
            [self.train_data[prov_col], self.train_data[city_col]]).sum()
        city_df = pd.DataFrame({
            "sum_w": city_sumw,
            "sum_yw": city_sumyw,
        })
        city_df["prior"] = city_df.index.get_level_values(0).map(
            prov_mean).fillna(global_mean)
        city_df["effect"] = (
            city_df["sum_yw"] + alpha * city_df["prior"]
        ) / (city_df["sum_w"] + alpha).clip(lower=EPS)
        city_effect = city_df["effect"]

        def lookup_effect(df: pd.DataFrame) -> pd.Series:
            idx = pd.MultiIndex.from_frame(df[[prov_col, city_col]])
            effects = city_effect.reindex(idx).to_numpy(dtype=np.float64)
            prov_fallback = df[prov_col].map(
                prov_mean).fillna(global_mean).to_numpy(dtype=np.float64)
            effects = np.where(np.isfinite(effects), effects, prov_fallback)
            effects = np.where(np.isfinite(effects), effects, global_mean)
            return pd.Series(effects, index=df.index, dtype=np.float32)

        re_train = lookup_effect(self.train_data)
        re_test = lookup_effect(self.test_data)

        col_name = "region_effect"
        self.train_data[col_name] = re_train
        self.test_data[col_name] = re_test

        # Sync into one-hot and scaled variants.
        for df in [self.train_oht_data, self.test_oht_data]:
            if df is not None:
                df[col_name] = re_train if df is self.train_oht_data else re_test

        # Standardize region_effect and propagate.
        scaler = StandardScaler()
        re_train_s = scaler.fit_transform(
            re_train.values.reshape(-1, 1)).astype(np.float32).reshape(-1)
        re_test_s = scaler.transform(
            re_test.values.reshape(-1, 1)).astype(np.float32).reshape(-1)
        for df in [self.train_oht_scl_data, self.test_oht_scl_data]:
            if df is not None:
                df[col_name] = re_train_s if df is self.train_oht_scl_data else re_test_s

        # Update feature lists.
        if col_name not in self.factor_nmes:
            self.factor_nmes.append(col_name)
        if col_name not in self.num_features:
            self.num_features.append(col_name)
        if self.train_oht_scl_data is not None:
            excluded = {self.weight_nme, self.resp_nme}
            self.var_nmes = [
                col for col in self.train_oht_scl_data.columns if col not in excluded
            ]

    def _require_trainer(self, model_key: str) -> "TrainerBase":
        trainer = self.trainers.get(model_key)
        if trainer is None:
            raise KeyError(f"Unknown model key: {model_key}")
        return trainer

    def _pred_vector_columns(self, pred_prefix: str) -> List[str]:
        """Return vector feature columns like pred_<prefix>_0.. sorted by suffix."""
        col_prefix = f"pred_{pred_prefix}_"
        cols = [c for c in self.train_data.columns if c.startswith(col_prefix)]

        def sort_key(name: str):
            tail = name.rsplit("_", 1)[-1]
            try:
                return (0, int(tail))
            except Exception:
                return (1, tail)

        cols.sort(key=sort_key)
        return cols

    def _inject_pred_features(self, pred_prefix: str) -> List[str]:
        """Inject pred_<prefix> or pred_<prefix>_i columns into features and return names."""
        cols = self._pred_vector_columns(pred_prefix)
        if cols:
            self.add_numeric_features_from_columns(cols)
            return cols
        scalar_col = f"pred_{pred_prefix}"
        if scalar_col in self.train_data.columns:
            self.add_numeric_feature_from_column(scalar_col)
            return [scalar_col]
        return []

    def _maybe_load_best_params(self, model_key: str, trainer: "TrainerBase") -> None:
        # 1) If best_params_files is specified, load and skip tuning.
        best_params_files = getattr(self.config, "best_params_files", None) or {}
        best_params_file = best_params_files.get(model_key)
        if best_params_file and not trainer.best_params:
            trainer.best_params = IOUtils.load_params_file(best_params_file)
            trainer.best_trial = None
            _log(
                f"[Optuna][{trainer.label}] Loaded best_params from {best_params_file}; skip tuning."
            )

        # 2) If reuse_best_params is enabled, prefer version snapshots; else load legacy CSV.
        reuse_params = bool(getattr(self.config, "reuse_best_params", False))
        if reuse_params and not trainer.best_params:
            payload = self.version_manager.load_latest(f"{model_key}_best")
            best_params = None if payload is None else payload.get("best_params")
            if best_params:
                trainer.best_params = best_params
                trainer.best_trial = None
                trainer.study_name = payload.get(
                    "study_name") if isinstance(payload, dict) else None
                _log(
                    f"[Optuna][{trainer.label}] Reusing best_params from versions snapshot.")
                return

            params_path = best_params_csv_path(
                self.output_manager.result_dir,
                self.model_nme,
                trainer.label,
            )
            params = load_best_params_csv(
                self.output_manager.result_dir,
                self.model_nme,
                trainer.label,
            )
            if params is not None:
                trainer.best_params = params
                trainer.best_trial = None
                _log(
                    f"[Optuna][{trainer.label}] Reusing best_params from {params_path}."
                )

    # Generic optimization entry point.
    def optimize_model(self, model_key: str, max_evals: int = 100):
        if model_key not in self.trainers:
            _log(f"Warning: Unknown model key: {model_key}")
            return

        trainer = self._require_trainer(model_key)
        self._maybe_load_best_params(model_key, trainer)

        should_tune = not trainer.best_params
        if should_tune:
            if model_key == "ft" and str(self.config.ft_role) == "unsupervised_embedding":
                if hasattr(trainer, "cross_val_unsupervised"):
                    trainer.tune(
                        max_evals,
                        objective_fn=getattr(trainer, "cross_val_unsupervised")
                    )
                else:
                    raise RuntimeError(
                        "FT trainer does not support unsupervised Optuna objective.")
            else:
                trainer.tune(max_evals)

        if model_key == "ft" and str(self.config.ft_role) != "model":
            prefix = str(self.config.ft_feature_prefix or "ft_emb")
            role = str(self.config.ft_role)
            if role == "embedding":
                trainer.train_as_feature(
                    pred_prefix=prefix, feature_mode="embedding")
            elif role == "unsupervised_embedding":
                trainer.pretrain_unsupervised_as_feature(
                    pred_prefix=prefix,
                    params=trainer.best_params
                )
            else:
                raise ValueError(
                    f"Unsupported ft_role='{role}', expected 'model'/'embedding'/'unsupervised_embedding'.")

            # Inject generated prediction/embedding columns as features (scalar or vector).
            self._inject_pred_features(prefix)
            # Do not add FT as a standalone model label; downstream models handle evaluation.
        else:
            trainer.train()

        if bool(getattr(self.config, "final_ensemble", False)):
            k = int(getattr(self.config, "final_ensemble_k", 3) or 3)
            if (
                k > 1
                and not (model_key == "ft" and str(self.config.ft_role) != "model")
            ):
                if hasattr(trainer, "ensemble_predict"):
                    trainer.ensemble_predict(k)
                else:
                    _log(
                        f"[Ensemble] Trainer '{model_key}' does not support ensemble prediction.",
                        flush=True,
                    )

        # Update context fields for backward compatibility.
        setattr(self, f"{model_key}_best", trainer.model)
        setattr(self, f"best_{model_key}_params", trainer.best_params)
        setattr(self, f"best_{model_key}_trial", trainer.best_trial)
        # Save a snapshot for traceability.
        study_name = getattr(trainer, "study_name", None)
        if study_name is None and trainer.best_trial is not None:
            study_obj = getattr(trainer.best_trial, "study", None)
            study_name = getattr(study_obj, "study_name", None)
        snapshot = {
            "model_key": model_key,
            "timestamp": datetime.now().isoformat(),
            "best_params": trainer.best_params,
            "study_name": study_name,
            "config": asdict(self.config),
        }
        self.version_manager.save(f"{model_key}_best", snapshot)

    def add_numeric_feature_from_column(self, col_name: str) -> None:
        """Add an existing column as a feature and sync one-hot/scaled tables."""
        if col_name not in self.train_data.columns or col_name not in self.test_data.columns:
            raise KeyError(
                f"Column '{col_name}' must exist in both train_data and test_data.")

        if col_name not in self.factor_nmes:
            self.factor_nmes.append(col_name)
        if col_name not in self.config.factor_nmes:
            self.config.factor_nmes.append(col_name)

        if col_name not in self.cate_list and col_name not in self.num_features:
            self.num_features.append(col_name)

        if self.train_oht_data is not None and self.test_oht_data is not None:
            self.train_oht_data[col_name] = self.train_data[col_name].values
            self.test_oht_data[col_name] = self.test_data[col_name].values
        if self.train_oht_scl_data is not None and self.test_oht_scl_data is not None:
            scaler = StandardScaler()
            tr = self.train_data[col_name].to_numpy(
                dtype=np.float32, copy=False).reshape(-1, 1)
            te = self.test_data[col_name].to_numpy(
                dtype=np.float32, copy=False).reshape(-1, 1)
            self.train_oht_scl_data[col_name] = scaler.fit_transform(
                tr).reshape(-1)
            self.test_oht_scl_data[col_name] = scaler.transform(te).reshape(-1)

        if col_name not in self.var_nmes:
            self.var_nmes.append(col_name)

    def add_numeric_features_from_columns(self, col_names: List[str]) -> None:
        if not col_names:
            return

        missing = [
            col for col in col_names
            if col not in self.train_data.columns or col not in self.test_data.columns
        ]
        if missing:
            raise KeyError(
                f"Column(s) {missing} must exist in both train_data and test_data."
            )

        for col_name in col_names:
            if col_name not in self.factor_nmes:
                self.factor_nmes.append(col_name)
            if col_name not in self.config.factor_nmes:
                self.config.factor_nmes.append(col_name)
            if col_name not in self.cate_list and col_name not in self.num_features:
                self.num_features.append(col_name)
            if col_name not in self.var_nmes:
                self.var_nmes.append(col_name)

        if self.train_oht_data is not None and self.test_oht_data is not None:
            self.train_oht_data[col_names] = self.train_data[col_names].to_numpy(copy=False)
            self.test_oht_data[col_names] = self.test_data[col_names].to_numpy(copy=False)

        if self.train_oht_scl_data is not None and self.test_oht_scl_data is not None:
            scaler = StandardScaler()
            tr = self.train_data[col_names].to_numpy(dtype=np.float32, copy=False)
            te = self.test_data[col_names].to_numpy(dtype=np.float32, copy=False)
            self.train_oht_scl_data[col_names] = scaler.fit_transform(tr)
            self.test_oht_scl_data[col_names] = scaler.transform(te)

    def prepare_ft_as_feature(self, max_evals: int = 50, pred_prefix: str = "ft_feat") -> str:
        """Train FT as a feature generator and return the downstream column name."""
        ft_trainer = self._require_trainer("ft")
        ft_trainer.tune(max_evals=max_evals)
        if hasattr(ft_trainer, "train_as_feature"):
            ft_trainer.train_as_feature(pred_prefix=pred_prefix)
        else:
            ft_trainer.train()
        feature_col = f"pred_{pred_prefix}"
        self.add_numeric_feature_from_column(feature_col)
        return feature_col

    def prepare_ft_embedding_as_features(self, max_evals: int = 50, pred_prefix: str = "ft_emb") -> List[str]:
        """Train FT and inject pooled embeddings as vector features pred_<prefix>_0.. ."""
        ft_trainer = self._require_trainer("ft")
        ft_trainer.tune(max_evals=max_evals)
        if hasattr(ft_trainer, "train_as_feature"):
            ft_trainer.train_as_feature(
                pred_prefix=pred_prefix, feature_mode="embedding")
        else:
            raise RuntimeError(
                "FT trainer does not support embedding feature mode.")
        cols = self._pred_vector_columns(pred_prefix)
        if not cols:
            raise RuntimeError(
                f"No embedding columns were generated for prefix '{pred_prefix}'.")
        self.add_numeric_features_from_columns(cols)
        return cols

    def prepare_ft_unsupervised_embedding_as_features(self,
                                                      pred_prefix: str = "ft_uemb",
                                                      params: Optional[Dict[str,
                                                                            Any]] = None,
                                                      mask_prob_num: float = 0.15,
                                                      mask_prob_cat: float = 0.15,
                                                      num_loss_weight: float = 1.0,
                                                      cat_loss_weight: float = 1.0) -> List[str]:
        """Export embeddings after FT self-supervised masked reconstruction pretraining."""
        ft_trainer = self._require_trainer("ft")
        if not hasattr(ft_trainer, "pretrain_unsupervised_as_feature"):
            raise RuntimeError(
                "FT trainer does not support unsupervised pretraining.")
        ft_trainer.pretrain_unsupervised_as_feature(
            pred_prefix=pred_prefix,
            params=params,
            mask_prob_num=mask_prob_num,
            mask_prob_cat=mask_prob_cat,
            num_loss_weight=num_loss_weight,
            cat_loss_weight=cat_loss_weight
        )
        cols = self._pred_vector_columns(pred_prefix)
        if not cols:
            raise RuntimeError(
                f"No embedding columns were generated for prefix '{pred_prefix}'.")
        self.add_numeric_features_from_columns(cols)
        return cols

    # GLM Bayesian optimization wrapper.
    def bayesopt_glm(self, max_evals=50):
        self.optimize_model('glm', max_evals)

    # XGBoost Bayesian optimization wrapper.
    def bayesopt_xgb(self, max_evals=100):
        self.optimize_model('xgb', max_evals)

    # ResNet Bayesian optimization wrapper.
    def bayesopt_resnet(self, max_evals=100):
        self.optimize_model('resn', max_evals)

    # GNN Bayesian optimization wrapper.
    def bayesopt_gnn(self, max_evals=50):
        self.optimize_model('gnn', max_evals)

    # FT-Transformer Bayesian optimization wrapper.
    def bayesopt_ft(self, max_evals=50):
        self.optimize_model('ft', max_evals)

    def save_model(self, model_name=None):
        keys = [model_name] if model_name else self.trainers.keys()
        for key in keys:
            if key in self.trainers:
                self.trainers[key].save()
            else:
                if model_name:  # Only warn when the user specifies a model name.
                    _log(f"[save_model] Warning: Unknown model key {key}")

    def load_model(self, model_name=None):
        keys = [model_name] if model_name else self.trainers.keys()
        for key in keys:
            if key in self.trainers:
                self.trainers[key].load()
                # Sync context fields.
                trainer = self.trainers[key]
                if trainer.model is not None:
                    setattr(self, f"{key}_best", trainer.model)
                    # For legacy compatibility, also update xxx_load.
                    # Old versions only tracked xgb_load/resn_load/ft_load (not glm_load/gnn_load).
                    if key in ['xgb', 'resn', 'ft', 'gnn']:
                        setattr(self, f"{key}_load", trainer.model)
            else:
                if model_name:
                    _log(f"[load_model] Warning: Unknown model key {key}")


def _bind_legacy_trainer_aliases() -> None:
    """Bind legacy context attributes to trainer-owned state."""

    alias_specs: List[Tuple[str, str, str]] = [
        ("xgb_best", "xgb", "model"),
        ("resn_best", "resn", "model"),
        ("gnn_best", "gnn", "model"),
        ("glm_best", "glm", "model"),
        ("ft_best", "ft", "model"),
        ("best_xgb_params", "xgb", "best_params"),
        ("best_resn_params", "resn", "best_params"),
        ("best_gnn_params", "gnn", "best_params"),
        ("best_glm_params", "glm", "best_params"),
        ("best_ft_params", "ft", "best_params"),
        ("best_xgb_trial", "xgb", "best_trial"),
        ("best_resn_trial", "resn", "best_trial"),
        ("best_gnn_trial", "gnn", "best_trial"),
        ("best_glm_trial", "glm", "best_trial"),
        ("best_ft_trial", "ft", "best_trial"),
        ("xgb_load", "xgb", "model"),
        ("resn_load", "resn", "model"),
        ("gnn_load", "gnn", "model"),
        ("ft_load", "ft", "model"),
    ]

    def _make_alias(model_key: str, trainer_attr: str):
        def _get(self):
            trainer = getattr(self, "trainers", {}).get(model_key)
            if trainer is None:
                return None
            return getattr(trainer, trainer_attr, None)

        def _set(self, value):
            trainer = getattr(self, "trainers", {}).get(model_key)
            if trainer is None:
                return
            setattr(trainer, trainer_attr, value)

        return property(_get, _set)

    for alias_name, model_key, trainer_attr in alias_specs:
        setattr(BayesOptModel, alias_name, _make_alias(model_key, trainer_attr))


_bind_legacy_trainer_aliases()
