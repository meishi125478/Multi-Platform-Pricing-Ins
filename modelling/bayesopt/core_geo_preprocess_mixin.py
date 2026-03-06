from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ins_pricing.modelling.bayesopt.models import GraphNeuralNetSklearn
from ins_pricing.utils import EPS, get_logger, log_print
from ins_pricing.utils.losses import resolve_tweedie_power
from ins_pricing.utils.safe_pickle import restricted_pickle_load

_logger = get_logger("ins_pricing.modelling.bayesopt.core")


def _log(*args, **kwargs) -> None:
    log_print(_logger, *args, **kwargs)


class BayesOptGeoPreprocessMixin:
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
            payload = restricted_pickle_load(fh)
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

    def default_tweedie_power(self, obj: Optional[str] = None) -> Optional[float]:
        if self.task_type == "classification":
            return None
        loss_name = getattr(self, "loss_name", None)
        if loss_name:
            resolved = resolve_tweedie_power(str(loss_name), default=1.5)
            if resolved is not None:
                return resolved
        objective = obj or getattr(self, "obj", None)
        if objective == "count:poisson":
            return 1.0
        if objective == "reg:gamma":
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
        proc_train: Dict[str, np.ndarray] = {}
        proc_test: Dict[str, np.ndarray] = {}
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
                    s_test.astype(str), categories=cats.categories
                )
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
            index=self.train_data.index,
        )
        test_geo = pd.DataFrame(
            scaler.transform(test_geo_raw),
            columns=available,
            index=self.test_data.index,
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
                self.train_data[self.weight_nme],
            )
            train_embed = geo_gnn.encode(train_geo)
            test_embed = geo_gnn.encode(test_geo)
            cols = [f"geo_token_{i}" for i in range(train_embed.shape[1])]
            train_tokens = pd.DataFrame(
                train_embed, index=self.train_data.index, columns=cols
            )
            test_tokens = pd.DataFrame(
                test_embed, index=self.test_data.index, columns=cols
            )
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
            self.weight_nme
        ].sum()
        city_sumyw = yw_all.groupby(
            [self.train_data[prov_col], self.train_data[city_col]]
        ).sum()
        city_df = pd.DataFrame({
            "sum_w": city_sumw,
            "sum_yw": city_sumyw,
        })
        city_df["prior"] = city_df.index.get_level_values(0).map(
            prov_mean
        ).fillna(global_mean)
        city_df["effect"] = (
            city_df["sum_yw"] + alpha * city_df["prior"]
        ) / (city_df["sum_w"] + alpha).clip(lower=EPS)
        city_effect = city_df["effect"]

        def lookup_effect(df: pd.DataFrame) -> pd.Series:
            idx = pd.MultiIndex.from_frame(df[[prov_col, city_col]])
            effects = city_effect.reindex(idx).to_numpy(dtype=np.float64)
            prov_fallback = df[prov_col].map(
                prov_mean
            ).fillna(global_mean).to_numpy(dtype=np.float64)
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
            re_train.values.reshape(-1, 1)
        ).astype(np.float32).reshape(-1)
        re_test_s = scaler.transform(
            re_test.values.reshape(-1, 1)
        ).astype(np.float32).reshape(-1)
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


__all__ = ["BayesOptGeoPreprocessMixin"]
