from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ins_pricing.modelling.bayesopt.config_preprocess_helpers import (
    build_sparse_onehot_encoder as _build_sparse_onehot_encoder,
    normalize_required_columns as _normalize_required_columns,
)
from ins_pricing.modelling.bayesopt.config_runtime import PreprocessArtifacts
from ins_pricing.modelling.bayesopt.config_schema import BayesOptConfig
from ins_pricing.exceptions import DataValidationError
from ins_pricing.utils import get_logger, log_print

_logger = get_logger("ins_pricing.modelling.bayesopt.dataset_preprocessor")


def _log(*args, **kwargs) -> None:
    log_print(_logger, *args, **kwargs)
class DatasetPreprocessor:
    # Prepare shared train/test views for trainers.

    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                 config: BayesOptConfig) -> None:
        self.config = config
        # Copy inputs to avoid mutating caller-provided DataFrames.
        self.train_data = train_df.copy()
        self.test_data = test_df.copy()
        self.num_features: List[str] = []
        self.train_oht_data: Optional[pd.DataFrame] = None
        self.test_oht_data: Optional[pd.DataFrame] = None
        self.train_oht_scl_data: Optional[pd.DataFrame] = None
        self.test_oht_scl_data: Optional[pd.DataFrame] = None
        self.var_nmes: List[str] = []
        self.cat_categories_for_shap: Dict[str, List[Any]] = {}
        self.numeric_scalers: Dict[str, Dict[str, float]] = {}
        self.ohe_feature_names: List[str] = []
        self.train_cat_oht_csr = None
        self.test_cat_oht_csr = None
        self.oht_sparse_csr: bool = bool(getattr(config, "oht_sparse_csr", True))

    def run(self) -> "DatasetPreprocessor":
        """Run preprocessing: categorical encoding, target clipping, numeric scaling."""
        cfg = self.config
        _normalize_required_columns(
            self.train_data,
            [cfg.resp_nme, cfg.weight_nme, cfg.binary_resp_nme],
            df_label="Train data",
        )
        _normalize_required_columns(
            self.test_data,
            [cfg.resp_nme, cfg.weight_nme, cfg.binary_resp_nme],
            df_label="Test data",
        )
        missing_train = [
            col for col in (cfg.resp_nme, cfg.weight_nme)
            if col not in self.train_data.columns
        ]
        if missing_train:
            raise KeyError(
                f"Train data missing required columns: {missing_train}. "
                f"Available columns (first 50): {list(self.train_data.columns)[:50]}"
            )
        if cfg.binary_resp_nme and cfg.binary_resp_nme not in self.train_data.columns:
            raise DataValidationError(
                f"Train data missing binary response column: {cfg.binary_resp_nme}. "
                f"Available columns (first 50): {list(self.train_data.columns)[:50]}"
            )

        test_has_resp = cfg.resp_nme in self.test_data.columns
        test_has_weight = cfg.weight_nme in self.test_data.columns
        test_has_binary = bool(
            cfg.binary_resp_nme and cfg.binary_resp_nme in self.test_data.columns
        )
        if not test_has_weight:
            self.test_data[cfg.weight_nme] = 1.0
        if not test_has_resp:
            self.test_data[cfg.resp_nme] = np.nan
        if cfg.binary_resp_nme and cfg.binary_resp_nme not in self.test_data.columns:
            self.test_data[cfg.binary_resp_nme] = np.nan

        # Precompute weighted actuals for plots and validation checks.
        # Direct assignment is more efficient than .loc[:, col]
        self.train_data['w_act'] = self.train_data[cfg.resp_nme] * \
            self.train_data[cfg.weight_nme]
        if test_has_resp:
            self.test_data['w_act'] = self.test_data[cfg.resp_nme] * \
                self.test_data[cfg.weight_nme]
        if cfg.binary_resp_nme:
            self.train_data['w_binary_act'] = self.train_data[cfg.binary_resp_nme] * \
                self.train_data[cfg.weight_nme]
            if test_has_binary:
                self.test_data['w_binary_act'] = self.test_data[cfg.binary_resp_nme] * \
                    self.test_data[cfg.weight_nme]
        # High-quantile clipping absorbs outliers; removing it lets extremes dominate loss.
        q99 = self.train_data[cfg.resp_nme].quantile(0.999)
        self.train_data[cfg.resp_nme] = self.train_data[cfg.resp_nme].clip(
            upper=q99)
        cate_list = list(cfg.cate_list or [])
        if cate_list:
            for cate in cate_list:
                self.train_data[cate] = self.train_data[cate].astype(
                    'category')
                self.test_data[cate] = self.test_data[cate].astype('category')
                cats = self.train_data[cate].cat.categories
                self.cat_categories_for_shap[cate] = list(cats)
        self.num_features = [
            nme for nme in cfg.factor_nmes if nme not in cate_list]

        build_oht = bool(getattr(cfg, "build_oht", True))
        if not build_oht:
            _log("[Preprocess] build_oht=False; skip one-hot features.", flush=True)
            self.train_oht_data = None
            self.test_oht_data = None
            self.train_oht_scl_data = None
            self.test_oht_scl_data = None
            self.var_nmes = list(cfg.factor_nmes)
            return self

        keep_unscaled_oht = bool(getattr(cfg, "keep_unscaled_oht", True))
        keep_unscaled_env = os.environ.get("BAYESOPT_KEEP_UNSCALED_OHT")
        if keep_unscaled_env is not None:
            keep_unscaled_oht = str(keep_unscaled_env).strip().lower() in {
                "1", "true", "yes", "y", "on"
            }
        profile = str(
            getattr(cfg, "resource_profile", os.environ.get("BAYESOPT_RESOURCE_PROFILE", "auto"))
        ).strip().lower()
        world_size = 1
        try:
            world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
        except (TypeError, ValueError):
            world_size = 1
        if (
            keep_unscaled_oht
            and profile == "memory_saving"
            and world_size > 1
            and len(self.train_data) >= 500_000
        ):
            keep_unscaled_oht = False
            _log(
                "[Preprocess] Auto-set keep_unscaled_oht=False for DDP + memory_saving "
                "on large dataset to reduce host RAM pressure.",
                flush=True,
            )

        oht_cols = cfg.factor_nmes + [cfg.weight_nme] + [cfg.resp_nme]
        use_sparse_csr = bool(getattr(cfg, "oht_sparse_csr", True) and cate_list)

        if use_sparse_csr:
            dense_cols = [
                col for col in (self.num_features + [cfg.weight_nme, cfg.resp_nme])
                if col in oht_cols
            ]
            train_dense = self.train_data[dense_cols].copy()
            test_dense = self.test_data[dense_cols].copy()
            for col in dense_cols:
                train_dense[col] = pd.to_numeric(
                    train_dense[col], errors="coerce"
                ).astype(np.float32, copy=False)
                test_dense[col] = pd.to_numeric(
                    test_dense[col], errors="coerce"
                ).astype(np.float32, copy=False)

            train_cat = self.train_data[cate_list].copy()
            test_cat = self.test_data[cate_list].copy()
            for col in cate_list:
                train_cat[col] = train_cat[col].astype("object").where(
                    train_cat[col].notna(), "<NA>"
                )
                test_cat[col] = test_cat[col].astype("object").where(
                    test_cat[col].notna(), "<NA>"
                )

            try:
                encoder = _build_sparse_onehot_encoder(drop_first=True)
                train_cat_sparse = encoder.fit_transform(train_cat)
                test_cat_sparse = encoder.transform(test_cat)
                cat_feature_names = [
                    str(name) for name in encoder.get_feature_names_out(cate_list)
                ]
                self.train_cat_oht_csr = train_cat_sparse.tocsr()
                self.test_cat_oht_csr = test_cat_sparse.tocsr()
                self.ohe_feature_names = list(cat_feature_names)
                self.oht_sparse_csr = True
                _log(
                    f"[Preprocess] one-hot CSR enabled: {len(cat_feature_names)} columns.",
                    flush=True,
                )
            except Exception as exc:
                _log(
                    f"[Preprocess] CSR one-hot failed, fallback to dense get_dummies: {exc}",
                    flush=True,
                )
                use_sparse_csr = False

            if use_sparse_csr:
                train_cat_df = pd.DataFrame.sparse.from_spmatrix(
                    self.train_cat_oht_csr,
                    index=train_dense.index,
                    columns=self.ohe_feature_names,
                )
                test_cat_df = pd.DataFrame.sparse.from_spmatrix(
                    self.test_cat_oht_csr,
                    index=test_dense.index,
                    columns=self.ohe_feature_names,
                )

                if keep_unscaled_oht:
                    self.train_oht_data = pd.concat(
                        [train_dense, train_cat_df], axis=1
                    )
                    self.test_oht_data = pd.concat(
                        [test_dense, test_cat_df], axis=1
                    )
                    train_dense_scaled = train_dense.copy()
                    test_dense_scaled = test_dense.copy()
                else:
                    self.train_oht_data = None
                    self.test_oht_data = None
                    train_dense_scaled = train_dense
                    test_dense_scaled = test_dense

                for num_chr in self.num_features:
                    scaler = StandardScaler()
                    train_dense_scaled[num_chr] = scaler.fit_transform(
                        train_dense_scaled[num_chr].values.reshape(-1, 1)
                    ).astype(np.float32, copy=False).reshape(-1)
                    test_dense_scaled[num_chr] = scaler.transform(
                        test_dense_scaled[num_chr].values.reshape(-1, 1)
                    ).astype(np.float32, copy=False).reshape(-1)
                    scale_val = float(getattr(scaler, "scale_", [1.0])[0])
                    if scale_val == 0.0:
                        scale_val = 1.0
                    self.numeric_scalers[num_chr] = {
                        "mean": float(getattr(scaler, "mean_", [0.0])[0]),
                        "scale": scale_val,
                    }

                self.train_oht_scl_data = pd.concat(
                    [train_dense_scaled, train_cat_df], axis=1
                )
                self.test_oht_scl_data = pd.concat(
                    [test_dense_scaled, test_cat_df], axis=1
                )
                self.var_nmes = list(self.num_features) + list(self.ohe_feature_names)
                return self

        # Fallback path: dense one-hot via pandas get_dummies.
        self.train_cat_oht_csr = None
        self.test_cat_oht_csr = None
        self.oht_sparse_csr = False

        train_oht = self.train_data[oht_cols].copy()
        test_oht = self.test_data[oht_cols].copy()
        dense_float_cols = [
            col for col in (self.num_features + [cfg.weight_nme, cfg.resp_nme])
            if col in oht_cols
        ]
        for col in dense_float_cols:
            train_oht[col] = pd.to_numeric(
                train_oht[col], errors="coerce"
            ).astype(np.float32, copy=False)
            test_oht[col] = pd.to_numeric(
                test_oht[col], errors="coerce"
            ).astype(np.float32, copy=False)

        train_oht = pd.get_dummies(
            train_oht,
            columns=cate_list,
            drop_first=True,
            dtype=np.int8
        )
        test_oht = pd.get_dummies(
            test_oht,
            columns=cate_list,
            drop_first=True,
            dtype=np.int8
        )
        test_oht = test_oht.reindex(
            columns=train_oht.columns, fill_value=0, copy=False)

        if keep_unscaled_oht:
            self.train_oht_data = train_oht
            self.test_oht_data = test_oht
        else:
            self.train_oht_data = None
            self.test_oht_data = None

        if self.num_features:
            if keep_unscaled_oht:
                train_oht_scaled = train_oht.copy()
                test_oht_scaled = test_oht.copy()
            else:
                train_oht_scaled = train_oht
                test_oht_scaled = test_oht
        else:
            train_oht_scaled = train_oht
            test_oht_scaled = test_oht
        for num_chr in self.num_features:
            scaler = StandardScaler()
            train_oht_scaled[num_chr] = scaler.fit_transform(
                train_oht_scaled[num_chr].values.reshape(-1, 1)
            ).astype(np.float32, copy=False).reshape(-1)
            test_oht_scaled[num_chr] = scaler.transform(
                test_oht_scaled[num_chr].values.reshape(-1, 1)
            ).astype(np.float32, copy=False).reshape(-1)
            scale_val = float(getattr(scaler, "scale_", [1.0])[0])
            if scale_val == 0.0:
                scale_val = 1.0
            self.numeric_scalers[num_chr] = {
                "mean": float(getattr(scaler, "mean_", [0.0])[0]),
                "scale": scale_val,
            }
        if not test_oht_scaled.columns.equals(train_oht_scaled.columns):
            test_oht_scaled = test_oht_scaled.reindex(
                columns=train_oht_scaled.columns, fill_value=0, copy=False)
        self.train_oht_scl_data = train_oht_scaled
        self.test_oht_scl_data = test_oht_scaled
        excluded = {cfg.weight_nme, cfg.resp_nme}
        self.var_nmes = [
            col for col in train_oht_scaled.columns if col not in excluded
        ]
        self.ohe_feature_names = [
            col for col in self.var_nmes if col not in set(self.num_features)
        ]
        return self

    def export_artifacts(self) -> PreprocessArtifacts:
        dummy_columns: List[str] = []
        if self.train_oht_data is not None:
            dummy_columns = list(self.train_oht_data.columns)
        return PreprocessArtifacts(
            factor_nmes=list(self.config.factor_nmes),
            cate_list=list(self.config.cate_list or []),
            num_features=list(self.num_features),
            var_nmes=list(self.var_nmes),
            cat_categories=dict(self.cat_categories_for_shap),
            ohe_feature_names=list(self.ohe_feature_names),
            dummy_columns=dummy_columns,
            numeric_scalers=dict(self.numeric_scalers),
            weight_nme=str(self.config.weight_nme),
            resp_nme=str(self.config.resp_nme),
            binary_resp_nme=self.config.binary_resp_nme,
            drop_first=True,
            oht_sparse_csr=bool(self.oht_sparse_csr),
        )

    def save_artifacts(self, path: str | Path) -> str:
        payload = self.export_artifacts()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(payload), ensure_ascii=True, indent=2), encoding="utf-8")
        return str(target)

