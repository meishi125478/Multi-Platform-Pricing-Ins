from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, ShuffleSplit, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .config_preprocess import BayesOptConfig, DatasetPreprocessor, OutputManager, VersionManager
from .model_explain_mixin import BayesOptExplainMixin
from .model_plotting_mixin import BayesOptPlottingMixin
from .models import GraphNeuralNetSklearn
from .trainers import FTTrainer, GLMTrainer, GNNTrainer, ResNetTrainer, XGBTrainer
from .utils import EPS, infer_factor_and_cate_list, set_global_seed


class _CVSplitter:
    """Wrapper to carry optional groups or time order for CV splits."""

    def __init__(
        self,
        splitter,
        *,
        groups: Optional[pd.Series] = None,
        order: Optional[np.ndarray] = None,
    ) -> None:
        self._splitter = splitter
        self._groups = groups
        self._order = order

    def split(self, X, y=None, groups=None):
        if self._order is not None:
            order = np.asarray(self._order)
            X_ord = X.iloc[order] if hasattr(X, "iloc") else X[order]
            for tr_idx, val_idx in self._splitter.split(X_ord, y=y):
                yield order[tr_idx], order[val_idx]
            return
        use_groups = groups if groups is not None else self._groups
        for tr_idx, val_idx in self._splitter.split(X, y=y, groups=use_groups):
            yield tr_idx, val_idx

# BayesOpt orchestration and SHAP utilities
# =============================================================================
class BayesOptModel(BayesOptPlottingMixin, BayesOptExplainMixin):
    def __init__(self, train_data, test_data,
                 model_nme, resp_nme, weight_nme, factor_nmes: Optional[List[str]] = None, task_type='regression',
                 binary_resp_nme=None,
                 cate_list=None, prop_test=0.25, rand_seed=None,
                 epochs=100, use_gpu=True,
                 use_resn_data_parallel: bool = False, use_ft_data_parallel: bool = False,
                 use_gnn_data_parallel: bool = False,
                 use_resn_ddp: bool = False, use_ft_ddp: bool = False,
                 use_gnn_ddp: bool = False,
                 output_dir: Optional[str] = None,
                 gnn_use_approx_knn: bool = True,
                 gnn_approx_knn_threshold: int = 50000,
                 gnn_graph_cache: Optional[str] = None,
                 gnn_max_gpu_knn_nodes: Optional[int] = 200000,
                 gnn_knn_gpu_mem_ratio: float = 0.9,
                 gnn_knn_gpu_mem_overhead: float = 2.0,
                 ft_role: str = "model",
                 ft_feature_prefix: str = "ft_emb",
                 ft_num_numeric_tokens: Optional[int] = None,
                 infer_categorical_max_unique: int = 50,
                 infer_categorical_max_ratio: float = 0.05,
                 reuse_best_params: bool = False,
                 xgb_max_depth_max: int = 25,
                 xgb_n_estimators_max: int = 500,
                 resn_weight_decay: Optional[float] = None,
                 final_ensemble: bool = False,
                 final_ensemble_k: int = 3,
                 final_refit: bool = True,
                 optuna_storage: Optional[str] = None,
                 optuna_study_prefix: Optional[str] = None,
                 best_params_files: Optional[Dict[str, str]] = None,
                 cv_strategy: Optional[str] = None,
                 cv_splits: Optional[int] = None,
                 cv_group_col: Optional[str] = None,
                 cv_time_col: Optional[str] = None,
                 cv_time_ascending: bool = True,
                 ft_oof_folds: Optional[int] = None,
                 ft_oof_strategy: Optional[str] = None,
                 ft_oof_shuffle: bool = True,
                 save_preprocess: bool = False,
                 preprocess_artifact_path: Optional[str] = None,
                 plot_path_style: Optional[str] = None,
                 bo_sample_limit: Optional[int] = None,
                 cache_predictions: bool = False,
                 prediction_cache_dir: Optional[str] = None,
                 prediction_cache_format: Optional[str] = None,
                 region_province_col: Optional[str] = None,
                 region_city_col: Optional[str] = None,
                 region_effect_alpha: Optional[float] = None,
                 geo_feature_nmes: Optional[List[str]] = None,
                 geo_token_hidden_dim: Optional[int] = None,
                 geo_token_layers: Optional[int] = None,
                 geo_token_dropout: Optional[float] = None,
                 geo_token_k_neighbors: Optional[int] = None,
                 geo_token_learning_rate: Optional[float] = None,
                 geo_token_epochs: Optional[int] = None):
        """Orchestrate BayesOpt training across multiple trainers.

        Args:
            train_data: Training DataFrame.
            test_data: Test DataFrame.
            model_nme: Model name prefix used in outputs.
            resp_nme: Target column name.
            weight_nme: Sample weight column name.
            factor_nmes: Feature column list.
            task_type: "regression" or "classification".
            binary_resp_nme: Optional binary target for lift curves.
            cate_list: Categorical feature list.
            prop_test: Validation split ratio in CV.
            rand_seed: Random seed.
            epochs: NN training epochs.
            use_gpu: Prefer GPU when available.
            use_resn_data_parallel: Enable DataParallel for ResNet.
            use_ft_data_parallel: Enable DataParallel for FTTransformer.
            use_gnn_data_parallel: Enable DataParallel for GNN.
            use_resn_ddp: Enable DDP for ResNet.
            use_ft_ddp: Enable DDP for FTTransformer.
            use_gnn_ddp: Enable DDP for GNN.
            output_dir: Output root for models/results/plots.
            gnn_use_approx_knn: Use approximate kNN when available.
            gnn_approx_knn_threshold: Row threshold to switch to approximate kNN.
            gnn_graph_cache: Optional adjacency cache path.
            gnn_max_gpu_knn_nodes: Force CPU kNN above this node count to avoid OOM.
            gnn_knn_gpu_mem_ratio: Fraction of free GPU memory for kNN.
            gnn_knn_gpu_mem_overhead: Temporary memory multiplier for GPU kNN.
            ft_num_numeric_tokens: Number of numeric tokens for FT (None = auto).
            final_ensemble: Enable k-fold model averaging at the final stage.
            final_ensemble_k: Number of folds for averaging.
            final_refit: Refit on full data using best stopping point.
        """
        inferred_factors, inferred_cats = infer_factor_and_cate_list(
            train_df=train_data,
            test_df=test_data,
            resp_nme=resp_nme,
            weight_nme=weight_nme,
            binary_resp_nme=binary_resp_nme,
            factor_nmes=factor_nmes,
            cate_list=cate_list,
            infer_categorical_max_unique=int(infer_categorical_max_unique),
            infer_categorical_max_ratio=float(infer_categorical_max_ratio),
        )

        cfg = BayesOptConfig(
            model_nme=model_nme,
            task_type=task_type,
            resp_nme=resp_nme,
            weight_nme=weight_nme,
            factor_nmes=list(inferred_factors),
            binary_resp_nme=binary_resp_nme,
            cate_list=list(inferred_cats) if inferred_cats else None,
            prop_test=prop_test,
            rand_seed=rand_seed,
            epochs=epochs,
            use_gpu=use_gpu,
            xgb_max_depth_max=int(xgb_max_depth_max),
            xgb_n_estimators_max=int(xgb_n_estimators_max),
            use_resn_data_parallel=use_resn_data_parallel,
            use_ft_data_parallel=use_ft_data_parallel,
            use_resn_ddp=use_resn_ddp,
            use_gnn_data_parallel=use_gnn_data_parallel,
            use_ft_ddp=use_ft_ddp,
            use_gnn_ddp=use_gnn_ddp,
            gnn_use_approx_knn=gnn_use_approx_knn,
            gnn_approx_knn_threshold=gnn_approx_knn_threshold,
            gnn_graph_cache=gnn_graph_cache,
            gnn_max_gpu_knn_nodes=gnn_max_gpu_knn_nodes,
            gnn_knn_gpu_mem_ratio=gnn_knn_gpu_mem_ratio,
            gnn_knn_gpu_mem_overhead=gnn_knn_gpu_mem_overhead,
            output_dir=output_dir,
            optuna_storage=optuna_storage,
            optuna_study_prefix=optuna_study_prefix,
            best_params_files=best_params_files,
            ft_role=str(ft_role or "model"),
            ft_feature_prefix=str(ft_feature_prefix or "ft_emb"),
            ft_num_numeric_tokens=ft_num_numeric_tokens,
            reuse_best_params=bool(reuse_best_params),
            resn_weight_decay=float(resn_weight_decay)
            if resn_weight_decay is not None
            else 1e-4,
            final_ensemble=bool(final_ensemble),
            final_ensemble_k=int(final_ensemble_k),
            final_refit=bool(final_refit),
            cv_strategy=str(cv_strategy or "random"),
            cv_splits=cv_splits,
            cv_group_col=cv_group_col,
            cv_time_col=cv_time_col,
            cv_time_ascending=bool(cv_time_ascending),
            ft_oof_folds=ft_oof_folds,
            ft_oof_strategy=ft_oof_strategy,
            ft_oof_shuffle=bool(ft_oof_shuffle),
            save_preprocess=bool(save_preprocess),
            preprocess_artifact_path=preprocess_artifact_path,
            plot_path_style=str(plot_path_style or "nested"),
            bo_sample_limit=bo_sample_limit,
            cache_predictions=bool(cache_predictions),
            prediction_cache_dir=prediction_cache_dir,
            prediction_cache_format=str(prediction_cache_format or "parquet"),
            region_province_col=region_province_col,
            region_city_col=region_city_col,
            region_effect_alpha=float(region_effect_alpha)
            if region_effect_alpha is not None
            else 50.0,
            geo_feature_nmes=list(geo_feature_nmes)
            if geo_feature_nmes is not None
            else None,
            geo_token_hidden_dim=int(geo_token_hidden_dim)
            if geo_token_hidden_dim is not None
            else 32,
            geo_token_layers=int(geo_token_layers)
            if geo_token_layers is not None
            else 2,
            geo_token_dropout=float(geo_token_dropout)
            if geo_token_dropout is not None
            else 0.1,
            geo_token_k_neighbors=int(geo_token_k_neighbors)
            if geo_token_k_neighbors is not None
            else 10,
            geo_token_learning_rate=float(geo_token_learning_rate)
            if geo_token_learning_rate is not None
            else 1e-3,
            geo_token_epochs=int(geo_token_epochs)
            if geo_token_epochs is not None
            else 50,
        )
        self.config = cfg
        self.model_nme = cfg.model_nme
        self.task_type = cfg.task_type
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
        self.use_gpu = bool(cfg.use_gpu and torch.cuda.is_available())
        self.output_manager = OutputManager(
            cfg.output_dir or os.getcwd(), self.model_nme)

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
        self.geo_token_cols: List[str] = []
        self.train_geo_tokens: Optional[pd.DataFrame] = None
        self.test_geo_tokens: Optional[pd.DataFrame] = None
        self.geo_gnn_model: Optional[GraphNeuralNetSklearn] = None
        self._add_region_effect()

        self.cv = self._build_cv_splitter()
        if self.task_type == 'classification':
            self.obj = 'binary:logistic'
        else:  # regression task
            if 'f' in self.model_nme:
                self.obj = 'count:poisson'
            elif 's' in self.model_nme:
                self.obj = 'reg:gamma'
            elif 'bc' in self.model_nme:
                self.obj = 'reg:tweedie'
            else:
                self.obj = 'reg:tweedie'
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
        self.xgb_best = None
        self.resn_best = None
        self.gnn_best = None
        self.glm_best = None
        self.ft_best = None
        self.best_xgb_params = None
        self.best_resn_params = None
        self.best_gnn_params = None
        self.best_ft_params = None
        self.best_xgb_trial = None
        self.best_resn_trial = None
        self.best_gnn_trial = None
        self.best_ft_trial = None
        self.best_glm_params = None
        self.best_glm_trial = None
        self.xgb_load = None
        self.resn_load = None
        self.gnn_load = None
        self.ft_load = None
        self.version_manager = VersionManager(self.output_manager)

    def _build_cv_splitter(self) -> _CVSplitter:
        strategy = str(getattr(self.config, "cv_strategy", "random") or "random").strip().lower()
        val_ratio = float(self.prop_test) if self.prop_test is not None else 0.25
        if not (0.0 < val_ratio < 1.0):
            val_ratio = 0.25
        cv_splits = getattr(self.config, "cv_splits", None)
        if cv_splits is None:
            cv_splits = max(2, int(round(1 / val_ratio)))
        cv_splits = max(2, int(cv_splits))

        if strategy in {"group", "grouped"}:
            group_col = getattr(self.config, "cv_group_col", None)
            if not group_col:
                raise ValueError("cv_group_col is required for group cv_strategy.")
            if group_col not in self.train_data.columns:
                raise KeyError(f"cv_group_col '{group_col}' not in train_data.")
            groups = self.train_data[group_col]
            splitter = GroupKFold(n_splits=cv_splits)
            return _CVSplitter(splitter, groups=groups)

        if strategy in {"time", "timeseries", "temporal"}:
            time_col = getattr(self.config, "cv_time_col", None)
            if not time_col:
                raise ValueError("cv_time_col is required for time cv_strategy.")
            if time_col not in self.train_data.columns:
                raise KeyError(f"cv_time_col '{time_col}' not in train_data.")
            ascending = bool(getattr(self.config, "cv_time_ascending", True))
            order_index = self.train_data[time_col].sort_values(ascending=ascending).index
            order = self.train_data.index.get_indexer(order_index)
            splitter = TimeSeriesSplit(n_splits=cv_splits)
            return _CVSplitter(splitter, order=order)

        splitter = ShuffleSplit(
            n_splits=cv_splits,
            test_size=val_ratio,
            random_state=self.rand_seed,
        )
        return _CVSplitter(splitter)

    def default_tweedie_power(self, obj: Optional[str] = None) -> Optional[float]:
        if self.task_type == 'classification':
            return None
        objective = obj or getattr(self, "obj", None)
        if objective == 'count:poisson':
            return 1.0
        if objective == 'reg:gamma':
            return 2.0
        return 1.5

    def _build_geo_tokens(self, params_override: Optional[Dict[str, Any]] = None):
        """Internal builder; allows trial overrides and returns None on failure."""
        geo_cols = list(self.config.geo_feature_nmes or [])
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
                hidden_dim=cfg.get("geo_token_hidden_dim",
                                   self.config.geo_token_hidden_dim),
                num_layers=cfg.get("geo_token_layers",
                                   self.config.geo_token_layers),
                k_neighbors=cfg.get("geo_token_k_neighbors",
                                    self.config.geo_token_k_neighbors),
                dropout=cfg.get("geo_token_dropout",
                                self.config.geo_token_dropout),
                learning_rate=cfg.get(
                    "geo_token_learning_rate", self.config.geo_token_learning_rate),
                epochs=int(cfg.get("geo_token_epochs",
                           self.config.geo_token_epochs)),
                patience=5,
                task_type=self.task_type,
                tweedie_power=tw_power,
                use_data_parallel=False,
                use_ddp=False,
                use_approx_knn=self.config.gnn_use_approx_knn,
                approx_knn_threshold=self.config.gnn_approx_knn_threshold,
                graph_cache_path=None,
                max_gpu_knn_nodes=self.config.gnn_max_gpu_knn_nodes,
                knn_gpu_mem_ratio=self.config.gnn_knn_gpu_mem_ratio,
                knn_gpu_mem_overhead=self.config.gnn_knn_gpu_mem_overhead
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
            print(f"[GeoToken] Generation failed: {exc}")
            return None

    def _prepare_geo_tokens(self) -> None:
        """Build and persist geo tokens with default config values."""
        gnn_trainer = self.trainers.get("gnn")
        if gnn_trainer is not None and hasattr(gnn_trainer, "prepare_geo_tokens"):
            try:
                gnn_trainer.prepare_geo_tokens(force=False)  # type: ignore[attr-defined]
                return
            except Exception as exc:
                print(f"[GeoToken] GNNTrainer generation failed: {exc}")

        result = self._build_geo_tokens()
        if result is None:
            return
        train_tokens, test_tokens, cols, geo_gnn = result
        self.train_geo_tokens = train_tokens
        self.test_geo_tokens = test_tokens
        self.geo_token_cols = cols
        self.geo_gnn_model = geo_gnn
        print(f"[GeoToken] Generated {len(cols)}-dim geo tokens; injecting into FT.")

    def _add_region_effect(self) -> None:
        """Partial pooling over province/city to create a smoothed region_effect feature."""
        prov_col = self.config.region_province_col
        city_col = self.config.region_city_col
        if not prov_col or not city_col:
            return
        for col in [prov_col, city_col]:
            if col not in self.train_data.columns:
                print(f"[RegionEffect] Missing column {col}; skipped.")
                return

        def safe_mean(df: pd.DataFrame) -> float:
            w = df[self.weight_nme]
            y = df[self.resp_nme]
            denom = max(float(w.sum()), EPS)
            return float((y * w).sum() / denom)

        global_mean = safe_mean(self.train_data)
        alpha = max(float(self.config.region_effect_alpha), 0.0)

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
            print(
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
                print(
                    f"[Optuna][{trainer.label}] Reusing best_params from versions snapshot.")
                return

            params_path = self.output_manager.result_path(
                f'{self.model_nme}_bestparams_{trainer.label.lower()}.csv'
            )
            if os.path.exists(params_path):
                try:
                    trainer.best_params = IOUtils.load_params_file(params_path)
                    trainer.best_trial = None
                    print(
                        f"[Optuna][{trainer.label}] Reusing best_params from {params_path}.")
                except ValueError:
                    # Legacy compatibility: ignore empty files and continue tuning.
                    pass

    # Generic optimization entry point.
    def optimize_model(self, model_key: str, max_evals: int = 100):
        if model_key not in self.trainers:
            print(f"Warning: Unknown model key: {model_key}")
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
            if k > 1:
                if model_key == "ft" and str(self.config.ft_role) != "model":
                    pass
                elif hasattr(trainer, "ensemble_predict"):
                    trainer.ensemble_predict(k)
                else:
                    print(
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
                    print(f"[save_model] Warning: Unknown model key {key}")

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
                    print(f"[load_model] Warning: Unknown model key {key}")
