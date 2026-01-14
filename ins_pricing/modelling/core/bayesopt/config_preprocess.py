from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .utils import IOUtils
from ....exceptions import ConfigurationError, DataValidationError

# NOTE: Some CSV exports may contain invisible BOM characters or leading/trailing
# spaces in column names. Pandas requires exact matches, so we normalize a few
# "required" column names (response/weight/binary response) before validating.


def _clean_column_name(name: Any) -> Any:
    if not isinstance(name, str):
        return name
    return name.replace("\ufeff", "").strip()


def _normalize_required_columns(
    df: pd.DataFrame, required: List[Optional[str]], *, df_label: str
) -> None:
    required_names = [r for r in required if isinstance(r, str) and r.strip()]
    if not required_names:
        return

    mapping: Dict[Any, Any] = {}
    existing = set(df.columns)
    for col in df.columns:
        cleaned = _clean_column_name(col)
        if cleaned != col and cleaned not in existing:
            mapping[col] = cleaned
    if mapping:
        df.rename(columns=mapping, inplace=True)

    existing = set(df.columns)
    for req in required_names:
        if req in existing:
            continue
        candidates = [
            col
            for col in df.columns
            if isinstance(col, str) and _clean_column_name(col).lower() == req.lower()
        ]
        if len(candidates) == 1 and req not in existing:
            df.rename(columns={candidates[0]: req}, inplace=True)
            existing = set(df.columns)
        elif len(candidates) > 1:
            raise KeyError(
                f"{df_label} has multiple columns matching required {req!r} "
                f"(case/space-insensitive): {candidates}"
            )


# ===== Core components and training wrappers =================================

# =============================================================================
# Config, preprocessing, and trainer base types
# =============================================================================
@dataclass
class BayesOptConfig:
    """Configuration for Bayesian optimization-based model training.

    This dataclass holds all configuration parameters for the BayesOpt training
    pipeline, including model settings, distributed training options, and
    cross-validation strategies.

    Attributes:
        model_nme: Unique identifier for the model
        resp_nme: Column name for the response/target variable
        weight_nme: Column name for sample weights
        factor_nmes: List of feature column names
        task_type: Either 'regression' or 'classification'
        binary_resp_nme: Column name for binary response (optional)
        cate_list: List of categorical feature column names
        prop_test: Proportion of data for validation (0.0-1.0)
        rand_seed: Random seed for reproducibility
        epochs: Number of training epochs
        use_gpu: Whether to use GPU acceleration
        xgb_max_depth_max: Maximum tree depth for XGBoost tuning
        xgb_n_estimators_max: Maximum estimators for XGBoost tuning
        use_resn_data_parallel: Use DataParallel for ResNet
        use_ft_data_parallel: Use DataParallel for FT-Transformer
        use_resn_ddp: Use DDP for ResNet
        use_ft_ddp: Use DDP for FT-Transformer
        use_gnn_data_parallel: Use DataParallel for GNN
        use_gnn_ddp: Use DDP for GNN
        ft_role: FT-Transformer role ('model', 'embedding', 'unsupervised_embedding')
        cv_strategy: CV strategy ('random', 'group', 'time', 'stratified')

    Example:
        >>> config = BayesOptConfig(
        ...     model_nme="pricing_model",
        ...     resp_nme="claim_amount",
        ...     weight_nme="exposure",
        ...     factor_nmes=["age", "gender", "region"],
        ...     task_type="regression",
        ...     use_ft_ddp=True,
        ... )
    """

    # Required fields
    model_nme: str
    resp_nme: str
    weight_nme: str
    factor_nmes: List[str]

    # Task configuration
    task_type: str = 'regression'
    binary_resp_nme: Optional[str] = None
    cate_list: Optional[List[str]] = None

    # Training configuration
    prop_test: float = 0.25
    rand_seed: Optional[int] = None
    epochs: int = 100
    use_gpu: bool = True

    # XGBoost settings
    xgb_max_depth_max: int = 25
    xgb_n_estimators_max: int = 500

    # Distributed training settings
    use_resn_data_parallel: bool = False
    use_ft_data_parallel: bool = False
    use_resn_ddp: bool = False
    use_ft_ddp: bool = False
    use_gnn_data_parallel: bool = False
    use_gnn_ddp: bool = False

    # GNN settings
    gnn_use_approx_knn: bool = True
    gnn_approx_knn_threshold: int = 50000
    gnn_graph_cache: Optional[str] = None
    gnn_max_gpu_knn_nodes: Optional[int] = 200000
    gnn_knn_gpu_mem_ratio: float = 0.9
    gnn_knn_gpu_mem_overhead: float = 2.0

    # Region/Geo settings
    region_province_col: Optional[str] = None
    region_city_col: Optional[str] = None
    region_effect_alpha: float = 50.0
    geo_feature_nmes: Optional[List[str]] = None
    geo_token_hidden_dim: int = 32
    geo_token_layers: int = 2
    geo_token_dropout: float = 0.1
    geo_token_k_neighbors: int = 10
    geo_token_learning_rate: float = 1e-3
    geo_token_epochs: int = 50

    # Output settings
    output_dir: Optional[str] = None
    optuna_storage: Optional[str] = None
    optuna_study_prefix: Optional[str] = None
    best_params_files: Optional[Dict[str, str]] = None

    # FT-Transformer settings
    ft_role: str = "model"
    ft_feature_prefix: str = "ft_emb"
    ft_num_numeric_tokens: Optional[int] = None

    # Training workflow settings
    reuse_best_params: bool = False
    resn_weight_decay: float = 1e-4
    final_ensemble: bool = False
    final_ensemble_k: int = 3
    final_refit: bool = True

    # Cross-validation settings
    cv_strategy: str = "random"
    cv_splits: Optional[int] = None
    cv_group_col: Optional[str] = None
    cv_time_col: Optional[str] = None
    cv_time_ascending: bool = True
    ft_oof_folds: Optional[int] = None
    ft_oof_strategy: Optional[str] = None
    ft_oof_shuffle: bool = True

    # Caching and output settings
    save_preprocess: bool = False
    preprocess_artifact_path: Optional[str] = None
    plot_path_style: str = "nested"
    bo_sample_limit: Optional[int] = None
    cache_predictions: bool = False
    prediction_cache_dir: Optional[str] = None
    prediction_cache_format: str = "parquet"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate configuration values and raise errors for invalid combinations."""
        errors: List[str] = []

        # Validate task_type
        valid_task_types = {"regression", "classification"}
        if self.task_type not in valid_task_types:
            errors.append(
                f"task_type must be one of {valid_task_types}, got '{self.task_type}'"
            )

        # Validate prop_test
        if not 0.0 < self.prop_test < 1.0:
            errors.append(
                f"prop_test must be between 0 and 1, got {self.prop_test}"
            )

        # Validate epochs
        if self.epochs < 1:
            errors.append(f"epochs must be >= 1, got {self.epochs}")

        # Validate XGBoost settings
        if self.xgb_max_depth_max < 1:
            errors.append(
                f"xgb_max_depth_max must be >= 1, got {self.xgb_max_depth_max}"
            )
        if self.xgb_n_estimators_max < 1:
            errors.append(
                f"xgb_n_estimators_max must be >= 1, got {self.xgb_n_estimators_max}"
            )

        # Validate distributed training: can't use both DataParallel and DDP
        if self.use_resn_data_parallel and self.use_resn_ddp:
            errors.append(
                "Cannot use both use_resn_data_parallel and use_resn_ddp"
            )
        if self.use_ft_data_parallel and self.use_ft_ddp:
            errors.append(
                "Cannot use both use_ft_data_parallel and use_ft_ddp"
            )
        if self.use_gnn_data_parallel and self.use_gnn_ddp:
            errors.append(
                "Cannot use both use_gnn_data_parallel and use_gnn_ddp"
            )

        # Validate ft_role
        valid_ft_roles = {"model", "embedding", "unsupervised_embedding"}
        if self.ft_role not in valid_ft_roles:
            errors.append(
                f"ft_role must be one of {valid_ft_roles}, got '{self.ft_role}'"
            )

        # Validate cv_strategy
        valid_cv_strategies = {"random", "group", "grouped", "time", "timeseries", "temporal", "stratified"}
        if self.cv_strategy not in valid_cv_strategies:
            errors.append(
                f"cv_strategy must be one of {valid_cv_strategies}, got '{self.cv_strategy}'"
            )

        # Validate group CV requires group_col
        if self.cv_strategy in {"group", "grouped"} and not self.cv_group_col:
            errors.append(
                f"cv_group_col is required when cv_strategy is '{self.cv_strategy}'"
            )

        # Validate time CV requires time_col
        if self.cv_strategy in {"time", "timeseries", "temporal"} and not self.cv_time_col:
            errors.append(
                f"cv_time_col is required when cv_strategy is '{self.cv_strategy}'"
            )

        # Validate prediction_cache_format
        valid_cache_formats = {"parquet", "csv"}
        if self.prediction_cache_format not in valid_cache_formats:
            errors.append(
                f"prediction_cache_format must be one of {valid_cache_formats}, "
                f"got '{self.prediction_cache_format}'"
            )

        # Validate GNN memory settings
        if self.gnn_knn_gpu_mem_ratio <= 0 or self.gnn_knn_gpu_mem_ratio > 1.0:
            errors.append(
                f"gnn_knn_gpu_mem_ratio must be in (0, 1], got {self.gnn_knn_gpu_mem_ratio}"
            )

        if errors:
            raise ConfigurationError(
                "BayesOptConfig validation failed:\n  - " + "\n  - ".join(errors)
            )


@dataclass
class PreprocessArtifacts:
    factor_nmes: List[str]
    cate_list: List[str]
    num_features: List[str]
    var_nmes: List[str]
    cat_categories: Dict[str, List[Any]]
    dummy_columns: List[str]
    numeric_scalers: Dict[str, Dict[str, float]]
    weight_nme: str
    resp_nme: str
    binary_resp_nme: Optional[str] = None
    drop_first: bool = True


class OutputManager:
    # Centralize output paths for plots, results, and models.

    def __init__(self, root: Optional[str] = None, model_name: str = "model") -> None:
        self.root = Path(root or os.getcwd())
        self.model_name = model_name
        self.plot_dir = self.root / 'plot'
        self.result_dir = self.root / 'Results'
        self.model_dir = self.root / 'model'

    def _prepare(self, path: Path) -> str:
        IOUtils.ensure_parent_dir(str(path))
        return str(path)

    def plot_path(self, filename: str) -> str:
        return self._prepare(self.plot_dir / filename)

    def result_path(self, filename: str) -> str:
        return self._prepare(self.result_dir / filename)

    def model_path(self, filename: str) -> str:
        return self._prepare(self.model_dir / filename)


class VersionManager:
    """Lightweight versioning: save config and best-params snapshots for traceability."""

    def __init__(self, output: OutputManager) -> None:
        self.output = output
        self.version_dir = Path(self.output.result_dir) / "versions"
        IOUtils.ensure_parent_dir(str(self.version_dir))

    def save(self, tag: str, payload: Dict[str, Any]) -> str:
        safe_tag = tag.replace(" ", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.version_dir / f"{ts}_{safe_tag}.json"
        IOUtils.ensure_parent_dir(str(path))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        print(f"[Version] Saved snapshot: {path}")
        return str(path)

    def load_latest(self, tag: str) -> Optional[Dict[str, Any]]:
        """Load the latest snapshot for a tag (sorted by timestamp prefix)."""
        safe_tag = tag.replace(" ", "_")
        pattern = f"*_{safe_tag}.json"
        candidates = sorted(self.version_dir.glob(pattern))
        if not candidates:
            return None
        path = candidates[-1]
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[Version] Failed to load snapshot {path}: {exc}")
            return None


class DatasetPreprocessor:
    # Prepare shared train/test views for trainers.

    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                 config: BayesOptConfig) -> None:
        self.config = config
        # Use shallow copy to avoid unnecessary memory overhead
        # Deep copies only made when actually modifying data
        self.train_data = train_df.copy(deep=False)
        self.test_data = test_df.copy(deep=False)
        self.num_features: List[str] = []
        self.train_oht_data: Optional[pd.DataFrame] = None
        self.test_oht_data: Optional[pd.DataFrame] = None
        self.train_oht_scl_data: Optional[pd.DataFrame] = None
        self.test_oht_scl_data: Optional[pd.DataFrame] = None
        self.var_nmes: List[str] = []
        self.cat_categories_for_shap: Dict[str, List[Any]] = {}
        self.numeric_scalers: Dict[str, Dict[str, float]] = {}

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
            raise DataValidationError(
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

        # Memory optimization: Single copy + in-place operations
        train_oht = self.train_data[cfg.factor_nmes +
                                    [cfg.weight_nme] + [cfg.resp_nme]].copy()
        test_oht = self.test_data[cfg.factor_nmes +
                                  [cfg.weight_nme] + [cfg.resp_nme]].copy()
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

        # Fill missing dummy columns when reindexing to align train/test columns.
        test_oht = test_oht.reindex(columns=train_oht.columns, fill_value=0)

        # Keep unscaled one-hot data for fold-specific scaling to avoid leakage.
        # Store direct references - these won't be mutated
        self.train_oht_data = train_oht
        self.test_oht_data = test_oht

        # Only copy if we need to scale numeric features (memory optimization)
        if self.num_features:
            train_oht_scaled = train_oht.copy()
            test_oht_scaled = test_oht.copy()
        else:
            # No scaling needed, reuse original
            train_oht_scaled = train_oht
            test_oht_scaled = test_oht
        for num_chr in self.num_features:
            # Scale per column so features are on comparable ranges for NN stability.
            scaler = StandardScaler()
            train_oht_scaled[num_chr] = scaler.fit_transform(
                train_oht_scaled[num_chr].values.reshape(-1, 1))
            test_oht_scaled[num_chr] = scaler.transform(
                test_oht_scaled[num_chr].values.reshape(-1, 1))
            scale_val = float(getattr(scaler, "scale_", [1.0])[0])
            if scale_val == 0.0:
                scale_val = 1.0
            self.numeric_scalers[num_chr] = {
                "mean": float(getattr(scaler, "mean_", [0.0])[0]),
                "scale": scale_val,
            }
        # Fill missing dummy columns when reindexing to align train/test columns.
        test_oht_scaled = test_oht_scaled.reindex(
            columns=train_oht_scaled.columns, fill_value=0)
        self.train_oht_scl_data = train_oht_scaled
        self.test_oht_scl_data = test_oht_scaled
        excluded = {cfg.weight_nme, cfg.resp_nme}
        self.var_nmes = [
            col for col in train_oht_scaled.columns if col not in excluded
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
            dummy_columns=dummy_columns,
            numeric_scalers=dict(self.numeric_scalers),
            weight_nme=str(self.config.weight_nme),
            resp_nme=str(self.config.resp_nme),
            binary_resp_nme=self.config.binary_resp_nme,
            drop_first=True,
        )

    def save_artifacts(self, path: str | Path) -> str:
        payload = self.export_artifacts()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(payload), ensure_ascii=True, indent=2), encoding="utf-8")
        return str(target)
