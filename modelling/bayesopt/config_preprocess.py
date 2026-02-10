from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ins_pricing.modelling.bayesopt.config_components import (
    CVConfig,
    DistributedConfig,
    EnsembleConfig,
    FTOOFConfig,
    FTTransformerConfig,
    GNNConfig,
    GeoTokenConfig,
    OutputConfig,
    RegionConfig,
    TrainingConfig,
    XGBoostConfig,
)
from ins_pricing.utils.io import IOUtils
from ins_pricing.utils.losses import normalize_distribution_name, normalize_loss_name
from ins_pricing.exceptions import ConfigurationError, DataValidationError
from ins_pricing.utils import get_logger, log_print

_logger = get_logger("ins_pricing.modelling.bayesopt.config_preprocess")


def _log(*args, **kwargs) -> None:
    log_print(_logger, *args, **kwargs)

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
        distribution: Optional target distribution override (regression)
        loss_name: Regression loss ('auto', 'tweedie', 'poisson', 'gamma', 'mse', 'mae')
        prop_test: Proportion of data for validation (0.0-1.0)
        rand_seed: Random seed for reproducibility
        epochs: Number of training epochs
        use_gpu: Whether to use GPU acceleration
        xgb_max_depth_max: Maximum tree depth for XGBoost tuning
        xgb_n_estimators_max: Maximum estimators for XGBoost tuning
        xgb_gpu_id: GPU device id for XGBoost (None = default)
        xgb_cleanup_per_fold: Whether to cleanup GPU memory after each XGBoost fold
        xgb_cleanup_synchronize: Whether to synchronize CUDA during XGBoost cleanup
        xgb_use_dmatrix: Whether to use xgb.train with DMatrix/QuantileDMatrix
        xgb_chunk_size: Rows per chunk for XGBoost chunked incremental training
        ft_cleanup_per_fold: Whether to cleanup GPU memory after each FT fold
        ft_cleanup_synchronize: Whether to synchronize CUDA during FT cleanup
        resn_cleanup_per_fold: Whether to cleanup GPU memory after each ResNet fold
        resn_cleanup_synchronize: Whether to synchronize CUDA during ResNet cleanup
        resn_use_lazy_dataset: Whether ResNet uses lazy row-wise dataset to avoid full tensor materialization
        resn_predict_batch_size: Optional batch size for ResNet prediction (None = auto)
        ft_use_lazy_dataset: Whether FT-Transformer uses lazy row-wise dataset to avoid full tensor materialization
        ft_predict_batch_size: Optional batch size for FT-Transformer prediction (None = auto)
        gnn_cleanup_per_fold: Whether to cleanup GPU memory after each GNN fold
        gnn_cleanup_synchronize: Whether to synchronize CUDA during GNN cleanup
        optuna_cleanup_synchronize: Whether to synchronize CUDA during Optuna cleanup
        use_resn_data_parallel: Use DataParallel for ResNet
        use_ft_data_parallel: Use DataParallel for FT-Transformer
        use_resn_ddp: Use DDP for ResNet
        use_ft_ddp: Use DDP for FT-Transformer
        use_gnn_data_parallel: Use DataParallel for GNN
        use_gnn_ddp: Use DDP for GNN
        ft_role: FT-Transformer role ('model', 'embedding', 'unsupervised_embedding')
        cv_strategy: CV strategy ('random', 'group', 'time', 'stratified')
        build_oht: Whether to build one-hot encoded features (default True)

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
    distribution: Optional[str] = None
    loss_name: str = "auto"

    # Training configuration
    prop_test: float = 0.25
    rand_seed: Optional[int] = None
    epochs: int = 100
    use_gpu: bool = True

    # XGBoost settings
    xgb_max_depth_max: int = 25
    xgb_n_estimators_max: int = 500
    xgb_gpu_id: Optional[int] = None
    xgb_cleanup_per_fold: bool = False
    xgb_cleanup_synchronize: bool = False
    xgb_use_dmatrix: bool = True
    xgb_chunk_size: Optional[int] = None
    ft_cleanup_per_fold: bool = False
    ft_cleanup_synchronize: bool = False
    resn_cleanup_per_fold: bool = False
    resn_cleanup_synchronize: bool = False
    resn_use_lazy_dataset: bool = True
    resn_predict_batch_size: Optional[int] = None
    ft_use_lazy_dataset: bool = True
    ft_predict_batch_size: Optional[int] = None
    gnn_cleanup_per_fold: bool = False
    gnn_cleanup_synchronize: bool = False
    optuna_cleanup_synchronize: bool = False

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
    gnn_max_fit_rows: Optional[int] = None
    gnn_max_predict_rows: Optional[int] = None
    gnn_predict_chunk_rows: Optional[int] = None

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
    build_oht: bool = True
    cache_predictions: bool = False
    prediction_cache_dir: Optional[str] = None
    prediction_cache_format: str = "parquet"
    dataloader_workers: Optional[int] = None

    # Nested configuration views (synced from flat fields in __post_init__).
    _distributed: DistributedConfig = field(default_factory=DistributedConfig, init=False, repr=False)
    _gnn: GNNConfig = field(default_factory=GNNConfig, init=False, repr=False)
    _geo_token: GeoTokenConfig = field(default_factory=GeoTokenConfig, init=False, repr=False)
    _region: RegionConfig = field(default_factory=RegionConfig, init=False, repr=False)
    _ft_transformer: FTTransformerConfig = field(default_factory=FTTransformerConfig, init=False, repr=False)
    _xgboost: XGBoostConfig = field(default_factory=XGBoostConfig, init=False, repr=False)
    _cv: CVConfig = field(default_factory=CVConfig, init=False, repr=False)
    _ft_oof: FTOOFConfig = field(default_factory=FTOOFConfig, init=False, repr=False)
    _output: OutputConfig = field(default_factory=OutputConfig, init=False, repr=False)
    _ensemble: EnsembleConfig = field(default_factory=EnsembleConfig, init=False, repr=False)
    _training: TrainingConfig = field(default_factory=TrainingConfig, init=False, repr=False)
    _is_initialized: bool = field(default=False, init=False, repr=False)
    _sync_in_progress: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Synchronize nested config views and validate values."""
        self._sync_nested_components()
        self._validate()
        object.__setattr__(self, "_is_initialized", True)

    def __setattr__(self, name: str, value: Any) -> None:
        """Keep nested config views in sync when flat fields are mutated."""
        object.__setattr__(self, name, value)
        if name.startswith("_"):
            return
        if not getattr(self, "_is_initialized", False):
            return
        if getattr(self, "_sync_in_progress", False):
            return
        spec = self.__dataclass_fields__.get(name)
        if spec is None or not getattr(spec, "init", True):
            return
        self._sync_nested_components()

    @classmethod
    def from_flat_dict(cls, d: Dict[str, Any]) -> "BayesOptConfig":
        """Build config from a flat dict while ignoring unknown/None values."""
        fields = getattr(cls, "__dataclass_fields__", {})
        payload: Dict[str, Any] = {}
        for key, value in d.items():
            spec = fields.get(key)
            if spec is None or not getattr(spec, "init", True):
                continue
            if value is not None:
                payload[key] = value
        return cls(**payload)

    def _sync_nested_components(self) -> None:
        object.__setattr__(self, "_sync_in_progress", True)
        flat = {
            key: getattr(self, key)
            for key, spec in self.__dataclass_fields__.items()
            if getattr(spec, "init", True)
        }
        try:
            object.__setattr__(self, "_distributed", DistributedConfig.from_flat_dict(flat))
            object.__setattr__(self, "_gnn", GNNConfig.from_flat_dict(flat))
            object.__setattr__(self, "_geo_token", GeoTokenConfig.from_flat_dict(flat))
            object.__setattr__(self, "_region", RegionConfig.from_flat_dict(flat))
            object.__setattr__(self, "_ft_transformer", FTTransformerConfig.from_flat_dict(flat))
            object.__setattr__(self, "_xgboost", XGBoostConfig.from_flat_dict(flat))
            object.__setattr__(self, "_cv", CVConfig.from_flat_dict(flat))
            object.__setattr__(self, "_ft_oof", FTOOFConfig.from_flat_dict(flat))
            object.__setattr__(self, "_output", OutputConfig.from_flat_dict(flat))
            object.__setattr__(self, "_ensemble", EnsembleConfig.from_flat_dict(flat))
            object.__setattr__(self, "_training", TrainingConfig.from_flat_dict(flat))
        finally:
            object.__setattr__(self, "_sync_in_progress", False)

    @property
    def distributed(self) -> DistributedConfig:
        return self._distributed

    @property
    def gnn(self) -> GNNConfig:
        return self._gnn

    @property
    def geo_token(self) -> GeoTokenConfig:
        return self._geo_token

    @property
    def region(self) -> RegionConfig:
        return self._region

    @property
    def ft_transformer(self) -> FTTransformerConfig:
        return self._ft_transformer

    @property
    def xgboost(self) -> XGBoostConfig:
        return self._xgboost

    @property
    def cv(self) -> CVConfig:
        return self._cv

    @property
    def ft_oof(self) -> FTOOFConfig:
        return self._ft_oof

    @property
    def output(self) -> OutputConfig:
        return self._output

    @property
    def ensemble(self) -> EnsembleConfig:
        return self._ensemble

    @property
    def training(self) -> TrainingConfig:
        return self._training

    def _validate(self) -> None:
        """Validate configuration values and raise errors for invalid combinations."""
        errors: List[str] = []

        # Validate task_type
        valid_task_types = {"regression", "classification"}
        if self.task_type not in valid_task_types:
            errors.append(
                f"task_type must be one of {valid_task_types}, got '{self.task_type}'"
            )
        if self.dataloader_workers is not None:
            try:
                if int(self.dataloader_workers) < 0:
                    errors.append("dataloader_workers must be >= 0 when provided.")
            except (TypeError, ValueError):
                errors.append("dataloader_workers must be an integer when provided.")
        # Validate distribution
        try:
            normalized_distribution = normalize_distribution_name(
                self.distribution, self.task_type
            )
            self.distribution = None if normalized_distribution == "auto" else normalized_distribution
        except ValueError as exc:
            errors.append(str(exc))

        # Validate loss_name
        try:
            normalized_loss = normalize_loss_name(self.loss_name, self.task_type)
            if self.task_type == "classification" and normalized_loss not in {"auto", "logloss", "bce"}:
                errors.append(
                    "loss_name must be 'auto', 'logloss', or 'bce' for classification tasks."
                )
        except ValueError as exc:
            errors.append(str(exc))

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
        if self.xgb_gpu_id is not None:
            try:
                gpu_id = int(self.xgb_gpu_id)
            except (TypeError, ValueError):
                errors.append(f"xgb_gpu_id must be an integer, got {self.xgb_gpu_id!r}")
            else:
                if gpu_id < 0:
                    errors.append(f"xgb_gpu_id must be >= 0, got {gpu_id}")
        if self.xgb_chunk_size is not None:
            try:
                xgb_chunk_size = int(self.xgb_chunk_size)
            except (TypeError, ValueError):
                errors.append(
                    f"xgb_chunk_size must be a positive integer, got {self.xgb_chunk_size!r}"
                )
            else:
                if xgb_chunk_size < 1:
                    errors.append(
                        f"xgb_chunk_size must be >= 1 when provided, got {xgb_chunk_size}"
                    )
        if self.resn_predict_batch_size is not None:
            try:
                resn_predict_batch_size = int(self.resn_predict_batch_size)
            except (TypeError, ValueError):
                errors.append(
                    "resn_predict_batch_size must be a positive integer when provided."
                )
            else:
                if resn_predict_batch_size < 1:
                    errors.append(
                        f"resn_predict_batch_size must be >= 1 when provided, got {resn_predict_batch_size}"
                    )
        if self.ft_predict_batch_size is not None:
            try:
                ft_predict_batch_size = int(self.ft_predict_batch_size)
            except (TypeError, ValueError):
                errors.append(
                    "ft_predict_batch_size must be a positive integer when provided."
                )
            else:
                if ft_predict_batch_size < 1:
                    errors.append(
                        f"ft_predict_batch_size must be >= 1 when provided, got {ft_predict_batch_size}"
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
        if self.gnn_max_fit_rows is not None:
            try:
                gnn_max_fit_rows = int(self.gnn_max_fit_rows)
            except (TypeError, ValueError):
                errors.append("gnn_max_fit_rows must be a positive integer when provided.")
            else:
                if gnn_max_fit_rows < 1:
                    errors.append(
                        f"gnn_max_fit_rows must be >= 1 when provided, got {gnn_max_fit_rows}"
                    )
        if self.gnn_max_predict_rows is not None:
            try:
                gnn_max_predict_rows = int(self.gnn_max_predict_rows)
            except (TypeError, ValueError):
                errors.append("gnn_max_predict_rows must be a positive integer when provided.")
            else:
                if gnn_max_predict_rows < 1:
                    errors.append(
                        f"gnn_max_predict_rows must be >= 1 when provided, got {gnn_max_predict_rows}"
                    )
        if self.gnn_predict_chunk_rows is not None:
            try:
                gnn_predict_chunk_rows = int(self.gnn_predict_chunk_rows)
            except (TypeError, ValueError):
                errors.append("gnn_predict_chunk_rows must be a positive integer when provided.")
            else:
                if gnn_predict_chunk_rows < 1:
                    errors.append(
                        f"gnn_predict_chunk_rows must be >= 1 when provided, got {gnn_predict_chunk_rows}"
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
        _log(f"[Version] Saved snapshot: {path}")
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
            _log(f"[Version] Failed to load snapshot {path}: {exc}")
            return None


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
