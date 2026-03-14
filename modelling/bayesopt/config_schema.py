from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
from ins_pricing.utils.losses import normalize_distribution_name, normalize_loss_name
from ins_pricing.exceptions import ConfigurationError
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
        xgb_search_space: Optional XGBoost Optuna search-space override (JSON object)
        ft_cleanup_per_fold: Whether to cleanup GPU memory after each FT fold
        ft_cleanup_synchronize: Whether to synchronize CUDA during FT cleanup
        resn_cleanup_per_fold: Whether to cleanup GPU memory after each ResNet fold
        resn_cleanup_synchronize: Whether to synchronize CUDA during ResNet cleanup
        resn_use_lazy_dataset: Whether ResNet uses lazy row-wise dataset to avoid full tensor materialization
        resn_predict_batch_size: Optional batch size for ResNet prediction (None = auto)
        resn_search_space: Optional ResNet Optuna search-space override (JSON object)
        ft_use_lazy_dataset: Whether FT-Transformer uses lazy row-wise dataset to avoid full tensor materialization
        ft_predict_batch_size: Optional batch size for FT-Transformer prediction (None = auto)
        ft_search_space: Optional FT supervised Optuna search-space override (JSON object)
        ft_unsupervised_search_space: Optional FT unsupervised Optuna search-space override (JSON object)
        gnn_cleanup_per_fold: Whether to cleanup GPU memory after each GNN fold
        gnn_cleanup_synchronize: Whether to synchronize CUDA during GNN cleanup
        optuna_cleanup_synchronize: Whether to synchronize CUDA during Optuna cleanup
        use_resn_data_parallel: Use DataParallel for ResNet
        use_ft_data_parallel: Use DataParallel for FT-Transformer
        use_resn_ddp: Use DDP for ResNet
        use_ft_ddp: Use DDP for FT-Transformer
        use_gnn_data_parallel: Use DataParallel for GNN
        ft_role: FT-Transformer role ('model', 'embedding', 'unsupervised_embedding')
        cv_strategy: CV strategy ('random', 'group', 'time', 'stratified')
        invalid_param_policy: Handling for unsupported tuned params ('warn', 'error', 'ignore')
        build_oht: Whether to build one-hot encoded features (default True)
        oht_sparse_csr: Use OneHotEncoder CSR backend for categorical OHE (default True)
        keep_unscaled_oht: Keep unscaled one-hot copy in memory (default False)
        dataloader_multiprocessing_context: Optional DataLoader multiprocessing start method
            ('fork', 'spawn', or 'forkserver')

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
    xgb_search_space: Optional[Dict[str, Any]] = None
    ft_cleanup_per_fold: bool = False
    ft_cleanup_synchronize: bool = False
    resn_cleanup_per_fold: bool = False
    resn_cleanup_synchronize: bool = False
    resn_use_lazy_dataset: bool = True
    resn_predict_batch_size: Optional[int] = None
    resn_search_space: Optional[Dict[str, Any]] = None
    ft_use_lazy_dataset: bool = True
    ft_predict_batch_size: Optional[int] = None
    ft_search_space: Optional[Dict[str, Any]] = None
    ft_unsupervised_search_space: Optional[Dict[str, Any]] = None
    gnn_cleanup_per_fold: bool = False
    gnn_cleanup_synchronize: bool = False
    optuna_cleanup_synchronize: bool = False

    # Distributed training settings
    use_resn_data_parallel: bool = False
    use_ft_data_parallel: bool = False
    use_resn_ddp: bool = False
    use_ft_ddp: bool = False
    use_gnn_data_parallel: bool = False

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
    save_preprocess_bundle: bool = False
    load_preprocess_bundle: bool = False
    preprocess_bundle_path: Optional[str] = None
    plot_path_style: str = "nested"
    bo_sample_limit: Optional[int] = None
    invalid_param_policy: str = "warn"
    build_oht: bool = True
    oht_sparse_csr: bool = True
    keep_unscaled_oht: bool = False
    cache_predictions: bool = False
    prediction_cache_dir: Optional[str] = None
    prediction_cache_format: str = "parquet"
    classification_predict_api: str = "label"
    classification_prediction_outputs: str = "score"
    classification_plot_prediction: str = "score"
    classification_label_threshold: float = 0.5
    dataloader_workers: Optional[int] = None
    dataloader_multiprocessing_context: Optional[str] = None

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
        if self.dataloader_multiprocessing_context is not None:
            if not isinstance(self.dataloader_multiprocessing_context, str):
                errors.append(
                    "dataloader_multiprocessing_context must be a string when provided."
                )
            else:
                mp_ctx = self.dataloader_multiprocessing_context.strip().lower()
                valid_mp = {"fork", "spawn", "forkserver"}
                if mp_ctx not in valid_mp:
                    errors.append(
                        "dataloader_multiprocessing_context must be one of "
                        f"{sorted(valid_mp)}, got {self.dataloader_multiprocessing_context!r}."
                    )
        if not isinstance(self.keep_unscaled_oht, bool):
            errors.append("keep_unscaled_oht must be a boolean.")
        if not isinstance(self.oht_sparse_csr, bool):
            errors.append("oht_sparse_csr must be a boolean.")
        if not isinstance(self.save_preprocess_bundle, bool):
            errors.append("save_preprocess_bundle must be a boolean.")
        if not isinstance(self.load_preprocess_bundle, bool):
            errors.append("load_preprocess_bundle must be a boolean.")
        if (
            self.load_preprocess_bundle
            and isinstance(self.preprocess_bundle_path, str)
            and not self.preprocess_bundle_path.strip()
        ):
            errors.append(
                "preprocess_bundle_path must be a non-empty string when "
                "load_preprocess_bundle=True."
            )
        if self.save_preprocess_bundle and self.load_preprocess_bundle:
            errors.append(
                "Cannot set both save_preprocess_bundle and load_preprocess_bundle to True."
            )
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
        valid_predict_api = {"label", "score"}
        predict_api = str(self.classification_predict_api).strip().lower()
        if predict_api not in valid_predict_api:
            errors.append(
                "classification_predict_api must be one of "
                f"{valid_predict_api}, got '{self.classification_predict_api}'"
            )
        else:
            self.classification_predict_api = predict_api
        valid_pred_outputs = {"score", "both"}
        pred_outputs = str(self.classification_prediction_outputs).strip().lower()
        if pred_outputs not in valid_pred_outputs:
            errors.append(
                "classification_prediction_outputs must be one of "
                f"{valid_pred_outputs}, got '{self.classification_prediction_outputs}'"
            )
        else:
            self.classification_prediction_outputs = pred_outputs
        valid_plot_pred = {"score", "label"}
        plot_pred = str(self.classification_plot_prediction).strip().lower()
        if plot_pred not in valid_plot_pred:
            errors.append(
                "classification_plot_prediction must be one of "
                f"{valid_plot_pred}, got '{self.classification_plot_prediction}'"
            )
        else:
            self.classification_plot_prediction = plot_pred
        try:
            label_threshold = float(self.classification_label_threshold)
        except (TypeError, ValueError):
            errors.append("classification_label_threshold must be a float in [0, 1].")
        else:
            if not (0.0 <= label_threshold <= 1.0):
                errors.append(
                    "classification_label_threshold must be in [0, 1], "
                    f"got {label_threshold}"
                )
            else:
                self.classification_label_threshold = label_threshold

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

        # Validate unsupported-parameter handling policy.
        invalid_param_policy = str(self.invalid_param_policy).strip().lower()
        if invalid_param_policy not in {"warn", "error", "ignore"}:
            errors.append(
                "invalid_param_policy must be one of {'warn', 'error', 'ignore'}, "
                f"got '{self.invalid_param_policy}'"
            )
        else:
            self.invalid_param_policy = invalid_param_policy

        # Validate JSON-driven Optuna search spaces.
        for field_name in (
            "xgb_search_space",
            "resn_search_space",
            "ft_search_space",
            "ft_unsupervised_search_space",
        ):
            raw_space = getattr(self, field_name, None)
            if raw_space is None:
                continue
            if not isinstance(raw_space, dict):
                errors.append(f"{field_name} must be a JSON object when provided.")
                continue
            for param_name, param_spec in raw_space.items():
                if not isinstance(param_name, str) or not param_name.strip():
                    errors.append(f"{field_name} has an invalid parameter name: {param_name!r}")
                    continue
                if isinstance(param_spec, dict):
                    param_type = str(param_spec.get("type", "")).strip().lower()
                    if param_type and param_type not in {"int", "float", "categorical"}:
                        errors.append(
                            f"{field_name}.{param_name}.type must be one of int/float/categorical."
                        )
                elif isinstance(param_spec, (list, tuple)):
                    if len(param_spec) == 0:
                        errors.append(f"{field_name}.{param_name} categorical choices cannot be empty.")
                # Scalars are treated as fixed values in trainer sampling.

        if errors:
            raise ConfigurationError(
                "BayesOptConfig validation failed:\n  - " + "\n  - ".join(errors)
            )

