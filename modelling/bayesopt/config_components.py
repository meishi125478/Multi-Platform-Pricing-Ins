"""Nested configuration components for BayesOptConfig.

This module provides focused configuration dataclasses that group related settings
together, improving maintainability and reducing the cognitive load of the main
BayesOptConfig class.

Usage:
    config = BayesOptConfig(
        model_nme="pricing_model",
        resp_nme="claim",
        weight_nme="exposure",
        factor_nmes=["age", "gender"],
        use_ft_ddp=True,
        gnn_use_approx_knn=False,
    )
    # Nested components are immutable views backed by flat fields.
    assert config.distributed.use_ft_ddp is True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _value_or_default(d: Dict[str, Any], key: str, default: Any) -> Any:
    value = d.get(key, default)
    return default if value is None else value


@dataclass(frozen=True)
class DistributedConfig:
    """Configuration for distributed training (DDP/DataParallel).

    Attributes:
        use_resn_data_parallel: Use DataParallel for ResNet
        use_ft_data_parallel: Use DataParallel for FT-Transformer
        use_gnn_data_parallel: Use DataParallel for GNN
        use_resn_ddp: Use DistributedDataParallel for ResNet
        use_ft_ddp: Use DistributedDataParallel for FT-Transformer
        use_gnn_ddp: Use DistributedDataParallel for GNN
    """

    use_resn_data_parallel: bool = False
    use_ft_data_parallel: bool = False
    use_gnn_data_parallel: bool = False
    use_resn_ddp: bool = False
    use_ft_ddp: bool = False
    use_gnn_ddp: bool = False

    @classmethod
    def from_flat_dict(cls, d: Dict[str, Any]) -> "DistributedConfig":
        """Create from a flat dictionary with prefixed keys."""
        return cls(
            use_resn_data_parallel=bool(_value_or_default(d, "use_resn_data_parallel", False)),
            use_ft_data_parallel=bool(_value_or_default(d, "use_ft_data_parallel", False)),
            use_gnn_data_parallel=bool(_value_or_default(d, "use_gnn_data_parallel", False)),
            use_resn_ddp=bool(_value_or_default(d, "use_resn_ddp", False)),
            use_ft_ddp=bool(_value_or_default(d, "use_ft_ddp", False)),
            use_gnn_ddp=bool(_value_or_default(d, "use_gnn_ddp", False)),
        )


@dataclass(frozen=True)
class GNNConfig:
    """Configuration for Graph Neural Network training.

    Attributes:
        use_approx_knn: Use approximate k-NN for graph construction
        approx_knn_threshold: Row count threshold for approximate k-NN
        graph_cache: Path to cache/load adjacency matrix
        max_gpu_knn_nodes: Max nodes for GPU k-NN construction
        knn_gpu_mem_ratio: Fraction of GPU memory for k-NN
        knn_gpu_mem_overhead: Temporary memory overhead multiplier
    """

    use_approx_knn: bool = True
    approx_knn_threshold: int = 50000
    graph_cache: Optional[str] = None
    max_gpu_knn_nodes: int = 200000
    knn_gpu_mem_ratio: float = 0.9
    knn_gpu_mem_overhead: float = 2.0

    @classmethod
    def from_flat_dict(cls, d: Dict[str, Any]) -> "GNNConfig":
        """Create from a flat dictionary with prefixed keys."""
        return cls(
            use_approx_knn=bool(_value_or_default(d, "gnn_use_approx_knn", True)),
            approx_knn_threshold=int(_value_or_default(d, "gnn_approx_knn_threshold", 50000)),
            graph_cache=d.get("gnn_graph_cache"),
            max_gpu_knn_nodes=int(_value_or_default(d, "gnn_max_gpu_knn_nodes", 200000)),
            knn_gpu_mem_ratio=float(_value_or_default(d, "gnn_knn_gpu_mem_ratio", 0.9)),
            knn_gpu_mem_overhead=float(_value_or_default(d, "gnn_knn_gpu_mem_overhead", 2.0)),
        )


@dataclass(frozen=True)
class GeoTokenConfig:
    """Configuration for geographic token embeddings.

    Attributes:
        feature_nmes: Feature column names for geo tokens
        hidden_dim: Hidden dimension for geo token network
        layers: Number of layers in geo token network
        dropout: Dropout rate
        k_neighbors: Number of neighbors for geo tokens
        learning_rate: Learning rate for geo token training
        epochs: Training epochs for geo tokens
    """

    feature_nmes: Optional[List[str]] = None
    hidden_dim: int = 32
    layers: int = 2
    dropout: float = 0.1
    k_neighbors: int = 10
    learning_rate: float = 1e-3
    epochs: int = 50

    @classmethod
    def from_flat_dict(cls, d: Dict[str, Any]) -> "GeoTokenConfig":
        """Create from a flat dictionary with prefixed keys."""
        return cls(
            feature_nmes=d.get("geo_feature_nmes"),
            hidden_dim=int(_value_or_default(d, "geo_token_hidden_dim", 32)),
            layers=int(_value_or_default(d, "geo_token_layers", 2)),
            dropout=float(_value_or_default(d, "geo_token_dropout", 0.1)),
            k_neighbors=int(_value_or_default(d, "geo_token_k_neighbors", 10)),
            learning_rate=float(_value_or_default(d, "geo_token_learning_rate", 1e-3)),
            epochs=int(_value_or_default(d, "geo_token_epochs", 50)),
        )


@dataclass(frozen=True)
class RegionConfig:
    """Configuration for region/geographic effects.

    Attributes:
        province_col: Column name for province/state
        city_col: Column name for city
        effect_alpha: Regularization alpha for region effects
    """

    province_col: Optional[str] = None
    city_col: Optional[str] = None
    effect_alpha: float = 50.0

    @classmethod
    def from_flat_dict(cls, d: Dict[str, Any]) -> "RegionConfig":
        """Create from a flat dictionary with prefixed keys."""
        return cls(
            province_col=d.get("region_province_col"),
            city_col=d.get("region_city_col"),
            effect_alpha=float(_value_or_default(d, "region_effect_alpha", 50.0)),
        )


@dataclass(frozen=True)
class FTTransformerConfig:
    """Configuration for FT-Transformer model.

    Attributes:
        role: Model role ('model', 'embedding', 'unsupervised_embedding')
        feature_prefix: Prefix for generated embedding features
        num_numeric_tokens: Number of numeric tokens
    """

    role: str = "model"
    feature_prefix: str = "ft_emb"
    num_numeric_tokens: Optional[int] = None

    @classmethod
    def from_flat_dict(cls, d: Dict[str, Any]) -> "FTTransformerConfig":
        """Create from a flat dictionary with prefixed keys."""
        return cls(
            role=str(_value_or_default(d, "ft_role", "model")),
            feature_prefix=str(_value_or_default(d, "ft_feature_prefix", "ft_emb")),
            num_numeric_tokens=d.get("ft_num_numeric_tokens"),
        )


@dataclass(frozen=True)
class XGBoostConfig:
    """Configuration for XGBoost model.

    Attributes:
        max_depth_max: Maximum tree depth for hyperparameter tuning
        n_estimators_max: Maximum number of estimators for tuning
        gpu_id: GPU device id for XGBoost (None = default)
        cleanup_per_fold: Whether to cleanup GPU memory after each fold
        cleanup_synchronize: Whether to synchronize CUDA during cleanup
        use_dmatrix: Whether to use xgb.train with DMatrix/QuantileDMatrix
    """

    max_depth_max: int = 25
    n_estimators_max: int = 500
    gpu_id: Optional[int] = None
    cleanup_per_fold: bool = False
    cleanup_synchronize: bool = False
    use_dmatrix: bool = True

    @classmethod
    def from_flat_dict(cls, d: Dict[str, Any]) -> "XGBoostConfig":
        """Create from a flat dictionary with prefixed keys."""
        return cls(
            max_depth_max=int(_value_or_default(d, "xgb_max_depth_max", 25)),
            n_estimators_max=int(_value_or_default(d, "xgb_n_estimators_max", 500)),
            gpu_id=d.get("xgb_gpu_id"),
            cleanup_per_fold=bool(_value_or_default(d, "xgb_cleanup_per_fold", False)),
            cleanup_synchronize=bool(_value_or_default(d, "xgb_cleanup_synchronize", False)),
            use_dmatrix=bool(_value_or_default(d, "xgb_use_dmatrix", True)),
        )


@dataclass(frozen=True)
class CVConfig:
    """Configuration for cross-validation.

    Attributes:
        strategy: CV strategy ('random', 'group', 'time', 'stratified')
        splits: Number of CV splits
        group_col: Column for group-based CV
        time_col: Column for time-based CV
        time_ascending: Whether to sort time ascending
    """

    strategy: str = "random"
    splits: Optional[int] = None
    group_col: Optional[str] = None
    time_col: Optional[str] = None
    time_ascending: bool = True

    @classmethod
    def from_flat_dict(cls, d: Dict[str, Any]) -> "CVConfig":
        """Create from a flat dictionary with prefixed keys."""
        return cls(
            strategy=str(_value_or_default(d, "cv_strategy", "random")),
            splits=d.get("cv_splits"),
            group_col=d.get("cv_group_col"),
            time_col=d.get("cv_time_col"),
            time_ascending=bool(_value_or_default(d, "cv_time_ascending", True)),
        )


@dataclass(frozen=True)
class FTOOFConfig:
    """Configuration for FT-Transformer out-of-fold predictions.

    Attributes:
        folds: Number of OOF folds
        strategy: OOF strategy
        shuffle: Whether to shuffle data
    """

    folds: Optional[int] = None
    strategy: Optional[str] = None
    shuffle: bool = True

    @classmethod
    def from_flat_dict(cls, d: Dict[str, Any]) -> "FTOOFConfig":
        """Create from a flat dictionary with prefixed keys."""
        return cls(
            folds=d.get("ft_oof_folds"),
            strategy=d.get("ft_oof_strategy"),
            shuffle=bool(_value_or_default(d, "ft_oof_shuffle", True)),
        )


@dataclass(frozen=True)
class OutputConfig:
    """Configuration for output and caching.

    Attributes:
        output_dir: Base output directory
        optuna_storage: Optuna study storage path
        optuna_study_prefix: Prefix for Optuna study names
        best_params_files: Mapping of trainer keys to param files
        save_preprocess: Whether to save preprocessing artifacts
        preprocess_artifact_path: Path for preprocessing artifacts
        plot_path_style: Plot path style ('nested' or 'flat')
        cache_predictions: Whether to cache predictions
        prediction_cache_dir: Directory for prediction cache
        prediction_cache_format: Format for prediction cache ('parquet' or 'csv')
    """

    output_dir: Optional[str] = None
    optuna_storage: Optional[str] = None
    optuna_study_prefix: Optional[str] = None
    best_params_files: Optional[Dict[str, str]] = None
    save_preprocess: bool = False
    preprocess_artifact_path: Optional[str] = None
    plot_path_style: str = "nested"
    cache_predictions: bool = False
    prediction_cache_dir: Optional[str] = None
    prediction_cache_format: str = "parquet"

    @classmethod
    def from_flat_dict(cls, d: Dict[str, Any]) -> "OutputConfig":
        """Create from a flat dictionary with prefixed keys."""
        return cls(
            output_dir=d.get("output_dir"),
            optuna_storage=d.get("optuna_storage"),
            optuna_study_prefix=d.get("optuna_study_prefix"),
            best_params_files=d.get("best_params_files"),
            save_preprocess=bool(_value_or_default(d, "save_preprocess", False)),
            preprocess_artifact_path=d.get("preprocess_artifact_path"),
            plot_path_style=str(_value_or_default(d, "plot_path_style", "nested")),
            cache_predictions=bool(_value_or_default(d, "cache_predictions", False)),
            prediction_cache_dir=d.get("prediction_cache_dir"),
            prediction_cache_format=str(_value_or_default(d, "prediction_cache_format", "parquet")),
        )


@dataclass(frozen=True)
class EnsembleConfig:
    """Configuration for ensemble training.

    Attributes:
        final_ensemble: Whether to use final ensemble
        final_ensemble_k: Number of models in ensemble
        final_refit: Whether to refit after ensemble
    """

    final_ensemble: bool = False
    final_ensemble_k: int = 3
    final_refit: bool = True

    @classmethod
    def from_flat_dict(cls, d: Dict[str, Any]) -> "EnsembleConfig":
        """Create from a flat dictionary with prefixed keys."""
        return cls(
            final_ensemble=bool(_value_or_default(d, "final_ensemble", False)),
            final_ensemble_k=int(_value_or_default(d, "final_ensemble_k", 3)),
            final_refit=bool(_value_or_default(d, "final_refit", True)),
        )


@dataclass(frozen=True)
class TrainingConfig:
    """Core training configuration.

    Attributes:
        prop_test: Proportion of data for validation
        rand_seed: Random seed for reproducibility
        epochs: Number of training epochs
        use_gpu: Whether to use GPU
        reuse_best_params: Whether to reuse best params
        resn_weight_decay: Weight decay for ResNet
        bo_sample_limit: Sample limit for Bayesian optimization
    """

    prop_test: float = 0.25
    rand_seed: Optional[int] = None
    epochs: int = 100
    use_gpu: bool = True
    reuse_best_params: bool = False
    resn_weight_decay: float = 1e-4
    bo_sample_limit: Optional[int] = None

    @classmethod
    def from_flat_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        """Create from a flat dictionary with prefixed keys."""
        return cls(
            prop_test=float(_value_or_default(d, "prop_test", 0.25)),
            rand_seed=d.get("rand_seed"),
            epochs=int(_value_or_default(d, "epochs", 100)),
            use_gpu=bool(_value_or_default(d, "use_gpu", True)),
            reuse_best_params=bool(_value_or_default(d, "reuse_best_params", False)),
            resn_weight_decay=float(_value_or_default(d, "resn_weight_decay", 1e-4)),
            bo_sample_limit=d.get("bo_sample_limit"),
        )
