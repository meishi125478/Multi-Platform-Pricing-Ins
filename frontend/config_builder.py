"""
Configuration Builder for Insurance Pricing Models
Generates complete configuration dictionaries from UI parameters.
"""

from typing import List, Optional, Dict, Any


class ConfigBuilder:
    """Build configuration dictionaries for model training."""

    def __init__(self):
        self.default_config = self._get_default_config()

    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Get default configuration template."""
        return {
            "data_format": "csv",
            "data_path_template": "{model_name}.{ext}",
            "dtype_map": None,
            "binary_resp_nme": None,
            "split_group_col": None,
            "split_time_col": None,
            "split_time_ascending": True,
            "cv_strategy": None,
            "cv_group_col": None,
            "cv_time_col": None,
            "cv_time_ascending": True,
            "cv_splits": None,
            "plot_path_style": "nested",
            "save_preprocess": False,
            "preprocess_artifact_path": None,
            "bo_sample_limit": None,
            "build_oht": True,
            "cache_predictions": False,
            "prediction_cache_dir": None,
            "prediction_cache_format": "parquet",
            "plot_curves": False,
            "plot": {
                "enable": False,
                "n_bins": 10,
                "oneway": False,
                "oneway_pred": False,
                "pre_oneway": False,
                "lift_models": [],
                "double_lift": False,
                "double_lift_pairs": []
            },
            "env": {
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1"
            },
            "use_resn_data_parallel": False,
            "use_ft_data_parallel": False,
            "use_gnn_data_parallel": False,
            "use_resn_ddp": True,
            "use_ft_ddp": True,
            "use_gnn_ddp": True,
            "ddp_min_rows": 50000,
            "ft_role": "model",
            "ft_feature_prefix": "ft_emb",
            "ft_num_numeric_tokens": None,
            "ft_oof_folds": None,
            "ft_oof_strategy": None,
            "ft_oof_shuffle": True,
            "resn_weight_decay": 0.0001,
            "final_ensemble": False,
            "final_ensemble_k": 3,
            "final_refit": True,
            "infer_categorical_max_unique": 50,
            "infer_categorical_max_ratio": 0.05,
            "optuna_study_prefix": "pricing",
            "reuse_best_params": False,
            "best_params_files": {},
            "gnn_use_approx_knn": True,
            "gnn_approx_knn_threshold": 50000,
            "gnn_graph_cache": None,
            "gnn_max_gpu_knn_nodes": 200000,
            "gnn_knn_gpu_mem_ratio": 0.9,
            "gnn_knn_gpu_mem_overhead": 2.0,
            "geo_feature_nmes": [],
            "region_province_col": None,
            "region_city_col": None,
            "region_effect_alpha": 0.0,
            "geo_token_hidden_dim": 32,
            "geo_token_layers": 2,
            "geo_token_dropout": 0.1,
            "geo_token_k_neighbors": 10,
            "geo_token_learning_rate": 0.001,
            "geo_token_epochs": 50,
            "report_output_dir": "./Results/reports",
            "report_group_cols": [],
            "report_time_col": None,
            "report_time_freq": "M",
            "report_time_ascending": True,
            "psi_bins": 10,
            "psi_strategy": "quantile",
            "psi_features": [],
            "calibration": {
                "enable": False,
                "method": "sigmoid",
                "max_rows": None,
                "seed": 13
            },
            "threshold": {
                "enable": False,
                "value": None,
                "metric": "f1",
                "min_positive_rate": None,
                "grid": 99,
                "max_rows": None,
                "seed": 13
            },
            "bootstrap": {
                "enable": False,
                "metrics": [],
                "n_samples": 200,
                "ci": 0.95,
                "seed": 13
            },
            "register_model": False,
            "registry_path": "./Results/model_registry.json",
            "registry_tags": {},
            "registry_status": "candidate",
            "data_fingerprint_max_bytes": 10485760,
        }

    def build_config(
        self,
        data_dir: str,
        model_list: List[str],
        model_categories: List[str],
        target: str,
        weight: str,
        feature_list: List[str],
        categorical_features: List[str],
        task_type: str = "regression",
        prop_test: float = 0.25,
        holdout_ratio: float = 0.25,
        val_ratio: float = 0.25,
        split_strategy: str = "random",
        rand_seed: int = 13,
        epochs: int = 50,
        output_dir: str = "./Results",
        use_gpu: bool = True,
        model_keys: Optional[List[str]] = None,
        max_evals: int = 50,
        xgb_max_depth_max: int = 25,
        xgb_n_estimators_max: int = 500,
        nproc_per_node: int = 2,
    ) -> Dict[str, Any]:
        """
        Build a complete configuration dictionary.

        Args:
            data_dir: Directory containing data files
            model_list: List of model names
            model_categories: List of model categories
            target: Target column name
            weight: Weight column name
            feature_list: List of feature names
            categorical_features: List of categorical feature names
            task_type: Type of task (regression, binary, multiclass)
            prop_test: Proportion of data for testing
            holdout_ratio: Holdout ratio for validation
            val_ratio: Validation ratio
            split_strategy: Strategy for splitting data
            rand_seed: Random seed for reproducibility
            epochs: Number of training epochs
            output_dir: Directory for output files
            use_gpu: Whether to use GPU
            model_keys: List of model types to train
            max_evals: Maximum number of evaluations for optimization
            xgb_max_depth_max: Maximum depth for XGBoost
            xgb_n_estimators_max: Maximum estimators for XGBoost
            nproc_per_node: Number of processes per node

        Returns:
            Complete configuration dictionary
        """
        if model_keys is None:
            model_keys = ["xgb", "resn"]

        config = self.default_config.copy()

        # Update with user-provided values
        config.update({
            "data_dir": data_dir,
            "model_list": model_list,
            "model_categories": model_categories,
            "target": target,
            "weight": weight,
            "feature_list": feature_list,
            "categorical_features": categorical_features,
            "task_type": task_type,
            "prop_test": prop_test,
            "holdout_ratio": holdout_ratio,
            "val_ratio": val_ratio,
            "split_strategy": split_strategy,
            "rand_seed": rand_seed,
            "epochs": epochs,
            "output_dir": output_dir,
            "use_gpu": use_gpu,
            "xgb_max_depth_max": xgb_max_depth_max,
            "xgb_n_estimators_max": xgb_n_estimators_max,
            "optuna_storage": f"{output_dir}/optuna/bayesopt.sqlite3",
            "stack_model_keys": model_keys,
        })

        # Add runner configuration
        config["runner"] = {
            "mode": "entry",
            "model_keys": model_keys,
            "nproc_per_node": nproc_per_node,
            "max_evals": max_evals,
            "plot_curves": False,
            "ft_role": None,
            "use_watchdog": False,
            "idle_seconds": 7200,
            "max_restarts": 50,
            "restart_delay_seconds": 10,
            "incremental_args": [
                "--incremental-dir",
                "./IncrementalBatches",
                "--incremental-template",
                "{model_name}_2025Q1.csv",
                "--merge-keys",
                "policy_id",
                "vehicle_id",
                "--model-keys",
                "glm",
                "xgb",
                "ft",
                "--max-evals",
                "25",
                "--update-base-data"
            ]
        }

        return config

    def build_explain_config(
        self,
        base_config: Dict[str, Any],
        model_keys: Optional[List[str]] = None,
        methods: Optional[List[str]] = None,
        on_train: bool = False,
        permutation_n_repeats: int = 5,
        permutation_max_rows: int = 5000,
        shap_n_background: int = 500,
        shap_n_samples: int = 200,
    ) -> Dict[str, Any]:
        """
        Build or update configuration for explain mode.

        Args:
            base_config: Base configuration dictionary
            model_keys: Models to explain (e.g., ['xgb', 'resn'])
            methods: Explanation methods (e.g., ['permutation', 'shap'])
            on_train: Whether to run on training set (vs validation)
            permutation_n_repeats: Number of repeats for permutation
            permutation_max_rows: Max rows for permutation
            shap_n_background: Background samples for SHAP
            shap_n_samples: Samples for SHAP explanation

        Returns:
            Configuration with explain settings
        """
        config = base_config.copy()

        if model_keys is None:
            model_keys = ["xgb"]
        if methods is None:
            methods = ["permutation"]

        # Set runner mode to explain
        runner = config.get('runner', {})
        runner['mode'] = 'explain'
        config['runner'] = runner

        # Add explain configuration
        explain = {
            "model_keys": model_keys,
            "methods": methods,
            "on_train": on_train,
            "validation_path": None,
            "train_path": None,
            "save_dir": f"{config.get('output_dir', './Results')}/explain",
            "model_dir": None,
            "result_dir": None,
            "permutation": {
                "metric": "auto",
                "n_repeats": permutation_n_repeats,
                "max_rows": permutation_max_rows,
                "random_state": config.get('rand_seed', 13)
            },
            "shap": {
                "n_background": shap_n_background,
                "n_samples": shap_n_samples,
                "save_values": False
            },
            "integrated_gradients": {
                "steps": 50,
                "batch_size": 256,
                "target": None,
                "baseline": None,
                "baseline_num": None,
                "baseline_geo": None,
                "save_values": False
            }
        }
        config['explain'] = explain

        return config

    def validate_config(self, config: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate configuration dictionary.

        Args:
            config: Configuration to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = [
            "data_dir",
            "model_list",
            "target",
            "weight",
            "feature_list"
        ]

        for field in required_fields:
            if field not in config:
                return False, f"Missing required field: {field}"

            if not config[field]:
                return False, f"Empty value for required field: {field}"

        # Validate model_list and model_categories have same length
        if len(config.get("model_list", [])) != len(config.get("model_categories", [])):
            return False, "model_list and model_categories must have the same length"

        # Validate categorical features are subset of features
        features = set(config.get("feature_list", []))
        cat_features = set(config.get("categorical_features", []))
        if not cat_features.issubset(features):
            invalid = cat_features - features
            return False, f"Categorical features not in feature_list: {invalid}"

        return True, "Configuration is valid"
