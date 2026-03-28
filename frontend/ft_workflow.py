"""
FT-Transformer Two-Step Workflow Helper
Automates the FT -> XGB/ResN two-step training process.
"""

import json
import copy
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd


class FTWorkflowHelper:
    """
    Helper for FT-Transformer two-step workflow.

    Step 1: Train FT as unsupervised embedding generator
    Step 2: Merge embeddings with raw data and train XGB/ResN
    """

    def __init__(self):
        self.step1_config = None
        self.step2_configs = {}
        self._supported_step2_models = {"xgb", "resn"}

    def prepare_step1_config(
        self,
        base_config: Dict[str, Any],
        output_dir: str = "./ResultsFTUnsupervisedDDP",
        ft_feature_prefix: str = "ft_emb",
        use_ddp: bool = True,
        nproc_per_node: int = 2,
    ) -> Dict[str, Any]:
        """
        Prepare configuration for Step 1: FT unsupervised embedding.

        Args:
            base_config: Base configuration dictionary
            output_dir: Output directory for FT embeddings
            ft_feature_prefix: Prefix for embedding column names
            use_ddp: Whether to use DDP for FT training
            nproc_per_node: Number of processes for DDP

        Returns:
            Step 1 configuration
        """
        config = copy.deepcopy(base_config)

        # Set FT role to unsupervised embedding
        config['ft_role'] = 'unsupervised_embedding'
        config['ft_feature_prefix'] = ft_feature_prefix
        config['output_dir'] = output_dir
        config['cache_predictions'] = True
        config['prediction_cache_format'] = 'csv'

        # Disable other models in step 1
        config['stack_model_keys'] = []

        # DDP settings
        config['use_ft_ddp'] = use_ddp
        config['use_resn_ddp'] = False
        config['use_ft_data_parallel'] = False
        config['use_resn_data_parallel'] = False
        config['use_gnn_data_parallel'] = False

        # Optuna storage
        config['optuna_storage'] = f"{output_dir}/optuna/bayesopt.sqlite3"
        config['optuna_study_prefix'] = 'pricing_ft_unsup'

        # Runner config
        runner = config.get('runner', {})
        runner['mode'] = 'entry'
        runner['model_keys'] = ['ft']
        runner['nproc_per_node'] = nproc_per_node if use_ddp else 1
        runner['plot_curves'] = False
        config['runner'] = runner

        # Disable plotting
        config['plot_curves'] = False
        plot_cfg = config.get('plot', {})
        plot_cfg['enable'] = False
        config['plot'] = plot_cfg

        self.step1_config = config
        return config

    def _resolve_prediction_alignment_indices(
        self,
        raw: pd.DataFrame,
        cfg: Dict[str, Any],
        *,
        pred_train_rows: int,
        pred_test_rows: int,
    ) -> Tuple[pd.Index, pd.Index]:
        """Rebuild the Step 1 holdout split so cached embeddings can be reattached.

        Falls back to the streaming random split mask used by the runtime when
        `memory_saving` auto-enables `stream_split_csv`.
        """
        try:
            from ins_pricing.cli.utils.cli_common import (
                split_train_test,
                resolve_split_config,
            )
        except ImportError:
            raise ImportError(
                "Cannot import split_train_test/resolve_split_config. "
                "Ensure ins_pricing is installed."
            )

        split_cfg = resolve_split_config(cfg)
        holdout_ratio = float(split_cfg["holdout_ratio"])
        split_strategy = str(split_cfg["split_strategy"] or "random").strip().lower()
        split_group_col = split_cfg["split_group_col"]
        split_time_col = split_cfg["split_time_col"]
        split_time_ascending = bool(split_cfg["split_time_ascending"])
        rand_seed = cfg.get("rand_seed", 13)

        # Keep original row indices for embedding reattachment.
        train_df, test_df = split_train_test(
            raw,
            holdout_ratio=holdout_ratio,
            strategy=split_strategy,
            group_col=split_group_col,
            time_col=split_time_col,
            time_ascending=split_time_ascending,
            rand_seed=rand_seed,
            reset_index_mode="none",
            ratio_label="holdout_ratio",
        )
        if len(train_df) == pred_train_rows and len(test_df) == pred_test_rows:
            return train_df.index, test_df.index

        if split_strategy == "random":
            # Runtime may auto-enable stream_split_csv for memory_saving profile.
            rng = np.random.default_rng(rand_seed if rand_seed is not None else 13)
            mask_test = rng.random(len(raw)) < holdout_ratio
            train_index = raw.index[~mask_test]
            test_index = raw.index[mask_test]
            if len(train_index) == pred_train_rows and len(test_index) == pred_test_rows:
                return train_index, test_index

        raise ValueError(
            "Prediction rows do not match reconstructed split sizes. "
            f"cached train/test=({pred_train_rows}, {pred_test_rows}), "
            f"reconstructed train/test=({len(train_df)}, {len(test_df)}). "
            "Check split settings (and whether Step 1 used stream_split_csv or "
            "memory_saving auto-stream split)."
        )

    def generate_step2_configs(
        self,
        step1_config_path: str,
        target_models: List[str] = None,
        augmented_data_dir: str = "./DataFTUnsupervised",
        xgb_overrides: Optional[Dict[str, Any]] = None,
        resn_overrides: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Generate Step 2 configurations for XGB and/or ResN.

        This requires that Step 1 has completed and embeddings are cached.

        Args:
            step1_config_path: Path to the Step 1 config file
            target_models: Models to train in step 2 (e.g., ['xgb', 'resn'])
            augmented_data_dir: Directory to save augmented data with embeddings
            xgb_overrides: Optional overrides merged into generated XGB Step-2 config
            resn_overrides: Optional overrides merged into generated ResN Step-2 config

        Returns:
            Tuple of (xgb_config, resn_config) - None if not in target_models
        """
        target_models = self._normalize_target_models(target_models)

        # Load step 1 config
        cfg_path = Path(step1_config_path)
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)

        # Read raw data and split
        model_name = f"{cfg['model_list'][0]}_{cfg['model_categories'][0]}"
        data_dir = (cfg_path.parent / cfg["data_dir"]).resolve()
        raw_path = data_dir / f"{model_name}.csv"

        if not raw_path.exists():
            raise FileNotFoundError(f"Data file not found: {raw_path}")

        raw = pd.read_csv(raw_path)

        # Load cached embeddings
        out_root = (cfg_path.parent / cfg["output_dir"]).resolve()
        pred_prefix = cfg.get("ft_feature_prefix", "ft_emb")
        pred_dir = out_root / "Results" / "predictions"

        train_pred_path = pred_dir / f"{model_name}_{pred_prefix}_train.csv"
        test_pred_path = pred_dir / f"{model_name}_{pred_prefix}_test.csv"

        if not train_pred_path.exists() or not test_pred_path.exists():
            raise FileNotFoundError(
                f"Embedding files not found. Run Step 1 first.\n"
                f"Expected: {train_pred_path} and {test_pred_path}"
            )

        pred_train = pd.read_csv(train_pred_path)
        pred_test = pd.read_csv(test_pred_path)
        train_index, test_index = self._resolve_prediction_alignment_indices(
            raw,
            cfg,
            pred_train_rows=len(pred_train),
            pred_test_rows=len(pred_test),
        )

        # Merge embeddings with raw data without duplicating all raw columns first.
        embed_cols = list(pred_train.columns)
        emb_full = pd.DataFrame(
            np.nan,
            index=raw.index,
            columns=embed_cols,
            dtype=np.float32,
        )
        emb_full.loc[train_index, embed_cols] = pred_train.to_numpy(
            dtype=np.float32, copy=False
        )
        emb_full.loc[test_index, embed_cols] = pred_test.to_numpy(
            dtype=np.float32, copy=False
        )
        raw_base = raw.drop(columns=embed_cols, errors="ignore")
        aug = pd.concat(
            [raw_base.reset_index(drop=True), emb_full.reset_index(drop=True)],
            axis=1,
            copy=False,
        )

        # Save augmented data
        data_out_dir = cfg_path.parent / augmented_data_dir
        data_out_dir.mkdir(parents=True, exist_ok=True)
        aug_path = data_out_dir / f"{model_name}.csv"
        aug.to_csv(aug_path, index=False)

        # Generate configs
        xgb_config = None
        resn_config = None

        if 'xgb' in target_models:
            xgb_config = self._build_xgb_config(
                cfg,
                embed_cols,
                augmented_data_dir,
                overrides=xgb_overrides,
            )
            self.step2_configs['xgb'] = xgb_config

        if 'resn' in target_models:
            resn_config = self._build_resn_config(
                cfg,
                embed_cols,
                augmented_data_dir,
                overrides=resn_overrides,
            )
            self.step2_configs['resn'] = resn_config

        return xgb_config, resn_config

    def _build_xgb_config(
        self,
        base_cfg: Dict[str, Any],
        embed_cols: List[str],
        data_dir: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build XGB config for Step 2."""
        return self._build_step2_model_config(
            base_cfg=base_cfg,
            embed_cols=embed_cols,
            data_dir=data_dir,
            model_key="xgb",
            output_dir="./ResultsXGBFromFTUnsupervised",
            study_prefix="pricing_ft_unsup_xgb",
            runner_nproc=1,
            use_resn_ddp=False,
            build_oht=False,
            final_refit=False,
            overrides=overrides,
        )

    def _build_resn_config(
        self,
        base_cfg: Dict[str, Any],
        embed_cols: List[str],
        data_dir: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build ResNet config for Step 2."""
        return self._build_step2_model_config(
            base_cfg=base_cfg,
            embed_cols=embed_cols,
            data_dir=data_dir,
            model_key="resn",
            output_dir="./ResultsResNFromFTUnsupervised",
            study_prefix="pricing_ft_unsup_resn_ddp",
            runner_nproc=2,
            use_resn_ddp=True,
            build_oht=True,
            final_refit=None,
            overrides=overrides,
        )

    def _build_step2_model_config(
        self,
        *,
        base_cfg: Dict[str, Any],
        embed_cols: List[str],
        data_dir: str,
        model_key: str,
        output_dir: str,
        study_prefix: str,
        runner_nproc: int,
        use_resn_ddp: bool,
        build_oht: Optional[bool],
        final_refit: Optional[bool],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cfg = copy.deepcopy(base_cfg)
        feature_list, categorical_features = self._resolve_step2_feature_space(
            base_cfg, embed_cols
        )

        cfg["data_dir"] = str(data_dir)
        cfg["feature_list"] = feature_list
        cfg["categorical_features"] = categorical_features
        cfg["ft_role"] = "model"
        cfg["stack_model_keys"] = [model_key]
        cfg["cache_predictions"] = False

        cfg["use_resn_ddp"] = bool(use_resn_ddp)
        cfg["use_ft_ddp"] = False
        cfg["use_resn_data_parallel"] = False
        cfg["use_ft_data_parallel"] = False
        cfg["use_gnn_data_parallel"] = False

        cfg["output_dir"] = output_dir
        cfg["optuna_storage"] = f"{output_dir}/optuna/bayesopt.sqlite3"
        cfg["optuna_study_prefix"] = study_prefix
        cfg["loss_name"] = "mse"
        if build_oht is not None:
            cfg["build_oht"] = bool(build_oht)
        if final_refit is not None:
            cfg["final_refit"] = bool(final_refit)

        runner_cfg = dict(cfg.get("runner", {}) or {})
        runner_cfg["model_keys"] = [model_key]
        runner_cfg["nproc_per_node"] = int(runner_nproc)
        runner_cfg["plot_curves"] = False
        cfg["runner"] = runner_cfg

        cfg["plot_curves"] = False
        plot_cfg = dict(cfg.get("plot", {}) or {})
        plot_cfg["enable"] = False
        cfg["plot"] = plot_cfg

        if overrides:
            self._deep_update(cfg, overrides)

        return cfg

    @staticmethod
    def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge updates into base dictionary in-place."""
        for key, value in (updates or {}).items():
            if (
                isinstance(value, dict)
                and isinstance(base.get(key), dict)
            ):
                FTWorkflowHelper._deep_update(base[key], value)
            else:
                base[key] = value
        return base

    def _normalize_target_models(self, target_models: Optional[List[str]]) -> List[str]:
        if target_models is None:
            return ["xgb", "resn"]

        normalized = []
        seen = set()
        invalid = []
        for model in target_models:
            key = str(model or "").strip().lower()
            if not key:
                continue
            if key not in self._supported_step2_models:
                invalid.append(key)
                continue
            if key in seen:
                continue
            seen.add(key)
            normalized.append(key)

        if invalid:
            supported = ", ".join(sorted(self._supported_step2_models))
            raise ValueError(
                f"Unsupported Step 2 models: {invalid}. Supported values: {supported}."
            )
        if not normalized:
            raise ValueError("No valid target_models provided for Step 2 config generation.")
        return normalized

    @staticmethod
    def _dedup_preserve_order(values: List[str]) -> List[str]:
        """Deduplicate values while preserving first-seen order."""
        seen = set()
        ordered = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered

    def _resolve_step2_feature_space(
        self,
        base_cfg: Dict[str, Any],
        embed_cols: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Resolve Step-2 feature/categorical lists after FT embedding.

        By default FT embeddings are assumed to be generated from all base features.
        Override this via optional config key `ft_embedding_source_features` when
        embeddings are generated from only a subset of raw features.
        """
        base_features = list(base_cfg.get("feature_list", []) or [])
        embed_source_raw = base_cfg.get("ft_embedding_source_features")
        if embed_source_raw is None:
            embed_source_features = base_features
        else:
            embed_source_features = [str(col) for col in (embed_source_raw or [])]

        embed_source_set = set(embed_source_features)
        remaining_raw_features = [
            col for col in base_features if col not in embed_source_set
        ]

        feature_list = self._dedup_preserve_order(
            remaining_raw_features + list(embed_cols)
        )

        base_categorical = list(base_cfg.get("categorical_features", []) or [])
        remaining_raw_set = set(remaining_raw_features)
        categorical_features = [
            col for col in base_categorical if col in remaining_raw_set
        ]
        categorical_features = self._dedup_preserve_order(categorical_features)

        return feature_list, categorical_features

    def save_configs(self, output_dir: str = ".") -> Dict[str, str]:
        """
        Save generated configs to files.

        Args:
            output_dir: Directory to save config files

        Returns:
            Dictionary mapping model names to saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        if self.step1_config:
            step1_path = output_path / "config_ft_step1_unsupervised.json"
            with open(step1_path, 'w', encoding='utf-8') as f:
                json.dump(self.step1_config, f, indent=2)
            saved_files['ft_step1'] = str(step1_path)

        if 'xgb' in self.step2_configs:
            xgb_path = output_path / "config_xgb_from_ft_step2.json"
            with open(xgb_path, 'w', encoding='utf-8') as f:
                json.dump(self.step2_configs['xgb'], f, indent=2)
            saved_files['xgb_step2'] = str(xgb_path)

        if 'resn' in self.step2_configs:
            resn_path = output_path / "config_resn_from_ft_step2.json"
            with open(resn_path, 'w', encoding='utf-8') as f:
                json.dump(self.step2_configs['resn'], f, indent=2)
            saved_files['resn_step2'] = str(resn_path)

        return saved_files
