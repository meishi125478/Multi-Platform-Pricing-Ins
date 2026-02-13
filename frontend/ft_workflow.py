"""
FT-Transformer Two-Step Workflow Helper
Automates the FT -> XGB/ResN two-step training process.
"""

import json
import copy
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
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

    def generate_step2_configs(
        self,
        step1_config_path: str,
        target_models: List[str] = None,
        augmented_data_dir: str = "./DataFTUnsupervised"
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Generate Step 2 configurations for XGB and/or ResN.

        This requires that Step 1 has completed and embeddings are cached.

        Args:
            step1_config_path: Path to the Step 1 config file
            target_models: Models to train in step 2 (e.g., ['xgb', 'resn'])
            augmented_data_dir: Directory to save augmented data with embeddings

        Returns:
            Tuple of (xgb_config, resn_config) - None if not in target_models
        """
        if target_models is None:
            target_models = ['xgb', 'resn']

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

        # Import split function
        try:
            from ins_pricing.cli.utils.cli_common import split_train_test
        except ImportError:
            raise ImportError("Cannot import split_train_test. Ensure ins_pricing is installed.")

        # Split data using same settings as step 1
        holdout_ratio = cfg.get("holdout_ratio", cfg.get("prop_test", 0.25))
        split_strategy = cfg.get("split_strategy", "random")
        split_group_col = cfg.get("split_group_col")
        split_time_col = cfg.get("split_time_col")
        split_time_ascending = cfg.get("split_time_ascending", True)
        rand_seed = cfg.get("rand_seed", 13)

        train_df, test_df = split_train_test(
            raw,
            holdout_ratio=holdout_ratio,
            strategy=split_strategy,
            group_col=split_group_col,
            time_col=split_time_col,
            time_ascending=split_time_ascending,
            rand_seed=rand_seed,
            reset_index_mode="time_group",
            ratio_label="holdout_ratio",
        )

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

        if len(pred_train) != len(train_df) or len(pred_test) != len(test_df):
            raise ValueError(
                "Prediction rows do not match split sizes; check split settings.")

        # Merge embeddings with raw data
        aug = raw.copy()
        aug.loc[train_df.index, pred_train.columns] = pred_train.values
        aug.loc[test_df.index, pred_test.columns] = pred_test.values

        # Save augmented data
        data_out_dir = cfg_path.parent / augmented_data_dir
        data_out_dir.mkdir(parents=True, exist_ok=True)
        aug_path = data_out_dir / f"{model_name}.csv"
        aug.to_csv(aug_path, index=False)

        # Get embedding column names
        embed_cols = list(pred_train.columns)

        # Generate configs
        xgb_config = None
        resn_config = None

        if 'xgb' in target_models:
            xgb_config = self._build_xgb_config(cfg, cfg_path, embed_cols, augmented_data_dir)
            self.step2_configs['xgb'] = xgb_config

        if 'resn' in target_models:
            resn_config = self._build_resn_config(cfg, cfg_path, embed_cols, augmented_data_dir)
            self.step2_configs['resn'] = resn_config

        return xgb_config, resn_config

    def _build_xgb_config(
        self,
        base_cfg: Dict[str, Any],
        cfg_path: Path,
        embed_cols: List[str],
        data_dir: str
    ) -> Dict[str, Any]:
        """Build XGB config for Step 2."""
        xgb_cfg = copy.deepcopy(base_cfg)

        xgb_cfg["data_dir"] = str(data_dir)
        xgb_cfg["feature_list"] = base_cfg["feature_list"] + embed_cols
        xgb_cfg["ft_role"] = "model"
        xgb_cfg["stack_model_keys"] = ["xgb"]
        xgb_cfg["cache_predictions"] = False

        # Disable DDP for XGB
        xgb_cfg["use_resn_ddp"] = False
        xgb_cfg["use_ft_ddp"] = False
        xgb_cfg["use_resn_data_parallel"] = False
        xgb_cfg["use_ft_data_parallel"] = False
        xgb_cfg["use_gnn_data_parallel"] = False

        xgb_cfg["output_dir"] = "./ResultsXGBFromFTUnsupervised"
        xgb_cfg["optuna_storage"] = "./ResultsXGBFromFTUnsupervised/optuna/bayesopt.sqlite3"
        xgb_cfg["optuna_study_prefix"] = "pricing_ft_unsup_xgb"
        xgb_cfg["loss_name"] = "mse"

        runner_cfg = xgb_cfg.get("runner", {})
        runner_cfg["model_keys"] = ["xgb"]
        runner_cfg["nproc_per_node"] = 1
        runner_cfg["plot_curves"] = False
        xgb_cfg["runner"] = runner_cfg

        xgb_cfg["plot_curves"] = False
        plot_cfg = xgb_cfg.get("plot", {})
        plot_cfg["enable"] = False
        xgb_cfg["plot"] = plot_cfg

        return xgb_cfg

    def _build_resn_config(
        self,
        base_cfg: Dict[str, Any],
        cfg_path: Path,
        embed_cols: List[str],
        data_dir: str
    ) -> Dict[str, Any]:
        """Build ResNet config for Step 2."""
        resn_cfg = copy.deepcopy(base_cfg)

        resn_cfg["data_dir"] = str(data_dir)
        resn_cfg["feature_list"] = base_cfg["feature_list"] + embed_cols
        resn_cfg["ft_role"] = "model"
        resn_cfg["stack_model_keys"] = ["resn"]
        resn_cfg["cache_predictions"] = False

        # Enable DDP for ResNet
        resn_cfg["use_resn_ddp"] = True
        resn_cfg["use_ft_ddp"] = False
        resn_cfg["use_resn_data_parallel"] = False
        resn_cfg["use_ft_data_parallel"] = False
        resn_cfg["use_gnn_data_parallel"] = False

        resn_cfg["output_dir"] = "./ResultsResNFromFTUnsupervised"
        resn_cfg["optuna_storage"] = "./ResultsResNFromFTUnsupervised/optuna/bayesopt.sqlite3"
        resn_cfg["optuna_study_prefix"] = "pricing_ft_unsup_resn_ddp"
        resn_cfg["loss_name"] = "mse"

        runner_cfg = resn_cfg.get("runner", {})
        runner_cfg["model_keys"] = ["resn"]
        runner_cfg["nproc_per_node"] = 2
        runner_cfg["plot_curves"] = False
        resn_cfg["runner"] = runner_cfg

        resn_cfg["plot_curves"] = False
        plot_cfg = resn_cfg.get("plot", {})
        plot_cfg["enable"] = False
        resn_cfg["plot"] = plot_cfg

        return resn_cfg

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
