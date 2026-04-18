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

from ins_pricing.frontend.config_builder import ConfigBuilder
from ins_pricing.split_cache import (
    load_split_cache,
    resolve_model_scoped_path,
    validate_split_cache_metadata,
    validate_split_indices,
)


class FTWorkflowHelper:
    """
    Helper for FT-Transformer two-step workflow.

    Step 1: Train FT as embedding generator
    Step 2: Merge embeddings with raw data and train XGB/ResN
    """

    def __init__(self):
        self.step1_config = None
        self.step2_configs = {}
        self._supported_step2_models = {"xgb", "resn"}

    def prepare_step1_config(
        self,
        base_config: Dict[str, Any],
        output_dir: str = "./ResultsFTEmbedDDP",
        ft_feature_prefix: str = "ft_emb",
        ft_role: str = "embedding",
        use_ddp: bool = True,
        nproc_per_node: int = 2,
    ) -> Dict[str, Any]:
        """
        Prepare configuration for Step 1: FT embedding.

        Args:
            base_config: Base configuration dictionary
            output_dir: Output directory for FT embeddings
            ft_feature_prefix: Prefix for embedding column names
            ft_role: FT role for step-1 embeddings.
                Primary value is ``embedding``; ``unsupervised_embedding`` is
                accepted for backward compatibility.
            use_ddp: Whether to use DDP for FT training
            nproc_per_node: Number of processes for DDP

        Returns:
            Step 1 configuration
        """
        config = copy.deepcopy(base_config)

        role = str(ft_role or "embedding").strip().lower()
        valid_roles = {"embedding", "unsupervised_embedding"}
        if role not in valid_roles:
            raise ValueError(
                f"Invalid ft_role for Step 1: {ft_role!r}. "
                f"Expected one of {sorted(valid_roles)}."
            )

        # Set FT role for embedding generation.
        config['ft_role'] = role
        config['ft_feature_prefix'] = ft_feature_prefix
        config['output_dir'] = output_dir
        config['cache_predictions'] = True
        config['prediction_cache_format'] = 'csv'
        # Step-1 embedding flow does not require OHT build.
        config['build_oht'] = False
        config['oht_sparse_csr'] = False
        config['keep_unscaled_oht'] = False

        # Disable other models in step 1
        config['stack_model_keys'] = []

        # DDP settings
        config['use_ft_ddp'] = use_ddp
        config['use_resn_ddp'] = False
        config['use_ft_data_parallel'] = False
        config['use_resn_data_parallel'] = False
        config['use_gnn_data_parallel'] = False

        config['optuna_study_prefix'] = (
            'pricing_ft_embed'
            if role == "embedding"
            else 'pricing_ft_unsup'
        )

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

    def _resolve_prediction_alignment_indices_from_split_cache(
        self,
        raw: pd.DataFrame,
        cfg: Dict[str, Any],
        *,
        cfg_path: Path,
        model_name: str,
        pred_train_rows: int,
        pred_test_rows: int,
    ) -> Optional[Tuple[pd.Index, pd.Index]]:
        prop_test = cfg.get("prop_test", 0.25)
        holdout_ratio = cfg.get("holdout_ratio", prop_test)
        if holdout_ratio is None:
            holdout_ratio = prop_test
        split_strategy = str(cfg.get("split_strategy", "random")).strip().lower()
        split_cache_path = resolve_model_scoped_path(
            cfg.get("split_cache_path"),
            model_name=model_name,
            base_dir=cfg_path.parent,
        )
        if split_cache_path is None:
            return None
        if not split_cache_path.exists():
            # Fall back to split reconstruction when cache path is configured but absent.
            return None

        train_idx, test_idx, cached_row_count, cache_meta = load_split_cache(split_cache_path)
        validate_split_cache_metadata(
            cache_path=split_cache_path,
            cached_row_count=cached_row_count,
            current_row_count=int(len(raw)),
            cache_meta=cache_meta,
            split_strategy=split_strategy,
            holdout_ratio=float(holdout_ratio),
            rand_seed=cfg.get("rand_seed", 13),
        )

        validate_split_indices(
            train_idx=train_idx,
            test_idx=test_idx,
            row_count=int(len(raw)),
            cache_path=split_cache_path,
        )
        if train_idx.size != int(pred_train_rows) or test_idx.size != int(pred_test_rows):
            raise ValueError(
                "Prediction rows do not match split cache sizes. "
                f"cached train/test indices=({train_idx.size}, {test_idx.size}), "
                f"prediction train/test rows=({pred_train_rows}, {pred_test_rows})."
            )
        return raw.index[train_idx], raw.index[test_idx]

    @staticmethod
    def _normalize_table_format(
        value: Optional[str],
        *,
        context: str,
        allow_auto: bool = False,
    ) -> str:
        raw = str(value or "").strip().lower()
        if raw in {"parquet", "pq"}:
            return "parquet"
        if raw in {"feather", "ft"}:
            return "feather"
        if raw in {"csv"}:
            return "csv"
        if allow_auto and raw in {"", "auto"}:
            return "auto"
        valid = ["csv", "parquet", "feather"] + (["auto"] if allow_auto else [])
        raise ValueError(f"{context} must be one of {valid}, got: {value!r}")

    @staticmethod
    def _format_to_ext(fmt: str) -> str:
        if fmt == "parquet":
            return "parquet"
        if fmt == "feather":
            return "feather"
        return "csv"

    @staticmethod
    def _load_table(path: Path, *, data_format: str = "auto") -> pd.DataFrame:
        fmt = FTWorkflowHelper._normalize_table_format(
            data_format,
            context="data_format",
            allow_auto=True,
        )
        if fmt == "auto":
            suffix = path.suffix.lower()
            if suffix in {".csv"}:
                fmt = "csv"
            elif suffix in {".parquet", ".pq"}:
                fmt = "parquet"
            elif suffix in {".feather", ".ft"}:
                fmt = "feather"
            else:
                for candidate_fmt in ("parquet", "feather", "csv"):
                    try:
                        return FTWorkflowHelper._load_table(path, data_format=candidate_fmt)
                    except Exception:
                        continue
                raise ValueError(f"Unable to infer table format for file: {path}")
        if fmt == "csv":
            return pd.read_csv(path, low_memory=False)
        if fmt == "parquet":
            return pd.read_parquet(path)
        if fmt == "feather":
            return pd.read_feather(path)
        raise ValueError(f"Unsupported read data format: {fmt!r}")

    @staticmethod
    def _write_table(df: pd.DataFrame, path: Path, *, data_format: str) -> None:
        if data_format == "csv":
            df.to_csv(path, index=False)
            return
        if data_format == "parquet":
            df.to_parquet(path, index=False)
            return
        if data_format == "feather":
            df.reset_index(drop=True).to_feather(path)
            return
        raise ValueError(f"Unsupported write data format: {data_format!r}")

    @staticmethod
    def _resolve_input_data_path(
        *,
        data_dir: Path,
        model_name: str,
        data_format: str,
        data_path_template: Optional[str],
        label: str,
    ) -> Path:
        fmt = FTWorkflowHelper._normalize_table_format(
            data_format,
            context="data_format",
            allow_auto=True,
        )
        template = str(data_path_template or "").strip()

        def _build_path(target_fmt: str) -> Path:
            ext = FTWorkflowHelper._format_to_ext(target_fmt)
            if template:
                filename = template.format(model_name=model_name, ext=ext)
            else:
                filename = f"{model_name}.{ext}"
            return (data_dir / filename).resolve()

        if fmt == "auto":
            for candidate_fmt in ("csv", "parquet", "feather"):
                candidate = _build_path(candidate_fmt)
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(f"{label} file not found under {data_dir}: {model_name}")

        path = _build_path(fmt)
        if path.exists():
            return path
        for candidate_fmt in ("csv", "parquet", "feather"):
            candidate = _build_path(candidate_fmt)
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"{label} file not found: {path}")

    @staticmethod
    def _resolve_output_data_path(
        *,
        output_dir: Path,
        model_name: str,
        data_format: str,
        data_path_template: str,
    ) -> Path:
        try:
            filename = data_path_template.format(
                model_name=model_name,
                ext=FTWorkflowHelper._format_to_ext(data_format),
            )
        except Exception as exc:
            raise ValueError(
                "Invalid augmented_data_path_template. "
                "Must support {model_name} and/or {ext} placeholders."
            ) from exc
        path = (output_dir / filename).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _to_config_relative_path(path: Path, *, base_dir: Path) -> str:
        resolved = path.resolve()
        base_resolved = base_dir.resolve()
        try:
            rel = resolved.relative_to(base_resolved)
            return f"./{rel.as_posix()}"
        except Exception:
            return str(resolved)

    def _resolve_prediction_cache_paths(
        self,
        *,
        pred_dir: Path,
        model_name: str,
        pred_prefix: str,
        prediction_cache_format: str,
    ) -> Tuple[Path, Path]:
        fmt = self._normalize_table_format(
            prediction_cache_format,
            context="prediction_cache_format",
            allow_auto=True,
        )

        all_formats = ["parquet", "feather", "csv"]
        if fmt == "auto":
            candidates = list(all_formats)
        else:
            candidates = [fmt] + [cand for cand in all_formats if cand != fmt]

        def _find_for_split(split: str) -> Optional[Path]:
            for cand_fmt in candidates:
                ext = self._format_to_ext(cand_fmt)
                path = pred_dir / f"{model_name}_{pred_prefix}_{split}.{ext}"
                if path.exists():
                    return path
            return None

        train_path = _find_for_split("train")
        test_path = _find_for_split("test")
        if train_path is None or test_path is None:
            expected = [
                str(pred_dir / f"{model_name}_{pred_prefix}_{split}.{self._format_to_ext(cand_fmt)}")
                for split in ("train", "test")
                for cand_fmt in candidates
            ]
            raise FileNotFoundError(
                "Embedding files not found. Run Step 1 first.\n"
                + "Expected one of:\n"
                + "\n".join(expected)
            )
        return train_path, test_path

    def generate_step2_configs(
        self,
        step1_config_path: str,
        target_models: List[str] = None,
        augmented_data_dir: str = "./DataFTEmbed",
        augmented_data_format: str = "csv",
        augmented_data_path_template: str = "{model_name}.{ext}",
        use_prediction_cache_splits: bool = False,
        split_data_path_template: str = "{model_name}_{split}.{ext}",
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
            augmented_data_format: Output format for augmented step-2 data
            augmented_data_path_template: Path template under augmented_data_dir
            use_prediction_cache_splits: If True, write compact train/test files
                (with row key + target/weight + required features + embeddings)
                instead of one full augmented dataset copy.
            split_data_path_template: Filename template used when
                use_prediction_cache_splits=True. Supports placeholders:
                {model_name}, {split}, {ext}.
            xgb_overrides: Optional overrides merged into generated XGB Step-2 config
            resn_overrides: Optional overrides merged into generated ResN Step-2 config

        Returns:
            Tuple of (xgb_config, resn_config) - None if not in target_models
        """
        target_models = self._normalize_target_models(target_models)
        aug_data_format = self._normalize_table_format(
            augmented_data_format,
            context="augmented_data_format",
            allow_auto=False,
        )
        aug_data_path_template = str(augmented_data_path_template or "").strip()
        if not aug_data_path_template:
            aug_data_path_template = "{model_name}.{ext}"

        # Load step 1 config
        cfg_path = Path(step1_config_path)
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)

        # Read raw data and split
        model_name = f"{cfg['model_list'][0]}_{cfg['model_categories'][0]}"
        data_dir = (cfg_path.parent / cfg["data_dir"]).resolve()
        raw_data_format = self._normalize_table_format(
            cfg.get("data_format", "csv"),
            context="data_format",
            allow_auto=True,
        )
        raw_data_template = cfg.get("data_path_template")
        raw_path = self._resolve_input_data_path(
            data_dir=data_dir,
            model_name=model_name,
            data_format=raw_data_format,
            data_path_template=raw_data_template,
            label="Data",
        )
        raw = self._load_table(raw_path, data_format="auto")

        # Load cached embeddings
        out_root = (cfg_path.parent / cfg["output_dir"]).resolve()
        pred_prefix = cfg.get("ft_feature_prefix", "ft_emb")
        result_root = out_root / "Results"
        pred_cache_dir = cfg.get("prediction_cache_dir")
        if pred_cache_dir:
            pred_dir = Path(str(pred_cache_dir))
            if not pred_dir.is_absolute():
                pred_dir = result_root / pred_dir
        else:
            pred_dir = result_root / "predictions"
        pred_dir = pred_dir.resolve()

        train_pred_path, test_pred_path = self._resolve_prediction_cache_paths(
            pred_dir=pred_dir,
            model_name=model_name,
            pred_prefix=pred_prefix,
            prediction_cache_format=str(cfg.get("prediction_cache_format", "parquet")),
        )
        pred_train = self._load_table(train_pred_path, data_format="auto")
        pred_test = self._load_table(test_pred_path, data_format="auto")
        split_cache_indices = self._resolve_prediction_alignment_indices_from_split_cache(
            raw,
            cfg,
            cfg_path=cfg_path,
            model_name=model_name,
            pred_train_rows=len(pred_train),
            pred_test_rows=len(pred_test),
        )
        if split_cache_indices is not None:
            train_index, test_index = split_cache_indices
        else:
            train_index, test_index = self._resolve_prediction_alignment_indices(
                raw,
                cfg,
                pred_train_rows=len(pred_train),
                pred_test_rows=len(pred_test),
            )

        data_out_dir = cfg_path.parent / augmented_data_dir
        data_out_dir.mkdir(parents=True, exist_ok=True)
        split_key_col = str(cfg.get("split_cache_key_col") or "_row_id").strip() or "_row_id"
        pred_key_col = None
        if split_key_col in pred_train.columns and split_key_col in pred_test.columns:
            pred_key_col = split_key_col

        embed_cols = [str(col) for col in pred_train.columns if str(col) != pred_key_col]
        if not embed_cols:
            raise ValueError(
                "No embedding prediction columns found in cached FT outputs. "
                f"columns={list(pred_train.columns)}"
            )
        if len(embed_cols) != len(pred_test.columns) - (1 if pred_key_col else 0):
            raise ValueError("Train/test cached embedding column mismatch.")

        step2_train_data_path: Optional[str] = None
        step2_test_data_path: Optional[str] = None

        if use_prediction_cache_splits:
            template = str(split_data_path_template or "").strip() or "{model_name}_{split}.{ext}"
            ext = self._format_to_ext(aug_data_format)

            def _split_path(split: str) -> Path:
                try:
                    filename = template.format(model_name=model_name, split=split, ext=ext)
                except Exception as exc:
                    raise ValueError(
                        "Invalid split_data_path_template. Must support {model_name}, "
                        "{split}, and/or {ext} placeholders."
                    ) from exc
                path = (data_out_dir / filename).resolve()
                path.parent.mkdir(parents=True, exist_ok=True)
                return path

            required_supervision_cols = self._dedup_preserve_order(
                [
                    str(cfg.get("target", "") or "").strip(),
                    str(cfg.get("weight", "") or "").strip(),
                    str(cfg.get("binary_target") or cfg.get("binary_resp_nme") or "").strip(),
                ]
            )
            feature_list_step2, _ = self._resolve_step2_feature_space(cfg, embed_cols)
            raw_required = self._dedup_preserve_order(
                [split_key_col, *required_supervision_cols, *feature_list_step2]
            )
            raw_available = [c for c in raw_required if c in raw.columns]
            train_raw = raw.loc[train_index, raw_available].copy()
            test_raw = raw.loc[test_index, raw_available].copy()

            if pred_key_col is not None and split_key_col in train_raw.columns:
                train_pred_key = pd.Index(pred_train[pred_key_col])
                test_pred_key = pd.Index(pred_test[pred_key_col])
                if not train_pred_key.is_unique or not test_pred_key.is_unique:
                    raise ValueError("Prediction cache key column contains duplicate values.")
                train_pos = train_pred_key.get_indexer(train_raw[split_key_col])
                test_pos = test_pred_key.get_indexer(test_raw[split_key_col])
                if np.any(train_pos < 0) or np.any(test_pos < 0):
                    raise ValueError(
                        f"Cannot align prediction cache rows by key column {split_key_col!r}."
                    )
                pred_train_embed = pred_train.iloc[train_pos][embed_cols].reset_index(drop=True)
                pred_test_embed = pred_test.iloc[test_pos][embed_cols].reset_index(drop=True)
            else:
                if len(train_raw) != len(pred_train) or len(test_raw) != len(pred_test):
                    raise ValueError(
                        "Prediction cache rows do not match split rows; "
                        "cannot align without key column."
                    )
                pred_train_embed = pred_train[embed_cols].reset_index(drop=True)
                pred_test_embed = pred_test[embed_cols].reset_index(drop=True)

            train_out = train_raw.drop(columns=embed_cols, errors="ignore").reset_index(drop=True)
            test_out = test_raw.drop(columns=embed_cols, errors="ignore").reset_index(drop=True)
            train_out = pd.concat([train_out, pred_train_embed], axis=1, copy=False)
            test_out = pd.concat([test_out, pred_test_embed], axis=1, copy=False)

            train_path = _split_path("train")
            test_path = _split_path("test")
            self._write_table(train_out, train_path, data_format=aug_data_format)
            self._write_table(test_out, test_path, data_format=aug_data_format)

            step2_train_data_path = self._to_config_relative_path(
                train_path, base_dir=cfg_path.parent
            )
            step2_test_data_path = self._to_config_relative_path(
                test_path, base_dir=cfg_path.parent
            )
        else:
            # Merge embeddings with one preallocated matrix to reduce intermediate copies.
            train_values = pred_train[embed_cols].to_numpy(dtype=np.float32, copy=False)
            test_values = pred_test[embed_cols].to_numpy(dtype=np.float32, copy=False)
            if train_values.ndim == 1:
                train_values = train_values.reshape(-1, 1)
            if test_values.ndim == 1:
                test_values = test_values.reshape(-1, 1)

            train_pos = raw.index.get_indexer(train_index)
            test_pos = raw.index.get_indexer(test_index)
            if np.any(train_pos < 0) or np.any(test_pos < 0):
                raise ValueError(
                    "Failed to map reconstructed split indices back to raw dataset rows."
                )

            embed_values = np.full((len(raw), len(embed_cols)), np.nan, dtype=np.float32)
            embed_values[train_pos, :] = train_values
            embed_values[test_pos, :] = test_values

            raw_base = raw.drop(columns=embed_cols, errors="ignore").reset_index(drop=True)
            embed_frame = pd.DataFrame(embed_values, columns=embed_cols, copy=False)
            aug = pd.concat([raw_base, embed_frame], axis=1, copy=False)

            aug_path = self._resolve_output_data_path(
                output_dir=data_out_dir,
                model_name=model_name,
                data_format=aug_data_format,
                data_path_template=aug_data_path_template,
            )
            self._write_table(aug, aug_path, data_format=aug_data_format)

        # Generate configs
        xgb_config = None
        resn_config = None

        if 'xgb' in target_models:
            xgb_config = self._build_xgb_config(
                cfg,
                embed_cols,
                augmented_data_dir,
                data_format=aug_data_format,
                data_path_template=aug_data_path_template,
                train_data_path=step2_train_data_path,
                test_data_path=step2_test_data_path,
                overrides=xgb_overrides,
            )
            self.step2_configs['xgb'] = xgb_config

        if 'resn' in target_models:
            resn_config = self._build_resn_config(
                cfg,
                embed_cols,
                augmented_data_dir,
                data_format=aug_data_format,
                data_path_template=aug_data_path_template,
                train_data_path=step2_train_data_path,
                test_data_path=step2_test_data_path,
                overrides=resn_overrides,
            )
            self.step2_configs['resn'] = resn_config

        return xgb_config, resn_config

    def _build_xgb_config(
        self,
        base_cfg: Dict[str, Any],
        embed_cols: List[str],
        data_dir: str,
        data_format: str,
        data_path_template: str,
        train_data_path: Optional[str] = None,
        test_data_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build XGB config for Step 2."""
        return self._build_step2_model_config(
            base_cfg=base_cfg,
            embed_cols=embed_cols,
            data_dir=data_dir,
            data_format=data_format,
            data_path_template=data_path_template,
            train_data_path=train_data_path,
            test_data_path=test_data_path,
            model_key="xgb",
            output_dir="./ResultsXGBFromFTEmbed",
            study_prefix="pricing_ft_embed_xgb",
            runner_nproc=1,
            use_resn_ddp=False,
            build_oht=False,
            final_refit=None,
            overrides=overrides,
        )

    def _build_resn_config(
        self,
        base_cfg: Dict[str, Any],
        embed_cols: List[str],
        data_dir: str,
        data_format: str,
        data_path_template: str,
        train_data_path: Optional[str] = None,
        test_data_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build ResNet config for Step 2."""
        return self._build_step2_model_config(
            base_cfg=base_cfg,
            embed_cols=embed_cols,
            data_dir=data_dir,
            data_format=data_format,
            data_path_template=data_path_template,
            train_data_path=train_data_path,
            test_data_path=test_data_path,
            model_key="resn",
            output_dir="./ResultsResNFromFTEmbed",
            study_prefix="pricing_ft_embed_resn",
            runner_nproc=1,
            use_resn_ddp=False,
            build_oht=False,
            final_refit=None,
            overrides=overrides,
        )

    def _build_step2_model_config(
        self,
        *,
        base_cfg: Dict[str, Any],
        embed_cols: List[str],
        data_dir: str,
        data_format: str,
        data_path_template: str,
        train_data_path: Optional[str],
        test_data_path: Optional[str],
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
        cfg["data_format"] = str(data_format)
        cfg["data_path_template"] = str(data_path_template)
        if train_data_path and test_data_path:
            cfg["train_data_path"] = str(train_data_path)
            cfg["test_data_path"] = str(test_data_path)
            cfg["split_cache_path"] = None
            cfg["split_cache_force_rebuild"] = False
        cfg["feature_list"] = feature_list
        cfg["categorical_features"] = categorical_features
        cfg["ft_role"] = "model"
        cfg["stack_model_keys"] = [model_key]
        cfg["cache_predictions"] = bool(base_cfg.get("cache_predictions", True))
        cfg["prediction_cache_format"] = str(
            base_cfg.get("prediction_cache_format", "csv")
        )

        cfg["use_resn_ddp"] = bool(use_resn_ddp)
        cfg["use_ft_ddp"] = False
        cfg["use_resn_data_parallel"] = False
        cfg["use_ft_data_parallel"] = False
        cfg["use_gnn_data_parallel"] = False

        cfg["output_dir"] = output_dir
        cfg["optuna_study_prefix"] = study_prefix
        cfg["loss_name"] = "mse"
        if build_oht is not None:
            cfg["build_oht"] = bool(build_oht)
        if final_refit is not None:
            cfg["final_refit"] = bool(final_refit)
        if model_key == "resn":
            cfg["resn_use_lazy_dataset"] = bool(cfg.get("resn_use_lazy_dataset", True))
            cfg["dataloader_workers"] = int(cfg.get("dataloader_workers", 0))

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

        # Step-2 should tune the downstream model, not carry Step-1 FT spaces.
        cfg["ft_search_space"] = {}
        cfg["ft_unsupervised_search_space"] = {}
        if model_key == "xgb":
            cfg["xgb_search_space"] = self._resolve_xgb_search_space(cfg)
            cfg["resn_search_space"] = {}
        elif model_key == "resn":
            cfg["resn_search_space"] = self._resolve_resn_search_space(cfg)
            cfg["xgb_search_space"] = {}

        return cfg

    @staticmethod
    def _resolve_xgb_search_space(cfg: Dict[str, Any]) -> Dict[str, Any]:
        search_space = cfg.get("xgb_search_space")
        if isinstance(search_space, dict) and search_space:
            return copy.deepcopy(search_space)
        max_depth_max = int(cfg.get("xgb_max_depth_max", 25))
        n_estimators_max = int(cfg.get("xgb_n_estimators_max", 500))
        return ConfigBuilder._default_xgb_search_space(
            max_depth_max=max_depth_max,
            n_estimators_max=n_estimators_max,
        )

    @staticmethod
    def _resolve_resn_search_space(cfg: Dict[str, Any]) -> Dict[str, Any]:
        search_space = cfg.get("resn_search_space")
        if isinstance(search_space, dict) and search_space:
            return copy.deepcopy(search_space)
        return ConfigBuilder._default_resn_search_space()

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

        def _write_config(path: Path, payload: Dict[str, Any]) -> None:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)

        if self.step1_config:
            step1_path = output_path / "config_ft_step1_embed.json"
            _write_config(step1_path, self.step1_config)
            saved_files['ft_step1'] = str(step1_path)
            step1_legacy_path = output_path / "config_ft_step1_unsupervised.json"
            _write_config(step1_legacy_path, self.step1_config)
            saved_files['ft_step1_legacy'] = str(step1_legacy_path)

        if 'xgb' in self.step2_configs:
            xgb_payload = self.step2_configs['xgb']
            xgb_path = output_path / "config_xgb_from_ft_embed.json"
            _write_config(xgb_path, xgb_payload)
            saved_files['xgb_step2'] = str(xgb_path)
            xgb_legacy_unsup = output_path / "config_xgb_from_ft_unsupervised.json"
            _write_config(xgb_legacy_unsup, xgb_payload)
            xgb_legacy_step2 = output_path / "config_xgb_from_ft_step2.json"
            _write_config(xgb_legacy_step2, xgb_payload)
            saved_files['xgb_step2_legacy_unsupervised'] = str(xgb_legacy_unsup)
            saved_files['xgb_step2_legacy_step2'] = str(xgb_legacy_step2)

        if 'resn' in self.step2_configs:
            resn_payload = self.step2_configs['resn']
            resn_path = output_path / "config_resn_from_ft_embed.json"
            _write_config(resn_path, resn_payload)
            saved_files['resn_step2'] = str(resn_path)
            resn_legacy_unsup = output_path / "config_resn_from_ft_unsupervised.json"
            _write_config(resn_legacy_unsup, resn_payload)
            resn_legacy_step2 = output_path / "config_resn_from_ft_step2.json"
            _write_config(resn_legacy_step2, resn_payload)
            saved_files['resn_step2_legacy_unsupervised'] = str(resn_legacy_unsup)
            saved_files['resn_step2_legacy_step2'] = str(resn_legacy_step2)

        return saved_files
