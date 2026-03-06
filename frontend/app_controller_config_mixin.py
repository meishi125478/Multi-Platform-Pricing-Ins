"""Configuration and path-resolution methods for PricingApp."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

class AppControllerConfigMixin:
    def load_json_config(self, file_path) -> tuple[str, Dict[str, Any], str]:
        """Load configuration from uploaded JSON file."""
        if not file_path:
            return "No file uploaded", {}, ""

        try:
            path = self._resolve_user_path(str(file_path), base_dir=self.working_dir)
            with open(path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.current_config = config
            self.current_config_path = path
            self.current_config_dir = path.parent
            config_json = json.dumps(config, indent=2, ensure_ascii=False)
            return f"Configuration loaded successfully from {path.name}", config, config_json
        except Exception as e:
            return f"Error loading config: {str(e)}", {}, ""

    def set_working_dir(self, working_dir: str) -> tuple[str, str]:
        """Set working directory used for relative paths and generated configs."""
        try:
            raw = str(working_dir or "").strip()
            if not raw:
                resolved = Path.cwd().resolve()
            else:
                candidate = Path(raw).expanduser()
                if not candidate.is_absolute():
                    candidate = (self.working_dir / candidate).resolve()
                else:
                    candidate = candidate.resolve()
                if not candidate.exists():
                    return f"Working directory does not exist: {candidate}", str(self.working_dir)
                if not candidate.is_dir():
                    return f"Working directory is not a folder: {candidate}", str(self.working_dir)
                resolved = candidate

            self.working_dir = resolved
            return f"Working directory set to: {resolved}", str(resolved)
        except Exception as e:
            return f"Error setting working directory: {str(e)}", str(self.working_dir)

    def list_directory_candidates(
        self,
        root_dir: str,
        *,
        max_depth: int = 2,
        max_items: int = 300,
    ) -> tuple[str, list[str], str]:
        """List directories for manual working-directory selection."""
        try:
            raw_root = str(root_dir or "").strip()
            candidate = Path(raw_root).expanduser() if raw_root else self.working_dir
            if not candidate.is_absolute():
                candidate = (self.working_dir / candidate).resolve()
            else:
                candidate = candidate.resolve()

            if not candidate.exists():
                fallback = str(self.working_dir)
                return f"Browse root does not exist: {candidate}", [fallback], fallback
            if not candidate.is_dir():
                fallback = str(self.working_dir)
                return f"Browse root is not a folder: {candidate}", [fallback], fallback

            depth_limit = max(0, int(max_depth))
            item_limit = max(1, int(max_items))
            root = candidate
            root_parts = len(root.parts)
            dirs: list[str] = []

            for current, child_dirs, _ in os.walk(root):
                current_path = Path(current).resolve()
                rel_depth = len(current_path.parts) - root_parts
                dirs.append(str(current_path))

                if rel_depth >= depth_limit:
                    child_dirs[:] = []

                if len(dirs) >= item_limit:
                    break

            dirs = sorted(dict.fromkeys(dirs))
            selected = str(self.working_dir) if str(self.working_dir) in dirs else str(root)
            truncated = len(dirs) >= item_limit
            status = f"Found {len(dirs)} folders under: {root}"
            if truncated:
                status += f" (limited to first {item_limit})"
            return status, dirs, selected
        except Exception as e:
            fallback = str(self.working_dir)
            return f"Error listing folders: {str(e)}", [fallback], fallback

    def _resolve_user_path(self, value: str, *, base_dir: Optional[Path] = None) -> Path:
        raw = str(value or "").strip()
        if not raw:
            raise ValueError("Path value is empty.")
        path = Path(raw).expanduser()
        root = (base_dir or self.working_dir or Path.cwd()).resolve()
        if not path.is_absolute():
            path = (root / path).resolve()
        else:
            path = path.resolve()
        return self._validate_allowed_user_path(path)

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _allowed_user_roots(self) -> list[Path]:
        candidates: list[Optional[Path]] = [
            self.working_dir,
            self.current_config_dir,
            self.current_workflow_config_dir,
            Path.cwd(),
            Path(tempfile.gettempdir()),
        ]
        roots: list[Path] = []
        seen: set[str] = set()
        for item in candidates:
            if item is None:
                continue
            resolved = Path(item).resolve()
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            roots.append(resolved)
        return roots

    def _validate_allowed_user_path(self, path: Path) -> Path:
        roots = self._allowed_user_roots()
        if any(self._is_relative_to(path, root) for root in roots):
            return path
        root_preview = ", ".join(str(root) for root in roots[:4])
        raise ValueError(
            f"Path is outside allowed roots: {path}. Allowed roots include: {root_preview}"
        )

    def _default_base_dir(self, preferred: Optional[Path] = None) -> Path:
        return (preferred or self.working_dir or Path.cwd()).resolve()

    def _resolve_override_path(
        self,
        manual_path: Optional[str],
        uploaded_file: Optional[Any] = None,
    ) -> Optional[str]:
        """Resolve override path from manual textbox or uploaded file path."""
        manual = str(manual_path or "").strip()
        resolved_raw: Optional[str] = manual or None
        if resolved_raw is None:
            if uploaded_file is None:
                return None

            if isinstance(uploaded_file, str):
                path_val = uploaded_file.strip()
                resolved_raw = path_val or None
            elif isinstance(uploaded_file, dict):
                for key in ("path", "name"):
                    value = uploaded_file.get(key)
                    if isinstance(value, str) and value.strip():
                        resolved_raw = value.strip()
                        break
            else:
                # Gradio may return file-like objects in some versions.
                name_attr = getattr(uploaded_file, "name", None)
                if isinstance(name_attr, str) and name_attr.strip():
                    resolved_raw = name_attr.strip()

        if not resolved_raw:
            return None
        resolved = self._resolve_user_path(resolved_raw, base_dir=self.working_dir)
        return str(resolved)

    @staticmethod
    def _parse_json_dict(raw_json: str, field_name: str) -> Dict[str, Any]:
        text = str(raw_json or "").strip()
        if not text:
            return {}
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError(f"{field_name} must be a JSON object.")
        return obj

    @staticmethod
    def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge override into base (override takes precedence)."""
        for key, value in override.items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)
            ):
                AppControllerConfigMixin._deep_merge_dict(base[key], value)
            else:
                base[key] = value
        return base

    @staticmethod
    def _normalize_feature_values(raw: Any) -> list[str]:
        if raw is None:
            return []
        values: list[str] = []
        if isinstance(raw, str):
            values = [x.strip() for x in raw.split(",") if x.strip()]
        elif isinstance(raw, Sequence):
            values = [str(x).strip() for x in raw if str(x).strip()]
        else:
            values = [str(raw).strip()] if str(raw).strip() else []
        seen: set[str] = set()
        deduped: list[str] = []
        for item in values:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    def suggest_oneway_factors(self, cfg_path: str) -> tuple[str, list[str], list[str]]:
        """Load oneway factor candidates from config.feature_list."""
        raw = str(cfg_path or "").strip()
        if not raw:
            return "Plot config path is empty.", [], []

        try:
            path_obj = self._resolve_user_path(raw, base_dir=self.working_dir)
            if not path_obj.exists():
                raw_path = Path(raw).expanduser()
                if not raw_path.is_absolute():
                    examples_candidate = (self.working_dir / "examples" / raw_path).resolve()
                    if examples_candidate.exists():
                        path_obj = examples_candidate
                    else:
                        return f"Config not found: {path_obj}", [], []
                else:
                    return f"Config not found: {path_obj}", [], []

            payload = json.loads(path_obj.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return f"Invalid config format: {path_obj}", [], []

            features = self._normalize_feature_values(payload.get("feature_list"))
            if not features:
                return f"No feature_list found in: {path_obj.name}", [], []

            default_selected = features[:1]
            return (
                f"Loaded {len(features)} factors from {path_obj.name} (default selects first factor).",
                features,
                default_selected,
            )
        except Exception as exc:
            return f"Failed to load oneway factors: {exc}", [], []

    def build_config_from_ui(
        self,
        data_dir: str,
        model_list: str,
        model_categories: str,
        target: str,
        weight: str,
        feature_list: str,
        categorical_features: str,
        task_type: str,
        prop_test: float,
        holdout_ratio: float,
        val_ratio: float,
        split_strategy: str,
        rand_seed: int,
        epochs: int,
        output_dir: str,
        use_gpu: bool,
        model_keys: str,
        max_evals: int,
        xgb_max_depth_max: int,
        xgb_n_estimators_max: int,
        xgb_gpu_id: int,
        xgb_cleanup_per_fold: bool,
        xgb_cleanup_synchronize: bool,
        xgb_use_dmatrix: bool,
        xgb_chunk_size: int,
        xgb_search_space_json: str,
        resn_search_space_json: str,
        ft_search_space_json: str,
        ft_unsupervised_search_space_json: str,
        ft_cleanup_per_fold: bool,
        ft_cleanup_synchronize: bool,
        resn_cleanup_per_fold: bool,
        resn_cleanup_synchronize: bool,
        resn_use_lazy_dataset: bool,
        resn_predict_batch_size: int,
        gnn_cleanup_per_fold: bool,
        gnn_cleanup_synchronize: bool,
        optuna_cleanup_synchronize: bool,
        config_overrides_json: str,
    ) -> tuple[str, str]:
        """Build configuration from UI parameters."""
        try:
            # Parse comma-separated lists
            model_list = [x.strip()
                          for x in model_list.split(',') if x.strip()]
            model_categories = [x.strip()
                                for x in model_categories.split(',') if x.strip()]
            feature_list = [x.strip()
                            for x in feature_list.split(',') if x.strip()]
            categorical_features = [
                x.strip() for x in categorical_features.split(',') if x.strip()]
            model_keys = [x.strip()
                          for x in model_keys.split(',') if x.strip()]
            parsed_xgb_chunk_size: Optional[int] = None
            try:
                chunk_size_val = int(xgb_chunk_size)
            except (TypeError, ValueError):
                chunk_size_val = 0
            if chunk_size_val > 0:
                parsed_xgb_chunk_size = chunk_size_val
            parsed_resn_predict_batch_size: Optional[int] = None
            try:
                resn_pred_bs_val = int(resn_predict_batch_size)
            except (TypeError, ValueError):
                resn_pred_bs_val = 0
            if resn_pred_bs_val > 0:
                parsed_resn_predict_batch_size = resn_pred_bs_val
            xgb_search_space = self._parse_json_dict(
                xgb_search_space_json,
                "xgb_search_space_json",
            )
            resn_search_space = self._parse_json_dict(
                resn_search_space_json,
                "resn_search_space_json",
            )
            ft_search_space = self._parse_json_dict(
                ft_search_space_json,
                "ft_search_space_json",
            )
            ft_unsupervised_search_space = self._parse_json_dict(
                ft_unsupervised_search_space_json,
                "ft_unsupervised_search_space_json",
            )
            config_overrides = self._parse_json_dict(
                config_overrides_json,
                "config_overrides_json",
            )

            config = self.config_builder.build_config(
                data_dir=data_dir,
                model_list=model_list,
                model_categories=model_categories,
                target=target,
                weight=weight,
                feature_list=feature_list,
                categorical_features=categorical_features,
                task_type=task_type,
                prop_test=prop_test,
                holdout_ratio=holdout_ratio,
                val_ratio=val_ratio,
                split_strategy=split_strategy,
                rand_seed=rand_seed,
                epochs=epochs,
                output_dir=output_dir,
                use_gpu=use_gpu,
                model_keys=model_keys,
                max_evals=max_evals,
                xgb_max_depth_max=xgb_max_depth_max,
                xgb_n_estimators_max=xgb_n_estimators_max,
                xgb_gpu_id=xgb_gpu_id,
                xgb_cleanup_per_fold=xgb_cleanup_per_fold,
                xgb_cleanup_synchronize=xgb_cleanup_synchronize,
                xgb_use_dmatrix=xgb_use_dmatrix,
                xgb_chunk_size=parsed_xgb_chunk_size,
                xgb_search_space=xgb_search_space,
                resn_search_space=resn_search_space,
                ft_search_space=ft_search_space,
                ft_unsupervised_search_space=ft_unsupervised_search_space,
                ft_cleanup_per_fold=ft_cleanup_per_fold,
                ft_cleanup_synchronize=ft_cleanup_synchronize,
                resn_cleanup_per_fold=resn_cleanup_per_fold,
                resn_cleanup_synchronize=resn_cleanup_synchronize,
                resn_use_lazy_dataset=resn_use_lazy_dataset,
                resn_predict_batch_size=parsed_resn_predict_batch_size,
                gnn_cleanup_per_fold=gnn_cleanup_per_fold,
                gnn_cleanup_synchronize=gnn_cleanup_synchronize,
                optuna_cleanup_synchronize=optuna_cleanup_synchronize,
            )
            if config_overrides:
                config = self._deep_merge_dict(config, config_overrides)

            is_valid, msg = self.config_builder.validate_config(config)
            if not is_valid:
                return f"Validation failed: {msg}", ""

            self.current_config = config
            self.current_config_path = None
            self.current_config_dir = self.working_dir
            config_json = json.dumps(config, indent=2, ensure_ascii=False)
            return "Configuration built successfully", config_json

        except Exception as e:
            return f"Error building config: {str(e)}", ""

    def save_config(self, config_json: str, filename: str) -> str:
        """Save current configuration to file."""
        if not config_json:
            return "No configuration to save"

        try:
            config_path = self._resolve_user_path(filename, base_dir=self._default_base_dir(self.current_config_dir))
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(json.loads(config_json), f,
                          indent=2, ensure_ascii=False)
            return f"Configuration saved to {config_path}"
        except Exception as e:
            return f"Error saving config: {str(e)}"

    def load_workflow_config(self, file_path) -> tuple[str, str]:
        """Load workflow configuration (plot/predict/compare/pre-oneway)."""
        if not file_path:
            return "No file uploaded", ""

        try:
            path = self._resolve_user_path(str(file_path), base_dir=self.working_dir)
            with open(path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.current_workflow_config = config
            self.current_workflow_config_path = path
            self.current_workflow_config_dir = path.parent
            config_json = json.dumps(config, indent=2, ensure_ascii=False)
            return f"Workflow config loaded from {path.name}", config_json
        except Exception as e:
            return f"Error loading workflow config: {str(e)}", ""

    @staticmethod
    def _to_csv(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(x).strip() for x in value if str(x).strip())
        return str(value)

    def _resolve_path_value(
        self,
        base_dir: Path,
        value: Any,
        field_name: str,
        *,
        required: bool = True,
    ) -> Optional[str]:
        raw = str(value or "").strip()
        if not raw:
            if required:
                raise ValueError(f"{field_name} is required.")
            return None
        path = self._resolve_user_path(raw, base_dir=base_dir)
        return str(path)

    def _run_workflow_from_config(self, config: Dict[str, Any], base_dir: Path) -> str:
        workflow_cfg = config.get("workflow", config)
        if not isinstance(workflow_cfg, dict):
            raise ValueError("workflow must be a JSON object.")
        workflows_module = self._load_workflows_module()

        mode = str(workflow_cfg.get("mode", "")).strip().lower()
        if not mode:
            raise ValueError(
                "workflow.mode is required. Supported modes: "
                "pre_oneway, plot_direct, plot_embed, predict_ft_embed, compare_xgb, compare_resn, compare, double_lift."
            )

        if mode in {"pre_oneway", "pre-oneway", "oneway_pre"}:
            holdout_ratio_raw = workflow_cfg.get("holdout_ratio", 0.25)
            holdout_ratio = None if holdout_ratio_raw is None else float(holdout_ratio_raw)
            output_dir = self._resolve_path_value(
                base_dir, workflow_cfg.get("output_dir"), "output_dir", required=False
            )
            return workflows_module.run_pre_oneway(
                data_path=self._resolve_path_value(base_dir, workflow_cfg.get("data_path"), "data_path"),
                model_name=str(workflow_cfg.get("model_name", "")).strip(),
                target_col=str(workflow_cfg.get("target_col", "")).strip(),
                weight_col=str(workflow_cfg.get("weight_col", "")).strip(),
                feature_list=self._to_csv(workflow_cfg.get("feature_list", "")),
                categorical_features=self._to_csv(workflow_cfg.get("categorical_features", "")),
                n_bins=int(workflow_cfg.get("n_bins", 10)),
                holdout_ratio=holdout_ratio,
                rand_seed=int(workflow_cfg.get("rand_seed", 13)),
                output_dir=output_dir,
            )

        if mode == "plot_direct":
            cfg_path = workflow_cfg.get("cfg_path", workflow_cfg.get("plot_cfg_path"))
            return workflows_module.run_plot_direct(
                cfg_path=self._resolve_path_value(base_dir, cfg_path, "cfg_path"),
                xgb_cfg_path=self._resolve_path_value(base_dir, workflow_cfg.get("xgb_cfg_path"), "xgb_cfg_path"),
                resn_cfg_path=self._resolve_path_value(base_dir, workflow_cfg.get("resn_cfg_path"), "resn_cfg_path"),
                model_search_dir=str(self.working_dir),
            )

        if mode == "plot_embed":
            cfg_path = workflow_cfg.get("cfg_path", workflow_cfg.get("plot_cfg_path"))
            return workflows_module.run_plot_embed(
                cfg_path=self._resolve_path_value(base_dir, cfg_path, "cfg_path"),
                xgb_cfg_path=self._resolve_path_value(base_dir, workflow_cfg.get("xgb_cfg_path"), "xgb_cfg_path"),
                resn_cfg_path=self._resolve_path_value(base_dir, workflow_cfg.get("resn_cfg_path"), "resn_cfg_path"),
                ft_cfg_path=self._resolve_path_value(base_dir, workflow_cfg.get("ft_cfg_path"), "ft_cfg_path"),
                use_runtime_ft_embedding=bool(workflow_cfg.get("use_runtime_ft_embedding", False)),
                model_search_dir=str(self.working_dir),
            )

        if mode in {"predict_ft_embed", "predict"}:
            xgb_cfg_path = self._resolve_path_value(
                base_dir, workflow_cfg.get("xgb_cfg_path"), "xgb_cfg_path", required=False
            )
            resn_cfg_path = self._resolve_path_value(
                base_dir, workflow_cfg.get("resn_cfg_path"), "resn_cfg_path", required=False
            )
            return workflows_module.run_predict_ft_embed(
                ft_cfg_path=self._resolve_path_value(base_dir, workflow_cfg.get("ft_cfg_path"), "ft_cfg_path"),
                xgb_cfg_path=xgb_cfg_path,
                resn_cfg_path=resn_cfg_path,
                input_path=self._resolve_path_value(base_dir, workflow_cfg.get("input_path"), "input_path"),
                output_path=self._resolve_path_value(base_dir, workflow_cfg.get("output_path"), "output_path"),
                model_name=(str(workflow_cfg.get("model_name", "")).strip() or None),
                model_keys=self._to_csv(workflow_cfg.get("model_keys", "")),
                model_search_dir=str(self.working_dir),
            )

        if mode in {"compare_xgb", "compare_resn", "compare"}:
            if mode == "compare_xgb":
                model_key = "xgb"
            elif mode == "compare_resn":
                model_key = "resn"
            else:
                model_key = str(workflow_cfg.get("model_key", "xgb")).strip().lower()
            if model_key not in {"xgb", "resn"}:
                raise ValueError("compare mode only supports model_key in {'xgb', 'resn'}.")

            n_bins_override_raw = workflow_cfg.get("n_bins_override", 10)
            n_bins_override = None if n_bins_override_raw is None else int(n_bins_override_raw)
            return workflows_module.run_compare_ft_embed(
                direct_cfg_path=self._resolve_path_value(base_dir, workflow_cfg.get("direct_cfg_path"), "direct_cfg_path"),
                ft_cfg_path=self._resolve_path_value(base_dir, workflow_cfg.get("ft_cfg_path"), "ft_cfg_path"),
                ft_embed_cfg_path=self._resolve_path_value(base_dir, workflow_cfg.get("ft_embed_cfg_path"), "ft_embed_cfg_path"),
                model_key=model_key,
                label_direct=str(workflow_cfg.get("label_direct", "Direct")).strip(),
                label_ft=str(workflow_cfg.get("label_ft", "FT")).strip(),
                use_runtime_ft_embedding=bool(workflow_cfg.get("use_runtime_ft_embedding", False)),
                n_bins_override=n_bins_override,
                model_search_dir=str(self.working_dir),
            )

        if mode in {"double_lift", "double-lift"}:
            holdout_ratio_raw = workflow_cfg.get("holdout_ratio", 0.0)
            holdout_ratio = None if holdout_ratio_raw is None else float(holdout_ratio_raw)
            output_path = self._resolve_path_value(
                base_dir, workflow_cfg.get("output_path"), "output_path", required=False
            )
            return workflows_module.run_double_lift_from_file(
                data_path=self._resolve_path_value(base_dir, workflow_cfg.get("data_path"), "data_path"),
                pred_col_1=str(workflow_cfg.get("pred_col_1", workflow_cfg.get("pred_col1", ""))).strip(),
                pred_col_2=str(workflow_cfg.get("pred_col_2", workflow_cfg.get("pred_col2", ""))).strip(),
                target_col=str(workflow_cfg.get("target_col", workflow_cfg.get("target", ""))).strip(),
                weight_col=str(workflow_cfg.get("weight_col", workflow_cfg.get("weight", "weights"))).strip(),
                n_bins=int(workflow_cfg.get("n_bins", 10)),
                label1=str(workflow_cfg.get("label1", "")).strip() or None,
                label2=str(workflow_cfg.get("label2", "")).strip() or None,
                pred1_weighted=bool(workflow_cfg.get("pred1_weighted", False)),
                pred2_weighted=bool(workflow_cfg.get("pred2_weighted", False)),
                actual_weighted=bool(workflow_cfg.get("actual_weighted", False)),
                holdout_ratio=holdout_ratio,
                split_strategy=str(workflow_cfg.get("split_strategy", "random")).strip(),
                split_group_col=str(workflow_cfg.get("split_group_col", "")).strip() or None,
                split_time_col=str(workflow_cfg.get("split_time_col", "")).strip() or None,
                split_time_ascending=bool(workflow_cfg.get("split_time_ascending", True)),
                rand_seed=int(workflow_cfg.get("rand_seed", 13)),
                output_path=output_path,
            )

        raise ValueError(
            f"Unsupported workflow mode: {mode}. "
            "Supported: pre_oneway, plot_direct, plot_embed, predict_ft_embed, compare_xgb, compare_resn, compare, double_lift."
        )

