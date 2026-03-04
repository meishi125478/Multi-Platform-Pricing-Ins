"""
Insurance Pricing Model Training Frontend
A Gradio-based web interface for configuring and running insurance pricing models.
"""

import os
import platform
import subprocess
import time
from ins_pricing.frontend.ft_workflow import FTWorkflowHelper
from ins_pricing.frontend.config_builder import ConfigBuilder
import json
import tempfile
import inspect
import importlib.util
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Iterable, Generator, Sequence

def _ensure_repo_root() -> None:
    if __package__ not in {None, ""}:
        return
    if importlib.util.find_spec("ins_pricing") is not None:
        return
    bootstrap_path = Path(__file__).resolve().parents[1] / "cli" / "utils" / "bootstrap.py"
    spec = importlib.util.spec_from_file_location("ins_pricing.cli.utils.bootstrap", bootstrap_path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ensure_repo_root()


_ensure_repo_root()

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("GRADIO_TELEMETRY_ENABLED", "False")
os.environ.setdefault("GRADIO_CHECK_VERSION", "False")
os.environ.setdefault("GRADIO_VERSION_CHECK", "False")




class FrontendDependencyError(RuntimeError):
    pass


def _check_frontend_deps() -> None:
    """Fail fast with a clear message if frontend deps are incompatible."""
    try:
        import gradio  # noqa: F401
    except Exception as exc:
        raise FrontendDependencyError(f"Failed to import gradio: {exc}")

    try:
        import huggingface_hub as hf  # noqa: F401
    except Exception as exc:
        raise FrontendDependencyError(
            f"Failed to import huggingface_hub: {exc}. "
            "Pin version with `pip install 'huggingface_hub<0.24'`."
        )

    if not hasattr(hf, 'HfFolder'):
        raise FrontendDependencyError(
            'Incompatible huggingface_hub detected: missing HfFolder. '
            'Please install `huggingface_hub<0.24`.'
        )


class PricingApp:
    """Main application class for the insurance pricing model tasks interface."""

    def __init__(self):
        self.config_builder = ConfigBuilder()
        self.runner = None
        self.ft_workflow = FTWorkflowHelper()
        self.working_dir: Path = Path.cwd().resolve()
        self.current_config = {}
        self.current_step1_config = None
        self.current_step2_config_paths: Dict[str, str] = {}
        self.current_config_path: Optional[Path] = None
        self.current_config_dir: Optional[Path] = None
        self.current_workflow_config: Dict[str, Any] = {}
        self.current_workflow_config_path: Optional[Path] = None
        self.current_workflow_config_dir: Optional[Path] = None

    @staticmethod
    def _load_workflows_module():
        # Delay workflow imports so the UI can start even if optional
        # scientific stack versions are temporarily inconsistent.
        from ins_pricing.frontend import workflows as workflows_module
        return workflows_module

    def _get_runner(self):
        if self.runner is None:
            from ins_pricing.frontend.runner import TaskRunner
            self.runner = TaskRunner()
        return self.runner

    def load_json_config(self, file_path) -> tuple[str, Dict[str, Any], str]:
        """Load configuration from uploaded JSON file."""
        if not file_path:
            return "No file uploaded", {}, ""

        try:
            path = Path(file_path).resolve()
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
        return path

    def _default_base_dir(self, preferred: Optional[Path] = None) -> Path:
        return (preferred or self.working_dir or Path.cwd()).resolve()

    @staticmethod
    def _resolve_override_path(
        manual_path: Optional[str],
        uploaded_file: Optional[Any] = None,
    ) -> Optional[str]:
        """Resolve override path from manual textbox or uploaded file path."""
        manual = str(manual_path or "").strip()
        if manual:
            return manual
        if uploaded_file is None:
            return None

        if isinstance(uploaded_file, str):
            path_val = uploaded_file.strip()
            return path_val or None

        if isinstance(uploaded_file, dict):
            for key in ("path", "name"):
                value = uploaded_file.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return None

        # Gradio may return file-like objects in some versions.
        name_attr = getattr(uploaded_file, "name", None)
        if isinstance(name_attr, str) and name_attr.strip():
            return name_attr.strip()
        return None

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
                PricingApp._deep_merge_dict(base[key], value)
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
            path = Path(file_path).resolve()
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

    @staticmethod
    def _resolve_path_value(
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
        path = Path(raw)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        else:
            path = path.resolve()
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

    def run_workflow_config_ui(self, workflow_config_json: str) -> Generator[tuple[str, str], None, None]:
        """Run plotting/prediction/compare/pre-oneway from workflow JSON config."""
        try:
            if workflow_config_json:
                workflow_config = json.loads(workflow_config_json)
                base_dir = self._default_base_dir(self.current_workflow_config_dir)
                self.current_workflow_config = workflow_config
            elif self.current_workflow_config:
                workflow_config = self.current_workflow_config
                base_dir = self._default_base_dir(self.current_workflow_config_dir)
            else:
                yield "No workflow configuration provided", ""
                return

            mode = str(workflow_config.get("workflow", workflow_config).get("mode", "unknown")).strip().lower()
            runner = self._get_runner()
            log_generator = runner.run_callable(
                self._run_workflow_from_config,
                workflow_config,
                base_dir,
            )

            full_log = ""
            for log_line in log_generator:
                full_log += log_line + "\n"
                yield f"Workflow [{mode}] in progress...", full_log

            yield f"Workflow [{mode}] completed!", full_log

        except Exception as e:
            error_msg = f"Workflow config execution error: {str(e)}"
            yield error_msg, error_msg

    def run_training(self, config_json: str) -> Generator[tuple[str, str], None, None]:
        """
        Run task (training, explain, plotting, etc.) with the current configuration.

        The task type is automatically detected from config.runner.mode.
        Supported modes: entry (training), explain, incremental, watchdog, etc.
        """
        temp_config_path: Optional[Path] = None
        try:
            if config_json:
                config = json.loads(config_json)
                task_mode = config.get('runner', {}).get('mode', 'entry')
                base_dir = self._default_base_dir(self.current_config_dir)
                fd, temp_path = tempfile.mkstemp(prefix="temp_config_", suffix=".json", dir=base_dir)
                temp_config_path = Path(temp_path)
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)
                config_path = temp_config_path
            elif self.current_config_path and self.current_config_path.exists():
                config_path = self.current_config_path
                config = json.loads(config_path.read_text(encoding="utf-8"))
                task_mode = config.get('runner', {}).get('mode', 'entry')
            elif self.current_config:
                config = self.current_config
                task_mode = config.get('runner', {}).get('mode', 'entry')
                base_dir = self._default_base_dir(self.current_config_dir)
                fd, temp_path = tempfile.mkstemp(prefix="temp_config_", suffix=".json", dir=base_dir)
                temp_config_path = Path(temp_path)
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)
                config_path = temp_config_path
            else:
                yield "No configuration provided", ""
                return

            runner = self._get_runner()
            log_generator = runner.run_task(str(config_path))

            # Collect logs
            full_log = ""
            for log_line in log_generator:
                full_log += log_line + "\n"
                yield f"Task [{task_mode}] in progress...", full_log

            yield f"Task [{task_mode}] completed!", full_log

        except Exception as e:
            error_msg = f"Error during task execution: {str(e)}"
            yield error_msg, error_msg
        finally:
            if temp_config_path is not None:
                try:
                    temp_config_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def prepare_ft_step1(self, config_json: str, use_ddp: bool, nproc: int) -> tuple[str, str]:
        """Prepare FT Step 1 configuration."""
        if not config_json:
            return "No configuration provided", ""

        try:
            config = json.loads(config_json)
            step1_config = self.ft_workflow.prepare_step1_config(
                base_config=config,
                use_ddp=use_ddp,
                nproc_per_node=int(nproc)
            )

            # Save to temp file
            base_dir = self._default_base_dir(self.current_config_dir)
            temp_path = (base_dir / "temp_ft_step1_config.json").resolve()
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(step1_config, f, indent=2)

            self.current_step1_config = str(temp_path)
            step1_json = json.dumps(step1_config, indent=2, ensure_ascii=False)

            return "Step 1 config prepared. Click 'Run Step 1' to train FT embeddings.", step1_json

        except Exception as e:
            return f"Error preparing Step 1 config: {str(e)}", ""

    def prepare_ft_step2(
        self,
        step1_config_path: str,
        target_models: str,
        augmented_data_dir: str,
        xgb_overrides_json: str,
        resn_overrides_json: str,
    ) -> tuple[str, str, str]:
        """Prepare FT Step 2 configurations."""
        if not step1_config_path:
            return "Step 1 config not found. Run Step 1 first.", "", ""

        try:
            step1_path = Path(step1_config_path).expanduser()
            if not step1_path.is_absolute():
                candidate_dirs: Iterable[Path] = (
                    self.current_config_dir,
                    self.working_dir,
                    Path.cwd(),
                )
                resolved_candidate = None
                for root in candidate_dirs:
                    if root is None:
                        continue
                    candidate = (Path(root).resolve() / step1_path).resolve()
                    if candidate.exists():
                        resolved_candidate = candidate
                        break
                if resolved_candidate is None:
                    return "Step 1 config not found. Run Step 1 first.", "", ""
                step1_path = resolved_candidate
            else:
                step1_path = step1_path.resolve()
                if not step1_path.exists():
                    return "Step 1 config not found. Run Step 1 first.", "", ""

            models = [m.strip() for m in target_models.split(',') if m.strip()]
            data_dir_value = str(augmented_data_dir or "").strip() or "./DataFTUnsupervised"
            xgb_overrides = self._parse_json_dict(
                xgb_overrides_json,
                "xgb_overrides_json",
            )
            resn_overrides = self._parse_json_dict(
                resn_overrides_json,
                "resn_overrides_json",
            )
            xgb_cfg, resn_cfg = self.ft_workflow.generate_step2_configs(
                step1_config_path=str(step1_path),
                target_models=models,
                augmented_data_dir=data_dir_value,
                xgb_overrides=xgb_overrides,
                resn_overrides=resn_overrides,
            )
            save_dir = step1_path.parent
            saved_paths: Dict[str, str] = {}

            if xgb_cfg:
                xgb_cfg_path = save_dir / "config_xgb_from_ft_unsupervised.json"
                xgb_cfg_path.write_text(
                    json.dumps(xgb_cfg, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                saved_paths["xgb"] = str(xgb_cfg_path)

            if resn_cfg:
                resn_cfg_path = save_dir / "config_resn_from_ft_unsupervised.json"
                resn_cfg_path.write_text(
                    json.dumps(resn_cfg, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                saved_paths["resn"] = str(resn_cfg_path)

            self.current_step2_config_paths = saved_paths

            status_lines = [
                f"Step 2 configs prepared for: {', '.join(models)}",
                f"Augmented data dir: {data_dir_value}",
            ]
            if "xgb" in saved_paths:
                status_lines.append(f"Saved XGB config: {saved_paths['xgb']}")
            if "resn" in saved_paths:
                status_lines.append(f"Saved ResN config: {saved_paths['resn']}")
            status_msg = "\n".join(status_lines)
            xgb_json = json.dumps(
                xgb_cfg, indent=2, ensure_ascii=False) if xgb_cfg else ""
            resn_json = json.dumps(
                resn_cfg, indent=2, ensure_ascii=False) if resn_cfg else ""

            return status_msg, xgb_json, resn_json

        except FileNotFoundError as e:
            return f"Error: {str(e)}\n\nMake sure Step 1 completed successfully.", "", ""
        except Exception as e:
            return f"Error preparing Step 2 configs: {str(e)}", "", ""

    def open_results_folder(self, config_json: str) -> str:
        """Open the results folder in file explorer."""
        try:
            if config_json:
                config = json.loads(config_json)
                output_dir = config.get('output_dir', './Results')
                out = Path(str(output_dir))
                if out.is_absolute():
                    results_path = out.resolve()
                else:
                    results_path = (self._default_base_dir(self.current_config_dir) / out).resolve()
            elif self.current_config_path and self.current_config_path.exists():
                config = json.loads(
                    self.current_config_path.read_text(encoding="utf-8"))
                output_dir = config.get('output_dir', './Results')
                results_path = (
                    self.current_config_path.parent / output_dir).resolve()
            elif self.current_config:
                output_dir = self.current_config.get('output_dir', './Results')
                out = Path(str(output_dir))
                if out.is_absolute():
                    results_path = out.resolve()
                else:
                    results_path = (self._default_base_dir(self.current_config_dir) / out).resolve()
            else:
                return "No configuration loaded"

            if not results_path.exists():
                return f"Results folder does not exist yet: {results_path}"

            # Open folder based on OS
            system = platform.system()
            if system == "Windows":
                os.startfile(results_path)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", str(results_path)])
            else:  # Linux
                subprocess.run(["xdg-open", str(results_path)])

            return f"Opened folder: {results_path}"

        except Exception as e:
            return f"Error opening folder: {str(e)}"

    def _run_workflow(self, label: str, func: Callable, *args, **kwargs):
        """Run a workflow function and stream logs."""
        try:
            runner = self._get_runner()
            log_generator = runner.run_callable(func, *args, **kwargs)
            full_log = ""
            for log_line in log_generator:
                full_log += log_line + "\n"
                yield f"{label} in progress...", full_log
            yield f"{label} completed!", full_log
        except Exception as e:
            error_msg = f"{label} error: {str(e)}"
            yield error_msg, error_msg

    def _run_workflow_with_preview(
        self,
        label: str,
        func: Callable,
        preview_collector: Callable[[float], list[str]],
        *args,
        **kwargs,
    ):
        """Run a workflow and return status/logs plus generated image previews."""
        try:
            runner = self._get_runner()
            started_at = time.time()
            log_generator = runner.run_callable(func, *args, **kwargs)
            full_log = ""
            for log_line in log_generator:
                full_log += log_line + "\n"
                yield f"{label} in progress...", full_log, []

            previews: list[str] = []
            try:
                previews = preview_collector(started_at)
            except Exception as preview_exc:
                full_log += f"[Warn] Failed to collect image previews: {preview_exc}\n"

            yield f"{label} completed!", full_log, previews
        except Exception as e:
            error_msg = f"{label} error: {str(e)}"
            yield error_msg, error_msg, []

    def _load_json_file(self, path_value: str) -> tuple[Path, Dict[str, Any]]:
        path_obj = self._resolve_user_path(path_value, base_dir=self.working_dir)
        if not path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {path_obj}")
        payload = json.loads(path_obj.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Config must be a JSON object: {path_obj}")
        return path_obj, payload

    @staticmethod
    def _resolve_cfg_output_dir(cfg_obj: Dict[str, Any], cfg_path: Path) -> Path:
        raw_output = str(cfg_obj.get("output_dir", "./Results") or "./Results").strip()
        out_path = Path(raw_output).expanduser()
        if not out_path.is_absolute():
            out_path = (cfg_path.parent / out_path).resolve()
        else:
            out_path = out_path.resolve()
        return out_path

    @staticmethod
    def _collect_png_paths(
        candidates: Iterable[Path],
        *,
        min_mtime: Optional[float] = None,
        limit: int = 40,
    ) -> list[str]:
        ranked: list[tuple[float, str]] = []
        seen: set[str] = set()
        for candidate in candidates:
            path_obj = Path(candidate).resolve()
            if not path_obj.exists():
                continue
            if path_obj.is_file():
                file_iter = [path_obj] if path_obj.suffix.lower() == ".png" else []
            else:
                file_iter = path_obj.rglob("*.png")
            for file_path in file_iter:
                if not file_path.is_file():
                    continue
                resolved = str(file_path.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                try:
                    mtime = file_path.stat().st_mtime
                except OSError:
                    continue
                ranked.append((mtime, resolved))

        ranked.sort(key=lambda x: x[0], reverse=True)
        if min_mtime is not None:
            recent = [path for mtime, path in ranked if mtime >= min_mtime]
            if recent:
                return recent[:limit]
        return [path for _, path in ranked[:limit]]

    def _collect_prediction_plot_images(
        self,
        *,
        cfg_path: str,
        xgb_cfg_path: str,
        resn_cfg_path: str,
        started_at: float,
    ) -> list[str]:
        cfg_file, cfg_obj = self._load_json_file(cfg_path)
        xgb_file, xgb_obj = self._load_json_file(xgb_cfg_path)
        resn_file, resn_obj = self._load_json_file(resn_cfg_path)

        model_list = cfg_obj.get("model_list") or []
        model_categories = cfg_obj.get("model_categories") or []
        model_name = ""
        if model_list and model_categories:
            model_name = f"{model_list[0]}_{model_categories[0]}"

        candidate_roots: list[Path] = []
        for file_path, cfg_item in (
            (cfg_file, cfg_obj),
            (xgb_file, xgb_obj),
            (resn_file, resn_obj),
        ):
            output_root = self._resolve_cfg_output_dir(cfg_item, file_path)
            plot_root = output_root / "plot"
            candidate_roots.append(plot_root)
            if model_name:
                candidate_roots.append(plot_root / model_name)

        return self._collect_png_paths(
            candidate_roots,
            min_mtime=started_at - 1.0,
        )

    def _collect_pre_oneway_images(
        self,
        *,
        data_path: str,
        train_data_path: str,
        test_data_path: str,
        model_name: str,
        output_dir: str,
        started_at: float,
    ) -> list[str]:
        if str(output_dir or "").strip():
            out_dir = self._resolve_user_path(output_dir, base_dir=self.working_dir)
        else:
            base_path: Optional[Path] = None
            if str(train_data_path or "").strip():
                base_path = self._resolve_user_path(train_data_path, base_dir=self.working_dir)
            elif str(test_data_path or "").strip():
                base_path = self._resolve_user_path(test_data_path, base_dir=self.working_dir)
            elif str(data_path or "").strip():
                base_path = self._resolve_user_path(data_path, base_dir=self.working_dir)
            root_dir = base_path.parent if base_path is not None else self.working_dir
            out_dir = (root_dir / "Results" / "plot" / model_name / "oneway" / "pre").resolve()
        return self._collect_png_paths([out_dir], min_mtime=started_at - 1.0)

    def _collect_compare_images(
        self,
        *,
        direct_cfg_path: str,
        started_at: float,
    ) -> list[str]:
        cfg_file, cfg_obj = self._load_json_file(direct_cfg_path)
        output_root = self._resolve_cfg_output_dir(cfg_obj, cfg_file)
        plot_root = output_root / "plot"

        model_list = cfg_obj.get("model_list") or []
        model_categories = cfg_obj.get("model_categories") or []
        candidates: list[Path] = [plot_root]
        if model_list and model_categories:
            model_name = f"{model_list[0]}_{model_categories[0]}"
            candidates.append(plot_root / model_name / "double_lift")

        return self._collect_png_paths(candidates, min_mtime=started_at - 1.0)

    def _collect_double_lift_images(
        self,
        *,
        data_path: str,
        output_path: str,
        started_at: float,
    ) -> list[str]:
        raw_output_path = str(output_path or "").strip()
        if raw_output_path:
            target_path = self._resolve_user_path(raw_output_path, base_dir=self.working_dir)
            candidates = [target_path]
        else:
            data_obj = self._resolve_user_path(data_path, base_dir=self.working_dir)
            candidates = [(data_obj.parent / "Results" / "plot").resolve()]
        return self._collect_png_paths(candidates, min_mtime=started_at - 1.0)

    def run_pre_oneway_ui(
        self,
        data_path: str,
        model_name: str,
        target_col: str,
        weight_col: str,
        feature_list: str,
        oneway_factors: Optional[Sequence[str]],
        categorical_features: str,
        n_bins: int,
        holdout_ratio: float,
        rand_seed: int,
        output_dir: str,
        train_data_path: str,
        test_data_path: str,
    ):
        workflows_module = self._load_workflows_module()
        selected_factors = self._normalize_feature_values(oneway_factors)
        resolved_feature_list = ",".join(selected_factors) if selected_factors else feature_list
        yield from self._run_workflow_with_preview(
            "Pre-Oneway Plot",
            workflows_module.run_pre_oneway,
            lambda started_at: self._collect_pre_oneway_images(
                data_path=data_path,
                train_data_path=train_data_path,
                test_data_path=test_data_path,
                model_name=model_name,
                output_dir=output_dir,
                started_at=started_at,
            ),
            data_path=data_path,
            model_name=model_name,
            target_col=target_col,
            weight_col=weight_col,
            feature_list=resolved_feature_list,
            categorical_features=categorical_features,
            n_bins=n_bins,
            holdout_ratio=holdout_ratio,
            rand_seed=rand_seed,
            output_dir=output_dir or None,
            train_data_path=train_data_path or None,
            test_data_path=test_data_path or None,
        )

    def run_plot_direct_ui(
        self,
        cfg_path: str,
        xgb_cfg_path: str,
        resn_cfg_path: str,
        oneway_factors: Optional[Sequence[str]],
        train_data_path: str,
        test_data_path: str,
        xgb_model_file: Optional[Any] = None,
        resn_model_file: Optional[Any] = None,
        xgb_model_path: Optional[str] = None,
        resn_model_path: Optional[str] = None,
    ):
        workflows_module = self._load_workflows_module()
        selected_factors = self._normalize_feature_values(oneway_factors)
        resolved_xgb_model_path = self._resolve_override_path(
            xgb_model_path, xgb_model_file
        )
        resolved_resn_model_path = self._resolve_override_path(
            resn_model_path, resn_model_file
        )
        yield from self._run_workflow_with_preview(
            "Direct Plot",
            workflows_module.run_plot_direct,
            lambda started_at: self._collect_prediction_plot_images(
                cfg_path=cfg_path,
                xgb_cfg_path=xgb_cfg_path,
                resn_cfg_path=resn_cfg_path,
                started_at=started_at,
            ),
            cfg_path=cfg_path,
            xgb_cfg_path=xgb_cfg_path,
            resn_cfg_path=resn_cfg_path,
            oneway_features=selected_factors or None,
            train_data_path=train_data_path or None,
            test_data_path=test_data_path or None,
            xgb_model_path=resolved_xgb_model_path,
            resn_model_path=resolved_resn_model_path,
            model_search_dir=str(self.working_dir),
        )

    def run_plot_embed_ui(
        self,
        cfg_path: str,
        xgb_cfg_path: str,
        resn_cfg_path: str,
        ft_cfg_path: str,
        use_runtime_ft_embedding: bool,
        oneway_factors: Optional[Sequence[str]],
        train_data_path: str,
        test_data_path: str,
        xgb_model_file: Optional[Any] = None,
        resn_model_file: Optional[Any] = None,
        ft_model_file: Optional[Any] = None,
        xgb_model_path: Optional[str] = None,
        resn_model_path: Optional[str] = None,
        ft_model_path: Optional[str] = None,
    ):
        workflows_module = self._load_workflows_module()
        selected_factors = self._normalize_feature_values(oneway_factors)
        resolved_xgb_model_path = self._resolve_override_path(
            xgb_model_path, xgb_model_file
        )
        resolved_resn_model_path = self._resolve_override_path(
            resn_model_path, resn_model_file
        )
        resolved_ft_model_path = self._resolve_override_path(
            ft_model_path, ft_model_file
        )
        yield from self._run_workflow_with_preview(
            "Embed Plot",
            workflows_module.run_plot_embed,
            lambda started_at: self._collect_prediction_plot_images(
                cfg_path=cfg_path,
                xgb_cfg_path=xgb_cfg_path,
                resn_cfg_path=resn_cfg_path,
                started_at=started_at,
            ),
            cfg_path=cfg_path,
            xgb_cfg_path=xgb_cfg_path,
            resn_cfg_path=resn_cfg_path,
            ft_cfg_path=ft_cfg_path,
            use_runtime_ft_embedding=use_runtime_ft_embedding,
            oneway_features=selected_factors or None,
            train_data_path=train_data_path or None,
            test_data_path=test_data_path or None,
            xgb_model_path=resolved_xgb_model_path,
            resn_model_path=resolved_resn_model_path,
            ft_model_path=resolved_ft_model_path,
            model_search_dir=str(self.working_dir),
        )

    def run_predict_ui(
        self,
        ft_cfg_path: str,
        xgb_cfg_path: str,
        resn_cfg_path: str,
        input_path: str,
        output_path: str,
        model_name: str,
        model_keys: str,
        ft_model_file: Optional[Any] = None,
        xgb_model_file: Optional[Any] = None,
        resn_model_file: Optional[Any] = None,
        ft_model_path: Optional[str] = None,
        xgb_model_path: Optional[str] = None,
        resn_model_path: Optional[str] = None,
    ):
        workflows_module = self._load_workflows_module()
        resolved_ft_model_path = self._resolve_override_path(
            ft_model_path, ft_model_file
        )
        resolved_xgb_model_path = self._resolve_override_path(
            xgb_model_path, xgb_model_file
        )
        resolved_resn_model_path = self._resolve_override_path(
            resn_model_path, resn_model_file
        )
        yield from self._run_workflow(
            "Prediction",
            workflows_module.run_predict_ft_embed,
            ft_cfg_path=ft_cfg_path,
            xgb_cfg_path=xgb_cfg_path or None,
            resn_cfg_path=resn_cfg_path or None,
            input_path=input_path,
            output_path=output_path,
            model_name=model_name or None,
            model_keys=model_keys,
            ft_model_path=resolved_ft_model_path,
            xgb_model_path=resolved_xgb_model_path,
            resn_model_path=resolved_resn_model_path,
            model_search_dir=str(self.working_dir),
        )

    def run_compare_ui(
        self,
        model_key: str,
        direct_cfg_path: str,
        ft_cfg_path: str,
        ft_embed_cfg_path: str,
        label_direct: str,
        label_ft: str,
        use_runtime_ft_embedding: bool,
        n_bins_override: int,
        train_data_path: str,
        test_data_path: str,
        direct_model_file: Optional[Any] = None,
        ft_embed_model_file: Optional[Any] = None,
        ft_model_file: Optional[Any] = None,
        direct_model_path: Optional[str] = None,
        ft_embed_model_path: Optional[str] = None,
        ft_model_path: Optional[str] = None,
    ):
        model_key_norm = str(model_key or "").strip().lower()
        if model_key_norm not in {"xgb", "resn"}:
            raise ValueError("model_key must be one of: xgb, resn.")
        label = "Compare XGB" if model_key_norm == "xgb" else "Compare ResNet"
        resolved_direct_model_path = self._resolve_override_path(
            direct_model_path, direct_model_file
        )
        resolved_ft_embed_model_path = self._resolve_override_path(
            ft_embed_model_path, ft_embed_model_file
        )
        resolved_ft_model_path = self._resolve_override_path(
            ft_model_path, ft_model_file
        )
        yield from self._run_compare_ui(
            model_key=model_key_norm,
            label=label,
            direct_cfg_path=direct_cfg_path,
            ft_cfg_path=ft_cfg_path,
            ft_embed_cfg_path=ft_embed_cfg_path,
            label_direct=label_direct,
            label_ft=label_ft,
            use_runtime_ft_embedding=use_runtime_ft_embedding,
            n_bins_override=n_bins_override,
            train_data_path=train_data_path,
            test_data_path=test_data_path,
            direct_model_path=resolved_direct_model_path or "",
            ft_embed_model_path=resolved_ft_embed_model_path or "",
            ft_model_path=resolved_ft_model_path or "",
        )

    def _run_compare_ui(
        self,
        *,
        model_key: str,
        label: str,
        direct_cfg_path: str,
        ft_cfg_path: str,
        ft_embed_cfg_path: str,
        label_direct: str,
        label_ft: str,
        use_runtime_ft_embedding: bool,
        n_bins_override: int,
        train_data_path: str,
        test_data_path: str,
        direct_model_path: str,
        ft_embed_model_path: str,
        ft_model_path: str,
    ):
        workflows_module = self._load_workflows_module()
        yield from self._run_workflow_with_preview(
            label,
            workflows_module.run_compare_ft_embed,
            lambda started_at: self._collect_compare_images(
                direct_cfg_path=direct_cfg_path,
                started_at=started_at,
            ),
            direct_cfg_path=direct_cfg_path,
            ft_cfg_path=ft_cfg_path,
            ft_embed_cfg_path=ft_embed_cfg_path,
            model_key=model_key,
            label_direct=label_direct,
            label_ft=label_ft,
            use_runtime_ft_embedding=use_runtime_ft_embedding,
            n_bins_override=n_bins_override,
            train_data_path=train_data_path or None,
            test_data_path=test_data_path or None,
            direct_model_path=direct_model_path or None,
            ft_embed_model_path=ft_embed_model_path or None,
            ft_model_path=ft_model_path or None,
            model_search_dir=str(self.working_dir),
        )

    def run_double_lift_ui(
        self,
        data_path: str,
        pred_col_1: str,
        pred_col_2: str,
        target_col: str,
        weight_col: str,
        n_bins: int,
        label1: str,
        label2: str,
        pred1_weighted: bool,
        pred2_weighted: bool,
        actual_weighted: bool,
        holdout_ratio: float,
        split_strategy: str,
        split_group_col: str,
        split_time_col: str,
        split_time_ascending: bool,
        rand_seed: int,
        output_path: str,
    ):
        workflows_module = self._load_workflows_module()
        yield from self._run_workflow_with_preview(
            "Double Lift",
            workflows_module.run_double_lift_from_file,
            lambda started_at: self._collect_double_lift_images(
                data_path=data_path,
                output_path=output_path,
                started_at=started_at,
            ),
            data_path=data_path,
            pred_col_1=pred_col_1,
            pred_col_2=pred_col_2,
            target_col=target_col,
            weight_col=weight_col,
            n_bins=n_bins,
            label1=label1 or None,
            label2=label2 or None,
            pred1_weighted=pred1_weighted,
            pred2_weighted=pred2_weighted,
            actual_weighted=actual_weighted,
            holdout_ratio=holdout_ratio,
            split_strategy=split_strategy,
            split_group_col=split_group_col or None,
            split_time_col=split_time_col or None,
            split_time_ascending=split_time_ascending,
            rand_seed=rand_seed,
            output_path=output_path or None,
        )


