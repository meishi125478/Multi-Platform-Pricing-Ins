"""
Insurance Pricing Model Training Frontend
A Gradio-based web interface for configuring and running insurance pricing models.
"""

import os
import platform
import subprocess
from ins_pricing.frontend.workflows import (
    run_compare_ft_embed,
    run_double_lift_from_file,
    run_plot_direct,
    run_plot_embed,
    run_predict_ft_embed,
    run_pre_oneway,
)
from ins_pricing.frontend.ft_workflow import FTWorkflowHelper
from ins_pricing.frontend.runner import TaskRunner
from ins_pricing.frontend.config_builder import ConfigBuilder
import json
import tempfile
import inspect
import importlib.util
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Iterable, Generator

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
        self.runner = TaskRunner()
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
    def _parse_json_dict(raw_json: str, field_name: str) -> Dict[str, Any]:
        text = str(raw_json or "").strip()
        if not text:
            return {}
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError(f"{field_name} must be a JSON object.")
        return obj

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
            return run_pre_oneway(
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
            return run_plot_direct(
                cfg_path=self._resolve_path_value(base_dir, cfg_path, "cfg_path"),
                xgb_cfg_path=self._resolve_path_value(base_dir, workflow_cfg.get("xgb_cfg_path"), "xgb_cfg_path"),
                resn_cfg_path=self._resolve_path_value(base_dir, workflow_cfg.get("resn_cfg_path"), "resn_cfg_path"),
            )

        if mode == "plot_embed":
            cfg_path = workflow_cfg.get("cfg_path", workflow_cfg.get("plot_cfg_path"))
            return run_plot_embed(
                cfg_path=self._resolve_path_value(base_dir, cfg_path, "cfg_path"),
                xgb_cfg_path=self._resolve_path_value(base_dir, workflow_cfg.get("xgb_cfg_path"), "xgb_cfg_path"),
                resn_cfg_path=self._resolve_path_value(base_dir, workflow_cfg.get("resn_cfg_path"), "resn_cfg_path"),
                ft_cfg_path=self._resolve_path_value(base_dir, workflow_cfg.get("ft_cfg_path"), "ft_cfg_path"),
                use_runtime_ft_embedding=bool(workflow_cfg.get("use_runtime_ft_embedding", False)),
            )

        if mode in {"predict_ft_embed", "predict"}:
            xgb_cfg_path = self._resolve_path_value(
                base_dir, workflow_cfg.get("xgb_cfg_path"), "xgb_cfg_path", required=False
            )
            resn_cfg_path = self._resolve_path_value(
                base_dir, workflow_cfg.get("resn_cfg_path"), "resn_cfg_path", required=False
            )
            return run_predict_ft_embed(
                ft_cfg_path=self._resolve_path_value(base_dir, workflow_cfg.get("ft_cfg_path"), "ft_cfg_path"),
                xgb_cfg_path=xgb_cfg_path,
                resn_cfg_path=resn_cfg_path,
                input_path=self._resolve_path_value(base_dir, workflow_cfg.get("input_path"), "input_path"),
                output_path=self._resolve_path_value(base_dir, workflow_cfg.get("output_path"), "output_path"),
                model_name=(str(workflow_cfg.get("model_name", "")).strip() or None),
                model_keys=self._to_csv(workflow_cfg.get("model_keys", "")),
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
            return run_compare_ft_embed(
                direct_cfg_path=self._resolve_path_value(base_dir, workflow_cfg.get("direct_cfg_path"), "direct_cfg_path"),
                ft_cfg_path=self._resolve_path_value(base_dir, workflow_cfg.get("ft_cfg_path"), "ft_cfg_path"),
                ft_embed_cfg_path=self._resolve_path_value(base_dir, workflow_cfg.get("ft_embed_cfg_path"), "ft_embed_cfg_path"),
                model_key=model_key,
                label_direct=str(workflow_cfg.get("label_direct", "Direct")).strip(),
                label_ft=str(workflow_cfg.get("label_ft", "FT")).strip(),
                use_runtime_ft_embedding=bool(workflow_cfg.get("use_runtime_ft_embedding", False)),
                n_bins_override=n_bins_override,
            )

        if mode in {"double_lift", "double-lift"}:
            holdout_ratio_raw = workflow_cfg.get("holdout_ratio", 0.0)
            holdout_ratio = None if holdout_ratio_raw is None else float(holdout_ratio_raw)
            output_path = self._resolve_path_value(
                base_dir, workflow_cfg.get("output_path"), "output_path", required=False
            )
            return run_double_lift_from_file(
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
            log_generator = self.runner.run_callable(
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

            log_generator = self.runner.run_task(str(config_path))

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
            log_generator = self.runner.run_callable(func, *args, **kwargs)
            full_log = ""
            for log_line in log_generator:
                full_log += log_line + "\n"
                yield f"{label} in progress...", full_log
            yield f"{label} completed!", full_log
        except Exception as e:
            error_msg = f"{label} error: {str(e)}"
            yield error_msg, error_msg

    def run_pre_oneway_ui(
        self,
        data_path: str,
        model_name: str,
        target_col: str,
        weight_col: str,
        feature_list: str,
        categorical_features: str,
        n_bins: int,
        holdout_ratio: float,
        rand_seed: int,
        output_dir: str,
    ):
        yield from self._run_workflow(
            "Pre-Oneway Plot",
            run_pre_oneway,
            data_path=data_path,
            model_name=model_name,
            target_col=target_col,
            weight_col=weight_col,
            feature_list=feature_list,
            categorical_features=categorical_features,
            n_bins=n_bins,
            holdout_ratio=holdout_ratio,
            rand_seed=rand_seed,
            output_dir=output_dir or None,
        )

    def run_plot_direct_ui(self, cfg_path: str, xgb_cfg_path: str, resn_cfg_path: str):
        yield from self._run_workflow(
            "Direct Plot",
            run_plot_direct,
            cfg_path=cfg_path,
            xgb_cfg_path=xgb_cfg_path,
            resn_cfg_path=resn_cfg_path,
        )

    def run_plot_embed_ui(
        self,
        cfg_path: str,
        xgb_cfg_path: str,
        resn_cfg_path: str,
        ft_cfg_path: str,
        use_runtime_ft_embedding: bool,
    ):
        yield from self._run_workflow(
            "Embed Plot",
            run_plot_embed,
            cfg_path=cfg_path,
            xgb_cfg_path=xgb_cfg_path,
            resn_cfg_path=resn_cfg_path,
            ft_cfg_path=ft_cfg_path,
            use_runtime_ft_embedding=use_runtime_ft_embedding,
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
    ):
        yield from self._run_workflow(
            "Prediction",
            run_predict_ft_embed,
            ft_cfg_path=ft_cfg_path,
            xgb_cfg_path=xgb_cfg_path or None,
            resn_cfg_path=resn_cfg_path or None,
            input_path=input_path,
            output_path=output_path,
            model_name=model_name or None,
            model_keys=model_keys,
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
    ):
        model_key_norm = str(model_key or "").strip().lower()
        if model_key_norm not in {"xgb", "resn"}:
            raise ValueError("model_key must be one of: xgb, resn.")
        label = "Compare XGB" if model_key_norm == "xgb" else "Compare ResNet"
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
    ):
        yield from self._run_workflow(
            label,
            run_compare_ft_embed,
            direct_cfg_path=direct_cfg_path,
            ft_cfg_path=ft_cfg_path,
            ft_embed_cfg_path=ft_embed_cfg_path,
            model_key=model_key,
            label_direct=label_direct,
            label_ft=label_ft,
            use_runtime_ft_embedding=use_runtime_ft_embedding,
            n_bins_override=n_bins_override,
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
        yield from self._run_workflow(
            "Double Lift",
            run_double_lift_from_file,
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


def create_ui():
    """Create the Gradio interface."""
    _check_frontend_deps()
    import gradio as gr

    app = PricingApp()
    xgb_search_space_template = json.dumps(
        app.config_builder._default_xgb_search_space(),
        indent=2,
        ensure_ascii=False,
    )
    resn_search_space_template = json.dumps(
        app.config_builder._default_resn_search_space(),
        indent=2,
        ensure_ascii=False,
    )
    ft_search_space_template = json.dumps(
        app.config_builder._default_ft_search_space(),
        indent=2,
        ensure_ascii=False,
    )
    ft_unsupervised_search_space_template = json.dumps(
        app.config_builder._default_ft_unsupervised_search_space(),
        indent=2,
        ensure_ascii=False,
    )
    workflow_template = json.dumps(
        {
            "workflow": {
                "mode": "plot_direct",
                "cfg_path": "config_plot.json",
                "xgb_cfg_path": "config_xgb_direct.json",
                "resn_cfg_path": "config_resn_direct.json",
            }
        },
        indent=2,
        ensure_ascii=False,
    )
    xgb_step2_overrides_template = json.dumps(
        {
            "output_dir": "./ResultsXGBFromFTUnsupervised",
            "optuna_storage": "./ResultsXGBFromFTUnsupervised/optuna/bayesopt.sqlite3",
            "optuna_study_prefix": "pricing_ft_unsup_xgb",
            "loss_name": "mse",
            "build_oht": False,
            "final_refit": False,
            "runner": {
                "model_keys": ["xgb"],
                "nproc_per_node": 1,
                "plot_curves": False,
            },
            "plot_curves": False,
            "plot": {"enable": False},
        },
        indent=2,
        ensure_ascii=False,
    )
    resn_step2_overrides_template = json.dumps(
        {
            "use_resn_ddp": True,
            "output_dir": "./ResultsResNFromFTUnsupervised",
            "optuna_storage": "./ResultsResNFromFTUnsupervised/optuna/bayesopt.sqlite3",
            "optuna_study_prefix": "pricing_ft_unsup_resn_ddp",
            "loss_name": "mse",
            "build_oht": True,
            "runner": {
                "model_keys": ["resn"],
                "nproc_per_node": 2,
                "plot_curves": False,
            },
            "plot_curves": False,
            "plot": {"enable": False},
        },
        indent=2,
        ensure_ascii=False,
    )
    dir_status_init, dir_choices_init, dir_value_init = app.list_directory_candidates(
        str(app.working_dir)
    )

    def _set_working_dir_ui(path_text: str):
        status, resolved = app.set_working_dir(path_text)
        _, choices, selected = app.list_directory_candidates(resolved)
        return (
            status,
            resolved,
            resolved,
            gr.update(choices=choices, value=selected),
        )

    def _refresh_working_dir_choices_ui(root_dir: str):
        status, choices, selected = app.list_directory_candidates(root_dir)
        return status, gr.update(choices=choices, value=selected)

    def _suggest_compare_defaults(model_key: str):
        key = str(model_key or "").strip().lower()
        if key == "resn":
            return (
                "config_resn_direct.json",
                "config_resn_from_ft_unsupervised.json",
                "ResN_raw",
                "ResN_ft_embed",
            )
        return (
            "config_xgb_direct.json",
            "config_xgb_from_ft_unsupervised.json",
            "XGB_raw",
            "XGB_ft_embed",
        )

    layout_css = """
    .gradio-container {
        max-width: 1480px !important;
        margin: 0 auto !important;
    }
    .gradio-container .tabitem {
        padding-top: 8px;
    }
    .gradio-container .gr-form {
        gap: 10px !important;
    }
    .gradio-container .gr-row {
        align-items: stretch;
    }
    .gradio-container .gr-column {
        min-width: 0;
    }
    .gradio-container textarea {
        line-height: 1.35;
    }
    """

    with gr.Blocks(
        title="Insurance Pricing Model Training",
        theme=gr.themes.Soft(),
        css=layout_css,
    ) as demo:
        gr.Markdown(
            """
            # Insurance Pricing Model Training Interface
            Configure and train insurance pricing models with an easy-to-use interface.

            **Two ways to configure:**
            1. **Upload JSON Config**: Upload an existing configuration file
            2. **Manual Configuration**: Fill in the parameters below
            """
        )

        with gr.Row():
            working_dir_input = gr.Textbox(
                label="Working Directory",
                value=str(app.working_dir),
                placeholder="Type a path, or select from the folder list below",
                scale=3,
            )
            set_working_dir_btn = gr.Button(
                "Set Working Directory", variant="secondary", scale=1)

        with gr.Row():
            working_dir_browse_root = gr.Textbox(
                label="Browse Root",
                value=str(app.working_dir),
                placeholder="List folders under this path (depth=2)",
                scale=3,
            )
            refresh_working_dir_btn = gr.Button(
                "Refresh Folder List", variant="secondary", scale=1
            )

        with gr.Row():
            working_dir_picker = gr.Dropdown(
                label="Select Existing Folder",
                choices=dir_choices_init,
                value=dir_value_init,
                scale=3,
            )
            use_selected_working_dir_btn = gr.Button(
                "Use Selected Folder", variant="secondary", scale=1
            )

        working_dir_status = gr.Textbox(
            label="Working Directory Status", value=dir_status_init, interactive=False)

        with gr.Tab("Configuration"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=2):
                    gr.Markdown("### Load Configuration")
                    json_file = gr.File(
                        label="Upload JSON Config File",
                        file_types=[".json"],
                        type="filepath"
                    )
                    load_btn = gr.Button("Load Config", variant="primary")
                    load_status = gr.Textbox(
                        label="Load Status", interactive=False)

                with gr.Column(scale=5):
                    gr.Markdown("### Current Configuration")
                    config_display = gr.JSON(label="Configuration", value={})

            gr.Markdown("---")
            gr.Markdown("### Manual Configuration")

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    gr.Markdown("#### Data Settings")
                    data_dir = gr.Textbox(
                        label="Data Directory", value="./Data")
                    model_list = gr.Textbox(
                        label="Model List (comma-separated)", value="od")
                    model_categories = gr.Textbox(
                        label="Model Categories (comma-separated)", value="bc")
                    target = gr.Textbox(
                        label="Target Column", value="response")
                    weight = gr.Textbox(label="Weight Column", value="weights")

                    gr.Markdown("#### Features")
                    feature_list = gr.Textbox(
                        label="Feature List (comma-separated)",
                        placeholder="feature_1, feature_2, feature_3",
                        lines=4
                    )
                    categorical_features = gr.Textbox(
                        label="Categorical Features (comma-separated)",
                        placeholder="feature_2, feature_3",
                        lines=3
                    )

                with gr.Column(scale=1):
                    gr.Markdown("#### Model Settings")
                    task_type = gr.Dropdown(
                        label="Task Type",
                        choices=["regression", "binary", "multiclass"],
                        value="regression"
                    )
                    split_strategy = gr.Dropdown(
                        label="Split Strategy",
                        choices=["random", "stratified", "time", "group"],
                        value="random"
                    )
                    rand_seed = gr.Number(
                        label="Random Seed", value=13, precision=0)
                    epochs = gr.Number(label="Epochs", value=50, precision=0)
                    prop_test = gr.Slider(
                        label="Test Proportion", minimum=0.1, maximum=0.5, value=0.25, step=0.05)
                    holdout_ratio = gr.Slider(
                        label="Holdout Ratio", minimum=0.1, maximum=0.5, value=0.25, step=0.05)
                    val_ratio = gr.Slider(
                        label="Validation Ratio", minimum=0.1, maximum=0.5, value=0.25, step=0.05)

                    gr.Markdown("#### Training Settings")
                    output_dir = gr.Textbox(
                        label="Output Directory", value="./Results")
                    use_gpu = gr.Checkbox(label="Use GPU", value=True)
                    model_keys = gr.Textbox(
                        label="Model Keys (comma-separated)",
                        value="xgb, resn",
                        placeholder="xgb, resn, ft, gnn"
                    )
                    max_evals = gr.Number(
                        label="Max Evaluations", value=50, precision=0)

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### XGBoost Runtime")
                            xgb_max_depth_max = gr.Number(
                                label="XGB Max Depth", value=25, precision=0)
                            xgb_n_estimators_max = gr.Number(
                                label="XGB Max Estimators", value=500, precision=0)
                            xgb_gpu_id = gr.Number(
                                label="XGB GPU ID", value=0, precision=0)
                            xgb_use_dmatrix = gr.Checkbox(
                                label="XGB Use DMatrix", value=True)
                            xgb_chunk_size = gr.Number(
                                label="XGB Chunk Size (rows, 0=off)", value=0, precision=0)
                            resn_use_lazy_dataset = gr.Checkbox(
                                label="ResNet Lazy Dataset", value=True)
                            resn_predict_batch_size = gr.Number(
                                label="ResNet Predict Batch Size (0=auto)", value=0, precision=0)

                        with gr.Column(scale=1):
                            gr.Markdown("#### Cleanup Controls")
                            xgb_cleanup_per_fold = gr.Checkbox(
                                label="XGB Cleanup Per Fold", value=False)
                            xgb_cleanup_synchronize = gr.Checkbox(
                                label="XGB Cleanup Synchronize", value=False)
                            ft_cleanup_per_fold = gr.Checkbox(
                                label="FT Cleanup Per Fold", value=False)
                            ft_cleanup_synchronize = gr.Checkbox(
                                label="FT Cleanup Synchronize", value=False)
                            resn_cleanup_per_fold = gr.Checkbox(
                                label="ResNet Cleanup Per Fold", value=False)
                            resn_cleanup_synchronize = gr.Checkbox(
                                label="ResNet Cleanup Synchronize", value=False)
                            gnn_cleanup_per_fold = gr.Checkbox(
                                label="GNN Cleanup Per Fold", value=False)
                            gnn_cleanup_synchronize = gr.Checkbox(
                                label="GNN Cleanup Synchronize", value=False)
                            optuna_cleanup_synchronize = gr.Checkbox(
                                label="Optuna Cleanup Synchronize", value=False)

            gr.Markdown("#### Bayesian Optimization Search Spaces (JSON)")
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    xgb_search_space_json = gr.Textbox(
                        label="XGB Search Space",
                        value=xgb_search_space_template,
                        lines=10,
                        max_lines=20,
                    )
                with gr.Column(scale=1):
                    resn_search_space_json = gr.Textbox(
                        label="ResNet Search Space",
                        value=resn_search_space_template,
                        lines=10,
                        max_lines=20,
                    )
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    ft_search_space_json = gr.Textbox(
                        label="FT Supervised Search Space",
                        value=ft_search_space_template,
                        lines=10,
                        max_lines=20,
                    )
                with gr.Column(scale=1):
                    ft_unsupervised_search_space_json = gr.Textbox(
                        label="FT Unsupervised Search Space",
                        value=ft_unsupervised_search_space_template,
                        lines=10,
                        max_lines=20,
                    )

            with gr.Row():
                build_btn = gr.Button(
                    "Build Configuration", variant="primary", size="lg")
                save_config_btn = gr.Button(
                    "Save Configuration", variant="secondary", size="lg")

            build_status = gr.Textbox(label="Status", interactive=False)
            config_json = gr.Textbox(
                label="Generated Config (JSON)", lines=12, max_lines=24)

            with gr.Row(equal_height=True):
                save_filename = gr.Textbox(
                    label="Save Filename", value="my_config.json", scale=3)
                save_status = gr.Textbox(
                    label="Save Status", interactive=False, scale=4)

        with gr.Tab("Run Task"):
            gr.Markdown(
                """
                ### Run Model Task
                Click the button below to execute the task defined in your configuration.
                Task type is automatically detected from `config.runner.mode`:
                - **entry**: Standard model training
                - **explain**: Model explanation (permutation, SHAP, integrated gradients)
                - **incremental**: Incremental training
                - **watchdog**: Watchdog mode

                Task logs will appear in real-time below.
                """
            )

            with gr.Row():
                run_btn = gr.Button("Run Task", variant="primary", size="lg")
                run_status = gr.Textbox(label="Task Status", interactive=False)

            gr.Markdown("### Task Logs")
            log_output = gr.Textbox(
                label="Logs",
                lines=25,
                max_lines=50,
                interactive=False,
                autoscroll=True
            )

            gr.Markdown("---")
            with gr.Row():
                open_folder_btn = gr.Button("Open Results Folder", size="lg")
                folder_status = gr.Textbox(
                    label="Status", interactive=False, scale=2)

        with gr.Tab("FT Two-Step Workflow"):
            gr.Markdown(
                """
                ### FT-Transformer Two-Step Training

                Automates the FT -> XGB/ResN workflow:
                1. **Step 1**: Train FT-Transformer as unsupervised embedding generator
                2. **Step 2**: Merge embeddings with raw data and train XGB/ResN

                **Instructions**:
                1. Load or build a base configuration in the Configuration tab
                2. Prepare Step 1 config (FT embeddings)
                3. Run Step 1 to generate embeddings
                4. Prepare Step 2 configs (XGB/ResN using embeddings)
                5. Run Step 2 with the generated configs
                """
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Step 1: FT Embedding Generation")
                    ft_use_ddp = gr.Checkbox(
                        label="Use DDP for FT", value=True)
                    ft_nproc = gr.Number(
                        label="Number of Processes (DDP)", value=2, precision=0)

                    prepare_step1_btn = gr.Button(
                        "Prepare Step 1 Config", variant="primary")
                    step1_status = gr.Textbox(
                        label="Status", interactive=False)
                    step1_config_display = gr.Textbox(
                        label="Step 1 Config (FT Embedding)",
                        lines=15,
                        max_lines=25
                    )

                with gr.Column():
                    gr.Markdown("### Step 2: Train XGB/ResN with Embeddings")
                    target_models_input = gr.Textbox(
                        label="Target Models (comma-separated)",
                        value="xgb, resn",
                        placeholder="xgb, resn"
                    )
                    augmented_data_dir_input = gr.Textbox(
                        label="Augmented Data Directory",
                        value="./DataFTUnsupervised",
                        placeholder="./DataFTUnsupervised",
                    )
                    gr.Markdown(
                        "Step-2 parameter overrides (JSON). "
                        "Defaults match `02 Train_FT_Embed_XGBResN.ipynb`."
                    )
                    xgb_overrides_input = gr.Textbox(
                        label="XGB Step 2 Overrides (JSON)",
                        value=xgb_step2_overrides_template,
                        lines=10,
                        max_lines=20,
                    )
                    resn_overrides_input = gr.Textbox(
                        label="ResN Step 2 Overrides (JSON)",
                        value=resn_step2_overrides_template,
                        lines=10,
                        max_lines=20,
                    )

                    prepare_step2_btn = gr.Button(
                        "Prepare Step 2 Configs", variant="primary")
                    step2_status = gr.Textbox(
                        label="Status", interactive=False)

                    with gr.Tab("XGB Config"):
                        xgb_config_display = gr.Textbox(
                            label="XGB Step 2 Config",
                            lines=15,
                            max_lines=25
                        )

                    with gr.Tab("ResN Config"):
                        resn_config_display = gr.Textbox(
                            label="ResN Step 2 Config",
                            lines=15,
                            max_lines=25
                        )

            gr.Markdown("---")
            gr.Markdown(
                """
                ### Quick Actions
                After preparing configs, you can:
                - Copy the Step 1 config and paste it in the **Configuration** tab, then run it in **Run Task** tab
                - After Step 1 completes, click **Prepare Step 2 Configs**
                - Step 2 configs are auto-saved as:
                  - `config_xgb_from_ft_unsupervised.json`
                  - `config_resn_from_ft_unsupervised.json`
                - You can edit XGB/ResN override JSONs before generation to customize Step-2 parameters
                - Run those configs in **Run Task** tab (or keep using the JSON text boxes below)
                """
            )

        with gr.Tab("Workflow Config"):
            gr.Markdown(
                """
                ### Config-Driven Plotting / Prediction / Compare / Pre-Oneway
                Use a JSON config file to run frontend workflows without manual field-by-field input.

                Supported `workflow.mode` values:
                - `pre_oneway`
                - `plot_direct`
                - `plot_embed`
                - `predict_ft_embed`
                - `compare_xgb`
                - `compare_resn`
                - `compare` (requires `model_key`: `xgb` or `resn`)
                - `double_lift`
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    workflow_file = gr.File(
                        label="Upload Workflow Config",
                        file_types=[".json"],
                        type="filepath",
                    )
                    workflow_load_btn = gr.Button(
                        "Load Workflow Config", variant="secondary")
                    workflow_load_status = gr.Textbox(
                        label="Load Status", interactive=False)

                with gr.Column(scale=2):
                    workflow_config_json = gr.Textbox(
                        label="Workflow Config (JSON)",
                        value=workflow_template,
                        lines=18,
                        max_lines=30,
                    )

            workflow_run_btn = gr.Button(
                "Run Workflow Config", variant="primary", size="lg")
            workflow_status = gr.Textbox(label="Workflow Status", interactive=False)
            workflow_log = gr.Textbox(
                label="Workflow Logs",
                lines=18,
                max_lines=40,
                interactive=False,
                autoscroll=True,
            )

        with gr.Tab("Plotting"):
            gr.Markdown(
                """
                ### Plotting Workflows
                Run the plotting steps from the example notebooks.
                """
            )

            with gr.Tab("Pre Oneway"):
                with gr.Row():
                    with gr.Column():
                        pre_data_path = gr.Textbox(
                            label="Data Path", value="./Data/od_bc.csv")
                        pre_model_name = gr.Textbox(
                            label="Model Name", value="od_bc")
                        pre_target = gr.Textbox(
                            label="Target Column", value="response")
                        pre_weight = gr.Textbox(
                            label="Weight Column", value="weights")
                        pre_output_dir = gr.Textbox(
                            label="Output Dir (optional)", value="")
                    with gr.Column():
                        pre_feature_list = gr.Textbox(
                            label="Feature List (comma-separated)",
                            lines=4,
                            placeholder="feature_1, feature_2, feature_3",
                        )
                        pre_categorical = gr.Textbox(
                            label="Categorical Features (comma-separated, optional)",
                            lines=3,
                            placeholder="feature_2, feature_3",
                        )
                        pre_n_bins = gr.Number(
                            label="Bins", value=10, precision=0)
                        pre_holdout = gr.Slider(
                            label="Holdout Ratio",
                            minimum=0.0,
                            maximum=0.5,
                            value=0.25,
                            step=0.05,
                        )
                        pre_seed = gr.Number(
                            label="Random Seed", value=13, precision=0)

                pre_run_btn = gr.Button("Run Pre Oneway", variant="primary")
                pre_status = gr.Textbox(label="Status", interactive=False)
                pre_log = gr.Textbox(label="Logs", lines=15,
                                     max_lines=40, interactive=False)

            with gr.Tab("Direct Plot"):
                with gr.Row():
                    with gr.Column(scale=3):
                        direct_cfg_path = gr.Textbox(
                            label="Plot Config", value="config_plot.json")
                        direct_run_btn = gr.Button(
                            "Run Direct Plot", variant="primary")
                        direct_status = gr.Textbox(
                            label="Status", interactive=False)
                    with gr.Column(scale=4):
                        direct_xgb_cfg = gr.Textbox(
                            label="XGB Config", value="config_xgb_direct.json")
                        direct_resn_cfg = gr.Textbox(
                            label="ResN Config", value="config_resn_direct.json")
                direct_log = gr.Textbox(
                    label="Logs", lines=15, max_lines=40, interactive=False)

            with gr.Tab("Embed Plot"):
                with gr.Row():
                    with gr.Column(scale=3):
                        embed_cfg_path = gr.Textbox(
                            label="Plot Config", value="config_plot.json")
                        embed_ft_cfg = gr.Textbox(
                            label="FT Embed Config", value="config_ft_unsupervised_ddp_embed.json")
                        embed_runtime = gr.Checkbox(
                            label="Use Runtime FT Embedding", value=False)
                        embed_run_btn = gr.Button(
                            "Run Embed Plot", variant="primary")
                        embed_status = gr.Textbox(
                            label="Status", interactive=False)
                    with gr.Column(scale=4):
                        embed_xgb_cfg = gr.Textbox(
                            label="XGB Embed Config", value="config_xgb_from_ft_unsupervised.json")
                        embed_resn_cfg = gr.Textbox(
                            label="ResN Embed Config", value="config_resn_from_ft_unsupervised.json")
                embed_log = gr.Textbox(
                    label="Logs", lines=15, max_lines=40, interactive=False)

            with gr.Tab("Double Lift"):
                gr.Markdown(
                    """
                    Draw a double-lift curve from any CSV file with two prediction columns.
                    """
                )
                with gr.Row():
                    with gr.Column():
                        dl_data_path = gr.Textbox(
                            label="Data Path (CSV)", value="./Data/od_bc.csv")
                        dl_pred_col_1 = gr.Textbox(
                            label="Prediction Column 1", value="pred_xgb")
                        dl_pred_col_2 = gr.Textbox(
                            label="Prediction Column 2", value="pred_resn")
                        dl_target_col = gr.Textbox(
                            label="Target Column", value="reponse")
                        dl_weight_col = gr.Textbox(
                            label="Weight Column", value="weights")
                        dl_output_path = gr.Textbox(
                            label="Output Image Path (optional)", value="")
                    with gr.Column():
                        dl_label1 = gr.Textbox(label="Label 1", value="Model 1")
                        dl_label2 = gr.Textbox(label="Label 2", value="Model 2")
                        dl_n_bins = gr.Number(label="Bins", value=10, precision=0)
                        dl_holdout_ratio = gr.Slider(
                            label="Holdout Ratio (0 = all data, >0 = train/test split)",
                            minimum=0.0,
                            maximum=0.5,
                            value=0.0,
                            step=0.05,
                        )
                        dl_split_strategy = gr.Dropdown(
                            label="Split Strategy",
                            choices=["random", "stratified", "time", "group"],
                            value="random",
                        )
                        dl_split_group_col = gr.Textbox(
                            label="Group Column (optional, for group split)",
                            value="",
                        )
                        dl_split_time_col = gr.Textbox(
                            label="Time Column (optional, for time split)",
                            value="",
                        )
                        dl_split_time_ascending = gr.Checkbox(
                            label="Time Ascending",
                            value=True,
                        )
                        dl_rand_seed = gr.Number(
                            label="Random Seed", value=13, precision=0)
                        dl_pred1_weighted = gr.Checkbox(
                            label="Prediction 1 Is Weighted",
                            value=False,
                        )
                        dl_pred2_weighted = gr.Checkbox(
                            label="Prediction 2 Is Weighted",
                            value=False,
                        )
                        dl_actual_weighted = gr.Checkbox(
                            label="Actual Is Weighted",
                            value=False,
                        )

                dl_run_btn = gr.Button("Run Double Lift", variant="primary")
                dl_status = gr.Textbox(label="Status", interactive=False)
                dl_log = gr.Textbox(
                    label="Logs", lines=15, max_lines=40, interactive=False)

            with gr.Tab("FT-Embed Compare"):
                gr.Markdown("Compare Direct vs FT-Embed models and draw double-lift curves.")
                with gr.Row():
                    with gr.Column(scale=4):
                        cmp_model_key = gr.Dropdown(
                            label="Model Key",
                            choices=["xgb", "resn"],
                            value="xgb",
                        )
                        cmp_direct_cfg = gr.Textbox(
                            label="Direct Model Config", value="config_xgb_direct.json")
                        cmp_ft_cfg = gr.Textbox(
                            label="FT Config", value="config_ft_unsupervised_ddp_embed.json")
                        cmp_embed_cfg = gr.Textbox(
                            label="FT-Embed Model Config", value="config_xgb_from_ft_unsupervised.json")
                    with gr.Column(scale=3):
                        cmp_label_direct = gr.Textbox(
                            label="Direct Label", value="XGB_raw")
                        cmp_label_ft = gr.Textbox(
                            label="FT Label", value="XGB_ft_embed")
                        cmp_runtime = gr.Checkbox(
                            label="Use Runtime FT Embedding", value=False)
                        cmp_bins = gr.Number(
                            label="Bins Override", value=10, precision=0)
                        cmp_run_btn = gr.Button("Run Compare", variant="primary")
                        cmp_status = gr.Textbox(label="Status", interactive=False)
                cmp_log = gr.Textbox(
                    label="Logs", lines=15, max_lines=40, interactive=False)

        with gr.Tab("Prediction"):
            gr.Markdown("### FT Embed Prediction")
            with gr.Row():
                with gr.Column(scale=4):
                    pred_ft_cfg = gr.Textbox(
                        label="FT Config", value="config_ft_unsupervised_ddp_embed.json")
                    pred_xgb_cfg = gr.Textbox(
                        label="XGB Config (optional)", value="config_xgb_from_ft_unsupervised.json")
                    pred_resn_cfg = gr.Textbox(
                        label="ResN Config (optional)", value="config_resn_from_ft_unsupervised.json")
                    pred_model_name = gr.Textbox(
                        label="Model Name (optional)", value="")
                    pred_model_keys = gr.Textbox(
                        label="Model Keys", value="xgb, resn")
                with gr.Column(scale=3):
                    pred_input = gr.Textbox(
                        label="Input Data", value="./Data/od_bc_new.csv")
                    pred_output = gr.Textbox(
                        label="Output CSV", value="./Results/predictions_ft_xgb.csv")
                    pred_run_btn = gr.Button("Run Prediction", variant="primary")
                    pred_status = gr.Textbox(label="Status", interactive=False)
            pred_log = gr.Textbox(label="Logs", lines=15,
                                  max_lines=40, interactive=False)

        # Event handlers
        set_working_dir_btn.click(
            fn=_set_working_dir_ui,
            inputs=[working_dir_input],
            outputs=[
                working_dir_status,
                working_dir_input,
                working_dir_browse_root,
                working_dir_picker,
            ],
        )

        refresh_working_dir_btn.click(
            fn=_refresh_working_dir_choices_ui,
            inputs=[working_dir_browse_root],
            outputs=[working_dir_status, working_dir_picker],
        )

        use_selected_working_dir_btn.click(
            fn=_set_working_dir_ui,
            inputs=[working_dir_picker],
            outputs=[
                working_dir_status,
                working_dir_input,
                working_dir_browse_root,
                working_dir_picker,
            ],
        )

        working_dir_picker.change(
            fn=lambda path: str(path or ""),
            inputs=[working_dir_picker],
            outputs=[working_dir_input],
        )

        load_btn.click(
            fn=app.load_json_config,
            inputs=[json_file],
            outputs=[load_status, config_display, config_json]
        )

        build_btn.click(
            fn=app.build_config_from_ui,
            inputs=[
                data_dir, model_list, model_categories, target, weight,
                feature_list, categorical_features, task_type, prop_test,
                holdout_ratio, val_ratio, split_strategy, rand_seed, epochs,
                output_dir, use_gpu, model_keys, max_evals,
                xgb_max_depth_max, xgb_n_estimators_max,
                xgb_gpu_id, xgb_cleanup_per_fold, xgb_cleanup_synchronize,
                xgb_use_dmatrix, xgb_chunk_size,
                xgb_search_space_json, resn_search_space_json,
                ft_search_space_json, ft_unsupervised_search_space_json,
                ft_cleanup_per_fold, ft_cleanup_synchronize,
                resn_cleanup_per_fold, resn_cleanup_synchronize,
                resn_use_lazy_dataset, resn_predict_batch_size,
                gnn_cleanup_per_fold, gnn_cleanup_synchronize,
                optuna_cleanup_synchronize
            ],
            outputs=[build_status, config_json]
        )

        save_config_btn.click(
            fn=app.save_config,
            inputs=[config_json, save_filename],
            outputs=[save_status]
        )

        run_btn.click(
            fn=app.run_training,
            inputs=[config_json],
            outputs=[run_status, log_output]
        )

        open_folder_btn.click(
            fn=app.open_results_folder,
            inputs=[config_json],
            outputs=[folder_status]
        )

        workflow_load_btn.click(
            fn=app.load_workflow_config,
            inputs=[workflow_file],
            outputs=[workflow_load_status, workflow_config_json]
        )

        workflow_run_btn.click(
            fn=app.run_workflow_config_ui,
            inputs=[workflow_config_json],
            outputs=[workflow_status, workflow_log]
        )

        prepare_step1_btn.click(
            fn=app.prepare_ft_step1,
            inputs=[config_json, ft_use_ddp, ft_nproc],
            outputs=[step1_status, step1_config_display]
        )

        prepare_step2_btn.click(
            fn=app.prepare_ft_step2,
            inputs=[gr.State(
                lambda: app.current_step1_config or "temp_ft_step1_config.json"),
                target_models_input,
                augmented_data_dir_input,
                xgb_overrides_input,
                resn_overrides_input,
            ],
            outputs=[step2_status, xgb_config_display, resn_config_display]
        )

        pre_run_btn.click(
            fn=app.run_pre_oneway_ui,
            inputs=[
                pre_data_path, pre_model_name, pre_target, pre_weight,
                pre_feature_list, pre_categorical, pre_n_bins,
                pre_holdout, pre_seed, pre_output_dir
            ],
            outputs=[pre_status, pre_log]
        )

        direct_run_btn.click(
            fn=app.run_plot_direct_ui,
            inputs=[direct_cfg_path, direct_xgb_cfg, direct_resn_cfg],
            outputs=[direct_status, direct_log]
        )

        embed_run_btn.click(
            fn=app.run_plot_embed_ui,
            inputs=[embed_cfg_path, embed_xgb_cfg,
                    embed_resn_cfg, embed_ft_cfg, embed_runtime],
            outputs=[embed_status, embed_log]
        )

        dl_run_btn.click(
            fn=app.run_double_lift_ui,
            inputs=[
                dl_data_path, dl_pred_col_1, dl_pred_col_2, dl_target_col, dl_weight_col,
                dl_n_bins, dl_label1, dl_label2,
                dl_pred1_weighted, dl_pred2_weighted, dl_actual_weighted,
                dl_holdout_ratio, dl_split_strategy, dl_split_group_col, dl_split_time_col,
                dl_split_time_ascending, dl_rand_seed, dl_output_path
            ],
            outputs=[dl_status, dl_log]
        )

        pred_run_btn.click(
            fn=app.run_predict_ui,
            inputs=[
                pred_ft_cfg, pred_xgb_cfg, pred_resn_cfg, pred_input,
                pred_output, pred_model_name, pred_model_keys
            ],
            outputs=[pred_status, pred_log]
        )

        cmp_model_key.change(
            fn=_suggest_compare_defaults,
            inputs=[cmp_model_key],
            outputs=[
                cmp_direct_cfg,
                cmp_embed_cfg,
                cmp_label_direct,
                cmp_label_ft,
            ],
        )

        cmp_run_btn.click(
            fn=app.run_compare_ui,
            inputs=[
                cmp_model_key,
                cmp_direct_cfg,
                cmp_ft_cfg,
                cmp_embed_cfg,
                cmp_label_direct,
                cmp_label_ft,
                cmp_runtime,
                cmp_bins,
            ],
            outputs=[cmp_status, cmp_log]
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": 7860,
        "share": False,
        "show_error": True,
    }
    if "analytics_enabled" in inspect.signature(demo.launch).parameters:
        launch_kwargs["analytics_enabled"] = False
    demo.launch(**launch_kwargs)
