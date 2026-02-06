"""
Insurance Pricing Model Training Frontend
A Gradio-based web interface for configuring and running insurance pricing models.
"""

import os
import platform
import subprocess
from ins_pricing.frontend.example_workflows import (
    run_compare_ft_embed,
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
import sys
import inspect
import importlib.util
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Iterable, Tuple, Generator
import threading
import queue
import time

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
        self.current_config = {}
        self.current_step1_config = None
        self.current_config_path: Optional[Path] = None
        self.current_config_dir: Optional[Path] = None

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
        ft_cleanup_per_fold: bool,
        ft_cleanup_synchronize: bool,
        resn_cleanup_per_fold: bool,
        resn_cleanup_synchronize: bool,
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
                ft_cleanup_per_fold=ft_cleanup_per_fold,
                ft_cleanup_synchronize=ft_cleanup_synchronize,
                resn_cleanup_per_fold=resn_cleanup_per_fold,
                resn_cleanup_synchronize=resn_cleanup_synchronize,
                gnn_cleanup_per_fold=gnn_cleanup_per_fold,
                gnn_cleanup_synchronize=gnn_cleanup_synchronize,
                optuna_cleanup_synchronize=optuna_cleanup_synchronize,
            )

            is_valid, msg = self.config_builder.validate_config(config)
            if not is_valid:
                return f"Validation failed: {msg}", ""

            self.current_config = config
            self.current_config_path = None
            self.current_config_dir = None
            config_json = json.dumps(config, indent=2, ensure_ascii=False)
            return "Configuration built successfully", config_json

        except Exception as e:
            return f"Error building config: {str(e)}", ""

    def save_config(self, config_json: str, filename: str) -> str:
        """Save current configuration to file."""
        if not config_json:
            return "No configuration to save"

        try:
            config_path = Path(filename)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(json.loads(config_json), f,
                          indent=2, ensure_ascii=False)
            return f"Configuration saved to {config_path}"
        except Exception as e:
            return f"Error saving config: {str(e)}"

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
                base_dir = self.current_config_dir or Path.cwd()
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
                base_dir = Path.cwd()
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
            temp_path = Path("temp_ft_step1_config.json")
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(step1_config, f, indent=2)

            self.current_step1_config = str(temp_path)
            step1_json = json.dumps(step1_config, indent=2, ensure_ascii=False)

            return "Step 1 config prepared. Click 'Run Step 1' to train FT embeddings.", step1_json

        except Exception as e:
            return f"Error preparing Step 1 config: {str(e)}", ""

    def prepare_ft_step2(self, step1_config_path: str, target_models: str) -> tuple[str, str, str]:
        """Prepare FT Step 2 configurations."""
        if not step1_config_path or not os.path.exists(step1_config_path):
            return "Step 1 config not found. Run Step 1 first.", "", ""

        try:
            models = [m.strip() for m in target_models.split(',') if m.strip()]
            xgb_cfg, resn_cfg = self.ft_workflow.generate_step2_configs(
                step1_config_path=step1_config_path,
                target_models=models
            )

            status_msg = f"Step 2 configs prepared for: {', '.join(models)}"
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
                results_path = Path(output_dir).resolve()
            elif self.current_config_path and self.current_config_path.exists():
                config = json.loads(
                    self.current_config_path.read_text(encoding="utf-8"))
                output_dir = config.get('output_dir', './Results')
                results_path = (
                    self.current_config_path.parent / output_dir).resolve()
            elif self.current_config:
                output_dir = self.current_config.get('output_dir', './Results')
                results_path = Path(output_dir).resolve()
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

    def run_compare_xgb_ui(
        self,
        direct_cfg_path: str,
        ft_cfg_path: str,
        ft_embed_cfg_path: str,
        label_direct: str,
        label_ft: str,
        use_runtime_ft_embedding: bool,
        n_bins_override: int,
    ):
        yield from self._run_workflow(
            "Compare XGB",
            run_compare_ft_embed,
            direct_cfg_path=direct_cfg_path,
            ft_cfg_path=ft_cfg_path,
            ft_embed_cfg_path=ft_embed_cfg_path,
            model_key="xgb",
            label_direct=label_direct,
            label_ft=label_ft,
            use_runtime_ft_embedding=use_runtime_ft_embedding,
            n_bins_override=n_bins_override,
        )

    def run_compare_resn_ui(
        self,
        direct_cfg_path: str,
        ft_cfg_path: str,
        ft_embed_cfg_path: str,
        label_direct: str,
        label_ft: str,
        use_runtime_ft_embedding: bool,
        n_bins_override: int,
    ):
        yield from self._run_workflow(
            "Compare ResNet",
            run_compare_ft_embed,
            direct_cfg_path=direct_cfg_path,
            ft_cfg_path=ft_cfg_path,
            ft_embed_cfg_path=ft_embed_cfg_path,
            model_key="resn",
            label_direct=label_direct,
            label_ft=label_ft,
            use_runtime_ft_embedding=use_runtime_ft_embedding,
            n_bins_override=n_bins_override,
        )


def create_ui():
    """Create the Gradio interface."""
    _check_frontend_deps()
    import gradio as gr

    app = PricingApp()

    with gr.Blocks(title="Insurance Pricing Model Training", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Insurance Pricing Model Training Interface
            Configure and train insurance pricing models with an easy-to-use interface.

            **Two ways to configure:**
            1. **Upload JSON Config**: Upload an existing configuration file
            2. **Manual Configuration**: Fill in the parameters below
            """
        )

        with gr.Tab("Configuration"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Load Configuration")
                    json_file = gr.File(
                        label="Upload JSON Config File",
                        file_types=[".json"],
                        type="filepath"
                    )
                    load_btn = gr.Button("Load Config", variant="primary")
                    load_status = gr.Textbox(
                        label="Load Status", interactive=False)

                with gr.Column(scale=2):
                    gr.Markdown("### Current Configuration")
                    config_display = gr.JSON(label="Configuration", value={})

            gr.Markdown("---")
            gr.Markdown("### Manual Configuration")

            with gr.Row():
                with gr.Column():
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
                        lines=3
                    )
                    categorical_features = gr.Textbox(
                        label="Categorical Features (comma-separated)",
                        placeholder="feature_2, feature_3",
                        lines=2
                    )

                with gr.Column():
                    gr.Markdown("#### Model Settings")
                    task_type = gr.Dropdown(
                        label="Task Type",
                        choices=["regression", "binary", "multiclass"],
                        value="regression"
                    )
                    prop_test = gr.Slider(
                        label="Test Proportion", minimum=0.1, maximum=0.5, value=0.25, step=0.05)
                    holdout_ratio = gr.Slider(
                        label="Holdout Ratio", minimum=0.1, maximum=0.5, value=0.25, step=0.05)
                    val_ratio = gr.Slider(
                        label="Validation Ratio", minimum=0.1, maximum=0.5, value=0.25, step=0.05)
                    split_strategy = gr.Dropdown(
                        label="Split Strategy",
                        choices=["random", "stratified", "time", "group"],
                        value="random"
                    )
                    rand_seed = gr.Number(
                        label="Random Seed", value=13, precision=0)
                    epochs = gr.Number(label="Epochs", value=50, precision=0)

            with gr.Row():
                with gr.Column():
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

                with gr.Column():
                    gr.Markdown("#### XGBoost Settings")
                    xgb_max_depth_max = gr.Number(
                        label="XGB Max Depth", value=25, precision=0)
                    xgb_n_estimators_max = gr.Number(
                        label="XGB Max Estimators", value=500, precision=0)
                    xgb_gpu_id = gr.Number(
                        label="XGB GPU ID", value=0, precision=0)
                    xgb_cleanup_per_fold = gr.Checkbox(
                        label="XGB Cleanup Per Fold", value=False)
                    xgb_cleanup_synchronize = gr.Checkbox(
                        label="XGB Cleanup Synchronize", value=False)
                    xgb_use_dmatrix = gr.Checkbox(
                        label="XGB Use DMatrix", value=True)
                    gr.Markdown("#### Fold Cleanup")
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

            with gr.Row():
                build_btn = gr.Button(
                    "Build Configuration", variant="primary", size="lg")
                save_config_btn = gr.Button(
                    "Save Configuration", variant="secondary", size="lg")

            with gr.Row():
                build_status = gr.Textbox(label="Status", interactive=False)
                config_json = gr.Textbox(
                    label="Generated Config (JSON)", lines=10, max_lines=20)

            save_filename = gr.Textbox(
                label="Save Filename", value="my_config.json")
            save_status = gr.Textbox(label="Save Status", interactive=False)

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
                - Copy the Step 2 configs (XGB or ResN) and run them in **Run Task** tab
                """
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
                direct_cfg_path = gr.Textbox(
                    label="Plot Config", value="config_plot.json")
                direct_xgb_cfg = gr.Textbox(
                    label="XGB Config", value="config_xgb_direct.json")
                direct_resn_cfg = gr.Textbox(
                    label="ResN Config", value="config_resn_direct.json")
                direct_run_btn = gr.Button(
                    "Run Direct Plot", variant="primary")
                direct_status = gr.Textbox(label="Status", interactive=False)
                direct_log = gr.Textbox(
                    label="Logs", lines=15, max_lines=40, interactive=False)

            with gr.Tab("Embed Plot"):
                embed_cfg_path = gr.Textbox(
                    label="Plot Config", value="config_plot.json")
                embed_xgb_cfg = gr.Textbox(
                    label="XGB Embed Config", value="config_xgb_from_ft_unsupervised.json")
                embed_resn_cfg = gr.Textbox(
                    label="ResN Embed Config", value="config_resn_from_ft_unsupervised.json")
                embed_ft_cfg = gr.Textbox(
                    label="FT Embed Config", value="config_ft_unsupervised_ddp_embed.json")
                embed_runtime = gr.Checkbox(
                    label="Use Runtime FT Embedding", value=False)
                embed_run_btn = gr.Button("Run Embed Plot", variant="primary")
                embed_status = gr.Textbox(label="Status", interactive=False)
                embed_log = gr.Textbox(
                    label="Logs", lines=15, max_lines=40, interactive=False)

        with gr.Tab("Prediction"):
            gr.Markdown("### FT Embed Prediction")
            pred_ft_cfg = gr.Textbox(
                label="FT Config", value="config_ft_unsupervised_ddp_embed.json")
            pred_xgb_cfg = gr.Textbox(
                label="XGB Config (optional)", value="config_xgb_from_ft_unsupervised.json")
            pred_resn_cfg = gr.Textbox(
                label="ResN Config (optional)", value="config_resn_from_ft_unsupervised.json")
            pred_input = gr.Textbox(
                label="Input Data", value="./Data/od_bc_new.csv")
            pred_output = gr.Textbox(
                label="Output CSV", value="./Results/predictions_ft_xgb.csv")
            pred_model_name = gr.Textbox(
                label="Model Name (optional)", value="")
            pred_model_keys = gr.Textbox(label="Model Keys", value="xgb, resn")
            pred_run_btn = gr.Button("Run Prediction", variant="primary")
            pred_status = gr.Textbox(label="Status", interactive=False)
            pred_log = gr.Textbox(label="Logs", lines=15,
                                  max_lines=40, interactive=False)

        with gr.Tab("Compare"):
            gr.Markdown("### Compare Direct vs FT-Embed Models")

            with gr.Tab("Compare XGB"):
                cmp_xgb_direct_cfg = gr.Textbox(
                    label="Direct XGB Config", value="config_xgb_direct.json")
                cmp_xgb_ft_cfg = gr.Textbox(
                    label="FT Config", value="config_ft_unsupervised_ddp_embed.json")
                cmp_xgb_embed_cfg = gr.Textbox(
                    label="FT-Embed XGB Config", value="config_xgb_from_ft_unsupervised.json")
                cmp_xgb_label_direct = gr.Textbox(
                    label="Direct Label", value="XGB_raw")
                cmp_xgb_label_ft = gr.Textbox(
                    label="FT Label", value="XGB_ft_embed")
                cmp_xgb_runtime = gr.Checkbox(
                    label="Use Runtime FT Embedding", value=False)
                cmp_xgb_bins = gr.Number(
                    label="Bins Override", value=10, precision=0)
                cmp_xgb_run_btn = gr.Button(
                    "Run XGB Compare", variant="primary")
                cmp_xgb_status = gr.Textbox(label="Status", interactive=False)
                cmp_xgb_log = gr.Textbox(
                    label="Logs", lines=15, max_lines=40, interactive=False)

            with gr.Tab("Compare ResNet"):
                cmp_resn_direct_cfg = gr.Textbox(
                    label="Direct ResN Config", value="config_resn_direct.json")
                cmp_resn_ft_cfg = gr.Textbox(
                    label="FT Config", value="config_ft_unsupervised_ddp_embed.json")
                cmp_resn_embed_cfg = gr.Textbox(
                    label="FT-Embed ResN Config", value="config_resn_from_ft_unsupervised.json")
                cmp_resn_label_direct = gr.Textbox(
                    label="Direct Label", value="ResN_raw")
                cmp_resn_label_ft = gr.Textbox(
                    label="FT Label", value="ResN_ft_embed")
                cmp_resn_runtime = gr.Checkbox(
                    label="Use Runtime FT Embedding", value=False)
                cmp_resn_bins = gr.Number(
                    label="Bins Override", value=10, precision=0)
                cmp_resn_run_btn = gr.Button(
                    "Run ResNet Compare", variant="primary")
                cmp_resn_status = gr.Textbox(label="Status", interactive=False)
                cmp_resn_log = gr.Textbox(
                    label="Logs", lines=15, max_lines=40, interactive=False)

        # Event handlers
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
                xgb_use_dmatrix, ft_cleanup_per_fold, ft_cleanup_synchronize,
                resn_cleanup_per_fold, resn_cleanup_synchronize,
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

        prepare_step1_btn.click(
            fn=app.prepare_ft_step1,
            inputs=[config_json, ft_use_ddp, ft_nproc],
            outputs=[step1_status, step1_config_display]
        )

        prepare_step2_btn.click(
            fn=app.prepare_ft_step2,
            inputs=[gr.State(
                lambda: app.current_step1_config or "temp_ft_step1_config.json"), target_models_input],
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

        pred_run_btn.click(
            fn=app.run_predict_ui,
            inputs=[
                pred_ft_cfg, pred_xgb_cfg, pred_resn_cfg, pred_input,
                pred_output, pred_model_name, pred_model_keys
            ],
            outputs=[pred_status, pred_log]
        )

        cmp_xgb_run_btn.click(
            fn=app.run_compare_xgb_ui,
            inputs=[
                cmp_xgb_direct_cfg, cmp_xgb_ft_cfg, cmp_xgb_embed_cfg,
                cmp_xgb_label_direct, cmp_xgb_label_ft,
                cmp_xgb_runtime, cmp_xgb_bins
            ],
            outputs=[cmp_xgb_status, cmp_xgb_log]
        )

        cmp_resn_run_btn.click(
            fn=app.run_compare_resn_ui,
            inputs=[
                cmp_resn_direct_cfg, cmp_resn_ft_cfg, cmp_resn_embed_cfg,
                cmp_resn_label_direct, cmp_resn_label_ft,
                cmp_resn_runtime, cmp_resn_bins
            ],
            outputs=[cmp_resn_status, cmp_resn_log]
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
