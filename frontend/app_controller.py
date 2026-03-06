"""
Insurance Pricing Model Training Frontend
A Gradio-based web interface for configuring and running insurance pricing models.
"""

import os
from ins_pricing.frontend.ft_workflow import FTWorkflowHelper
from ins_pricing.frontend.config_builder import ConfigBuilder
from ins_pricing.frontend.app_controller_config_mixin import AppControllerConfigMixin
from ins_pricing.frontend.app_controller_runtime_mixin import AppControllerRuntimeMixin
import importlib.util
from pathlib import Path
from typing import Optional, Dict, Any
from types import SimpleNamespace

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


class PricingApp(AppControllerRuntimeMixin, AppControllerConfigMixin):
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
        from ins_pricing.frontend.workflows_compare import (
            run_compare_ft_embed,
            run_double_lift_from_file,
        )
        from ins_pricing.frontend.workflows_plot import (
            run_plot_direct,
            run_plot_embed,
            run_pre_oneway,
        )
        from ins_pricing.frontend.workflows_predict import run_predict_ft_embed

        return SimpleNamespace(
            run_pre_oneway=run_pre_oneway,
            run_plot_direct=run_plot_direct,
            run_plot_embed=run_plot_embed,
            run_predict_ft_embed=run_predict_ft_embed,
            run_compare_ft_embed=run_compare_ft_embed,
            run_double_lift_from_file=run_double_lift_from_file,
        )

    def _get_runner(self):
        if self.runner is None:
            from ins_pricing.frontend.runner import TaskRunner
            self.runner = TaskRunner()
        return self.runner

