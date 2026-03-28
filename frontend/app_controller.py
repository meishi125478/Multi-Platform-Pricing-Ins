"""
Insurance Pricing Model Training Frontend.
Frontend controller for configuring and running insurance pricing models.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

from ins_pricing.frontend.app_controller_config_mixin import AppControllerConfigMixin
from ins_pricing.frontend.app_controller_runtime_mixin import AppControllerRuntimeMixin
from ins_pricing.frontend.config_builder import ConfigBuilder
from ins_pricing.frontend.ft_workflow import FTWorkflowHelper


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
