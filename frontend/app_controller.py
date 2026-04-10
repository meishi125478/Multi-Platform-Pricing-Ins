"""
Insurance Pricing Model Training Frontend.
Frontend controller for configuring and running insurance pricing models.
"""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

from ins_pricing.frontend.access_control import (
    AuthorizationError,
    FrontendAuthStore,
)
from ins_pricing.frontend.app_controller_config_mixin import AppControllerConfigMixin
from ins_pricing.frontend.app_controller_runtime_mixin import AppControllerRuntimeMixin
from ins_pricing.frontend.config_builder import ConfigBuilder
from ins_pricing.frontend.ft_workflow import FTWorkflowHelper
from ins_pricing.frontend.system_status import collect_system_status


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
        self.xgb_direct_cfg_name: str = "config_xgb_direct.json"
        self.user_workspaces_root: Path = Path(
            os.environ.get(
                "INS_PRICING_FRONTEND_USER_WORKSPACES_ROOT",
                str((Path.cwd() / "workspaces").resolve()),
            )
        ).resolve()
        self.user_workspaces_root.mkdir(parents=True, exist_ok=True)
        self.active_username: Optional[str] = None
        self.user_workspace_root: Optional[Path] = None

        auth_file = os.environ.get(
            "INS_PRICING_FRONTEND_AUTH_FILE",
            str((Path.home() / ".ins_pricing" / "frontend_users.json").resolve()),
        )
        self.auth_store = FrontendAuthStore(auth_file)

    @staticmethod
    def _safe_workspace_tag(username: str) -> str:
        text = str(username or "").strip()
        if not text:
            raise ValueError("username is required")
        safe = (
            text.replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace("..", "_")
        )
        return safe

    def _workspace_dir_for_user(self, username: str) -> Path:
        safe_name = self._safe_workspace_tag(username)
        return (self.user_workspaces_root / safe_name).resolve()

    @staticmethod
    def _is_admin_roles(roles: list[str]) -> bool:
        return any(str(role).strip().lower() == "admin" for role in (roles or []))

    def ensure_user_workspace(self, username: str) -> Path:
        workspace = self._workspace_dir_for_user(username)
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def activate_user_workspace(self, username: str, *, is_admin: bool = False) -> Path:
        # Always keep per-user folder created for consistency.
        self.ensure_user_workspace(username)
        workspace = (
            self.user_workspaces_root.resolve()
            if bool(is_admin)
            else self._workspace_dir_for_user(username)
        )
        self.active_username = str(username).strip()
        self.user_workspace_root = workspace
        self.working_dir = workspace
        return workspace

    def clear_active_user_workspace(self) -> None:
        self.active_username = None
        self.user_workspace_root = None
        self.working_dir = Path.cwd().resolve()

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

    def _runner_busy(self) -> bool:
        if self.runner is None:
            return False
        is_running = getattr(self.runner, "is_running", None)
        if callable(is_running):
            return bool(is_running())
        thread = getattr(self.runner, "task_thread", None)
        return bool(thread and thread.is_alive())

    def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        user = self.auth_store.authenticate(username, password)
        workspace = self.activate_user_workspace(
            user.username,
            is_admin=self._is_admin_roles(list(user.roles)),
        )
        return {
            "username": user.username,
            "roles": list(user.roles),
            "permissions": list(user.permissions),
            "workspace_dir": str(workspace),
        }

    def has_permission(self, username: Optional[str], permission: str) -> bool:
        user_name = str(username or "").strip()
        if not user_name:
            return False
        return self.auth_store.has_permission(user_name, permission)

    def require_permission(
        self,
        username: Optional[str],
        permission: str,
        *,
        allow_anonymous: bool = True,
    ) -> None:
        user_name = str(username or "").strip()
        if not user_name:
            if allow_anonymous:
                return
            raise AuthorizationError("Please sign in first.")
        self.auth_store.require_permission(user_name, permission)

    def get_system_status(self, *, actor: Optional[str] = None) -> Dict[str, Any]:
        self.require_permission(actor, "system:view", allow_anonymous=False)
        return collect_system_status(
            working_dir=self.working_dir,
            runner_busy=self._runner_busy(),
        )

    def auth_metadata(self) -> Dict[str, Any]:
        return {
            "auth_file": str(self.auth_store.store_path),
            "default_admin_password_in_use": bool(self.auth_store.default_admin_password_in_use),
        }

    def list_accounts(self, *, actor: Optional[str]) -> list[Dict[str, Any]]:
        self.require_permission(actor, "account:manage", allow_anonymous=False)
        users = self.auth_store.list_users()
        for item in users:
            username = str(item.get("username", "")).strip()
            if username:
                item["workspace_dir"] = str(self._workspace_dir_for_user(username))
        return users

    def list_account_roles(self, *, actor: Optional[str]) -> Dict[str, Dict[str, Any]]:
        self.require_permission(actor, "account:manage", allow_anonymous=False)
        return self.auth_store.list_roles()

    def create_account(
        self,
        *,
        actor: Optional[str],
        username: str,
        password: str,
        roles: list[str],
    ) -> Dict[str, Any]:
        self.require_permission(actor, "account:manage", allow_anonymous=False)
        account = self.auth_store.create_user(username=username, password=password, roles=roles)
        workspace = self.ensure_user_workspace(username)
        account["workspace_dir"] = str(workspace)
        return account

    def set_account_roles(
        self,
        *,
        actor: Optional[str],
        username: str,
        roles: list[str],
    ) -> Dict[str, Any]:
        self.require_permission(actor, "account:manage", allow_anonymous=False)
        return self.auth_store.set_user_roles(username=username, roles=roles)

    def set_account_active(
        self,
        *,
        actor: Optional[str],
        username: str,
        active: bool,
    ) -> Dict[str, Any]:
        self.require_permission(actor, "account:manage", allow_anonymous=False)
        return self.auth_store.set_user_active(username=username, active=active)

    def set_account_password(
        self,
        *,
        actor: Optional[str],
        username: str,
        new_password: str,
    ) -> Dict[str, Any]:
        self.require_permission(actor, "account:manage", allow_anonymous=False)
        return self.auth_store.set_user_password(username=username, new_password=new_password)

    def change_own_password(
        self,
        *,
        actor: Optional[str],
        current_password: str,
        new_password: str,
    ) -> Dict[str, Any]:
        actor_name = str(actor or "").strip()
        if not actor_name:
            raise AuthorizationError("Please sign in first.")
        self.auth_store.authenticate(actor_name, current_password)
        return self.auth_store.set_user_password(username=actor_name, new_password=new_password)
