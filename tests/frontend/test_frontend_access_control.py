from __future__ import annotations

import pytest

from ins_pricing.frontend.access_control import AuthorizationError, FrontendAuthStore


def test_bootstrap_admin_user_can_authenticate(tmp_path) -> None:
    store = FrontendAuthStore(
        tmp_path / "frontend_users.json",
        bootstrap_admin_password="secret123",
    )

    user = store.authenticate("admin", "secret123")
    assert user.username == "admin"
    assert "account:manage" in user.permissions
    assert "task:run" in user.permissions


def test_role_update_and_disable_flow(tmp_path) -> None:
    store = FrontendAuthStore(
        tmp_path / "frontend_users.json",
        bootstrap_admin_password="secret123",
    )
    store.create_user("viewer_1", "pass123", ["viewer"])
    assert store.has_permission("viewer_1", "system:view")
    assert not store.has_permission("viewer_1", "task:run")

    store.set_user_roles("viewer_1", ["operator"])
    assert store.has_permission("viewer_1", "task:run")

    store.set_user_active("viewer_1", False)
    with pytest.raises(AuthorizationError):
        store.authenticate("viewer_1", "pass123")
