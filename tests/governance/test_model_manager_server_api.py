from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("nicegui")

from fastapi.testclient import TestClient
from nicegui import app

from model_manager_tool.server import configure_app


def test_model_manager_api_end_to_end(tmp_path: Path) -> None:
    data_root = tmp_path / "model_manager_api"
    configure_app(data_root=data_root, enable_purge_all=True)
    client = TestClient(app)

    whoami = client.get("/api/whoami")
    assert whoami.status_code == 200
    assert whoami.json() == {"authenticated": False}

    login = client.post("/api/login", json={"username": "admin", "password": "admin123!"})
    assert login.status_code == 200
    assert login.json().get("ok") is True

    list_empty = client.get("/api/models")
    assert list_empty.status_code == 200
    assert list_empty.json().get("count") == 0

    files = [("files", ("model.bin", b"binary-model", "application/octet-stream"))]
    data = {
        "model_name": "pricing_model",
        "version": "1.0.0",
        "model_type": "xgb",
        "notes": "api import",
        "tags": "{}",
        "metrics": "{}",
    }
    imported = client.post("/api/models/import", data=data, files=files)
    assert imported.status_code == 200
    assert imported.json().get("ok") is True

    listed = client.get("/api/models")
    assert listed.status_code == 200
    listed_payload = listed.json()
    assert listed_payload.get("count") == 1
    assert listed_payload["models"][0]["model_name"] == "pricing_model"

    deployed = client.post(
        "/api/deploy",
        json={
            "env": "staging",
            "model_name": "pricing_model",
            "version": "1.0.0",
            "note": "api smoke deploy",
        },
    )
    assert deployed.status_code == 200
    assert deployed.json().get("ok") is True

    env_state = client.get("/api/env/staging")
    assert env_state.status_code == 200
    assert env_state.json()["active"]["version"] == "1.0.0"

    audit = client.get("/api/audit?limit=20")
    assert audit.status_code == 200
    assert int(audit.json().get("count", 0)) >= 1

    roles = client.get("/api/admin/roles")
    assert roles.status_code == 200
    assert "admin" in roles.json().get("roles", {})

    purge = client.post("/api/admin/purge-models", json={"clear_audit": False})
    assert purge.status_code == 200
    assert purge.json().get("ok") is True

    listed_after_purge = client.get("/api/models")
    assert listed_after_purge.status_code == 200
    assert listed_after_purge.json().get("count") == 0

    logout = client.post("/api/logout")
    assert logout.status_code == 200
    assert logout.json().get("ok") is True

    whoami_after = client.get("/api/whoami")
    assert whoami_after.status_code == 200
    assert whoami_after.json() == {"authenticated": False}
