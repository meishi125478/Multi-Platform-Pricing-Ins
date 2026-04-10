from __future__ import annotations

from ins_pricing.frontend.app_controller import PricingApp


class _FakeRunner:
    def run_task(self, _config_path: str):
        for i in range(120):
            yield f"line-{i:03d} " + ("x" * 20)


def test_run_training_caps_log_tail(monkeypatch) -> None:
    monkeypatch.setenv("INS_PRICING_FRONTEND_LOG_MAX_CHARS", "240")
    app = PricingApp()
    app.runner = _FakeRunner()

    updates = list(app.run_training('{"runner": {"mode": "entry"}}'))
    assert updates

    final_status, final_log = updates[-1]
    assert "completed" in final_status.lower()
    assert len(final_log) <= 240
    assert "line-000" not in final_log
    assert "line-119" in final_log


def test_run_training_denies_viewer_actor(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INS_PRICING_FRONTEND_AUTH_FILE", str(tmp_path / "frontend_users.json"))
    app = PricingApp()
    app.auth_store.create_user("viewer_1", "pass123", ["viewer"])
    app.runner = _FakeRunner()

    updates = list(app.run_training('{"runner": {"mode": "entry"}}', actor="viewer_1"))
    assert updates
    final_status, final_log = updates[-1]
    assert "permission denied" in final_status.lower()
    assert "task:run" in final_log.lower()


def test_run_training_allows_operator_actor(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INS_PRICING_FRONTEND_AUTH_FILE", str(tmp_path / "frontend_users.json"))
    app = PricingApp()
    app.auth_store.create_user("operator_1", "pass123", ["operator"])
    app.runner = _FakeRunner()

    updates = list(app.run_training('{"runner": {"mode": "entry"}}', actor="operator_1"))
    assert updates
    final_status, final_log = updates[-1]
    assert "completed" in final_status.lower()
    assert "line-119" in final_log
