from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from ins_pricing.frontend import system_status


class _HealthyPsutil:
    @staticmethod
    def cpu_percent(interval=None):
        return 24.5

    @staticmethod
    def cpu_count(logical=True):
        return 16

    @staticmethod
    def virtual_memory():
        gib = 1024 * 1024 * 1024
        return SimpleNamespace(
            percent=36.0,
            used=12 * gib,
            total=32 * gib,
        )

    @staticmethod
    def disk_usage(_path):
        gib = 1024 * 1024 * 1024
        return SimpleNamespace(
            percent=41.0,
            used=410 * gib,
            total=1000 * gib,
        )

    @staticmethod
    def Process(_pid):
        return SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=512 * 1024 * 1024))


class _BusyPsutil:
    @staticmethod
    def cpu_percent(interval=None):
        return 96.0

    @staticmethod
    def cpu_count(logical=True):
        return 16

    @staticmethod
    def virtual_memory():
        gib = 1024 * 1024 * 1024
        return SimpleNamespace(
            percent=92.0,
            used=29.44 * gib,
            total=32 * gib,
        )

    @staticmethod
    def disk_usage(_path):
        gib = 1024 * 1024 * 1024
        return SimpleNamespace(
            percent=97.0,
            used=970 * gib,
            total=1000 * gib,
        )

    @staticmethod
    def Process(_pid):
        return SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=1024 * 1024 * 1024))


def test_collect_system_status_recommends_run_when_metrics_healthy(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(system_status, "psutil", _HealthyPsutil)
    monkeypatch.setattr(
        system_status,
        "_collect_gpu_status",
        lambda: {"available": False, "count": 0, "devices": []},
    )

    snapshot = system_status.collect_system_status(
        working_dir=Path(tmp_path),
        runner_busy=False,
    )

    assert snapshot["probe_available"] is True
    assert snapshot["can_start_task"] is True
    assert "ready" in snapshot["decision"].lower()
    assert snapshot["cpu_percent"] < 85
    assert snapshot["memory_used_mb"] > 0
    assert snapshot["memory_total_mb"] > snapshot["memory_used_mb"]
    assert snapshot["disk_used_gb"] > 0
    assert snapshot["disk_total_gb"] > snapshot["disk_used_gb"]
    assert snapshot["gpu"]["available"] is False


def test_collect_system_status_blocks_when_busy_or_over_threshold(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(system_status, "psutil", _BusyPsutil)
    monkeypatch.setattr(
        system_status,
        "_collect_gpu_status",
        lambda: {"available": False, "count": 0, "devices": []},
    )

    snapshot = system_status.collect_system_status(
        working_dir=Path(tmp_path),
        runner_busy=True,
    )

    assert snapshot["probe_available"] is True
    assert snapshot["can_start_task"] is False
    reasons = " ".join(snapshot.get("reasons", [])).lower()
    assert "task is currently running" in reasons
    assert "cpu usage" in reasons


def test_collect_system_status_includes_gpu_metrics(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(system_status, "psutil", _HealthyPsutil)
    monkeypatch.setattr(
        system_status,
        "_collect_gpu_status",
        lambda: {
            "available": True,
            "count": 1,
            "devices": [
                {
                    "index": 0,
                    "name": "RTX",
                    "utilization_percent": 12.5,
                    "memory_used_mb": 2048.0,
                    "memory_total_mb": 24576.0,
                    "memory_percent": 8.3,
                    "temperature_c": 45.0,
                }
            ],
            "max_utilization_percent": 12.5,
            "max_memory_percent": 8.3,
        },
    )

    snapshot = system_status.collect_system_status(
        working_dir=Path(tmp_path),
        runner_busy=False,
    )

    assert snapshot["gpu"]["available"] is True
    assert snapshot["gpu"]["count"] == 1
    device = snapshot["gpu"]["devices"][0]
    assert device["memory_used_mb"] == 2048.0
    assert device["utilization_percent"] == 12.5


def test_collect_gpu_status_skips_on_non_linux_by_default(monkeypatch) -> None:
    monkeypatch.setattr(system_status.platform, "system", lambda: "Windows")
    monkeypatch.delenv("INS_PRICING_FRONTEND_GPU_MONITOR_FORCE", raising=False)

    called = {"value": False}

    def _should_not_run(*args, **kwargs):
        called["value"] = True
        raise AssertionError("subprocess.run should not be called on non-Linux by default")

    monkeypatch.setattr(system_status.subprocess, "run", _should_not_run)
    snapshot = system_status._collect_gpu_status()

    assert snapshot["available"] is False
    assert called["value"] is False


def test_collect_gpu_status_force_enable_on_non_linux(monkeypatch) -> None:
    monkeypatch.setattr(system_status.platform, "system", lambda: "Windows")
    monkeypatch.setenv("INS_PRICING_FRONTEND_GPU_MONITOR_FORCE", "true")

    class _Completed:
        stdout = "0, RTX 4090, 22, 12000, 24576, 55\n"

    monkeypatch.setattr(
        system_status.subprocess,
        "run",
        lambda *args, **kwargs: _Completed(),
    )

    snapshot = system_status._collect_gpu_status()
    assert snapshot["available"] is True
    assert snapshot["count"] == 1
    assert snapshot["devices"][0]["name"] == "RTX 4090"
