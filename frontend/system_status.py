"""System status sampling for frontend runtime capacity checks."""

from __future__ import annotations

import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - environment-dependent import
    psutil = None


def _env_threshold(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _should_collect_gpu() -> bool:
    """GPU monitoring is Linux-first; allow explicit override for debugging."""
    force_raw = os.environ.get("INS_PRICING_FRONTEND_GPU_MONITOR_FORCE", "").strip().lower()
    if force_raw in {"1", "true", "yes", "on"}:
        return True
    if force_raw in {"0", "false", "no", "off"}:
        return False
    return platform.system().lower() == "linux"


def _collect_gpu_status() -> Dict[str, Any]:
    """Collect GPU metrics via nvidia-smi. Returns unavailable when absent."""
    base: Dict[str, Any] = {"available": False, "count": 0, "devices": []}
    if not _should_collect_gpu():
        return base

    nvidia_smi_bin = (
        os.environ.get("INS_PRICING_FRONTEND_NVIDIA_SMI_PATH", "").strip() or "nvidia-smi"
    )
    query = (
        "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu"
    )
    try:
        completed = subprocess.run(
            [
                nvidia_smi_bin,
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
    except Exception:
        return base

    devices = []
    for raw_line in completed.stdout.splitlines():
        line = str(raw_line or "").strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            index = int(parts[0])
            name = parts[1]
            utilization_percent = float(parts[2])
            memory_used_mb = float(parts[3])
            memory_total_mb = float(parts[4])
            temperature_raw = str(parts[5] or "").strip().lower()
            temperature_c = (
                None if temperature_raw in {"", "n/a", "na"} else float(parts[5])
            )
        except (TypeError, ValueError):
            continue

        memory_percent = (
            (memory_used_mb / memory_total_mb) * 100.0 if memory_total_mb > 0 else 0.0
        )
        devices.append(
            {
                "index": index,
                "name": name,
                "utilization_percent": utilization_percent,
                "memory_used_mb": memory_used_mb,
                "memory_total_mb": memory_total_mb,
                "memory_percent": memory_percent,
                "temperature_c": temperature_c,
            }
        )

    if not devices:
        return base

    return {
        "available": True,
        "count": len(devices),
        "devices": devices,
        "max_utilization_percent": max(item["utilization_percent"] for item in devices),
        "max_memory_percent": max(item["memory_percent"] for item in devices),
    }


def collect_system_status(*, working_dir: Path, runner_busy: bool) -> Dict[str, Any]:
    """Collect a lightweight system snapshot and task admission suggestion."""
    cpu_limit = _env_threshold("INS_PRICING_FRONTEND_CPU_LIMIT_PCT", 85.0)
    memory_limit = _env_threshold("INS_PRICING_FRONTEND_MEMORY_LIMIT_PCT", 85.0)
    disk_limit = _env_threshold("INS_PRICING_FRONTEND_DISK_LIMIT_PCT", 95.0)
    gpu_util_limit = _env_threshold("INS_PRICING_FRONTEND_GPU_UTIL_LIMIT_PCT", 95.0)
    gpu_memory_limit = _env_threshold("INS_PRICING_FRONTEND_GPU_MEMORY_LIMIT_PCT", 95.0)

    base: Dict[str, Any] = {
        "timestamp": time.time(),
        "host": platform.node(),
        "runner_busy": bool(runner_busy),
        "thresholds": {
            "cpu_percent": cpu_limit,
            "memory_percent": memory_limit,
            "disk_percent": disk_limit,
            "gpu_util_percent": gpu_util_limit,
            "gpu_memory_percent": gpu_memory_limit,
        },
        "gpu": _collect_gpu_status(),
    }

    if psutil is None:
        decision = "Task running" if runner_busy else "Resource probe unavailable"
        reasons = ["psutil not installed"]
        if runner_busy:
            reasons.append("a task is currently running")
        base.update(
            {
                "probe_available": False,
                "cpu_percent": None,
                "memory_percent": None,
                "disk_percent": None,
                "process_memory_mb": None,
                "memory_used_mb": None,
                "memory_total_mb": None,
                "disk_used_gb": None,
                "disk_total_gb": None,
                "cpu_logical_cores": None,
                "can_start_task": not runner_busy,
                "decision": decision,
                "reasons": reasons,
            }
        )
        return base

    target_path = Path(working_dir).expanduser().resolve()
    if not target_path.exists():
        target_path = Path.cwd().resolve()

    cpu_percent = float(psutil.cpu_percent(interval=None))
    cpu_logical_cores = int(psutil.cpu_count(logical=True) or 0)
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(str(target_path))
    process_memory_mb = float(psutil.Process(os.getpid()).memory_info().rss) / 1024.0 / 1024.0
    memory_used_mb = float(vm.used) / 1024.0 / 1024.0
    memory_total_mb = float(vm.total) / 1024.0 / 1024.0
    disk_used_gb = float(disk.used) / 1024.0 / 1024.0 / 1024.0
    disk_total_gb = float(disk.total) / 1024.0 / 1024.0 / 1024.0

    reasons = []
    if runner_busy:
        reasons.append("a task is currently running")
    if cpu_percent >= cpu_limit:
        reasons.append(f"cpu usage {cpu_percent:.1f}% >= {cpu_limit:.1f}%")
    if float(vm.percent) >= memory_limit:
        reasons.append(f"memory usage {float(vm.percent):.1f}% >= {memory_limit:.1f}%")
    if float(disk.percent) >= disk_limit:
        reasons.append(f"disk usage {float(disk.percent):.1f}% >= {disk_limit:.1f}%")
    gpu_info = base.get("gpu", {}) if isinstance(base.get("gpu"), dict) else {}
    if bool(gpu_info.get("available", False)):
        for device in gpu_info.get("devices", []):
            util = float(device.get("utilization_percent", 0.0))
            mem_pct = float(device.get("memory_percent", 0.0))
            idx = int(device.get("index", 0))
            if util >= gpu_util_limit:
                reasons.append(f"gpu[{idx}] utilization {util:.1f}% >= {gpu_util_limit:.1f}%")
            if mem_pct >= gpu_memory_limit:
                reasons.append(f"gpu[{idx}] memory {mem_pct:.1f}% >= {gpu_memory_limit:.1f}%")

    can_start = len(reasons) == 0
    decision = "Ready for new task" if can_start else "Not recommended to start new task"

    base.update(
        {
            "probe_available": True,
            "cpu_percent": cpu_percent,
            "cpu_logical_cores": cpu_logical_cores,
            "memory_percent": float(vm.percent),
            "memory_used_mb": memory_used_mb,
            "memory_total_mb": memory_total_mb,
            "disk_percent": float(disk.percent),
            "disk_used_gb": disk_used_gb,
            "disk_total_gb": disk_total_gb,
            "process_memory_mb": process_memory_mb,
            "can_start_task": can_start,
            "decision": decision,
            "reasons": reasons or ["all metrics below thresholds"],
        }
    )
    return base
