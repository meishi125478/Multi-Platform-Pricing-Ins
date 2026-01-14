from __future__ import annotations

import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO

_LOGGING_CONFIGURED = False
_LOG_PATH: Optional[Path] = None
_LOG_FILE: Optional[TextIO] = None

_TRUTHY = {"1", "true", "yes", "y", "on"}


class _TeeStream:
    def __init__(self, primary: TextIO, secondary: TextIO) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, data: str) -> int:
        if not data:
            return 0
        try:
            self._primary.write(data)
        except Exception:
            pass
        try:
            self._secondary.write(data)
        except Exception:
            pass
        return len(data)

    def flush(self) -> None:
        for stream in (self._primary, self._secondary):
            try:
                stream.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return bool(getattr(self._primary, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self._primary.fileno()

    def __getattr__(self, name: str):
        return getattr(self._primary, name)


def _is_truthy(value: Optional[str]) -> bool:
    return str(value).strip().lower() in _TRUTHY


def _resolve_log_dir(log_dir: Optional[str | Path]) -> Optional[Path]:
    candidates: list[Path] = []
    if log_dir:
        candidates.append(Path(log_dir).expanduser())
    env_dir = os.environ.get("INS_PRICING_LOG_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.append(Path.cwd() / "logs")
    candidates.append(Path.home() / ".ins_pricing" / "logs")
    candidates.append(Path(tempfile.gettempdir()) / "ins_pricing_logs")

    for cand in candidates:
        try:
            cand.mkdir(parents=True, exist_ok=True)
            return cand
        except Exception:
            continue
    return None


def _build_log_filename(prefix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rank = (
        os.environ.get("RANK")
        or os.environ.get("LOCAL_RANK")
        or os.environ.get("SLURM_PROCID")
    )
    suffix = f"r{rank}" if rank is not None else f"pid{os.getpid()}"
    safe_prefix = "".join(
        ch if ch.isalnum() or ch in "-_." else "_" for ch in prefix)
    return f"{safe_prefix}_{ts}_{suffix}.log"


def configure_run_logging(
    *,
    prefix: str = "ins_pricing",
    log_dir: Optional[str | Path] = None,
    level: int = logging.INFO,
    announce: bool = True,
) -> Optional[Path]:
    global _LOGGING_CONFIGURED, _LOG_PATH, _LOG_FILE

    if _LOGGING_CONFIGURED:
        return _LOG_PATH
    if _is_truthy(os.environ.get("INS_PRICING_LOG_DISABLE")):
        return None

    resolved_dir = _resolve_log_dir(log_dir)
    if resolved_dir is None:
        return None

    log_path = resolved_dir / _build_log_filename(prefix)
    try:
        log_file = log_path.open("a", encoding="utf-8")
    except Exception:
        return None

    sys.stdout = _TeeStream(sys.stdout, log_file)  # type: ignore[assignment]
    sys.stderr = _TeeStream(sys.stderr, log_file)  # type: ignore[assignment]
    _LOG_FILE = log_file
    _LOG_PATH = log_path
    _LOGGING_CONFIGURED = True

    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
    else:
        root.setLevel(level)

    if announce:
        print(f"[ins_pricing] log saved to {log_path}", flush=True)

    return log_path
