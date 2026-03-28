"""
Unified Task Runner with Real-time Logging.

Executes model training and in-process frontend workflows while streaming logs
through task-scoped queues (without mutating global stdout/stderr).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import signal
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from ins_pricing.frontend.logging_utils import get_frontend_logger, log_print

_logger = get_frontend_logger("ins_pricing.frontend.runner")


def _log(*args, **kwargs) -> None:
    log_print(_logger, *args, **kwargs)


class LogCapture:
    """Task-scoped log sink backed by a queue."""

    def __init__(self):
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.stop_flag = threading.Event()

    def put(self, text: str) -> None:
        if text and text.strip():
            self.log_queue.put(text)


class _QueueLogHandler(logging.Handler):
    """Route log records to a LogCapture queue."""

    def __init__(self, capture: LogCapture):
        super().__init__(level=logging.INFO)
        self._capture = capture

    def emit(self, record: logging.LogRecord) -> None:
        try:
            rendered = self.format(record) if self.formatter else record.getMessage()
            self._capture.put(rendered)
        except Exception:
            return


class TaskRunner:
    """
    Run model tasks (training, explain, plotting, etc.) and capture logs.

    Supports all task modes defined in config.runner.mode:
    - entry: Standard model training
    - explain: Model explanation (permutation, SHAP, etc.)
    - incremental: Incremental training
    - watchdog: Watchdog mode for monitoring
    """

    def __init__(self):
        self.task_thread: Optional[threading.Thread] = None
        self.log_capture: Optional[LogCapture] = None
        self._proc: Optional[subprocess.Popen] = None

    def _detect_task_mode(self, config_path: str) -> str:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            runner_config = config.get("runner", {})
            mode = runner_config.get("mode", "entry")
            return str(mode).lower()
        except Exception as exc:
            _log(f"Warning: Could not detect task mode, defaulting to 'entry': {exc}")
            return "entry"

    def _build_cmd_from_config(self, config_path: str) -> Tuple[List[str], str, Dict[str, str]]:
        """Build the command to execute based on config.runner.mode."""
        from ins_pricing.cli.utils.notebook_utils import build_run_spec_from_config

        return build_run_spec_from_config(config_path)

    def _emit_task_log(self, text: str) -> None:
        if self.log_capture is not None:
            self.log_capture.put(text)

    def _stream_logs(
        self,
        *,
        exception_holder: List[Exception],
    ) -> Generator[str, None, None]:
        last_update = time.time()
        assert self.log_capture is not None
        while self.task_thread and (self.task_thread.is_alive() or not self.log_capture.log_queue.empty()):
            try:
                log_line = self.log_capture.log_queue.get(timeout=0.1)
                yield log_line
                last_update = time.time()
            except queue.Empty:
                if time.time() - last_update > 5:
                    yield "."
                    last_update = time.time()
                continue

        if self.task_thread:
            self.task_thread.join(timeout=1)
        if exception_holder:
            raise exception_holder[0]

    def run_task(self, config_path: str) -> Generator[str, None, None]:
        """
        Run task based on config file with real-time log capture.

        Args:
            config_path: Path to configuration JSON file

        Yields:
            Log lines as they are generated
        """
        self.log_capture = LogCapture()
        exception_holder: List[Exception] = []

        def task_worker() -> None:
            try:
                cmd, task_mode_inner, cmd_env = self._build_cmd_from_config(config_path)
                self._emit_task_log(
                    f"Starting task [{task_mode_inner}] with config: {config_path}"
                )
                self._emit_task_log("=" * 80)

                creationflags = 0
                start_new_session = False
                if os.name == "nt":
                    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                else:
                    start_new_session = True

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=str(Path(config_path).resolve().parent),
                    env=cmd_env,
                    creationflags=creationflags,
                    start_new_session=start_new_session,
                )
                self._proc = proc
                if proc.stdout is not None:
                    for line in proc.stdout:
                        self._emit_task_log(line.rstrip())
                return_code = proc.wait()
                if return_code != 0:
                    raise RuntimeError(f"Task exited with code {return_code}")

                self._emit_task_log("=" * 80)
                self._emit_task_log(f"Task [{task_mode_inner}] completed successfully!")
            except Exception as exc:
                exception_holder.append(exc)
                self._emit_task_log(f"Error during task execution: {exc}")
                self._emit_task_log(traceback.format_exc())
            finally:
                self._proc = None

        self.task_thread = threading.Thread(target=task_worker, daemon=True)
        self.task_thread.start()
        yield from self._stream_logs(exception_holder=exception_holder)

    def run_callable(self, func, *args, **kwargs) -> Generator[str, None, None]:
        """Run an in-process callable and stream task-scoped logger output."""
        self.log_capture = LogCapture()
        exception_holder: List[Exception] = []
        package_logger = logging.getLogger("ins_pricing")
        queue_handler = _QueueLogHandler(self.log_capture)
        queue_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        thread_id_holder: Dict[str, Optional[int]] = {"value": None}

        class _ThreadFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                target_id = thread_id_holder["value"]
                if target_id is None:
                    return False
                return int(record.thread) == int(target_id)

        queue_handler.addFilter(_ThreadFilter())

        def task_worker() -> None:
            thread_id_holder["value"] = threading.get_ident()
            package_logger.addHandler(queue_handler)
            try:
                func(*args, **kwargs)
            except Exception as exc:
                exception_holder.append(exc)
                self._emit_task_log(f"Error during task execution: {exc}")
                self._emit_task_log(traceback.format_exc())
            finally:
                try:
                    package_logger.removeHandler(queue_handler)
                except Exception:
                    pass
                try:
                    queue_handler.close()
                except Exception:
                    pass

        self.task_thread = threading.Thread(target=task_worker, daemon=True)
        self.task_thread.start()
        yield from self._stream_logs(exception_holder=exception_holder)

    def stop_task(self):
        """Stop the current task process."""
        if self.log_capture:
            self.log_capture.stop_flag.set()

        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                else:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except Exception:
                        proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    if os.name == "nt":
                        proc.kill()
                    else:
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        except Exception:
                            proc.kill()
            except Exception:
                try:
                    if os.name != "nt":
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        except Exception:
                            proc.kill()
                    else:
                        proc.kill()
                except Exception:
                    pass

        if self.task_thread and self.task_thread.is_alive():
            self.task_thread.join(timeout=5)


def setup_logger(name: str = "task") -> logging.Logger:
    """
    Set up a logger for task execution.

    Args:
        name: Logger name

    Returns:
        Configured logger instance
    """
    logger_name = name if "." in name else f"ins_pricing.frontend.{name}"
    logger = get_frontend_logger(logger_name)
    logger.setLevel(logging.INFO)
    return logger
