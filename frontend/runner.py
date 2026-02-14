"""
Unified Task Runner with Real-time Logging
Executes model training, explanation, plotting, and other tasks based on config.
"""

import sys
import os
import threading
import queue
import time
import json
import subprocess
import signal
from pathlib import Path
from typing import Generator, Optional, Dict, Any, List, Sequence, Tuple
import logging

from ins_pricing.utils import get_logger, log_print

_logger = get_logger("ins_pricing.frontend.runner")


def _log(*args, **kwargs) -> None:
    log_print(_logger, *args, **kwargs)

class LogCapture:
    """Capture stdout and stderr for real-time display."""

    def __init__(self):
        self.log_queue = queue.Queue()
        self.stop_flag = threading.Event()

    def write(self, text: str):
        """Write method for capturing output."""
        if text and text.strip():
            self.log_queue.put(text)

    def flush(self):
        """Flush method (required for file-like objects)."""
        pass


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
        self.task_thread = None
        self.log_capture = None
        self._proc: Optional[subprocess.Popen] = None

    def _detect_task_mode(self, config_path: str) -> str:
        """
        Detect the task mode from config file.

        Args:
            config_path: Path to configuration JSON file

        Returns:
            Task mode string (e.g., 'entry', 'explain', 'incremental', 'watchdog')
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            runner_config = config.get('runner', {})
            mode = runner_config.get('mode', 'entry')
            return str(mode).lower()

        except Exception as e:
            _log(f"Warning: Could not detect task mode, defaulting to 'entry': {e}")
            return 'entry'

    def _build_cmd_from_config(self, config_path: str) -> Tuple[List[str], str]:
        """Build the command to execute based on config.runner.mode."""
        from ins_pricing.cli.utils.notebook_utils import build_cmd_from_config

        return build_cmd_from_config(config_path)

    def run_task(self, config_path: str) -> Generator[str, None, None]:
        """
        Run task based on config file with real-time log capture.

        This method automatically detects the task mode from the config file
        (training, explain, plotting, etc.) and runs the appropriate task.

        Args:
            config_path: Path to configuration JSON file

        Yields:
            Log lines as they are generated
        """
        self.log_capture = LogCapture()

        # Configure logging to capture both file and stream output
        log_handler = logging.StreamHandler(self.log_capture)
        log_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        log_handler.setFormatter(formatter)

        # Add handler to root logger
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers.copy()
        root_logger.addHandler(log_handler)

        # Store original stdout/stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        try:
            # Detect task mode
            task_mode = self._detect_task_mode(config_path)

            # Start task in separate thread
            exception_holder = []

            def task_worker():
                try:
                    sys.stdout = self.log_capture
                    sys.stderr = self.log_capture

                    # Log start
                    cmd, task_mode = self._build_cmd_from_config(config_path)
                    _log(f"Starting task [{task_mode}] with config: {config_path}")
                    _log("=" * 80)

                    # Run subprocess with streamed output
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
                        creationflags=creationflags,
                        start_new_session=start_new_session,
                    )
                    self._proc = proc
                    if proc.stdout is not None:
                        for line in proc.stdout:
                            _log(line.rstrip())
                    return_code = proc.wait()
                    if return_code != 0:
                        raise RuntimeError(f"Task exited with code {return_code}")

                    _log("=" * 80)
                    _log(f"Task [{task_mode}] completed successfully!")

                except Exception as e:
                    exception_holder.append(e)
                    _log(f"Error during task execution: {str(e)}")
                    import traceback
                    _log(traceback.format_exc())

                finally:
                    self._proc = None
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr

            self.task_thread = threading.Thread(target=task_worker, daemon=True)
            self.task_thread.start()

            # Yield log lines as they come in
            last_update = time.time()
            while self.task_thread.is_alive() or not self.log_capture.log_queue.empty():
                try:
                    # Try to get log with timeout
                    log_line = self.log_capture.log_queue.get(timeout=0.1)
                    yield log_line
                    last_update = time.time()

                except queue.Empty:
                    # Send heartbeat every 5 seconds
                    if time.time() - last_update > 5:
                        yield "."
                        last_update = time.time()
                    continue

            # Wait for thread to complete
            self.task_thread.join(timeout=1)

            # Check for exceptions
            if exception_holder:
                raise exception_holder[0]

        finally:
            # Restore original stdout/stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr

            # Restore original logging handlers
            root_logger.handlers = original_handlers

    def run_callable(self, func, *args, **kwargs) -> Generator[str, None, None]:
        """Run an in-process callable and stream stdout/stderr."""
        self.log_capture = LogCapture()

        log_handler = logging.StreamHandler(self.log_capture)
        log_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        log_handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers.copy()
        root_logger.addHandler(log_handler)

        original_stdout = sys.stdout
        original_stderr = sys.stderr

        try:
            exception_holder = []

            def task_worker():
                try:
                    sys.stdout = self.log_capture
                    sys.stderr = self.log_capture
                    func(*args, **kwargs)
                except Exception as e:
                    exception_holder.append(e)
                    _log(f"Error during task execution: {str(e)}")
                    import traceback
                    _log(traceback.format_exc())
                finally:
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr

            self.task_thread = threading.Thread(target=task_worker, daemon=True)
            self.task_thread.start()

            last_update = time.time()
            while self.task_thread.is_alive() or not self.log_capture.log_queue.empty():
                try:
                    log_line = self.log_capture.log_queue.get(timeout=0.1)
                    yield log_line
                    last_update = time.time()
                except queue.Empty:
                    if time.time() - last_update > 5:
                        yield "."
                        last_update = time.time()
                    continue

            self.task_thread.join(timeout=1)
            if exception_holder:
                raise exception_holder[0]
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            root_logger.handlers = original_handlers

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


# Backward compatibility aliases
TrainingRunner = TaskRunner


class StreamToLogger:
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


def setup_logger(name: str = "task") -> logging.Logger:
    """
    Set up a logger for task execution.

    Args:
        name: Logger name

    Returns:
        Configured logger instance
    """
    logger_name = name if "." in name else f"ins_pricing.frontend.{name}"
    logger = get_logger(logger_name)
    logger.setLevel(logging.INFO)
    return logger
