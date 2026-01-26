"""
Unified Task Runner with Real-time Logging
Executes model training, explanation, plotting, and other tasks based on config.
"""

import sys
import threading
import queue
import time
import json
import subprocess
from pathlib import Path
from typing import Generator, Optional, Dict, Any, List, Sequence, Tuple
import logging


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
            print(f"Warning: Could not detect task mode, defaulting to 'entry': {e}")
            return 'entry'

    def _build_cmd_from_config(self, config_path: str) -> Tuple[List[str], str]:
        """
        Build the command to execute based on config.runner.mode, mirroring
        ins_pricing.cli.utils.notebook_utils.run_from_config behavior.
        """
        from ins_pricing.cli.utils.cli_config import set_env
        from ins_pricing.cli.utils.notebook_utils import (
            build_bayesopt_entry_cmd,
            build_incremental_cmd,
            build_explain_cmd,
            wrap_with_watchdog,
        )

        cfg_path = Path(config_path).resolve()
        raw = json.loads(cfg_path.read_text(encoding="utf-8", errors="replace"))
        set_env(raw.get("env", {}))
        runner = dict(raw.get("runner") or {})

        mode = str(runner.get("mode") or "entry").strip().lower()
        use_watchdog = bool(runner.get("use_watchdog", False))
        if mode == "watchdog":
            use_watchdog = True
            mode = "entry"

        idle_seconds = int(runner.get("idle_seconds", 7200))
        max_restarts = int(runner.get("max_restarts", 50))
        restart_delay_seconds = int(runner.get("restart_delay_seconds", 10))

        if mode == "incremental":
            inc_args = runner.get("incremental_args") or []
            if not isinstance(inc_args, list):
                raise ValueError("config.runner.incremental_args must be a list of strings.")
            cmd = build_incremental_cmd(cfg_path, extra_args=[str(x) for x in inc_args])
            if use_watchdog:
                cmd = wrap_with_watchdog(
                    cmd,
                    idle_seconds=idle_seconds,
                    max_restarts=max_restarts,
                    restart_delay_seconds=restart_delay_seconds,
                )
            return cmd, "incremental"

        if mode == "explain":
            exp_args = runner.get("explain_args") or []
            if not isinstance(exp_args, list):
                raise ValueError("config.runner.explain_args must be a list of strings.")
            cmd = build_explain_cmd(cfg_path, extra_args=[str(x) for x in exp_args])
            if use_watchdog:
                cmd = wrap_with_watchdog(
                    cmd,
                    idle_seconds=idle_seconds,
                    max_restarts=max_restarts,
                    restart_delay_seconds=restart_delay_seconds,
                )
            return cmd, "explain"

        if mode != "entry":
            raise ValueError(
                f"Unsupported runner.mode={mode!r}, expected 'entry', 'incremental', or 'explain'."
            )

        model_keys = runner.get("model_keys") or raw.get("model_keys") or ["ft"]
        if not isinstance(model_keys, list):
            raise ValueError("runner.model_keys must be a list of strings.")
        nproc_per_node = int(runner.get("nproc_per_node", 1))
        max_evals = int(runner.get("max_evals", raw.get("max_evals", 50)))
        plot_curves = bool(runner.get("plot_curves", raw.get("plot_curves", True)))
        ft_role = runner.get("ft_role", raw.get("ft_role"))

        extra_args: List[str] = ["--max-evals", str(max_evals)]
        if plot_curves:
            extra_args.append("--plot-curves")
        if ft_role:
            extra_args += ["--ft-role", str(ft_role)]

        cmd = build_bayesopt_entry_cmd(
            cfg_path,
            model_keys=[str(x) for x in model_keys],
            nproc_per_node=nproc_per_node,
            extra_args=extra_args,
        )
        if use_watchdog:
            cmd = wrap_with_watchdog(
                cmd,
                idle_seconds=idle_seconds,
                max_restarts=max_restarts,
                restart_delay_seconds=restart_delay_seconds,
            )
        return cmd, "entry"

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
                    print(f"Starting task [{task_mode}] with config: {config_path}")
                    print("=" * 80)

                    # Run subprocess with streamed output
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        cwd=str(Path(config_path).resolve().parent),
                    )
                    if proc.stdout is not None:
                        for line in proc.stdout:
                            print(line.rstrip())
                    return_code = proc.wait()
                    if return_code != 0:
                        raise RuntimeError(f"Task exited with code {return_code}")

                    print("=" * 80)
                    print(f"Task [{task_mode}] completed successfully!")

                except Exception as e:
                    exception_holder.append(e)
                    print(f"Error during task execution: {str(e)}")
                    import traceback
                    print(traceback.format_exc())

                finally:
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
                    print(f"Error during task execution: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
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

        if self.task_thread and self.task_thread.is_alive():
            # Note: Thread.join() will wait for completion
            # For forceful termination, you may need to use process-based approach
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
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    # Add handler to logger
    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger
