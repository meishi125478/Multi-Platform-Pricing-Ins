from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from typing import List, Optional

try:
    from .utils.run_logging import configure_run_logging  # type: ignore
except Exception:  # pragma: no cover
    try:
        from utils.run_logging import configure_run_logging  # type: ignore
    except Exception:  # pragma: no cover
        configure_run_logging = None  # type: ignore


def _split_argv(argv: List[str]) -> tuple[List[str], List[str]]:
    if "--" not in argv:
        raise ValueError("Missing '--' separator before the command to run.")
    idx = argv.index("--")
    return argv[:idx], argv[idx + 1 :]


def _kill_process_tree(pid: int) -> None:
    if pid <= 0:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    try:
        os.killpg(pid, 15)
        time.sleep(2)
        os.killpg(pid, 9)
    except Exception:
        try:
            os.kill(pid, 9)
        except Exception:
            pass


def _reader_thread(
    proc: subprocess.Popen, last_output_ts: dict, prefix: str = ""
) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        last_output_ts["ts"] = time.time()
        if prefix:
            sys.stdout.write(prefix)
        sys.stdout.write(line)
        sys.stdout.flush()


def _parse_args(before_cmd: List[str], cmd: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a command under a simple watchdog: if there is no stdout/stderr "
            "output for N seconds, kill the whole process tree and restart. "
            "Designed to pair with optuna_storage so BayesOpt can resume."
        )
    )
    parser.add_argument(
        "--idle-seconds",
        type=int,
        default=7200,
        help="Restart if there is no output for this many seconds (default: 7200).",
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=50,
        help="Maximum restart attempts (default: 50).",
    )
    parser.add_argument(
        "--restart-delay-seconds",
        type=int,
        default=10,
        help="Delay between restarts (default: 10).",
    )
    parser.add_argument(
        "--stop-on-nonzero-exit",
        action="store_true",
        help="If the command exits non-zero, stop instead of restarting.",
    )
    args = parser.parse_args(before_cmd)
    if not cmd:
        parser.error("Empty command after '--'.")
    return args


def run_with_watchdog(
    cmd: List[str],
    idle_seconds: int,
    max_restarts: int,
    restart_delay_seconds: int,
    stop_on_nonzero_exit: bool,
) -> int:
    idle_seconds = max(1, int(idle_seconds))
    max_restarts = max(0, int(max_restarts))
    restart_delay_seconds = max(0, int(restart_delay_seconds))

    attempt = 0
    while True:
        attempt += 1
        print(
            f"[watchdog] start attempt={attempt} idle_seconds={idle_seconds} cmd={cmd}",
            flush=True,
        )

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
            universal_newlines=True,
            creationflags=creationflags,
            start_new_session=start_new_session,
        )

        last_output_ts: dict = {"ts": time.time()}
        reader = threading.Thread(
            target=_reader_thread,
            args=(proc, last_output_ts),
            kwargs={"prefix": ""},
            daemon=True,
        )
        reader.start()

        killed_for_idle = False
        exit_code: Optional[int] = None
        while True:
            exit_code = proc.poll()
            if exit_code is not None:
                break
            idle_for = time.time() - float(last_output_ts["ts"])
            if idle_for > idle_seconds:
                killed_for_idle = True
                print(
                    f"[watchdog] idle>{idle_seconds}s (idle_for={int(idle_for)}s), killing pid={proc.pid}",
                    flush=True,
                )
                _kill_process_tree(proc.pid)
                break
            time.sleep(5)

        try:
            proc.wait(timeout=30)
        except Exception:
            _kill_process_tree(proc.pid)

        if exit_code is None:
            exit_code = proc.poll() or 1

        if exit_code == 0:
            print("[watchdog] finished with exit_code=0", flush=True)
            return 0

        if stop_on_nonzero_exit and not killed_for_idle:
            print(
                f"[watchdog] command exited non-zero (exit_code={exit_code}); stop.",
                flush=True,
            )
            return int(exit_code)

        if attempt > max_restarts + 1:
            print(
                f"[watchdog] exceeded max_restarts={max_restarts}; last exit_code={exit_code}",
                flush=True,
            )
            return int(exit_code)

        print(
            f"[watchdog] restart in {restart_delay_seconds}s (exit_code={exit_code}, killed_for_idle={killed_for_idle})",
            flush=True,
        )
        if restart_delay_seconds:
            time.sleep(restart_delay_seconds)


def main(argv: Optional[List[str]] = None) -> int:
    if configure_run_logging:
        configure_run_logging(prefix="watchdog")
    argv = list(sys.argv[1:] if argv is None else argv)
    before_cmd, cmd = _split_argv(argv)
    args = _parse_args(before_cmd, cmd)
    return run_with_watchdog(
        cmd=cmd,
        idle_seconds=args.idle_seconds,
        max_restarts=args.max_restarts,
        restart_delay_seconds=args.restart_delay_seconds,
        stop_on_nonzero_exit=bool(args.stop_on_nonzero_exit),
    )


if __name__ == "__main__":
    raise SystemExit(main())
