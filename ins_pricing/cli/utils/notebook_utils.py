from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, cast

try:
    from .cli_config import add_config_json_arg  # type: ignore
except Exception:  # pragma: no cover
    from cli_config import add_config_json_arg  # type: ignore


def _find_ins_pricing_dir(cwd: Optional[Path] = None) -> Path:
    cwd = (cwd or Path().resolve()).resolve()
    pkg_root = Path(__file__).resolve().parents[2]
    candidates = [pkg_root, cwd / "ins_pricing", cwd, cwd.parent / "ins_pricing"]
    for cand in candidates:
        cli_entry = cand / "cli" / "BayesOpt_entry.py"
        cli_watchdog = cand / "cli" / "watchdog_run.py"
        if cli_entry.exists() and cli_watchdog.exists():
            return cand
    raise FileNotFoundError(
        "Cannot locate ins_pricing directory (expected cli/BayesOpt_entry.py and "
        "cli/watchdog_run.py). "
        f"cwd={cwd}"
    )


def _stringify_cmd(cmd: Sequence[object]) -> List[str]:
    return [str(x) for x in cmd]


def build_bayesopt_entry_cmd(
    config_json: str | Path,
    model_keys: Sequence[str],
    *,
    nproc_per_node: int = 1,
    standalone: bool = True,
    entry_script: str | Path = "cli/BayesOpt_entry.py",
    extra_args: Optional[Sequence[str]] = None,
) -> List[str]:
    """Build a command to run cli/BayesOpt_entry.py (optional torchrun/DDP)."""
    pkg_dir = _find_ins_pricing_dir()
    entry_script_path = Path(entry_script)
    if entry_script_path.is_absolute():
        entry_path = entry_script_path.resolve()
    else:
        candidate = pkg_dir / entry_script_path
        legacy = pkg_dir / "modelling" / entry_script_path
        entry_path = (
            candidate.resolve()
            if candidate.exists()
            else legacy.resolve()
            if legacy.exists()
            else candidate.resolve()
        )
    config_path = Path(config_json)
    if not config_path.is_absolute():
        config_path = (pkg_dir / config_path).resolve() if (pkg_dir / config_path).exists() else config_path.resolve()

    cmd: List[object]
    if int(nproc_per_node) > 1:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            *(["--standalone"] if standalone else []),
            f"--nproc_per_node={int(nproc_per_node)}",
            str(entry_path),
        ]
    else:
        cmd = [sys.executable, str(entry_path)]

    cmd += ["--config-json", str(config_path), "--model-keys", *list(model_keys)]
    if extra_args:
        cmd += list(extra_args)
    return _stringify_cmd(cmd)


def build_incremental_cmd(
    config_json: str | Path,
    *,
    entry_script: str | Path = "cli/BayesOpt_incremental.py",
    extra_args: Optional[Sequence[str]] = None,
) -> List[str]:
    """Build a command to run cli/BayesOpt_incremental.py."""
    pkg_dir = _find_ins_pricing_dir()
    entry_script_path = Path(entry_script)
    if entry_script_path.is_absolute():
        entry_path = entry_script_path.resolve()
    else:
        candidate = pkg_dir / entry_script_path
        legacy = pkg_dir / "modelling" / entry_script_path
        entry_path = (
            candidate.resolve()
            if candidate.exists()
            else legacy.resolve()
            if legacy.exists()
            else candidate.resolve()
        )
    config_path = Path(config_json)
    if not config_path.is_absolute():
        config_path = (pkg_dir / config_path).resolve() if (pkg_dir / config_path).exists() else config_path.resolve()

    cmd: List[object] = [sys.executable, str(entry_path), "--config-json", str(config_path)]
    if extra_args:
        cmd += list(extra_args)
    return _stringify_cmd(cmd)


def build_explain_cmd(
    config_json: str | Path,
    *,
    entry_script: str | Path = "cli/Explain_entry.py",
    extra_args: Optional[Sequence[str]] = None,
) -> List[str]:
    """Build a command to run cli/Explain_entry.py."""
    pkg_dir = _find_ins_pricing_dir()
    entry_script_path = Path(entry_script)
    if entry_script_path.is_absolute():
        entry_path = entry_script_path.resolve()
    else:
        candidate = pkg_dir / entry_script_path
        legacy = pkg_dir / "modelling" / entry_script_path
        entry_path = (
            candidate.resolve()
            if candidate.exists()
            else legacy.resolve()
            if legacy.exists()
            else candidate.resolve()
        )
    config_path = Path(config_json)
    if not config_path.is_absolute():
        config_path = (pkg_dir / config_path).resolve() if (pkg_dir / config_path).exists() else config_path.resolve()

    cmd: List[object] = [sys.executable, str(entry_path), "--config-json", str(config_path)]
    if extra_args:
        cmd += list(extra_args)
    return _stringify_cmd(cmd)


def wrap_with_watchdog(
    cmd: Sequence[str],
    *,
    idle_seconds: int = 7200,
    max_restarts: int = 50,
    restart_delay_seconds: int = 10,
    stop_on_nonzero_exit: bool = True,
    watchdog_script: str | Path = "cli/watchdog_run.py",
) -> List[str]:
    """Wrap a command with watchdog: restart when idle_seconds elapses with no output."""
    pkg_dir = _find_ins_pricing_dir()
    watchdog_script_path = Path(watchdog_script)
    if watchdog_script_path.is_absolute():
        watchdog_path = watchdog_script_path.resolve()
    else:
        candidate = pkg_dir / watchdog_script_path
        legacy = pkg_dir / "modelling" / watchdog_script_path
        watchdog_path = (
            candidate.resolve()
            if candidate.exists()
            else legacy.resolve()
            if legacy.exists()
            else candidate.resolve()
        )
    wd_cmd: List[object] = [
        sys.executable,
        str(watchdog_path),
        "--idle-seconds",
        str(int(idle_seconds)),
        "--max-restarts",
        str(int(max_restarts)),
        "--restart-delay-seconds",
        str(int(restart_delay_seconds)),
    ]
    if stop_on_nonzero_exit:
        wd_cmd.append("--stop-on-nonzero-exit")
    wd_cmd.append("--")
    wd_cmd.extend(list(cmd))
    return _stringify_cmd(wd_cmd)


def run(cmd: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Run an external command from a notebook (blocking)."""
    return subprocess.run(list(cmd), check=check)


def _build_config_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_config_json_arg(
        parser,
        help_text="Path to config.json (relative paths are resolved from ins_pricing/ when possible).",
    )
    return parser


def run_from_config_cli(
    description: str,
    argv: Optional[Sequence[str]] = None,
) -> subprocess.CompletedProcess:
    parser = _build_config_parser(description)
    args = parser.parse_args(argv)
    return run_from_config(args.config_json)


def run_bayesopt_entry(
    *,
    config_json: str | Path,
    model_keys: Sequence[str],
    max_evals: int = 50,
    plot_curves: bool = True,
    ft_role: Optional[str] = None,
    nproc_per_node: int = 1,
    use_watchdog: bool = False,
    idle_seconds: int = 7200,
    max_restarts: int = 50,
    restart_delay_seconds: int = 10,
    extra_args: Optional[Sequence[str]] = None,
) -> subprocess.CompletedProcess:
    """Convenience wrapper: build and run BayesOpt_entry (optional torchrun + watchdog)."""
    args: List[str] = ["--max-evals", str(int(max_evals))]
    if plot_curves:
        args.append("--plot-curves")
    if ft_role:
        args += ["--ft-role", str(ft_role)]
    if extra_args:
        args += list(extra_args)

    cmd = build_bayesopt_entry_cmd(
        config_json=config_json,
        model_keys=model_keys,
        nproc_per_node=nproc_per_node,
        extra_args=args,
    )
    if use_watchdog:
        cmd = wrap_with_watchdog(
            cmd,
            idle_seconds=idle_seconds,
            max_restarts=max_restarts,
            restart_delay_seconds=restart_delay_seconds,
        )
    return run(cmd, check=True)


def run_from_config(config_json: str | Path) -> subprocess.CompletedProcess:
    """Notebook entry point: switch execution modes by editing config.json.

    Convention: config.json may include a `runner` section for notebook control:
    - runner.mode: "entry" (default), "incremental", or "explain"
    - runner.nproc_per_node: >1 enables torchrun/DDP (entry only)
    - runner.model_keys: list of models to run (entry only)
    - runner.max_evals / runner.plot_curves / runner.ft_role (entry only; override config fields)
    - runner.use_watchdog / runner.idle_seconds / runner.max_restarts / runner.restart_delay_seconds
    - runner.incremental_args: List[str] (incremental only; extra args for cli/BayesOpt_incremental.py)
    """
    pkg_dir = _find_ins_pricing_dir()
    config_path = Path(config_json)
    if not config_path.is_absolute():
        config_path = (pkg_dir / config_path).resolve() if (pkg_dir / config_path).exists() else config_path.resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8", errors="replace"))
    runner = cast(dict, raw.get("runner") or {})

    mode = str(runner.get("mode") or "entry").strip().lower()
    use_watchdog = bool(runner.get("use_watchdog", False))
    idle_seconds = int(runner.get("idle_seconds", 7200))
    max_restarts = int(runner.get("max_restarts", 50))
    restart_delay_seconds = int(runner.get("restart_delay_seconds", 10))

    if mode == "incremental":
        inc_args = runner.get("incremental_args") or []
        if not isinstance(inc_args, list):
            raise ValueError("config.runner.incremental_args must be a list of strings.")
        cmd = build_incremental_cmd(config_path, extra_args=[str(x) for x in inc_args])
        if use_watchdog:
            cmd = wrap_with_watchdog(
                cmd,
                idle_seconds=idle_seconds,
                max_restarts=max_restarts,
                restart_delay_seconds=restart_delay_seconds,
            )
        return run(cmd, check=True)

    if mode == "explain":
        exp_args = runner.get("explain_args") or []
        if not isinstance(exp_args, list):
            raise ValueError("config.runner.explain_args must be a list of strings.")
        cmd = build_explain_cmd(config_path, extra_args=[str(x) for x in exp_args])
        if use_watchdog:
            cmd = wrap_with_watchdog(
                cmd,
                idle_seconds=idle_seconds,
                max_restarts=max_restarts,
                restart_delay_seconds=restart_delay_seconds,
            )
        return run(cmd, check=True)

    if mode != "entry":
        raise ValueError(
            f"Unsupported runner.mode={mode!r}, expected 'entry', 'incremental', or 'explain'."
        )

    model_keys = runner.get("model_keys")
    if not model_keys:
        model_keys = raw.get("model_keys")
    if not model_keys:
        model_keys = ["ft"]
    if not isinstance(model_keys, list):
        raise ValueError("runner.model_keys must be a list of strings.")

    nproc_per_node = int(runner.get("nproc_per_node", 1))
    max_evals = int(runner.get("max_evals", raw.get("max_evals", 50)))
    plot_curves = bool(runner.get("plot_curves", raw.get("plot_curves", True)))
    ft_role = runner.get("ft_role", None)
    if ft_role is None:
        ft_role = raw.get("ft_role")

    cmd = build_bayesopt_entry_cmd(
        config_path,
        model_keys=[str(x) for x in model_keys],
        nproc_per_node=nproc_per_node,
        extra_args=[
            "--max-evals",
            str(max_evals),
            *(["--plot-curves"] if plot_curves else []),
            *(["--ft-role", str(ft_role)] if ft_role else []),
        ],
    )

    if use_watchdog:
        cmd = wrap_with_watchdog(
            cmd,
            idle_seconds=idle_seconds,
            max_restarts=max_restarts,
            restart_delay_seconds=restart_delay_seconds,
        )
    return run(cmd, check=True)
