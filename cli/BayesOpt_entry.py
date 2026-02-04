"""Thin wrapper for the BayesOpt CLI entry point.

The main implementation lives in bayesopt_entry_runner.py.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import os
import sys

def _ensure_repo_root() -> None:
    if __package__ not in {None, ""}:
        return
    if importlib.util.find_spec("ins_pricing") is not None:
        return
    bootstrap_path = Path(__file__).resolve().parents[1] / "utils" / "bootstrap.py"
    spec = importlib.util.spec_from_file_location("ins_pricing.cli.utils.bootstrap", bootstrap_path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ensure_repo_root()


_ensure_repo_root()

def _apply_env_from_config(argv: list[str]) -> None:
    if "--config-json" not in argv:
        return
    idx = argv.index("--config-json")
    if idx + 1 >= len(argv):
        return
    raw_path = argv[idx + 1]
    try:
        cfg_path = Path(raw_path).expanduser()
        if not cfg_path.is_absolute():
            cfg_path = cfg_path.resolve()
        if not cfg_path.exists():
            script_dir = Path(__file__).resolve().parents[1]
            candidate = (script_dir / raw_path).resolve()
            if candidate.exists():
                cfg_path = candidate
        if not cfg_path.exists():
            return
        cfg = json.loads(cfg_path.read_text(encoding="utf-8", errors="replace"))
        env = cfg.get("env", {})
        if isinstance(env, dict):
            for key, value in env.items():
                if key is None:
                    continue
                os.environ.setdefault(str(key), str(value))
    except Exception:
        return

_apply_env_from_config(sys.argv)

try:
    from ins_pricing.cli.bayesopt_entry_runner import main
except Exception:  # pragma: no cover
    from ins_pricing.cli.bayesopt_entry_runner import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
