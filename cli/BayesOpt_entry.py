"""Thin wrapper for the BayesOpt CLI entry point.

The main implementation lives in bayesopt_entry_runner.py.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import sys

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

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
    from .bayesopt_entry_runner import main
except Exception:  # pragma: no cover
    from ins_pricing.cli.bayesopt_entry_runner import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
