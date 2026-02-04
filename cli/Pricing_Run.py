from __future__ import annotations

from pathlib import Path
from typing import Optional
import importlib.util
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

try:
    from ins_pricing.cli.utils.notebook_utils import run_from_config, run_from_config_cli  # type: ignore
except Exception:  # pragma: no cover
    from utils.notebook_utils import run_from_config, run_from_config_cli  # type: ignore


def run(config_json: str | Path) -> None:
    """Unified entry point: run entry/incremental/watchdog/DDP based on config.json runner."""
    run_from_config(config_json)


def main(argv: Optional[list[str]] = None) -> None:
    run_from_config_cli(
        "Pricing_Run: run BayesOpt by config.json (entry/incremental/watchdog/DDP).",
        argv,
    )


if __name__ == "__main__":
    main()
