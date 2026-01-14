from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    from .utils.notebook_utils import run_from_config, run_from_config_cli  # type: ignore
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
