from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    from .utils.notebook_utils import run_from_config, run_from_config_cli  # type: ignore
except Exception:  # pragma: no cover
    from utils.notebook_utils import run_from_config, run_from_config_cli  # type: ignore


def run(config_json: str | Path) -> None:
    """Run explain by config.json (runner.mode=explain)."""
    run_from_config(config_json)


def main(argv: Optional[list[str]] = None) -> None:
    run_from_config_cli(
        "Explain_Run: run explain by config.json (runner.mode=explain).",
        argv,
    )


if __name__ == "__main__":
    main()
