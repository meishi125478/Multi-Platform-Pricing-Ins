"""Shared helpers for config-driven CLI entry scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence


def run_config(config_json: str | Path) -> None:
    from ins_pricing.cli.utils.notebook_utils import run_from_config

    run_from_config(config_json)


def run_config_main(
    description: str,
    argv: Optional[Sequence[str]] = None,
) -> None:
    from ins_pricing.cli.utils.notebook_utils import run_from_config_cli

    run_from_config_cli(description, argv)


__all__ = ["run_config", "run_config_main"]
