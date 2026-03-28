"""Shared bootstrap helpers for CLI scripts that may run outside package mode."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional


def ensure_repo_root(script_file: str, package_name: Optional[str]) -> None:
    """Ensure repo root is importable before importing ins_pricing.* modules."""
    if package_name not in {None, ""}:
        return
    if importlib.util.find_spec("ins_pricing") is not None:
        return

    script_path = Path(script_file).resolve()
    bootstrap_path = script_path.parent / "utils" / "bootstrap.py"
    if not bootstrap_path.exists():
        return

    spec = importlib.util.spec_from_file_location(
        "ins_pricing.cli.utils.bootstrap",
        bootstrap_path,
    )
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ensure = getattr(module, "ensure_repo_root", None)
    if callable(ensure):
        ensure()


__all__ = ["ensure_repo_root"]
