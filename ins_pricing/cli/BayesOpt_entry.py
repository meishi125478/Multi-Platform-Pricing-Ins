"""Thin wrapper for the BayesOpt CLI entry point.

The main implementation lives in bayesopt_entry_runner.py.
"""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

try:
    from .bayesopt_entry_runner import main
except Exception:  # pragma: no cover
    from ins_pricing.cli.bayesopt_entry_runner import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
