"""Bootstrap helpers for running CLI modules as scripts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


def _find_repo_root(start: Optional[Path] = None) -> Path:
    anchor = (start or Path(__file__)).resolve()
    for parent in [anchor] + list(anchor.parents):
        if (parent / "ins_pricing").is_dir():
            return parent
    return anchor.parent


def ensure_repo_root(repo_root: Optional[Path] = None) -> Path:
    """Ensure the repository root (parent of ins_pricing/) is on sys.path."""
    root = _find_repo_root(repo_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
