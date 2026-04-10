from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RowStore:
    """Canonical row-aligned views for runtime execution."""

    train_raw: pd.DataFrame
    test_raw: pd.DataFrame
    train_row_id: np.ndarray
    test_row_id: np.ndarray
    train_source_index: np.ndarray
    test_source_index: np.ndarray


@dataclass(frozen=True)
class FoldSlice:
    fold_id: int
    train_idx: np.ndarray
    val_idx: np.ndarray


@dataclass
class FoldResult:
    loss: float
    val_pred: np.ndarray
    val_idx: np.ndarray
    train_meta: Optional[dict[str, Any]] = None
