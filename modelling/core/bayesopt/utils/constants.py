"""Core constants and simple utility functions.

This module contains:
- EPS constant for numerical stability
- set_global_seed() for reproducibility
- ensure_parent_dir() for file operations
- compute_batch_size() for adaptive batching
- tweedie_loss() for regression loss
- infer_factor_and_cate_list() for automatic feature detection
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Constants
# =============================================================================
EPS = 1e-8
"""Small epsilon value for numerical stability."""


# Simple utility functions
# =============================================================================

def set_global_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent_dir(file_path: str) -> None:
    """Create parent directories when missing.

    Args:
        file_path: Path to file whose parent directory should be created
    """
    directory = Path(file_path).parent
    if directory and not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)


def compute_batch_size(data_size: int, learning_rate: float,
                      batch_num: int, minimum: int) -> int:
    """Compute adaptive batch size based on data size and learning rate.

    Args:
        data_size: Total number of samples
        learning_rate: Learning rate value
        batch_num: Target number of batches
        minimum: Minimum batch size

    Returns:
        Computed batch size
    """
    estimated = int((learning_rate / 1e-4) ** 0.5 *
                   (data_size / max(batch_num, 1)))
    return max(1, min(data_size, max(minimum, estimated)))


def tweedie_loss(pred, target, p=1.5, eps=1e-6, max_clip=1e6):
    """Compute Tweedie deviance loss for PyTorch.

    Reference: https://scikit-learn.org/stable/modules/model_evaluation.html

    Args:
        pred: Predicted values (tensor)
        target: True values (tensor)
        p: Tweedie power parameter (1.0-2.0)
        eps: Small epsilon for numerical stability
        max_clip: Maximum value for clipping

    Returns:
        Tweedie negative log-likelihood (tensor)
    """
    # Clamp predictions to positive values for stability
    pred_clamped = torch.clamp(pred, min=eps)

    if p == 1:
        # Poisson
        term1 = target * torch.log(target / pred_clamped + eps)
        term2 = -target + pred_clamped
        term3 = 0
    elif p == 0:
        # Gaussian
        term1 = 0.5 * torch.pow(target - pred_clamped, 2)
        term2 = 0
        term3 = 0
    elif p == 2:
        # Gamma
        term1 = torch.log(pred_clamped / target + eps)
        term2 = -target / pred_clamped + 1
        term3 = 0
    else:
        # General Tweedie
        term1 = torch.pow(target, 2 - p) / ((1 - p) * (2 - p))
        term2 = target * torch.pow(pred_clamped, 1 - p) / (1 - p)
        term3 = torch.pow(pred_clamped, 2 - p) / (2 - p)

    return torch.nan_to_num(
        2 * (term1 - term2 + term3),
        nan=eps,
        posinf=max_clip,
        neginf=-max_clip
    )


def infer_factor_and_cate_list(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    resp_nme: str,
    weight_nme: str,
    binary_resp_nme: Optional[str] = None,
    factor_nmes: Optional[List[str]] = None,
    cate_list: Optional[List[str]] = None,
    infer_categorical_max_unique: int = 50,
    infer_categorical_max_ratio: float = 0.05
) -> Tuple[List[str], List[str]]:
    """Infer factor_nmes/cate_list when feature names are not provided.

    Rules:
      - factor_nmes: start from shared train/test columns, exclude target/weight/(optional binary target).
      - cate_list: object/category/bool plus low-cardinality integer columns.
      - Always intersect with shared train/test columns to avoid mismatches.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        resp_nme: Response/target column name
        weight_nme: Sample weight column name
        binary_resp_nme: Optional binary response column name
        factor_nmes: Optional list of feature column names
        cate_list: Optional list of categorical feature names
        infer_categorical_max_unique: Max unique values for categorical inference
        infer_categorical_max_ratio: Max ratio of unique/total for categorical inference

    Returns:
        Tuple of (factor_nmes, cate_list)
    """
    excluded = {resp_nme, weight_nme}
    if binary_resp_nme:
        excluded.add(binary_resp_nme)

    common_cols = [c for c in train_df.columns if c in test_df.columns]
    if factor_nmes is None:
        factors = [c for c in common_cols if c not in excluded]
    else:
        factors = [
            c for c in factor_nmes if c in common_cols and c not in excluded
        ]

    if cate_list is not None:
        cats = [c for c in cate_list if c in factors]
        return factors, cats

    n_rows = max(1, len(train_df))
    cats: List[str] = []
    for col in factors:
        s = train_df[col]
        if (pd.api.types.is_bool_dtype(s) or
            pd.api.types.is_object_dtype(s) or
            isinstance(s.dtype, pd.CategoricalDtype)):
            cats.append(col)
            continue
        if pd.api.types.is_integer_dtype(s):
            nunique = int(s.nunique(dropna=True))
            if (nunique <= infer_categorical_max_unique or
                (nunique / n_rows) <= infer_categorical_max_ratio):
                cats.append(col)

    return factors, cats
