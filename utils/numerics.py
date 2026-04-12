"""Numerical utilities shared across ins_pricing.

This module centralizes small, dependency-light numerical helpers so that
other subpackages can reuse them without importing bayesopt-specific code.
"""

from __future__ import annotations

import random
import numpy as np


EPS = 1e-8
"""Small epsilon value for numerical stability."""


def safe_divide(
    numerator: float,
    denominator: float,
    *,
    default: float = np.nan,
) -> float:
    """Safely divide two numeric scalars.

    Returns `default` when denominator is zero or either input is non-finite.
    """
    num = float(numerator)
    den = float(denominator)
    if not np.isfinite(num) or not np.isfinite(den) or den == 0.0:
        return float(default)
    return float(num / den)


def set_global_seed(seed: int) -> None:
    """Set random seed for reproducibility across numpy/python/torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # local import to keep module import torch-optional
    except Exception:  # pragma: no cover - optional dependency
        torch = None  # type: ignore[assignment]
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def compute_batch_size(data_size: int, learning_rate: float, batch_num: int, minimum: int) -> int:
    """Compute adaptive batch size based on data size and learning rate."""
    estimated = int((learning_rate / 1e-4) ** 0.5 * (data_size / max(batch_num, 1)))
    return max(1, min(int(data_size), max(int(minimum), estimated)))


def tweedie_loss(
    pred,
    target,
    *,
    p: float = 1.5,
    eps: float = 1e-6,
    max_clip: float = 1e6,
):
    """Compute Tweedie deviance loss for PyTorch tensors."""
    try:
        import torch  # local import to keep module import torch-optional
    except Exception:  # pragma: no cover - optional dependency
        torch = None  # type: ignore[assignment]
    if torch is None:
        raise ImportError("tweedie_loss requires torch. Install optional dependency 'torch'.")
    pred_clamped = torch.clamp(pred, min=eps)

    if p == 1:
        term1 = target * torch.log(target / pred_clamped + eps)
        term2 = -target + pred_clamped
        term3 = 0
    elif p == 0:
        term1 = 0.5 * torch.pow(target - pred_clamped, 2)
        term2 = 0
        term3 = 0
    elif p == 2:
        term1 = torch.log(pred_clamped / target + eps)
        term2 = -target / pred_clamped + 1
        term3 = 0
    else:
        term1 = torch.pow(target, 2 - p) / ((1 - p) * (2 - p))
        term2 = target * torch.pow(pred_clamped, 1 - p) / (1 - p)
        term3 = torch.pow(pred_clamped, 2 - p) / (2 - p)

    return torch.nan_to_num(
        2 * (term1 - term2 + term3),
        nan=eps,
        posinf=max_clip,
        neginf=-max_clip,
    )
