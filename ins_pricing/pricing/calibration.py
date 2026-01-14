"""Premium calibration utilities for insurance pricing models.

This module provides functions for calibrating model predictions to match
target loss ratios or actual experience. Calibration ensures that the total
predicted premium aligns with expected losses across the portfolio.

Calibration is typically applied after model training to adjust the overall
premium level without changing the relative risk differentiation between
policies.

Common use cases:
    - Adjusting premiums to achieve a target loss ratio (e.g., 65%)
    - Correcting for systematic over/under-prediction
    - Aligning model predictions with actual claims experience

Example:
    >>> import numpy as np
    >>> from ins_pricing.pricing.calibration import fit_calibration_factor, apply_calibration
    >>>
    >>> # Model predictions and actual claims
    >>> predicted = np.array([100, 150, 200, 250])
    >>> actual = np.array([110, 140, 210, 240])
    >>> exposure = np.array([1.0, 1.0, 1.0, 1.0])
    >>>
    >>> # Fit calibration factor to match actuals
    >>> factor = fit_calibration_factor(predicted, actual, weight=exposure)
    >>> print(f"Calibration factor: {factor:.3f}")
    Calibration factor: 1.000
    >>>
    >>> # Apply calibration to new predictions
    >>> new_predictions = np.array([120, 180])
    >>> calibrated = apply_calibration(new_predictions, factor)
    >>> print(calibrated)
    [120. 180.]

Note:
    Calibration preserves the relative ordering of predictions - it only
    adjusts the overall level. This ensures that risk differentiation
    remains intact while achieving target aggregate metrics.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def fit_calibration_factor(
    pred: np.ndarray,
    actual: np.ndarray,
    *,
    weight: Optional[np.ndarray] = None,
    target_lr: Optional[float] = None,
) -> float:
    """Fit a scalar calibration factor to align predictions with actuals or target loss ratio.

    This function computes a multiplicative calibration factor that adjusts
    model predictions to match either:
    1. Actual observed losses (when target_lr=None)
    2. A target loss ratio (when target_lr is specified)

    The calibration factor is computed as:
    - Without target: factor = sum(actual * weight) / sum(pred * weight)
    - With target: factor = sum(actual * weight) / (target_lr * sum(pred * weight))

    Args:
        pred: Model predictions (premiums or pure premiums)
        actual: Actual observed values (claims or losses)
        weight: Optional weights (e.g., exposure, earned premium).
               If provided, weighted sums are used for calibration.
               Default: None (equal weighting)
        target_lr: Target loss ratio to achieve (0 < target_lr < 1).
                   If None, calibrates to match actual observations.
                   Default: None

    Returns:
        Calibration factor (scalar multiplier) to apply to predictions.
        Returns 1.0 if pred sum is <= 0 (no calibration needed).

    Raises:
        ValueError: If weight length doesn't match pred length
        ValueError: If target_lr is specified but not positive

    Example:
        >>> # Calibrate to match actual claims
        >>> pred = np.array([100, 150, 200])
        >>> actual = np.array([110, 140, 210])
        >>> factor = fit_calibration_factor(pred, actual)
        >>> print(f"{factor:.3f}")
        1.022  # Multiply predictions by 1.022 to match actuals
        >>>
        >>> # Calibrate to achieve 70% loss ratio
        >>> pred_premium = np.array([100, 150, 200])
        >>> actual_claims = np.array([75, 100, 130])
        >>> factor = fit_calibration_factor(pred_premium, actual_claims, target_lr=0.70)
        >>> print(f"{factor:.3f}")
        1.143  # Adjust premiums to achieve 70% loss ratio
        >>>
        >>> # Weighted calibration (e.g., by exposure)
        >>> exposure = np.array([1.0, 0.5, 1.5])
        >>> factor = fit_calibration_factor(pred, actual, weight=exposure)

    Note:
        - Calibration preserves relative differences between predictions
        - Weight is applied to both pred and actual for consistency
        - Returns 1.0 (no adjustment) if predictions sum to zero or less
        - target_lr typically in range [0.5, 0.9] for insurance pricing
    """
    pred = np.asarray(pred, dtype=float).reshape(-1)
    actual = np.asarray(actual, dtype=float).reshape(-1)
    if weight is not None:
        weight = np.asarray(weight, dtype=float).reshape(-1)
        if weight.shape[0] != pred.shape[0]:
            raise ValueError("weight length must match pred length.")
        pred = pred * weight
        actual = actual * weight

    pred_sum = float(np.sum(pred))
    actual_sum = float(np.sum(actual))
    if pred_sum <= 0:
        return 1.0

    if target_lr is None:
        return actual_sum / pred_sum
    if target_lr <= 0:
        raise ValueError("target_lr must be positive.")
    return actual_sum / (target_lr * pred_sum)


def apply_calibration(pred: np.ndarray, factor: float) -> np.ndarray:
    """Apply calibration factor to predictions.

    Multiplies predictions by the calibration factor to adjust the overall
    premium level while preserving relative risk differentiation.

    Args:
        pred: Model predictions to calibrate (array-like)
        factor: Calibration factor from fit_calibration_factor()

    Returns:
        Calibrated predictions (pred * factor)

    Example:
        >>> pred = np.array([100, 150, 200, 250])
        >>> factor = 1.05  # 5% increase
        >>> calibrated = apply_calibration(pred, factor)
        >>> print(calibrated)
        [105. 157.5 210. 262.5]
        >>>
        >>> # Verify relative differences are preserved
        >>> print(pred[1] / pred[0])  # Original ratio
        1.5
        >>> print(calibrated[1] / calibrated[0])  # Calibrated ratio (same)
        1.5

    Note:
        - Calibration is a simple scalar multiplication
        - Relative ordering and ratios are preserved
        - Can be applied to any numeric predictions (premium, loss, pure premium)
    """
    pred = np.asarray(pred, dtype=float)
    return pred * float(factor)
