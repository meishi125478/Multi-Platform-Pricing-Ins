"""Drift monitoring utilities for production.

This module re-exports PSI functions from the shared utils package.

Example:
    >>> from ins_pricing.production.drift import psi_report
    >>> report = psi_report(expected_df, actual_df)
"""

from __future__ import annotations

# Re-export from shared utils
from ins_pricing.utils.metrics import psi_report

__all__ = ["psi_report"]
