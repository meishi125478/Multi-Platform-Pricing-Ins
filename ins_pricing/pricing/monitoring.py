"""Population Stability Index (PSI) monitoring utilities.

This module re-exports PSI functions from the shared utils package
for backward compatibility.

Example:
    >>> from ins_pricing.pricing.monitoring import psi_report
    >>> report = psi_report(expected_df, actual_df)
"""

from __future__ import annotations

# Re-export from shared utils for backward compatibility
from ins_pricing.utils.metrics import (
    psi_numeric,
    psi_categorical,
    population_stability_index,
    psi_report,
)

__all__ = [
    "psi_numeric",
    "psi_categorical",
    "population_stability_index",
    "psi_report",
]
