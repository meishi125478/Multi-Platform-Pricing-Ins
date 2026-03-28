"""Tests for pricing exposure calculation module."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


@pytest.fixture
def sample_policy_dates():
    """Sample policy with start and end dates."""
    start_date = datetime(2023, 1, 1)
    return pd.DataFrame({
        "policy_id": range(100),
        "start_date": [start_date + timedelta(days=i) for i in range(100)],
        "end_date": [start_date + timedelta(days=i+365) for i in range(100)],
        "premium": np.random.uniform(200, 1000, 100)
    })


class TestExposureCalculation:
    """Test exposure calculation functions."""

    def test_calculate_policy_exposure_years(self, sample_policy_dates):
        """Test calculating exposure in years."""
        from ins_pricing.pricing.exposure import compute_exposure

        df = compute_exposure(
            sample_policy_dates,
            start_col="start_date",
            end_col="end_date",
            time_unit="years"
        )

        assert "exposure" in df.columns
        assert all(df["exposure"] > 0)
        assert all(df["exposure"] <= 1.1)  # Roughly 1 year

    def test_calculate_policy_exposure_days(self, sample_policy_dates):
        """Test calculating exposure in days."""
        from ins_pricing.pricing.exposure import compute_exposure

        df = compute_exposure(
            sample_policy_dates,
            start_col="start_date",
            end_col="end_date",
            time_unit="days"
        )

        assert all(df["exposure"] >= 365)
        assert all(df["exposure"] <= 366)

    def test_partial_period_exposure(self):
        """Test exposure for partial periods."""
        from ins_pricing.pricing.exposure import compute_exposure

        df = pd.DataFrame({
            "start_date": [datetime(2023, 1, 1)],
            "end_date": [datetime(2023, 6, 30)]  # 6 months
        })

        result = compute_exposure(df, "start_date", "end_date", time_unit="years")

        assert 0.48 < result["exposure"].iloc[0] < 0.52  # Roughly 0.5 years
