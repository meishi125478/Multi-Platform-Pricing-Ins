"""Tests for pricing factors module."""

import numpy as np
import pandas as pd
import pytest

from ins_pricing.exceptions import DataValidationError


@pytest.fixture
def sample_policy_data():
    """Sample insurance policy data."""
    np.random.seed(42)
    return pd.DataFrame({
        "policy_id": range(1000),
        "age": np.random.randint(18, 80, 1000),
        "gender": np.random.choice(["M", "F"], 1000),
        "region": np.random.choice(["North", "South", "East", "West"], 1000),
        "vehicle_age": np.random.randint(0, 15, 1000),
        "claim_amount": np.random.exponential(500, 1000),
        "exposure": np.random.uniform(0.5, 1.0, 1000),
        "premium": np.random.uniform(200, 1000, 1000)
    })


class TestFactorTableConstruction:
    """Test factor table construction."""

    def test_build_univariate_factor_table(self, sample_policy_data):
        """Test building a univariate factor table."""
        from ins_pricing.pricing.factors import build_factor_table

        factor_table = build_factor_table(
            df=sample_policy_data,
            factor_col="age",
            loss_col="claim_amount",
            exposure_col="exposure",
            method="quantile",
            n_bins=10
        )

        assert len(factor_table) <= 10
        assert "age_bin" in factor_table.columns
        assert "relativity" in factor_table.columns
        assert "claim_count" in factor_table.columns

    def test_equal_width_binning(self, sample_policy_data):
        """Test equal width binning strategy."""
        from ins_pricing.pricing.factors import build_factor_table

        factor_table = build_factor_table(
            df=sample_policy_data,
            factor_col="vehicle_age",
            loss_col="claim_amount",
            exposure_col="exposure",
            method="equal_width",
            n_bins=5
        )

        assert len(factor_table) == 5

    def test_categorical_factor_table(self, sample_policy_data):
        """Test factor table for categorical variables."""
        from ins_pricing.pricing.factors import build_factor_table

        factor_table = build_factor_table(
            df=sample_policy_data,
            factor_col="region",
            loss_col="claim_amount",
            exposure_col="exposure",
            method="categorical"
        )

        assert set(factor_table["region"]) == set(sample_policy_data["region"].unique())
        assert "relativity" in factor_table.columns


class TestFactorSmoothing:
    """Test factor smoothing techniques."""

    def test_credibility_weighting(self):
        """Test credibility-weighted smoothing."""
        from ins_pricing.pricing.factors import apply_credibility_smoothing

        raw_factors = pd.DataFrame({
            "bin": ["A", "B", "C"],
            "relativity": [1.2, 0.8, 1.5],
            "exposure": [100, 500, 50]  # C has low credibility
        })

        smoothed = apply_credibility_smoothing(raw_factors, base_relativity=1.0)

        # Low exposure bin should be pulled toward base
        assert abs(smoothed.loc[2, "relativity"] - 1.0) < abs(raw_factors.loc[2, "relativity"] - 1.0)

    def test_neighbor_smoothing(self):
        """Test smoothing using neighboring bins."""
        from ins_pricing.pricing.factors import apply_neighbor_smoothing

        factors = pd.DataFrame({
            "bin": [1, 2, 3, 4, 5],
            "relativity": [1.0, 1.2, 2.5, 1.4, 1.5]  # Bin 3 is outlier
        })

        smoothed = apply_neighbor_smoothing(factors)

        # Outlier should be smoothed
        assert smoothed.loc[2, "relativity"] < factors.loc[2, "relativity"]


class TestFactorApplication:
    """Test applying factors to new data."""

    def test_apply_factors_to_policies(self, sample_policy_data):
        """Test applying factor table to policies."""
        from ins_pricing.pricing.factors import build_factor_table, apply_factors

        # Build factor table
        age_factors = build_factor_table(
            df=sample_policy_data,
            factor_col="age",
            loss_col="claim_amount",
            exposure_col="exposure",
            n_bins=5
        )

        # Apply to new data
        result = apply_factors(sample_policy_data, age_factors, factor_col="age")

        assert "age_relativity" in result.columns
        assert result["age_relativity"].notna().all()


@pytest.mark.parametrize("method,n_bins", [
    ("quantile", 5),
    ("quantile", 10),
    ("equal_width", 5),
    ("equal_width", 10),
])
class TestBinningMethods:
    """Test different binning methods."""

    def test_binning_produces_expected_bins(self, sample_policy_data, method, n_bins):
        """Test that binning produces expected number of bins."""
        from ins_pricing.pricing.factors import build_factor_table

        factor_table = build_factor_table(
            df=sample_policy_data,
            factor_col="age",
            loss_col="claim_amount",
            exposure_col="exposure",
            method=method,
            n_bins=n_bins
        )

        assert len(factor_table) <= n_bins
