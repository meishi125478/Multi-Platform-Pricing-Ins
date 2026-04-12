"""Tests for rate table module."""

import numpy as np
import pandas as pd
import pytest


class TestRateTableGeneration:
    """Test rate table generation."""

    def test_generate_multidimensional_rate_table(self):
        """Test generating rate table with multiple dimensions."""
        from ins_pricing.pricing.rate_table import generate_rate_table

        factors = {
            "age": pd.DataFrame({"age_band": ["18-25", "26-35", "36+"], "relativity": [1.5, 1.0, 0.8]}),
            "region": pd.DataFrame({"region": ["North", "South"], "relativity": [1.2, 0.9]})
        }

        rate_table = generate_rate_table(factors, base_rate=100)

        assert len(rate_table) == 3 * 2  # 3 age bands × 2 regions
        assert "rate" in rate_table.columns

    def test_rate_lookup(self):
        """Test looking up rate for specific characteristics."""
        from ins_pricing.pricing.rate_table import lookup_rate

        rate_table = pd.DataFrame({
            "age_band": ["18-25", "26-35"],
            "region": ["North", "North"],
            "rate": [150, 120]
        })

        rate = lookup_rate(
            rate_table,
            characteristics={"age_band": "18-25", "region": "North"}
        )

        assert rate == 150

    def test_compute_base_rate_zero_exposure_returns_nan(self):
        """Base rate is undefined when total exposure is zero."""
        from ins_pricing.pricing.rate_table import compute_base_rate

        df = pd.DataFrame(
            {
                "loss": [100.0, 50.0],
                "exposure": [0.0, 0.0],
            }
        )

        rate = compute_base_rate(df, loss_col="loss", exposure_col="exposure")
        assert np.isnan(rate)

    def test_apply_factor_tables_supports_interval_levels(self):
        from ins_pricing.pricing.rate_table import apply_factor_tables

        df = pd.DataFrame({"age": [22.0, 30.0, 55.0]})
        intervals = pd.IntervalIndex.from_tuples([(0, 25), (25, 45), (45, 200)], closed="left")
        factor_tables = {
            "age": pd.DataFrame({"level": intervals, "relativity": [1.3, 1.0, 0.8]})
        }

        out = apply_factor_tables(df, factor_tables)
        assert np.allclose(out, np.array([1.3, 1.0, 0.8], dtype=float))

    def test_apply_factor_tables_rejects_invalid_relativity_values(self):
        from ins_pricing.pricing.rate_table import apply_factor_tables

        df = pd.DataFrame({"age": [22.0]})
        factor_tables = {
            "age": pd.DataFrame({"level": [22.0], "relativity": ["bad"]})
        }

        with pytest.raises(ValueError, match="invalid relativity"):
            apply_factor_tables(df, factor_tables)
