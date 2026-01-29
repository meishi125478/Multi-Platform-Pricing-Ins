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
