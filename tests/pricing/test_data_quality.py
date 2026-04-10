from __future__ import annotations

import pandas as pd

from ins_pricing.pricing.data_quality import detect_leakage


def test_detect_leakage_returns_empty_frame_with_columns_when_no_hits():
    df = pd.DataFrame(
        {
            "target_claim": [0.0, 1.0, 0.0, 1.0],
            "age": [25, 42, 31, 51],
            "vehicle_age": [2, 5, 3, 7],
        }
    )

    result = detect_leakage(df, target_col="target_claim", corr_threshold=0.99999)
    assert result.empty
    assert list(result.columns) == ["feature", "reason", "score"]


def test_detect_leakage_finds_identical_column():
    df = pd.DataFrame(
        {
            "target_claim": [0, 1, 0, 1],
            "leak_col": [0, 1, 0, 1],
            "premium": [1000, 1200, 1100, 1250],
        }
    )

    result = detect_leakage(df, target_col="target_claim")
    assert not result.empty
    assert result.iloc[0]["feature"] == "leak_col"
    assert result.iloc[0]["reason"] == "identical"
