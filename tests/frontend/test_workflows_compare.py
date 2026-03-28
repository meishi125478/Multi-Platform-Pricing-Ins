from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ins_pricing.frontend.workflows_compare import run_double_lift_from_file


def test_run_double_lift_rejects_all_na_required_columns(tmp_path: Path) -> None:
    data_path = tmp_path / "double_lift.csv"
    pd.DataFrame(
        {
            "pred_1": [None, None, None],
            "pred_2": [0.2, 0.4, 0.6],
            "target": [1.0, 2.0, 3.0],
            "weights": [1.0, 1.0, 1.0],
        }
    ).to_csv(data_path, index=False)

    with pytest.raises(ValueError, match="No valid rows remain"):
        run_double_lift_from_file(
            data_path=str(data_path),
            pred_col_1="pred_1",
            pred_col_2="pred_2",
            target_col="target",
            weight_col="weights",
            holdout_ratio=0.0,
        )
