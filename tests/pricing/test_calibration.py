"""Tests for pricing calibration module."""

import numpy as np
import pandas as pd
import pytest

from ins_pricing.exceptions import DataValidationError


@pytest.fixture
def sample_model_predictions():
    """Sample model predictions and actuals."""
    np.random.seed(42)
    return pd.DataFrame({
        "actual_loss": np.random.exponential(500, 1000),
        "predicted_loss": np.random.exponential(480, 1000),
        "exposure": np.ones(1000),
        "premium": np.random.uniform(200, 1000, 1000)
    })


class TestGlobalCalibration:
    """Test global calibration methods."""

    def test_fit_calibration_factor(self, sample_model_predictions):
        """Test multiplicative calibration factor fitting."""
        from ins_pricing.pricing.calibration import fit_calibration_factor

        calibration_factor = fit_calibration_factor(
            pred=sample_model_predictions["predicted_loss"],
            actual=sample_model_predictions["actual_loss"],
            weight=sample_model_predictions["exposure"],
        )

        assert isinstance(calibration_factor, (int, float, np.number))
        assert calibration_factor > 0

    def test_apply_calibration(self, sample_model_predictions):
        """Test applying a fitted calibration factor."""
        from ins_pricing.pricing.calibration import apply_calibration, fit_calibration_factor

        calibration_factor = fit_calibration_factor(
            pred=sample_model_predictions["predicted_loss"],
            actual=sample_model_predictions["actual_loss"],
            weight=sample_model_predictions["exposure"],
        )
        calibrated = apply_calibration(
            sample_model_predictions["predicted_loss"],
            calibration_factor,
        )

        assert isinstance(calibrated, np.ndarray)
        assert calibrated.shape[0] == sample_model_predictions.shape[0]

    def test_fit_calibration_factor_rejects_non_positive_prediction_sum(self):
        from ins_pricing.pricing.calibration import fit_calibration_factor

        pred = np.array([0.0, 0.0, 0.0])
        actual = np.array([1.0, 2.0, 3.0])

        with pytest.raises(DataValidationError, match="positive weighted prediction sum"):
            fit_calibration_factor(pred=pred, actual=actual)


class TestSegmentCalibration:
    """Test segment-specific calibration."""

    def test_calibrate_by_segment(self):
        """Test calibration within segments."""
        from ins_pricing.pricing.calibration import calibrate_by_segment

        df = pd.DataFrame({
            "segment": ["A", "B", "A", "B", "A"] * 200,
            "actual": np.random.exponential(500, 1000),
            "predicted": np.random.exponential(480, 1000),
            "exposure": np.ones(1000)
        })

        calibrated = calibrate_by_segment(
            df,
            actual_col="actual",
            pred_col="predicted",
            segment_col="segment",
            weight_col="exposure"
        )

        assert "calibration_factor" in calibrated.columns
        assert len(calibrated["segment"].unique()) == 2

    def test_calibrate_by_segment_marks_fallback_when_pred_sum_non_positive(self):
        from ins_pricing.pricing.calibration import calibrate_by_segment

        df = pd.DataFrame(
            {
                "segment": ["A", "A", "B", "B"],
                "actual": [10.0, 12.0, 8.0, 9.0],
                "predicted": [0.0, 0.0, 5.0, 6.0],
                "exposure": [1.0, 1.0, 1.0, 1.0],
            }
        )
        out = calibrate_by_segment(
            df,
            actual_col="actual",
            pred_col="predicted",
            segment_col="segment",
            weight_col="exposure",
        )

        row_a = out.loc[out["segment"] == "A"].iloc[0]
        assert row_a["calibration_factor"] == 1.0
        assert row_a["status"] == "fallback_default"
        assert isinstance(row_a["error"], str)
