"""Tests for production scoring module."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch

from ins_pricing.exceptions import DataValidationError


@pytest.fixture
def sample_predictions():
    """Sample prediction data."""
    return pd.DataFrame({
        "actual": [100, 150, 200, 250, 300],
        "predicted": [105, 145, 210, 240, 295],
        "weight": [1.0, 1.0, 1.0, 1.0, 1.0]
    })


@pytest.fixture
def classification_data():
    """Sample classification data."""
    return pd.DataFrame({
        "actual": [0, 1, 1, 0, 1, 0, 1, 1, 0, 0],
        "predicted_proba": [0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.85, 0.6, 0.15, 0.25],
        "predicted_class": [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
    })


class TestRegressionMetrics:
    """Test regression scoring metrics."""

    def test_weighted_mse(self, sample_predictions):
        """Test weighted mean squared error calculation."""
        from ins_pricing.production.scoring import weighted_mse

        mse = weighted_mse(
            sample_predictions['actual'],
            sample_predictions['predicted'],
            sample_predictions['weight']
        )

        assert isinstance(mse, (int, float, np.number))
        assert mse >= 0

    def test_weighted_mae(self, sample_predictions):
        """Test weighted mean absolute error calculation."""
        from ins_pricing.production.scoring import weighted_mae

        mae = weighted_mae(
            sample_predictions['actual'],
            sample_predictions['predicted'],
            sample_predictions['weight']
        )

        assert isinstance(mae, (int, float, np.number))
        assert mae >= 0

    def test_weighted_r2(self, sample_predictions):
        """Test weighted R² score calculation."""
        from ins_pricing.production.scoring import weighted_r2

        r2 = weighted_r2(
            sample_predictions['actual'],
            sample_predictions['predicted'],
            sample_predictions['weight']
        )

        assert isinstance(r2, (int, float, np.number))
        assert r2 <= 1.0

    def test_mape(self, sample_predictions):
        """Test mean absolute percentage error."""
        from ins_pricing.production.scoring import mape

        mape_score = mape(
            sample_predictions['actual'],
            sample_predictions['predicted']
        )

        assert isinstance(mape_score, (int, float, np.number))
        assert mape_score >= 0

    def test_metrics_with_zero_actuals(self):
        """Test metrics handling when actual values are zero."""
        from ins_pricing.production.scoring import mape

        data = pd.DataFrame({
            "actual": [0, 100, 200],
            "predicted": [10, 105, 195]
        })

        # MAPE should handle zeros gracefully
        with pytest.raises((ValueError, ZeroDivisionError)):
            mape(data['actual'], data['predicted'])


class TestClassificationMetrics:
    """Test classification scoring metrics."""

    def test_accuracy(self, classification_data):
        """Test accuracy calculation."""
        from ins_pricing.production.scoring import accuracy

        acc = accuracy(
            classification_data['actual'],
            classification_data['predicted_class']
        )

        assert 0 <= acc <= 1

    def test_precision_recall(self, classification_data):
        """Test precision and recall calculation."""
        from ins_pricing.production.scoring import precision_recall

        precision, recall = precision_recall(
            classification_data['actual'],
            classification_data['predicted_class']
        )

        assert 0 <= precision <= 1
        assert 0 <= recall <= 1

    def test_f1_score(self, classification_data):
        """Test F1 score calculation."""
        from ins_pricing.production.scoring import f1_score

        f1 = f1_score(
            classification_data['actual'],
            classification_data['predicted_class']
        )

        assert 0 <= f1 <= 1

    def test_roc_auc(self, classification_data):
        """Test ROC AUC calculation."""
        from ins_pricing.production.scoring import roc_auc

        auc = roc_auc(
            classification_data['actual'],
            classification_data['predicted_proba']
        )

        assert 0 <= auc <= 1

    def test_confusion_matrix(self, classification_data):
        """Test confusion matrix generation."""
        from ins_pricing.production.scoring import confusion_matrix

        cm = confusion_matrix(
            classification_data['actual'],
            classification_data['predicted_class']
        )

        assert cm.shape == (2, 2)
        assert np.all(cm >= 0)


class TestInsuranceMetrics:
    """Test insurance-specific metrics."""

    def test_loss_ratio(self):
        """Test loss ratio calculation."""
        from ins_pricing.production.scoring import loss_ratio

        data = pd.DataFrame({
            "claims": [100, 200, 150],
            "premiums": [120, 180, 160],
            "exposure": [1.0, 1.0, 1.0]
        })

        lr = loss_ratio(data['claims'], data['premiums'], data['exposure'])

        assert isinstance(lr, (int, float, np.number))
        assert lr >= 0

    def test_gini_coefficient(self, sample_predictions):
        """Test Gini coefficient calculation."""
        from ins_pricing.production.scoring import gini_coefficient

        gini = gini_coefficient(
            sample_predictions['actual'],
            sample_predictions['predicted']
        )

        assert -1 <= gini <= 1

    def test_lift_at_percentile(self, sample_predictions):
        """Test lift calculation at specific percentile."""
        from ins_pricing.production.scoring import lift_at_percentile

        lift = lift_at_percentile(
            sample_predictions['actual'],
            sample_predictions['predicted'],
            percentile=20
        )

        assert isinstance(lift, (int, float, np.number))


class TestMetricValidation:
    """Test metric input validation."""

    def test_mismatched_lengths(self):
        """Test error on mismatched array lengths."""
        from ins_pricing.production.scoring import weighted_mse
        from ins_pricing.utils.validation import validate_dataframe_not_empty

        actual = np.array([1, 2, 3])
        predicted = np.array([1, 2])  # Wrong length
        weights = np.array([1, 1, 1])

        with pytest.raises((ValueError, IndexError)):
            weighted_mse(actual, predicted, weights)

    def test_negative_weights(self):
        """Test handling of negative weights."""
        from ins_pricing.production.scoring import weighted_mse

        actual = np.array([100, 200, 300])
        predicted = np.array([105, 195, 310])
        weights = np.array([1.0, -1.0, 1.0])  # Negative weight

        with pytest.raises(ValueError):
            weighted_mse(actual, predicted, weights)

    def test_nan_values(self):
        """Test handling of NaN values."""
        from ins_pricing.production.scoring import weighted_mse

        actual = np.array([100, np.nan, 300])
        predicted = np.array([105, 195, 310])
        weights = np.array([1.0, 1.0, 1.0])

        with pytest.raises(ValueError):
            weighted_mse(actual, predicted, weights)


class TestScoringReport:
    """Test scoring report generation."""

    def test_generate_regression_report(self, sample_predictions):
        """Test comprehensive regression scoring report."""
        from ins_pricing.production.scoring import generate_scoring_report

        report = generate_scoring_report(
            actual=sample_predictions['actual'],
            predicted=sample_predictions['predicted'],
            weights=sample_predictions['weight'],
            task_type='regression'
        )

        assert 'mse' in report
        assert 'mae' in report
        assert 'r2' in report
        assert all(isinstance(v, (int, float, np.number)) for v in report.values())

    def test_generate_classification_report(self, classification_data):
        """Test comprehensive classification scoring report."""
        from ins_pricing.production.scoring import generate_scoring_report

        report = generate_scoring_report(
            actual=classification_data['actual'],
            predicted=classification_data['predicted_class'],
            predicted_proba=classification_data['predicted_proba'],
            task_type='classification'
        )

        assert 'accuracy' in report
        assert 'precision' in report
        assert 'recall' in report
        assert 'f1' in report
        assert 'roc_auc' in report

    def test_save_report_to_file(self, sample_predictions, tmp_path):
        """Test saving scoring report to file."""
        from ins_pricing.production.scoring import generate_scoring_report, save_report

        report = generate_scoring_report(
            actual=sample_predictions['actual'],
            predicted=sample_predictions['predicted'],
            task_type='regression'
        )

        output_path = tmp_path / "scoring_report.json"
        save_report(report, output_path)

        assert output_path.exists()


@pytest.mark.performance
class TestScoringPerformance:
    """Test scoring performance on large datasets."""

    def test_large_dataset_scoring(self):
        """Test scoring metrics on large dataset."""
        from ins_pricing.production.scoring import weighted_mse

        n = 1_000_000
        actual = np.random.uniform(100, 500, n)
        predicted = actual + np.random.normal(0, 20, n)
        weights = np.ones(n)

        import time
        start = time.time()
        mse = weighted_mse(actual, predicted, weights)
        elapsed = time.time() - start

        assert isinstance(mse, (int, float, np.number))
        assert elapsed < 1.0  # Should complete in under 1 second
