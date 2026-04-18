"""Tests for production monitoring module."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from ins_pricing.exceptions import DataValidationError


@pytest.fixture
def sample_production_data():
    """Sample production data with timestamps."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    return pd.DataFrame({
        "date": dates,
        "prediction": np.random.uniform(100, 500, 100),
        "actual": np.random.uniform(100, 500, 100),
        "feature_1": np.random.uniform(0, 1, 100),
        "feature_2": np.random.choice(['A', 'B', 'C'], 100)
    })


@pytest.fixture
def training_distribution():
    """Reference training data distribution."""
    return pd.DataFrame({
        "feature_1": np.random.uniform(0, 1, 1000),
        "feature_2": np.random.choice(['A', 'B', 'C'], 1000, p=[0.5, 0.3, 0.2])
    })


class TestDriftDetection:
    """Test data drift detection."""

    def test_psi_calculation(self, training_distribution, sample_production_data):
        """Test Population Stability Index (PSI) calculation."""
        from ins_pricing.production.monitoring import calculate_psi

        psi = calculate_psi(
            expected=training_distribution['feature_1'],
            actual=sample_production_data['feature_1'],
            buckets=10
        )

        assert isinstance(psi, (int, float, np.number))
        assert psi >= 0

    def test_psi_drift_detected(self):
        """Test PSI detects significant drift."""
        from ins_pricing.production.monitoring import calculate_psi

        # Create distributions with significant drift
        expected = np.random.uniform(0, 1, 1000)
        actual = np.random.uniform(0.5, 1.5, 1000)  # Shifted distribution

        psi = calculate_psi(expected, actual, buckets=10)

        # PSI > 0.2 typically indicates significant drift
        assert psi > 0.1

    def test_psi_no_drift(self):
        """Test PSI when no drift present."""
        from ins_pricing.production.monitoring import calculate_psi

        # Same distribution
        distribution = np.random.uniform(0, 1, 1000)
        expected = distribution[:500]
        actual = distribution[500:]

        psi = calculate_psi(expected, actual, buckets=10)

        # Should be very low PSI
        assert psi < 0.1

    def test_categorical_drift(self, training_distribution, sample_production_data):
        """Test drift detection for categorical features."""
        from ins_pricing.production.monitoring import categorical_drift

        drift_score = categorical_drift(
            expected=training_distribution['feature_2'],
            actual=sample_production_data['feature_2']
        )

        assert isinstance(drift_score, (int, float, np.number))
        assert drift_score >= 0

    def test_ks_test_drift(self):
        """Test Kolmogorov-Smirnov test for drift."""
        from ins_pricing.production.monitoring import ks_test

        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(0.5, 1, 1000)  # Shifted mean

        statistic, p_value = ks_test(expected, actual)

        assert 0 <= statistic <= 1
        assert 0 <= p_value <= 1


class TestPerformanceMonitoring:
    """Test model performance monitoring."""

    def test_rolling_metrics(self, sample_production_data):
        """Test calculation of rolling performance metrics."""
        from ins_pricing.production.monitoring import rolling_metrics

        metrics = rolling_metrics(
            df=sample_production_data,
            actual_col='actual',
            pred_col='prediction',
            window=7
        )

        assert 'rolling_mae' in metrics.columns
        assert 'rolling_mse' in metrics.columns
        assert len(metrics) == len(sample_production_data)

    def test_performance_degradation_alert(self, sample_production_data):
        """Test alerting on performance degradation."""
        from ins_pricing.production.monitoring import check_performance_degradation

        # Simulate degrading predictions
        sample_production_data.loc[50:, 'prediction'] = \
            sample_production_data.loc[50:, 'actual'] * 2  # Make worse

        is_degraded = check_performance_degradation(
            df=sample_production_data,
            actual_col='actual',
            pred_col='prediction',
            threshold=0.2  # 20% worse
        )

        assert isinstance(is_degraded, bool)

    def test_metric_comparison(self):
        """Test comparison of current vs baseline metrics."""
        from ins_pricing.production.monitoring import compare_metrics

        baseline = {'mse': 100, 'mae': 8, 'r2': 0.85}
        current = {'mse': 150, 'mae': 10, 'r2': 0.75}

        comparison = compare_metrics(baseline, current)

        assert 'mse_change' in comparison
        assert 'mae_change' in comparison
        assert 'r2_change' in comparison


class TestDataQualityChecks:
    """Test data quality monitoring."""

    def test_missing_value_detection(self):
        """Test detection of missing values in production data."""
        from ins_pricing.production.monitoring import check_missing_values

        data = pd.DataFrame({
            "col1": [1, 2, np.nan, 4],
            "col2": [1, 2, 3, 4],
            "col3": [np.nan, np.nan, 3, 4]
        })

        missing_report = check_missing_values(data)

        assert 'col1' in missing_report
        assert 'col3' in missing_report
        assert missing_report['col1']['count'] == 1
        assert missing_report['col3']['count'] == 2

    def test_outlier_detection(self):
        """Test outlier detection in production data."""
        from ins_pricing.production.monitoring import detect_outliers

        data = pd.Series([1, 2, 3, 4, 5, 100, 2, 3, 4, 5])  # 100 is outlier

        outliers = detect_outliers(data, method='iqr')

        assert len(outliers) > 0
        assert 100 in data[outliers].values

    def test_schema_validation(self):
        """Test schema validation for production data."""
        from ins_pricing.production.monitoring import validate_schema

        expected_schema = {
            "feature_1": "float64",
            "feature_2": "object",
            "prediction": "float64"
        }

        data = pd.DataFrame({
            "feature_1": [1.0, 2.0],
            "feature_2": ["A", "B"],
            "prediction": [100.0, 200.0]
        })

        is_valid = validate_schema(data, expected_schema)

        assert is_valid

    def test_schema_validation_failure(self):
        """Test schema validation catches type mismatches."""
        from ins_pricing.production.monitoring import validate_schema

        expected_schema = {
            "feature_1": "float64",
            "feature_2": "int64"  # Expect int
        }

        data = pd.DataFrame({
            "feature_1": [1.0, 2.0],
            "feature_2": ["A", "B"]  # Actually string
        })

        is_valid = validate_schema(data, expected_schema)

        assert not is_valid


class TestAlertingSystem:
    """Test monitoring alert generation."""

    def test_drift_alert(self):
        """Test alert generation for drift detection."""
        from ins_pricing.production.monitoring import generate_drift_alert

        alert = generate_drift_alert(
            feature='age',
            psi=0.35,
            threshold=0.25
        )

        assert alert['alert_type'] == 'drift'
        assert alert['feature'] == 'age'
        assert alert['severity'] == 'high'

    def test_performance_alert(self):
        """Test alert generation for performance degradation."""
        from ins_pricing.production.monitoring import generate_performance_alert

        alert = generate_performance_alert(
            metric='mae',
            baseline=10.0,
            current=15.0,
            threshold=0.2
        )

        assert alert['alert_type'] == 'performance'
        assert alert['metric'] == 'mae'

    @patch('ins_pricing.production.monitoring.send_email')
    def test_send_alert_email(self, mock_send):
        """Test sending alert via email."""
        from ins_pricing.production.monitoring import send_alert

        alert = {
            'alert_type': 'drift',
            'feature': 'age',
            'severity': 'high'
        }

        send_alert(alert, recipients=['team@example.com'])

        mock_send.assert_called_once()

    @patch('ins_pricing.production.monitoring.log_to_monitoring_system')
    def test_log_alert(self, mock_log):
        """Test logging alert to monitoring system."""
        from ins_pricing.production.monitoring import log_alert

        alert = {'alert_type': 'performance', 'severity': 'medium'}

        log_alert(alert)

        mock_log.assert_called_once()


class TestMonitoringDashboard:
    """Test monitoring dashboard data preparation."""

    def test_dashboard_metrics(self, sample_production_data):
        """Test preparation of dashboard metrics."""
        from ins_pricing.production.monitoring import prepare_dashboard_metrics

        metrics = prepare_dashboard_metrics(
            df=sample_production_data,
            actual_col='actual',
            pred_col='prediction',
            date_col='date'
        )

        assert 'daily_predictions' in metrics
        assert 'daily_mae' in metrics
        assert 'daily_mse' in metrics

    def test_feature_distribution_summary(self, sample_production_data):
        """Test feature distribution summary for dashboard."""
        from ins_pricing.production.monitoring import feature_distribution_summary

        summary = feature_distribution_summary(
            sample_production_data,
            features=['feature_1', 'feature_2']
        )

        assert 'feature_1' in summary
        assert 'feature_2' in summary
        assert 'mean' in summary['feature_1']
        assert 'std' in summary['feature_1']


class TestBatchMonitoring:
    """Test batch monitoring functionality."""

    def test_monitor_batch_predictions(self, sample_production_data, training_distribution):
        """Test monitoring a batch of predictions."""
        from ins_pricing.production.monitoring import monitor_batch

        report = monitor_batch(
            production_data=sample_production_data,
            reference_data=training_distribution,
            features=['feature_1', 'feature_2']
        )

        assert 'drift_scores' in report
        assert 'quality_checks' in report
        assert 'alerts' in report

    def test_scheduled_monitoring(self):
        """Test scheduled monitoring execution."""
        from ins_pricing.production.monitoring import run_scheduled_monitoring

        with patch('ins_pricing.production.monitoring.load_production_data') as mock_load:
            with patch('ins_pricing.production.monitoring.monitor_batch') as mock_monitor:
                mock_load.return_value = pd.DataFrame()
                mock_monitor.return_value = {'status': 'ok'}

                result = run_scheduled_monitoring(config={'schedule': 'daily'})

                assert result['status'] == 'ok'


@pytest.mark.integration
class TestMonitoringIntegration:
    """Integration tests for monitoring pipeline."""

    def test_full_monitoring_pipeline(self):
        """Test complete monitoring pipeline."""
        # Would require full setup with real data
        pass
