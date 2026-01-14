"""Tests for production preprocessing module."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch

from ins_pricing.exceptions import PreprocessingError, DataValidationError


@pytest.fixture
def sample_raw_data():
    """Sample raw input data."""
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "gender": ["M", "F", "M", "F", "M"],
        "region": ["North", "South", "East", "West", "North"],
        "premium": [100.0, 150.0, 200.0, 250.0, 300.0],
        "coverage": ["Basic", "Premium", "Basic", "Premium", "Premium"]
    })


class TestFeatureEngineering:
    """Test feature engineering transformations."""

    def test_create_age_bands(self, sample_raw_data):
        """Test age banding transformation."""
        from ins_pricing.production.preprocess import create_age_bands

        df = create_age_bands(sample_raw_data, 'age', bins=[0, 30, 40, 100])

        assert 'age_band' in df.columns
        assert df['age_band'].dtype == 'object' or pd.api.types.is_categorical_dtype(df['age_band'])

    def test_encode_categorical(self, sample_raw_data):
        """Test categorical encoding."""
        from ins_pricing.production.preprocess import encode_categorical

        df = encode_categorical(
            sample_raw_data,
            columns=['gender', 'region'],
            method='onehot'
        )

        # Check that encoded columns exist
        assert any('gender_' in col for col in df.columns)
        assert any('region_' in col for col in df.columns)

    def test_scale_numeric_features(self, sample_raw_data):
        """Test numeric feature scaling."""
        from ins_pricing.production.preprocess import scale_features

        df = scale_features(
            sample_raw_data,
            columns=['premium'],
            method='standard'
        )

        # Check that scaled values have mean ~0 and std ~1
        assert abs(df['premium'].mean()) < 0.1
        assert abs(df['premium'].std() - 1.0) < 0.1

    def test_create_interaction_features(self, sample_raw_data):
        """Test interaction feature creation."""
        from ins_pricing.production.preprocess import create_interactions

        df = create_interactions(
            sample_raw_data,
            feature_pairs=[('age', 'premium')]
        )

        assert 'age_x_premium' in df.columns

    def test_polynomial_features(self, sample_raw_data):
        """Test polynomial feature generation."""
        from ins_pricing.production.preprocess import create_polynomial_features

        df = create_polynomial_features(
            sample_raw_data,
            columns=['age'],
            degree=2
        )

        assert 'age_squared' in df.columns


class TestDataCleaning:
    """Test data cleaning operations."""

    def test_handle_missing_values(self):
        """Test missing value handling."""
        from ins_pricing.production.preprocess import handle_missing

        df = pd.DataFrame({
            "col1": [1, 2, np.nan, 4],
            "col2": [np.nan, 2, 3, 4],
            "col3": ["A", "B", None, "D"]
        })

        cleaned = handle_missing(df, strategy='mean', columns=['col1', 'col2'])

        assert cleaned['col1'].isna().sum() == 0
        assert cleaned['col2'].isna().sum() == 0

    def test_remove_outliers(self, sample_raw_data):
        """Test outlier removal."""
        from ins_pricing.production.preprocess import remove_outliers

        # Add outlier
        data_with_outlier = sample_raw_data.copy()
        data_with_outlier.loc[0, 'premium'] = 10000  # Extreme value

        cleaned = remove_outliers(data_with_outlier, column='premium', method='iqr')

        assert len(cleaned) < len(data_with_outlier)

    def test_deduplicate(self):
        """Test duplicate removal."""
        from ins_pricing.production.preprocess import deduplicate

        df = pd.DataFrame({
            "id": [1, 2, 2, 3],
            "value": [10, 20, 20, 30]
        })

        deduped = deduplicate(df)

        assert len(deduped) == 3

    def test_fix_data_types(self):
        """Test data type corrections."""
        from ins_pricing.production.preprocess import fix_data_types

        df = pd.DataFrame({
            "age": ["25", "30", "35"],  # String instead of int
            "premium": [100, 200, 300]
        })

        fixed = fix_data_types(df, type_spec={'age': 'int64'})

        assert fixed['age'].dtype == np.int64


class TestFeatureSelection:
    """Test feature selection operations."""

    def test_select_by_importance(self):
        """Test feature selection by importance."""
        from ins_pricing.production.preprocess import select_features_by_importance

        X = pd.DataFrame(np.random.rand(100, 10))
        y = pd.Series(np.random.rand(100))

        selected = select_features_by_importance(X, y, n_features=5)

        assert selected.shape[1] == 5

    def test_remove_low_variance(self):
        """Test removal of low variance features."""
        from ins_pricing.production.preprocess import remove_low_variance

        df = pd.DataFrame({
            "constant": [1, 1, 1, 1],  # Zero variance
            "low_var": [1, 1, 1, 2],   # Low variance
            "high_var": [1, 5, 10, 20] # High variance
        })

        filtered = remove_low_variance(df, threshold=0.01)

        assert 'constant' not in filtered.columns
        assert 'high_var' in filtered.columns

    def test_remove_correlated_features(self):
        """Test removal of highly correlated features."""
        from ins_pricing.production.preprocess import remove_correlated

        # Create correlated features
        df = pd.DataFrame({
            "feature_1": np.random.rand(100),
            "feature_2": np.random.rand(100)
        })
        df['feature_3'] = df['feature_1'] * 1.1  # Highly correlated

        filtered = remove_correlated(df, threshold=0.95)

        assert filtered.shape[1] < 3


class TestPipelineValidation:
    """Test preprocessing pipeline validation."""

    def test_validate_input_schema(self, sample_raw_data):
        """Test input schema validation."""
        from ins_pricing.production.preprocess import validate_input_schema

        expected_schema = {
            "age": "int64",
            "gender": "object",
            "premium": "float64"
        }

        # Should not raise
        validate_input_schema(sample_raw_data, expected_schema)

    def test_validate_input_schema_failure(self):
        """Test input schema validation catches errors."""
        from ins_pricing.production.preprocess import validate_input_schema
        from ins_pricing.utils.validation import validate_column_types

        df = pd.DataFrame({
            "age": ["not_a_number", "25"],  # Wrong type
            "premium": [100.0, 200.0]
        })

        expected_schema = {"age": "int64", "premium": "float64"}

        with pytest.raises(DataValidationError):
            validate_column_types(df, expected_schema, coerce=False)

    def test_validate_feature_range(self, sample_raw_data):
        """Test feature value range validation."""
        from ins_pricing.utils.validation import validate_value_range

        # Age should be positive
        validate_value_range(sample_raw_data, 'age', min_val=0, max_val=120)

        # Premium should be positive
        validate_value_range(sample_raw_data, 'premium', min_val=0)


class TestPreprocessorState:
    """Test preprocessor state management."""

    def test_save_preprocessor_state(self, sample_raw_data, tmp_path):
        """Test saving preprocessor state."""
        from ins_pricing.production.preprocess import Preprocessor

        preprocessor = Preprocessor()
        preprocessor.fit(sample_raw_data)

        state_path = tmp_path / "preprocessor_state.pkl"
        preprocessor.save(state_path)

        assert state_path.exists()

    def test_load_preprocessor_state(self, tmp_path):
        """Test loading preprocessor state."""
        from ins_pricing.production.preprocess import Preprocessor

        # Create and save
        preprocessor = Preprocessor()
        state_path = tmp_path / "preprocessor_state.pkl"
        preprocessor.save(state_path)

        # Load
        loaded = Preprocessor.load(state_path)

        assert loaded is not None

    def test_preprocessor_consistency(self, sample_raw_data, tmp_path):
        """Test that loaded preprocessor produces same results."""
        from ins_pricing.production.preprocess import Preprocessor

        # Fit and transform
        preprocessor = Preprocessor()
        preprocessor.fit(sample_raw_data)
        result1 = preprocessor.transform(sample_raw_data)

        # Save, load, transform
        state_path = tmp_path / "preprocessor_state.pkl"
        preprocessor.save(state_path)
        loaded = Preprocessor.load(state_path)
        result2 = loaded.transform(sample_raw_data)

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)


class TestTransformationPipeline:
    """Test complete transformation pipeline."""

    def test_full_pipeline(self, sample_raw_data):
        """Test complete preprocessing pipeline."""
        from ins_pricing.production.preprocess import PreprocessingPipeline

        pipeline = PreprocessingPipeline([
            ('handle_missing', {'strategy': 'mean'}),
            ('encode_categorical', {'columns': ['gender', 'region']}),
            ('scale_features', {'columns': ['premium'], 'method': 'standard'})
        ])

        transformed = pipeline.fit_transform(sample_raw_data)

        assert transformed is not None
        assert len(transformed) == len(sample_raw_data)

    def test_pipeline_error_handling(self):
        """Test pipeline handles errors gracefully."""
        from ins_pricing.production.preprocess import PreprocessingPipeline

        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["A", "B", "C"]
        })

        # Try to scale a categorical column (should fail)
        pipeline = PreprocessingPipeline([
            ('scale_features', {'columns': ['col2'], 'method': 'standard'})
        ])

        with pytest.raises(PreprocessingError):
            pipeline.fit_transform(df)


@pytest.mark.performance
class TestPreprocessingPerformance:
    """Test preprocessing performance."""

    def test_large_dataset_preprocessing(self):
        """Test preprocessing on large dataset."""
        from ins_pricing.production.preprocess import Preprocessor

        n = 100_000
        large_df = pd.DataFrame({
            "age": np.random.randint(18, 80, n),
            "premium": np.random.uniform(100, 1000, n),
            "region": np.random.choice(['A', 'B', 'C', 'D'], n)
        })

        preprocessor = Preprocessor()

        import time
        start = time.time()
        preprocessor.fit(large_df)
        transformed = preprocessor.transform(large_df)
        elapsed = time.time() - start

        assert len(transformed) == n
        assert elapsed < 5.0  # Should complete in under 5 seconds
