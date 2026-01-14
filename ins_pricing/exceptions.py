"""Custom exceptions for the ins_pricing package.

This module defines a hierarchy of exceptions used throughout the package
to provide more specific error handling and better error messages.

Example:
    >>> from ins_pricing.exceptions import ConfigurationError
    >>> if 'required_field' not in config:
    ...     raise ConfigurationError("Missing required field: required_field")
"""

from __future__ import annotations


class InsPricingError(Exception):
    """Base exception for all ins_pricing errors.

    All custom exceptions in this package inherit from this base class,
    allowing users to catch all package-specific errors with a single
    except clause.

    Example:
        >>> try:
        ...     # ins_pricing operations
        ...     pass
        ... except InsPricingError as e:
        ...     print(f"Package error: {e}")
    """

    pass


class ConfigurationError(InsPricingError):
    """Invalid configuration or missing required configuration parameters.

    Raised when:
    - Required configuration fields are missing
    - Configuration values are out of valid range
    - Configuration file is malformed or cannot be parsed
    - Conflicting configuration options are specified

    Example:
        >>> raise ConfigurationError(
        ...     "prop_test must be in [0, 1], got 1.5"
        ... )
    """

    pass


class DataValidationError(InsPricingError):
    """Data validation failed.

    Raised when:
    - Required DataFrame columns are missing
    - Column data types are incorrect
    - Data values are outside expected range
    - Data integrity checks fail

    Example:
        >>> raise DataValidationError(
        ...     "Missing required columns: ['age', 'premium']"
        ... )
    """

    pass


class ModelLoadError(InsPricingError):
    """Failed to load model or model artifacts.

    Raised when:
    - Model file does not exist
    - Model file is corrupted
    - Model version mismatch
    - Required model artifacts are missing

    Example:
        >>> raise ModelLoadError(
        ...     "Cannot load model from path/to/model.pkl: file not found"
        ... )
    """

    pass


class DistributedTrainingError(InsPricingError):
    """Distributed training failure.

    Raised when:
    - DDP initialization fails
    - Rank synchronization errors occur
    - Communication backend errors
    - Distributed barrier timeouts

    Example:
        >>> raise DistributedTrainingError(
        ...     "DDP barrier timeout after 1800s at rank 0"
        ... )
    """

    pass


class PreprocessingError(InsPricingError):
    """Data preprocessing failure.

    Raised when:
    - Feature engineering fails
    - Categorical encoding fails
    - Scaling/normalization fails
    - Data transformation produces invalid values

    Example:
        >>> raise PreprocessingError(
        ...     "StandardScaler produced NaN values for column 'income'"
        ... )
    """

    pass


class PredictionError(InsPricingError):
    """Model prediction failure.

    Raised when:
    - Prediction on new data fails
    - Input data incompatible with model
    - Model state is invalid
    - Prediction produces invalid values

    Example:
        >>> raise PredictionError(
        ...     "Model expects 50 features, got 48"
        ... )
    """

    pass


class GovernanceError(InsPricingError):
    """Model governance or registry operation failure.

    Raised when:
    - Model registration fails
    - Model approval workflow error
    - Audit logging fails
    - Release/deployment fails

    Example:
        >>> raise GovernanceError(
        ...     "Cannot register model: version 1.2.3 already exists"
        ... )
    """

    pass


# Convenience function for validation
def require_columns(df, required, df_name="DataFrame"):
    """Validate that DataFrame contains required columns.

    Args:
        df: pandas DataFrame to validate
        required: List of required column names
        df_name: Name of DataFrame for error message

    Raises:
        DataValidationError: If any required columns are missing

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> require_columns(df, ['a', 'b'], 'training_data')
        >>> require_columns(df, ['a', 'c'], 'training_data')
        Traceback (most recent call last):
            ...
        DataValidationError: training_data missing required columns: ['c']
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        available_preview = list(df.columns)[:50]
        raise DataValidationError(
            f"{df_name} missing required columns: {missing}. "
            f"Available columns (first 50): {available_preview}"
        )
