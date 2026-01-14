"""Data validation utilities for ins_pricing.

This module provides reusable validation functions to ensure data quality
and provide clear error messages when validation fails.

Example:
    >>> import pandas as pd
    >>> from ins_pricing.utils.validation import validate_required_columns
    >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    >>> validate_required_columns(df, ['a', 'b'], df_name='my_data')
    >>> # Raises DataValidationError if columns missing
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import pandas as pd

from ins_pricing.exceptions import DataValidationError


def validate_required_columns(
    df: pd.DataFrame,
    required: List[str],
    *,
    df_name: str = "DataFrame"
) -> None:
    """Validate that DataFrame contains all required columns.

    Args:
        df: DataFrame to validate
        required: List of required column names
        df_name: Name of DataFrame for error messages (default: "DataFrame")

    Raises:
        DataValidationError: If any required columns are missing

    Example:
        >>> df = pd.DataFrame({'age': [25, 30], 'premium': [100, 200]})
        >>> validate_required_columns(df, ['age', 'premium'], df_name='policy_data')
        >>> validate_required_columns(df, ['age', 'claim'], df_name='policy_data')
        Traceback (most recent call last):
            ...
        DataValidationError: policy_data missing required columns: ['claim']...
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        available_preview = list(df.columns)[:50]
        raise DataValidationError(
            f"{df_name} missing required columns: {missing}. "
            f"Available columns (first 50): {available_preview}"
        )


def validate_column_types(
    df: pd.DataFrame,
    type_spec: Dict[str, Union[type, str]],
    *,
    coerce: bool = False,
    df_name: str = "DataFrame"
) -> pd.DataFrame:
    """Validate and optionally coerce column data types.

    Args:
        df: DataFrame to validate
        type_spec: Dictionary mapping column names to expected types.
                  Types can be Python types (int, float, str) or pandas
                  dtype strings ('int64', 'float64', 'object', 'category')
        coerce: If True, attempt to convert columns to expected types.
               If False, raise error on type mismatch (default: False)
        df_name: Name of DataFrame for error messages

    Returns:
        DataFrame with validated (and possibly coerced) types

    Raises:
        DataValidationError: If column types don't match and coerce=False

    Example:
        >>> df = pd.DataFrame({'age': ['25', '30'], 'premium': [100.0, 200.0]})
        >>> df = validate_column_types(
        ...     df,
        ...     {'age': 'int64', 'premium': 'float64'},
        ...     coerce=True
        ... )
        >>> df['age'].dtype
        dtype('int64')
    """
    df = df.copy() if coerce else df

    for col, expected_type in type_spec.items():
        if col not in df.columns:
            continue

        current_dtype = df[col].dtype

        # Convert type spec to pandas dtype
        if isinstance(expected_type, type):
            if expected_type == int:
                expected_dtype = 'int64'
            elif expected_type == float:
                expected_dtype = 'float64'
            elif expected_type == str:
                expected_dtype = 'object'
            else:
                expected_dtype = str(expected_type)
        else:
            expected_dtype = expected_type

        # Check if types match
        type_matches = (
            str(current_dtype) == expected_dtype or
            current_dtype.name == expected_dtype
        )

        if not type_matches:
            if coerce:
                try:
                    df[col] = df[col].astype(expected_dtype)
                except (ValueError, TypeError) as e:
                    raise DataValidationError(
                        f"{df_name}: Cannot convert column '{col}' from "
                        f"{current_dtype} to {expected_dtype}: {e}"
                    )
            else:
                raise DataValidationError(
                    f"{df_name}: Column '{col}' has type {current_dtype}, "
                    f"expected {expected_dtype}"
                )

    return df


def validate_value_range(
    df: pd.DataFrame,
    col: str,
    *,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    df_name: str = "DataFrame"
) -> None:
    """Validate that numeric column values are within expected range.

    Args:
        df: DataFrame to validate
        col: Column name to check
        min_val: Minimum allowed value (inclusive), None for no minimum
        max_val: Maximum allowed value (inclusive), None for no maximum
        df_name: Name of DataFrame for error messages

    Raises:
        DataValidationError: If values are outside the specified range

    Example:
        >>> df = pd.DataFrame({'age': [25, 30, 150]})
        >>> validate_value_range(df, 'age', min_val=0, max_val=120)
        Traceback (most recent call last):
            ...
        DataValidationError: ...values outside range [0, 120]...
    """
    if col not in df.columns:
        raise DataValidationError(
            f"{df_name}: Column '{col}' not found for range validation"
        )

    if not pd.api.types.is_numeric_dtype(df[col]):
        raise DataValidationError(
            f"{df_name}: Column '{col}' is not numeric, cannot validate range"
        )

    violations = []

    if min_val is not None:
        below_min = df[col] < min_val
        if below_min.any():
            count = below_min.sum()
            min_found = df.loc[below_min, col].min()
            violations.append(
                f"{count} values below minimum {min_val} (min found: {min_found})"
            )

    if max_val is not None:
        above_max = df[col] > max_val
        if above_max.any():
            count = above_max.sum()
            max_found = df.loc[above_max, col].max()
            violations.append(
                f"{count} values above maximum {max_val} (max found: {max_found})"
            )

    if violations:
        range_str = f"[{min_val}, {max_val}]"
        raise DataValidationError(
            f"{df_name}: Column '{col}' has values outside range {range_str}: "
            f"{'; '.join(violations)}"
        )


def validate_no_nulls(
    df: pd.DataFrame,
    columns: List[str],
    *,
    df_name: str = "DataFrame"
) -> None:
    """Validate that specified columns contain no null values.

    Args:
        df: DataFrame to validate
        columns: List of column names to check for nulls
        df_name: Name of DataFrame for error messages

    Raises:
        DataValidationError: If any specified columns contain null values

    Example:
        >>> df = pd.DataFrame({'age': [25, None, 30], 'premium': [100, 200, 300]})
        >>> validate_no_nulls(df, ['age', 'premium'])
        Traceback (most recent call last):
            ...
        DataValidationError: ...contains null values: age (1 nulls)...
    """
    null_info = []

    for col in columns:
        if col not in df.columns:
            raise DataValidationError(
                f"{df_name}: Column '{col}' not found for null validation"
            )

        null_count = df[col].isna().sum()
        if null_count > 0:
            null_info.append(f"{col} ({null_count} nulls)")

    if null_info:
        raise DataValidationError(
            f"{df_name} contains null values: {', '.join(null_info)}"
        )


def validate_categorical_values(
    df: pd.DataFrame,
    col: str,
    allowed_values: List[str],
    *,
    df_name: str = "DataFrame"
) -> None:
    """Validate that categorical column contains only allowed values.

    Args:
        df: DataFrame to validate
        col: Column name to check
        allowed_values: List of allowed values
        df_name: Name of DataFrame for error messages

    Raises:
        DataValidationError: If column contains values not in allowed_values

    Example:
        >>> df = pd.DataFrame({'gender': ['M', 'F', 'X']})
        >>> validate_categorical_values(df, 'gender', ['M', 'F'])
        Traceback (most recent call last):
            ...
        DataValidationError: ...contains invalid values: ['X']...
    """
    if col not in df.columns:
        raise DataValidationError(
            f"{df_name}: Column '{col}' not found for categorical validation"
        )

    unique_values = df[col].dropna().unique()
    invalid_values = [v for v in unique_values if v not in allowed_values]

    if invalid_values:
        raise DataValidationError(
            f"{df_name}: Column '{col}' contains invalid values: {invalid_values}. "
            f"Allowed values: {allowed_values}"
        )


def validate_positive(
    df: pd.DataFrame,
    columns: List[str],
    *,
    allow_zero: bool = False,
    df_name: str = "DataFrame"
) -> None:
    """Validate that numeric columns contain only positive values.

    Args:
        df: DataFrame to validate
        columns: List of column names to check
        allow_zero: If True, allow zero values (default: False)
        df_name: Name of DataFrame for error messages

    Raises:
        DataValidationError: If columns contain non-positive values

    Example:
        >>> df = pd.DataFrame({'premium': [100, -50, 200], 'exposure': [1, 0, 2]})
        >>> validate_positive(df, ['premium', 'exposure'])
        Traceback (most recent call last):
            ...
        DataValidationError: ...contains non-positive values...
    """
    violations = []

    for col in columns:
        if col not in df.columns:
            raise DataValidationError(
                f"{df_name}: Column '{col}' not found for positivity validation"
            )

        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(
                f"{df_name}: Column '{col}' is not numeric"
            )

        if allow_zero:
            invalid = df[col] < 0
            msg = "negative"
        else:
            invalid = df[col] <= 0
            msg = "non-positive"

        if invalid.any():
            count = invalid.sum()
            min_val = df.loc[invalid, col].min()
            violations.append(f"{col} ({count} {msg} values, min: {min_val})")

    if violations:
        raise DataValidationError(
            f"{df_name} contains {msg} values: {', '.join(violations)}"
        )


def validate_dataframe_not_empty(
    df: pd.DataFrame,
    *,
    df_name: str = "DataFrame"
) -> None:
    """Validate that DataFrame is not empty.

    Args:
        df: DataFrame to validate
        df_name: Name of DataFrame for error messages

    Raises:
        DataValidationError: If DataFrame is empty

    Example:
        >>> df = pd.DataFrame()
        >>> validate_dataframe_not_empty(df, df_name='train_data')
        Traceback (most recent call last):
            ...
        DataValidationError: train_data is empty (0 rows)
    """
    if len(df) == 0:
        raise DataValidationError(f"{df_name} is empty (0 rows)")


def validate_date_range(
    df: pd.DataFrame,
    col: str,
    *,
    min_date: Optional[pd.Timestamp] = None,
    max_date: Optional[pd.Timestamp] = None,
    df_name: str = "DataFrame"
) -> None:
    """Validate that date column values are within expected range.

    Args:
        df: DataFrame to validate
        col: Column name to check (should be datetime type)
        min_date: Minimum allowed date, None for no minimum
        max_date: Maximum allowed date, None for no maximum
        df_name: Name of DataFrame for error messages

    Raises:
        DataValidationError: If dates are outside the specified range

    Example:
        >>> df = pd.DataFrame({'policy_date': pd.to_datetime(['2020-01-01', '2025-01-01'])})
        >>> validate_date_range(
        ...     df, 'policy_date',
        ...     min_date=pd.Timestamp('2020-01-01'),
        ...     max_date=pd.Timestamp('2023-12-31')
        ... )
        Traceback (most recent call last):
            ...
        DataValidationError: ...dates outside range...
    """
    if col not in df.columns:
        raise DataValidationError(
            f"{df_name}: Column '{col}' not found for date validation"
        )

    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        raise DataValidationError(
            f"{df_name}: Column '{col}' is not datetime type"
        )

    violations = []

    if min_date is not None:
        before_min = df[col] < min_date
        if before_min.any():
            count = before_min.sum()
            earliest = df.loc[before_min, col].min()
            violations.append(
                f"{count} dates before {min_date} (earliest: {earliest})"
            )

    if max_date is not None:
        after_max = df[col] > max_date
        if after_max.any():
            count = after_max.sum()
            latest = df.loc[after_max, col].max()
            violations.append(
                f"{count} dates after {max_date} (latest: {latest})"
            )

    if violations:
        raise DataValidationError(
            f"{df_name}: Column '{col}' has dates outside range: "
            f"{'; '.join(violations)}"
        )
