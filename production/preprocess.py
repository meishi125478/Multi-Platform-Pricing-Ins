"""Production preprocessing utilities for applying training-time transformations.

This module provides functions for loading and applying preprocessing artifacts
that were saved during model training. It ensures that production data undergoes
the same transformations as training data.

Typical workflow:
    1. Load preprocessing artifacts from training
    2. Prepare raw features (type conversion, missing value handling)
    3. Apply full preprocessing pipeline (one-hot encoding, scaling)

Example:
    >>> from ins_pricing.production.preprocess import load_preprocess_artifacts, apply_preprocess_artifacts
    >>>
    >>> # Load artifacts saved during training
    >>> artifacts = load_preprocess_artifacts("models/my_model/preprocess_artifacts.json")
    >>>
    >>> # Apply to new production data
    >>> import pandas as pd
    >>> raw_data = pd.read_csv("new_policies.csv")
    >>> preprocessed = apply_preprocess_artifacts(raw_data, artifacts)
    >>>
    >>> # Now ready for model prediction
    >>> predictions = model.predict(preprocessed)

Note:
    Preprocessing artifacts must match the exact configuration used during training
    to ensure consistency between training and production predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def load_preprocess_artifacts(path: str | Path) -> Dict[str, Any]:
    """Load preprocessing artifacts from a JSON file.

    Preprocessing artifacts contain all information needed to transform
    raw production data the same way as training data, including:
    - Feature names and types
    - Categorical feature categories
    - Numeric feature scaling parameters (mean, scale)
    - One-hot encoding configuration

    Args:
        path: Path to the preprocessing artifacts JSON file

    Returns:
        Dictionary containing preprocessing configuration and parameters:
        - factor_nmes: List of feature column names
        - cate_list: List of categorical feature names
        - num_features: List of numeric feature names
        - cat_categories: Dict mapping categorical features to their categories
        - numeric_scalers: Dict with scaling parameters (mean, scale) per feature
        - var_nmes: List of final column names after preprocessing
        - drop_first: Whether first category was dropped in one-hot encoding

    Raises:
        ValueError: If the artifact file is not a valid JSON dictionary
        FileNotFoundError: If the artifact file does not exist

    Example:
        >>> artifacts = load_preprocess_artifacts("models/xgb_model/preprocess.json")
        >>> print(artifacts.keys())
        dict_keys(['factor_nmes', 'cate_list', 'num_features', ...])
        >>> print(artifacts['factor_nmes'])
        ['age', 'gender', 'region', 'vehicle_age']
    """
    artifact_path = Path(path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid preprocess artifact: {artifact_path}")
    return payload


def prepare_raw_features(df: pd.DataFrame, artifacts: Dict[str, Any]) -> pd.DataFrame:
    """Prepare raw features for preprocessing by handling types and missing values.

    This function performs initial data preparation:
    1. Ensures all required features exist (adds missing columns with NA)
    2. Converts numeric features to numeric type (coercing errors to 0)
    3. Converts categorical features to proper categorical type
    4. Applies category constraints from training data

    Args:
        df: Raw input DataFrame with policy/claim data
        artifacts: Preprocessing artifacts from load_preprocess_artifacts()
                  Must contain: factor_nmes, cate_list, num_features, cat_categories

    Returns:
        DataFrame with:
        - Only feature columns (factor_nmes)
        - Numeric features as float64
        - Categorical features as object or Categorical
        - Missing columns filled with NA
        - Invalid numeric values filled with 0

    Example:
        >>> raw_df = pd.DataFrame({
        ...     'age': ['25', '30', 'invalid'],
        ...     'gender': ['M', 'F', 'X'],
        ...     'missing_feature': [1, 2, 3]
        ... })
        >>> artifacts = {
        ...     'factor_nmes': ['age', 'gender', 'region'],
        ...     'num_features': ['age'],
        ...     'cate_list': ['gender', 'region'],
        ...     'cat_categories': {'gender': ['M', 'F'], 'region': ['North', 'South']}
        ... }
        >>> prepared = prepare_raw_features(raw_df, artifacts)
        >>> print(prepared.dtypes)
        age        float64
        gender     category
        region     object
        >>> print(prepared['age'].tolist())
        [25.0, 30.0, 0.0]  # 'invalid' coerced to 0

    Note:
        - Missing numeric values are filled with 0 (not NaN)
        - Unknown categories are kept as-is (handled later in one-hot encoding)
        - Extra columns in input df are dropped
    """
    factor_nmes = list(artifacts.get("factor_nmes") or [])
    cate_list = list(artifacts.get("cate_list") or [])
    num_features = set(artifacts.get("num_features") or [])
    cat_categories = artifacts.get("cat_categories") or {}

    work = df.copy()
    for col in factor_nmes:
        if col not in work.columns:
            work[col] = pd.NA

    for col in factor_nmes:
        if col in num_features:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)
        else:
            series = work[col].astype("object").fillna("<NA>")
            cats = cat_categories.get(col)
            if isinstance(cats, list) and cats:
                series = pd.Categorical(series, categories=cats)
            work[col] = series

    if factor_nmes:
        work = work[factor_nmes]
    return work


def apply_preprocess_artifacts(df: pd.DataFrame, artifacts: Dict[str, Any]) -> pd.DataFrame:
    """Apply complete preprocessing pipeline to production data.

    This is the main preprocessing function that applies the full transformation
    pipeline used during training:
    1. Prepare raw features (via prepare_raw_features)
    2. One-hot encode categorical features
    3. Standardize numeric features using training statistics
    4. Align columns to match training data exactly

    The output is ready for model prediction and guaranteed to have the same
    column structure as the training data.

    Args:
        df: Raw input DataFrame with policy/claim data
        artifacts: Complete preprocessing artifacts dictionary containing:
            - factor_nmes: Feature names
            - cate_list: Categorical feature names
            - num_features: Numeric feature names
            - cat_categories: Categorical feature categories
            - numeric_scalers: Dict with 'mean' and 'scale' for each numeric feature
            - var_nmes: Final column names after preprocessing
            - drop_first: Whether to drop first category in one-hot encoding

    Returns:
        Preprocessed DataFrame ready for model prediction with:
        - One-hot encoded categorical features
        - Standardized numeric features: (value - mean) / scale
        - Exact column structure matching training data
        - Missing columns filled with 0
        - dtype: int8 for one-hot columns, float64 for numeric

    Raises:
        KeyError: If artifacts are missing required keys

    Example:
        >>> # Complete preprocessing pipeline
        >>> artifacts = load_preprocess_artifacts("models/xgb/preprocess.json")
        >>> raw_data = pd.DataFrame({
        ...     'age': [25, 30, 35],
        ...     'gender': ['M', 'F', 'M'],
        ...     'region': ['North', 'South', 'East']
        ... })
        >>> processed = apply_preprocess_artifacts(raw_data, artifacts)
        >>> print(processed.shape)
        (3, 50)  # More columns after one-hot encoding
        >>> print(processed.columns[:5])
        Index(['age', 'gender_F', 'gender_M', 'region_East', 'region_North'], dtype='object')
        >>> # Age is now standardized
        >>> print(processed['age'].tolist())
        [-0.52, 0.0, 0.52]  # Standardized values

    Note:
        - Categorical features not seen during training will be ignored (dropped in one-hot)
        - Numeric features are standardized using training mean and std
        - Output column order matches training data exactly
        - Use this function for production scoring to ensure consistency
    """
    cate_list = list(artifacts.get("cate_list") or [])
    num_features = list(artifacts.get("num_features") or [])
    var_nmes = list(artifacts.get("var_nmes") or [])
    numeric_scalers = artifacts.get("numeric_scalers") or {}
    drop_first = bool(artifacts.get("drop_first", True))

    work = prepare_raw_features(df, artifacts)
    oht = pd.get_dummies(
        work,
        columns=cate_list,
        drop_first=drop_first,
        dtype="int8",
    )

    for col in num_features:
        if col not in oht.columns:
            continue
        stats = numeric_scalers.get(col) or {}
        mean = float(stats.get("mean", 0.0))
        scale = float(stats.get("scale", 1.0))
        if scale == 0.0:
            scale = 1.0
        oht[col] = (oht[col] - mean) / scale

    if var_nmes:
        oht = oht.reindex(columns=var_nmes, fill_value=0)
    return oht
