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
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ins_pricing.exceptions import DataValidationError, PreprocessingError
from ins_pricing.utils.validation import validate_column_types


def _coerce_numeric(series: pd.Series, *, fill_value: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)


def _coerce_categorical(
    series: pd.Series,
    *,
    categories: Optional[Sequence[str]] = None,
    fill_value: str = "<NA>",
) -> pd.Series:
    out = series.astype("object").fillna(fill_value)
    if categories:
        out = pd.Categorical(out, categories=list(categories))
    return out


def _one_hot_encode(
    df: pd.DataFrame,
    *,
    columns: Sequence[str],
    drop_first: bool,
    dtype: str = "int8",
) -> pd.DataFrame:
    return pd.get_dummies(df, columns=list(columns), drop_first=drop_first, dtype=dtype)


def _apply_numeric_scalers(
    df: pd.DataFrame,
    *,
    columns: Sequence[str],
    numeric_scalers: Dict[str, Any],
) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        stats = numeric_scalers.get(col) or {}
        mean = float(stats.get("mean", 0.0))
        scale = float(stats.get("scale", 1.0))
        if scale == 0.0:
            scale = 1.0
        out[col] = (_coerce_numeric(out[col], fill_value=mean) - mean) / scale
    return out


def _align_columns(df: pd.DataFrame, *, columns: Sequence[str], fill_value: float = 0) -> pd.DataFrame:
    if not columns:
        return df
    return df.reindex(columns=list(columns), fill_value=fill_value)


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
            work[col] = _coerce_numeric(work[col], fill_value=0.0)
        else:
            cats = cat_categories.get(col)
            category_list = cats if isinstance(cats, list) and cats else None
            work[col] = _coerce_categorical(work[col], categories=category_list, fill_value="<NA>")

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
    oht = _one_hot_encode(work, columns=cate_list, drop_first=drop_first, dtype="int8")
    oht = _apply_numeric_scalers(oht, columns=num_features, numeric_scalers=numeric_scalers)
    oht = _align_columns(oht, columns=var_nmes, fill_value=0)
    return oht


def create_age_bands(
    df: pd.DataFrame,
    age_col: str,
    *,
    bins: Sequence[float],
    labels: Optional[Sequence[str]] = None,
    output_col: str = "age_band",
) -> pd.DataFrame:
    out = df.copy()
    out[output_col] = pd.cut(out[age_col], bins=bins, labels=labels, include_lowest=True)
    return out


def encode_categorical(
    df: pd.DataFrame,
    *,
    columns: Sequence[str],
    method: str = "onehot",
) -> pd.DataFrame:
    out = df.copy()
    method_name = str(method).strip().lower()
    if method_name == "onehot":
        return pd.get_dummies(out, columns=list(columns), dtype="int8")
    if method_name == "label":
        for col in columns:
            out[col] = pd.factorize(out[col])[0]
        return out
    raise PreprocessingError("method must be one of: onehot, label.")


def scale_features(
    df: pd.DataFrame,
    *,
    columns: Sequence[str],
    method: str = "standard",
) -> pd.DataFrame:
    out = df.copy()
    method_name = str(method).strip().lower()
    for col in columns:
        try:
            values = pd.to_numeric(out[col], errors="raise").astype(float)
        except Exception as exc:
            raise PreprocessingError(f"Column '{col}' is not numeric for scaling.") from exc

        if method_name == "standard":
            std = float(values.std(ddof=1))
            if std == 0:
                std = 1.0
            out[col] = (values - float(values.mean())) / std
        elif method_name == "minmax":
            min_val = float(values.min())
            max_val = float(values.max())
            scale = max_val - min_val
            if scale == 0:
                scale = 1.0
            out[col] = (values - min_val) / scale
        else:
            raise PreprocessingError("method must be one of: standard, minmax.")
    return out


def create_interactions(
    df: pd.DataFrame,
    *,
    feature_pairs: Sequence[Tuple[str, str]],
) -> pd.DataFrame:
    out = df.copy()
    for left, right in feature_pairs:
        out[f"{left}_x_{right}"] = pd.to_numeric(out[left], errors="coerce").fillna(0.0) * pd.to_numeric(
            out[right], errors="coerce"
        ).fillna(0.0)
    return out


def create_polynomial_features(
    df: pd.DataFrame,
    *,
    columns: Sequence[str],
    degree: int = 2,
) -> pd.DataFrame:
    if degree < 2:
        return df.copy()
    out = df.copy()
    for col in columns:
        base = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        for power in range(2, int(degree) + 1):
            suffix = "squared" if power == 2 else f"pow_{power}"
            out[f"{col}_{suffix}"] = base ** power
    return out


def handle_missing(
    df: pd.DataFrame,
    *,
    strategy: str = "mean",
    columns: Optional[Sequence[str]] = None,
    fill_value: Optional[float] = None,
) -> pd.DataFrame:
    out = df.copy()
    cols = list(columns) if columns is not None else list(out.columns)
    strategy_name = str(strategy).strip().lower()
    for col in cols:
        if col not in out.columns:
            continue
        if strategy_name == "mean":
            out[col] = out[col].fillna(pd.to_numeric(out[col], errors="coerce").mean())
        elif strategy_name == "median":
            out[col] = out[col].fillna(pd.to_numeric(out[col], errors="coerce").median())
        elif strategy_name == "mode":
            mode = out[col].mode(dropna=True)
            out[col] = out[col].fillna(mode.iloc[0] if not mode.empty else 0)
        elif strategy_name == "constant":
            out[col] = out[col].fillna(fill_value)
        else:
            raise PreprocessingError("strategy must be one of: mean, median, mode, constant.")
    return out


def remove_outliers(
    df: pd.DataFrame,
    *,
    column: str,
    method: str = "iqr",
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    values = pd.to_numeric(df[column], errors="coerce")
    if method == "iqr":
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (values >= lower) & (values <= upper)
    elif method == "zscore":
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if std == 0:
            mask = pd.Series(True, index=df.index)
        else:
            z = (values - mean) / std
            mask = z.abs() <= float(z_threshold)
    else:
        raise PreprocessingError("method must be one of: iqr, zscore.")
    return df.loc[mask].reset_index(drop=True)


def deduplicate(df: pd.DataFrame, *, subset: Optional[Sequence[str]] = None) -> pd.DataFrame:
    return df.drop_duplicates(subset=None if subset is None else list(subset)).reset_index(drop=True)


def fix_data_types(df: pd.DataFrame, *, type_spec: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for col, dtype in type_spec.items():
        if col in out.columns:
            out[col] = out[col].astype(dtype)
    return out


def select_features_by_importance(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_features: int,
) -> pd.DataFrame:
    target = pd.to_numeric(y, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    scores: Dict[Any, float] = {}
    for col in X.columns:
        values = pd.to_numeric(X[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if values.size == 0 or np.std(values) == 0:
            scores[col] = 0.0
            continue
        corr = np.corrcoef(values, target)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        scores[col] = float(abs(corr))
    ranked = sorted(scores, key=scores.get, reverse=True)[: int(max(1, n_features))]
    return X[ranked].copy()


def remove_low_variance(df: pd.DataFrame, *, threshold: float = 0.0) -> pd.DataFrame:
    variances = df.var(numeric_only=True)
    keep_numeric = variances[variances > float(threshold)].index.tolist()
    non_numeric = [c for c in df.columns if c not in variances.index]
    keep_cols = keep_numeric + non_numeric
    return df[keep_cols].copy()


def remove_correlated(df: pd.DataFrame, *, threshold: float = 0.95) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] <= 1:
        return df.copy()
    corr = numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > float(threshold))]
    return df.drop(columns=drop_cols)


def validate_input_schema(df: pd.DataFrame, expected_schema: Dict[str, str]) -> None:
    missing = [col for col in expected_schema.keys() if col not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    validate_column_types(df, expected_schema, coerce=False, df_name="input_data")


class Preprocessor:
    """Lightweight, serializable production preprocessor."""

    def __init__(self) -> None:
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.numeric_means: Dict[str, float] = {}
        self.numeric_stds: Dict[str, float] = {}
        self.categorical_fill: Dict[str, str] = {}
        self.categorical_levels: Dict[str, List[str]] = {}
        self.feature_names_: List[str] = []
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = [c for c in df.columns if c not in self.numeric_cols]

        for col in self.numeric_cols:
            values = pd.to_numeric(df[col], errors="coerce")
            mean = float(values.mean()) if values.notna().any() else 0.0
            std = float(values.std(ddof=1)) if values.notna().any() else 1.0
            self.numeric_means[col] = mean
            self.numeric_stds[col] = std if std != 0 else 1.0

        for col in self.categorical_cols:
            series = df[col].astype("object")
            mode = series.mode(dropna=True)
            fill = str(mode.iloc[0]) if not mode.empty else "__MISSING__"
            filled = series.fillna(fill).astype(str)
            self.categorical_fill[col] = fill
            self.categorical_levels[col] = sorted(filled.unique().tolist())

        transformed = self._transform_core(df)
        self.feature_names_ = transformed.columns.tolist()
        self._fitted = True
        return self

    def _transform_core(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.numeric_cols:
            if col not in out.columns:
                out[col] = self.numeric_means.get(col, 0.0)
            values = _coerce_numeric(out[col], fill_value=self.numeric_means.get(col, 0.0))
            out[col] = (values - self.numeric_means.get(col, 0.0)) / self.numeric_stds.get(col, 1.0)

        for col in self.categorical_cols:
            if col not in out.columns:
                out[col] = self.categorical_fill.get(col, "__MISSING__")
            fill = self.categorical_fill.get(col, "__MISSING__")
            levels = self.categorical_levels.get(col, [fill])
            source = out[col].astype("object").fillna(fill).astype(str)
            out[col] = _coerce_categorical(source, categories=levels, fill_value=fill)

        transformed = _one_hot_encode(out, columns=self.categorical_cols, drop_first=False, dtype="int8")
        return transformed

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise PreprocessingError("Preprocessor must be fit before transform.")
        transformed = self._transform_core(df)
        return transformed.reindex(columns=self.feature_names_, fill_value=0)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, path: str | Path) -> None:
        state_path = Path(path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with state_path.open("wb") as fh:
            pickle.dump(self.__dict__, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "Preprocessor":
        state_path = Path(path)
        obj = cls()
        if state_path.exists():
            with state_path.open("rb") as fh:
                payload = pickle.load(fh)
            if isinstance(payload, dict):
                obj.__dict__.update(payload)
        return obj


class PreprocessingPipeline:
    """Simple functional preprocessing pipeline."""

    _STEP_MAP = {
        "handle_missing": handle_missing,
        "encode_categorical": encode_categorical,
        "scale_features": scale_features,
        "create_age_bands": create_age_bands,
        "create_interactions": create_interactions,
        "create_polynomial_features": create_polynomial_features,
    }

    def __init__(self, steps: Sequence[Tuple[str, Dict[str, Any]]]):
        self.steps = list(steps)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for step_name, params in self.steps:
            fn = self._STEP_MAP.get(step_name)
            if fn is None:
                raise PreprocessingError(f"Unknown preprocessing step: {step_name}")
            try:
                out = fn(out, **dict(params))
            except PreprocessingError:
                raise
            except Exception as exc:
                raise PreprocessingError(f"Step '{step_name}' failed: {exc}") from exc
        return out
