"""Path resolution and data loading utilities.

This module consolidates path handling logic from:
- cli/utils/cli_common.py
- production/predict.py
- Various model loaders

Example:
    >>> from ins_pricing.utils import resolve_path, load_dataset
    >>> path = resolve_path("./data/train.csv", base_dir)
    >>> df = load_dataset(path, data_format="auto")
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd


# =============================================================================
# Model Label Mappings
# =============================================================================

PLOT_MODEL_LABELS: Dict[str, Tuple[str, str]] = {
    "glm": ("GLM", "pred_glm"),
    "xgb": ("Xgboost", "pred_xgb"),
    "resn": ("ResNet", "pred_resn"),
    "ft": ("FTTransformer", "pred_ft"),
    "gnn": ("GNN", "pred_gnn"),
}

PYTORCH_TRAINERS = {"resn", "ft", "gnn"}


# =============================================================================
# List Utilities
# =============================================================================


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    """Remove duplicates while preserving order.

    Args:
        items: Iterable of strings

    Returns:
        List with duplicates removed, original order preserved
    """
    seen = set()
    unique: List[str] = []
    for item in items:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique


def build_model_names(prefixes: Sequence[str], suffixes: Sequence[str]) -> List[str]:
    """Build model names from prefixes and suffixes.

    Args:
        prefixes: Model type prefixes (e.g., ['bi', 'od'])
        suffixes: Model category suffixes (e.g., ['bc', 'nc'])

    Returns:
        List of combined names (e.g., ['bi_bc', 'od_bc', 'bi_nc', 'od_nc'])
    """
    names: List[str] = []
    for suffix in suffixes:
        names.extend(f"{prefix}_{suffix}" for prefix in prefixes)
    return names


def parse_model_pairs(raw_pairs: List) -> List[Tuple[str, str]]:
    """Parse model pairs from various formats.

    Args:
        raw_pairs: List of pairs in various formats

    Returns:
        List of (model1, model2) tuples
    """
    pairs: List[Tuple[str, str]] = []
    for pair in raw_pairs:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            pairs.append((str(pair[0]), str(pair[1])))
        elif isinstance(pair, str):
            parts = [p.strip() for p in pair.split(",") if p.strip()]
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    return pairs


# =============================================================================
# Path Resolution
# =============================================================================


def resolve_path(value: Optional[str], base_dir: Path) -> Optional[Path]:
    """Resolve a path relative to a base directory.

    Args:
        value: Path string (absolute or relative)
        base_dir: Base directory for relative paths

    Returns:
        Resolved absolute Path, or None if value is None/empty
    """
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def resolve_dir_path(
    value: Optional[Union[str, Path]],
    base_dir: Path,
    *,
    create: bool = False,
) -> Optional[Path]:
    """Resolve a directory path, optionally creating it.

    Args:
        value: Path string or Path object
        base_dir: Base directory for relative paths
        create: Whether to create the directory if it doesn't exist

    Returns:
        Resolved absolute Path, or None if value is None/empty
    """
    if value is None:
        return None
    if isinstance(value, Path):
        path = value if value.is_absolute() else (base_dir / value).resolve()
    else:
        value_str = str(value)
        if not value_str.strip():
            return None
        path = resolve_path(value_str, base_dir)
        if path is None:
            path = Path(value_str)
            if not path.is_absolute():
                path = (base_dir / path).resolve()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _infer_format_from_path(path: Path) -> str:
    """Infer data format from file extension.

    Args:
        path: File path

    Returns:
        Format string ('parquet', 'feather', or 'csv')
    """
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    if suffix in {".feather", ".ft"}:
        return "feather"
    return "csv"


def resolve_data_path(
    data_dir: Path,
    model_name: str,
    *,
    data_format: str = "csv",
    path_template: Optional[str] = None,
) -> Path:
    """Resolve the path to a data file.

    Args:
        data_dir: Directory containing data files
        model_name: Model name for path substitution
        data_format: Data format ('csv', 'parquet', 'feather', 'auto')
        path_template: Template string with {model_name} and {ext} placeholders

    Returns:
        Resolved path to data file
    """
    fmt = str(data_format or "csv").strip().lower()
    template = path_template or "{model_name}.{ext}"

    if fmt == "auto":
        candidates = [
            data_dir / template.format(model_name=model_name, ext="parquet"),
            data_dir / template.format(model_name=model_name, ext="feather"),
            data_dir / template.format(model_name=model_name, ext="csv"),
        ]
        for cand in candidates:
            if cand.exists():
                return cand.resolve()
        return candidates[-1].resolve()

    ext = "csv" if fmt in {"csv"} else fmt
    return (data_dir / template.format(model_name=model_name, ext=ext)).resolve()


# =============================================================================
# Data Loading
# =============================================================================


def load_dataset(
    path: Path,
    *,
    data_format: str = "auto",
    dtype_map: Optional[Dict[str, Any]] = None,
    low_memory: bool = False,
) -> pd.DataFrame:
    """Load a dataset from various formats.

    Args:
        path: Path to data file
        data_format: Format ('csv', 'parquet', 'feather', 'auto')
        dtype_map: Column type mapping
        low_memory: Whether to use low memory mode for CSV

    Returns:
        Loaded DataFrame
    """
    fmt = str(data_format or "auto").strip().lower()
    if fmt == "auto":
        fmt = _infer_format_from_path(path)

    if fmt == "parquet":
        df = pd.read_parquet(path)
    elif fmt == "feather":
        df = pd.read_feather(path)
    elif fmt == "csv":
        df = pd.read_csv(path, low_memory=low_memory, dtype=dtype_map or None)
    else:
        raise ValueError(f"Unsupported data_format: {data_format}")

    if dtype_map:
        for col, dtype in dtype_map.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
    return df


def coerce_dataset_types(raw: pd.DataFrame) -> pd.DataFrame:
    """Coerce dataset types to numeric or object.

    Args:
        raw: Input DataFrame

    Returns:
        DataFrame with coerced types
    """
    data = raw.copy()
    for col in data.columns:
        s = data[col]
        if pd.api.types.is_numeric_dtype(s):
            data[col] = pd.to_numeric(s, errors="coerce").fillna(0)
        else:
            data[col] = s.astype("object").fillna("<NA>")
    return data


# =============================================================================
# File Fingerprinting
# =============================================================================


def fingerprint_file(path: Path, *, max_bytes: int = 10_485_760) -> Dict[str, Any]:
    """Generate a fingerprint for a file (for cache validation).

    Args:
        path: Path to file
        max_bytes: Maximum bytes to hash (default 10MB)

    Returns:
        Dictionary with path, size, mtime, and partial SHA256 hash
    """
    path = Path(path)
    stat = path.stat()
    h = hashlib.sha256()
    remaining = int(max_bytes)

    with path.open("rb") as fh:
        while remaining > 0:
            chunk = fh.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)

    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime": int(stat.st_mtime),
        "sha256_prefix": h.hexdigest(),
        "max_bytes": int(max_bytes),
    }
