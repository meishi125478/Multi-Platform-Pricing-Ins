"""Shared helpers for frontend workflow execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Union

import pandas as pd
from ins_pricing.frontend.logging_utils import get_frontend_logger, log_print

_logger = get_frontend_logger("ins_pricing.frontend.workflows_common")


def _log(*args, **kwargs) -> None:
    log_print(_logger, *args, **kwargs)

_MODEL_FILE_SPEC = {
    "xgb": ("Xgboost", "pkl"),
    "glm": ("GLM", "pkl"),
    "resn": ("ResNet", "pth"),
    "ft": ("FTTransformer", "pth"),
    "gnn": ("GNN", "pth"),
}


def _parse_csv_list(value: str) -> List[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def _dedupe_list(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in values or []:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _drop_duplicate_columns(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.columns.duplicated().any():
        dupes = [str(x) for x in df.columns[df.columns.duplicated()]]
        _log(f"[Warn] {label}: dropping duplicate columns: {sorted(set(dupes))}")
        return df.loc[:, ~df.columns.duplicated()].copy()
    return df


def _resolve_output_dir(cfg_obj: dict, cfg_file_path: Path) -> str:
    output_dir = cfg_obj.get("output_dir", "./Results")
    return str((cfg_file_path.parent / output_dir).resolve())


def _resolve_plot_style(cfg_obj: dict) -> str:
    return str(cfg_obj.get("plot_path_style", "nested") or "nested").strip().lower()


def _resolve_plot_path(output_root: str, plot_style: str, subdir: str, filename: str) -> str:
    plot_root = Path(output_root) / "plot"
    if plot_style in {"flat", "root"}:
        return str((plot_root / filename).resolve())
    if subdir:
        return str((plot_root / subdir / filename).resolve())
    return str((plot_root / filename).resolve())


def _resolve_double_lift_dir(model_name: Optional[str] = None) -> Path:
    """Unified output directory for double-lift charts.

    Path format:
      work_dir/Results/plot/{model_name}/double_lift
    """
    model_tag = _safe_tag(str(model_name or "").strip()) or "unknown_model"
    return (Path.cwd() / "Results" / "plot" / model_tag / "double_lift").resolve()


def _safe_tag(value: str) -> str:
    return (
        value.strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def _resolve_data_path(cfg: dict, cfg_path: Path, model_name: str) -> Path:
    data_dir = cfg.get("data_dir", ".")
    data_format = cfg.get("data_format", "csv")
    data_path_template = cfg.get("data_path_template", "{model_name}.{ext}")
    filename = data_path_template.format(model_name=model_name, ext=data_format)
    return (cfg_path.parent / data_dir / filename).resolve()


def _resolve_model_output_dir(path_value: Optional[str], label: str) -> Optional[Path]:
    raw_val = str(path_value or "").strip()
    if not raw_val:
        return None
    path_obj = Path(raw_val).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"{label} not found: {path_obj}")
    if path_obj.is_file():
        if path_obj.parent.name.lower() == "model":
            return path_obj.parent.parent.resolve()
        return path_obj.parent.resolve()
    return path_obj.resolve()


def _model_artifact_filename(model_name: str, model_key: str) -> str:
    spec = _MODEL_FILE_SPEC.get(str(model_key or "").strip().lower())
    if spec is None:
        raise ValueError(f"Unsupported model key for artifact lookup: {model_key!r}")
    prefix, ext = spec
    return f"01_{model_name}_{prefix}.{ext}"


def _build_search_roots(*roots: Optional[Union[str, Path]]) -> List[Path]:
    out: List[Path] = []
    seen: set[str] = set()
    for root in roots:
        if root is None:
            continue
        path_obj = Path(str(root)).expanduser().resolve()
        key = str(path_obj)
        if key in seen:
            continue
        seen.add(key)
        out.append(path_obj)
    return out


def _discover_model_file(
    *,
    model_name: str,
    model_key: str,
    search_roots: Sequence[Path],
    output_roots: Optional[Sequence[Path]] = None,
) -> Optional[Path]:
    filename = _model_artifact_filename(model_name, model_key)
    output_candidates: List[Path] = []
    search_candidates: List[Path] = []
    recursive_candidates: List[Path] = []
    seen: set[str] = set()

    def _add_candidate(path_obj: Path, *, from_output_root: bool = False) -> None:
        try:
            resolved = path_obj.resolve()
        except OSError:
            return
        key = str(resolved)
        if key in seen:
            return
        if not resolved.exists() or not resolved.is_file():
            return
        seen.add(key)
        if from_output_root:
            output_candidates.append(resolved)
        else:
            search_candidates.append(resolved)

    def _latest(paths: Sequence[Path]) -> Optional[Path]:
        if not paths:
            return None
        def _mtime_key(path_obj: Path) -> float:
            try:
                return float(path_obj.stat().st_mtime)
            except OSError:
                return 0.0
        ordered = sorted(paths, key=_mtime_key, reverse=True)
        return ordered[0]

    for output_root in output_roots or []:
        try:
            root_obj = Path(output_root).resolve()
        except OSError:
            continue
        _add_candidate(root_obj / "model" / filename, from_output_root=True)

    best_output = _latest(output_candidates)
    if best_output is not None:
        return best_output

    for root in search_roots:
        try:
            root_obj = Path(root).resolve()
        except OSError:
            continue
        if not root_obj.exists():
            continue
        if root_obj.is_file():
            if root_obj.name == filename:
                _add_candidate(root_obj, from_output_root=False)
            continue
        _add_candidate(root_obj / filename, from_output_root=False)
        _add_candidate(root_obj / "model" / filename, from_output_root=False)

    best_search = _latest(search_candidates)
    if best_search is not None:
        return best_search

    # Recursive lookup is expensive; only run when direct candidates miss.
    for root in search_roots:
        try:
            root_obj = Path(root).resolve()
        except OSError:
            continue
        if not root_obj.exists() or root_obj.is_file():
            continue
        try:
            for match in root_obj.glob(f"**/model/{filename}"):
                try:
                    resolved = match.resolve()
                except OSError:
                    continue
                key = str(resolved)
                if key in seen or not resolved.exists() or not resolved.is_file():
                    continue
                seen.add(key)
                recursive_candidates.append(resolved)
        except OSError:
            continue

    if not recursive_candidates:
        try:
            for root in search_roots:
                root_obj = Path(root).resolve()
                if not root_obj.exists() or root_obj.is_file():
                    continue
                for match in root_obj.glob(f"**/{filename}"):
                    try:
                        resolved = match.resolve()
                    except OSError:
                        continue
                    key = str(resolved)
                    if key in seen or not resolved.exists() or not resolved.is_file():
                        continue
                    seen.add(key)
                    recursive_candidates.append(resolved)
        except OSError:
            pass

    best_recursive = _latest(recursive_candidates)
    if best_recursive is not None:
        return best_recursive
    return None


def _load_ft_embedding_model(model_path: Path) -> Any:
    """Load FT checkpoint for embedding inference using secure defaults."""
    from ins_pricing.utils.model_loading import load_model_artifact_payload
    from ins_pricing.utils.model_rebuild import rebuild_ft_payload

    payload = load_model_artifact_payload(
        model_path,
        model_key="ft",
        map_location="cpu",
    )
    model, _best_params, kind = rebuild_ft_payload(payload=payload)
    if kind == "raw":
        raise ValueError(
            f"Unsupported FT checkpoint format for secure loading: {model_path}"
        )
    return model


def _load_pickled_model_payload(model_path: Path) -> Any:
    from ins_pricing.utils.model_loading import load_model_artifact_payload

    return load_model_artifact_payload(model_path, model_key="xgb")


def _infer_categorical_features(
    df: pd.DataFrame,
    feature_list: Sequence[str],
    *,
    max_unique: int = 50,
    max_ratio: float = 0.05,
) -> List[str]:
    categorical: List[str] = []
    n_rows = max(1, len(df))
    for feature in feature_list:
        if feature not in df.columns:
            continue
        nunique = int(df[feature].nunique(dropna=True))
        ratio = nunique / float(n_rows)
        if nunique <= max_unique or ratio <= max_ratio:
            categorical.append(feature)
    return categorical

