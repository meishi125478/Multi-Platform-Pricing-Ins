"""Utilities for organizing generated frontend plot images."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List


def _humanize_token(token: str) -> str:
    parts = [part for part in str(token).replace("-", "_").split("_") if part]
    if not parts:
        return "Image"
    acronyms = {"xgb", "ft", "resn", "glm", "gnn"}
    return " ".join(part.upper() if part.lower() in acronyms else part.capitalize() for part in parts)


def _detect_split_label(stem: str) -> str:
    lowered = stem.lower()
    if lowered.endswith("_train") or "_train_" in lowered:
        return "Train"
    if lowered.endswith("_test") or "_test_" in lowered:
        return "Test"
    if lowered.endswith("_valid") or "_valid_" in lowered or "_validation_" in lowered:
        return "Validation"
    if lowered.endswith("_all") or "_all_" in lowered:
        return "All Data"
    if "explicit_split" in lowered:
        return "Explicit Split"
    return ""


def _classify_image(path_obj: Path) -> str:
    stem = path_obj.stem.lower()
    parts = {part.lower() for part in path_obj.parts}
    if stem.startswith("double_lift_compare_"):
        return "FT-Embed Compare"
    if stem.startswith("double_lift_multi_"):
        return "Multi-Model Double Lift"
    if stem.startswith("double_lift_file_"):
        return "Double Lift"
    if stem.startswith("02_") or "_dlift_" in stem or "double_lift" in stem:
        return "Double Lift"
    if stem.startswith("01_") or stem.endswith("_lift"):
        return "Lift Curve"
    if stem.startswith("00_") or "oneway" in parts:
        if "pre" in parts and "oneway" in parts:
            return "Pre-Oneway"
        return "Oneway"
    return "Plot"


def _build_title(path_obj: Path, category: str) -> str:
    stem = path_obj.stem
    split_label = _detect_split_label(stem)
    model_hint = path_obj.parent.name.strip()

    if category == "Pre-Oneway":
        feature = stem
        for suffix in ("_train", "_test", "_valid", "_all"):
            if feature.lower().endswith(suffix):
                feature = feature[: -len(suffix)]
                break
        title = _humanize_token(feature)
    elif category == "Oneway" and stem.startswith("00_"):
        payload = stem[3:]
        left = payload.split("_oneway_", 1)[0]
        if model_hint and left.startswith(f"{model_hint}_"):
            left = left[len(model_hint) + 1 :]
        title = _humanize_token(left)
    elif category == "Lift Curve" and stem.startswith("01_"):
        payload = stem[3:]
        left = payload.rsplit("_lift", 1)[0]
        if model_hint and left.startswith(f"{model_hint}_"):
            left = left[len(model_hint) + 1 :]
        title = _humanize_token(left)
    elif " vs " in stem.lower():
        title = _humanize_token(stem)
    elif "_vs_" in stem:
        left, right = stem.split("_vs_", 1)
        title = f"{_humanize_token(left.split('_')[-1])} vs {_humanize_token(right)}"
    else:
        title = _humanize_token(stem)

    if split_label:
        return f"{title} [{split_label}]"
    return title


def build_generated_image_choices(paths: Iterable[Any]) -> List[Dict[str, str]]:
    """Convert png paths into dropdown-friendly metadata."""
    items: List[Dict[str, str]] = []
    seen: set[str] = set()
    for raw in paths:
        path_obj = Path(str(raw)).expanduser().resolve()
        if not path_obj.exists() or not path_obj.is_file():
            continue
        resolved = str(path_obj)
        if resolved in seen:
            continue
        seen.add(resolved)
        category = _classify_image(path_obj)
        title = _build_title(path_obj, category)
        items.append(
            {
                "path": resolved,
                "category": category,
                "title": title,
                "filename": path_obj.name,
                "option_label": f"{category} | {title}",
            }
        )
    return items
