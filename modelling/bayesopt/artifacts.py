from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ins_pricing.utils.io import IOUtils

MODEL_KEY_TO_TRAINER_LABEL = {
    "xgb": "xgboost",
    "resn": "resnet",
    "ft": "fttransformer",
    "glm": "glm",
    "gnn": "gnn",
}


def normalize_trainer_label(label: str) -> str:
    return str(label).strip().lower()


def trainer_label_from_model_key(model_key: str) -> str:
    key = str(model_key).strip().lower()
    return MODEL_KEY_TO_TRAINER_LABEL.get(key, key)


def best_params_filename(model_name: str, trainer_label: str) -> str:
    return f"{model_name}_bestparams_{normalize_trainer_label(trainer_label)}.csv"


def best_params_csv_path(
    result_dir: str | Path,
    model_name: str,
    trainer_label: str,
) -> Path:
    return Path(result_dir) / best_params_filename(model_name, trainer_label)


def extract_best_params_from_snapshot(payload: Any) -> Optional[Dict[str, Any]]:
    """Extract best params from a version snapshot payload.

    Supports both direct `best_params` snapshots and `best_params_payload.values`
    layout.
    """
    if not isinstance(payload, dict):
        return None

    direct = payload.get("best_params")
    if isinstance(direct, dict) and direct:
        return dict(direct)

    payload_block = payload.get("best_params_payload")
    if not isinstance(payload_block, dict):
        return None

    for key in ("values", "best_params", "params", "best_params_values"):
        nested = payload_block.get(key)
        if isinstance(nested, dict) and nested:
            return dict(nested)
    return None


def load_best_params_csv(
    result_dir: str | Path,
    model_name: str,
    trainer_label: str,
) -> Optional[Dict[str, Any]]:
    path = best_params_csv_path(result_dir, model_name, trainer_label)
    if not path.exists():
        return None
    try:
        params = IOUtils.load_params_file(str(path))
    except ValueError:
        return None
    return dict(params) if isinstance(params, dict) else None


def load_best_params(
    output_dir: str | Path,
    model_name: str,
    model_key: str,
) -> Optional[Dict[str, Any]]:
    output_path = Path(output_dir)
    trainer_label = trainer_label_from_model_key(model_key)
    labels = (trainer_label,)
    for label in labels:
        params = load_best_params_csv(
            output_path / "Results",
            model_name,
            label,
        )
        if params is not None:
            return params

    return None


__all__ = [
    "MODEL_KEY_TO_TRAINER_LABEL",
    "normalize_trainer_label",
    "trainer_label_from_model_key",
    "best_params_filename",
    "best_params_csv_path",
    "extract_best_params_from_snapshot",
    "load_best_params_csv",
    "load_best_params",
]
