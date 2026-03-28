from __future__ import annotations

import json
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
    versions_dir = output_path / "Results" / "versions"
    if versions_dir.exists():
        candidates = sorted(versions_dir.glob(f"*_{model_key}_best.json"))
        if candidates:
            payload = json.loads(candidates[-1].read_text(encoding="utf-8"))
            params = payload.get("best_params") if isinstance(payload, dict) else None
            if isinstance(params, dict) and params:
                return params

    return load_best_params_csv(
        output_path / "Results",
        model_name,
        trainer_label_from_model_key(model_key),
    )


__all__ = [
    "MODEL_KEY_TO_TRAINER_LABEL",
    "normalize_trainer_label",
    "trainer_label_from_model_key",
    "best_params_filename",
    "best_params_csv_path",
    "load_best_params_csv",
    "load_best_params",
]

