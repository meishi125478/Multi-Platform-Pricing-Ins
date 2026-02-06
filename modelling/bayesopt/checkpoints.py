from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def _to_serializable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return value
    return value


def serialize_ft_model_config(model: Any) -> Dict[str, Any]:
    return {
        "model_nme": getattr(model, "model_nme", ""),
        "num_cols": list(getattr(model, "num_cols", [])),
        "cat_cols": list(getattr(model, "cat_cols", [])),
        "d_model": getattr(model, "d_model", 64),
        "n_heads": getattr(model, "n_heads", 8),
        "n_layers": getattr(model, "n_layers", 4),
        "dropout": getattr(model, "dropout", 0.1),
        "task_type": getattr(model, "task_type", "regression"),
        "distribution": getattr(model, "distribution", None),
        "loss_name": getattr(model, "loss_name", None),
        "tw_power": getattr(model, "tw_power", 1.5),
        "num_geo": getattr(model, "num_geo", 0),
        "num_numeric_tokens": getattr(model, "num_numeric_tokens", None),
        "cat_cardinalities": getattr(model, "cat_cardinalities", None),
        "cat_categories": {
            k: list(v) for k, v in getattr(model, "cat_categories", {}).items()
        },
        "_num_mean": _to_serializable(getattr(model, "_num_mean", None)),
        "_num_std": _to_serializable(getattr(model, "_num_std", None)),
    }


def rebuild_ft_model_from_checkpoint(
    *,
    state_dict: Dict[str, Any],
    model_config: Dict[str, Any],
):
    if state_dict is None:
        raise ValueError("FT checkpoint is missing state_dict.")

    from ins_pricing.modelling.bayesopt.models import FTTransformerSklearn
    from ins_pricing.modelling.bayesopt.models.model_ft_components import FTTransformerCore

    model = FTTransformerSklearn(
        model_nme=model_config.get("model_nme", ""),
        num_cols=model_config.get("num_cols", []),
        cat_cols=model_config.get("cat_cols", []),
        d_model=model_config.get("d_model", 64),
        n_heads=model_config.get("n_heads", 8),
        n_layers=model_config.get("n_layers", 4),
        dropout=model_config.get("dropout", 0.1),
        task_type=model_config.get("task_type", "regression"),
        distribution=model_config.get("distribution", None),
        loss_name=model_config.get("loss_name", None),
        tweedie_power=model_config.get("tw_power", 1.5),
        num_numeric_tokens=model_config.get("num_numeric_tokens"),
        use_data_parallel=False,
        use_ddp=False,
        use_gpu=False,
    )

    model.num_geo = model_config.get("num_geo", 0)
    model.cat_cardinalities = model_config.get("cat_cardinalities")
    model.cat_categories = {
        k: pd.Index(v) for k, v in model_config.get("cat_categories", {}).items()
    }
    if model_config.get("_num_mean") is not None:
        model._num_mean = np.array(model_config["_num_mean"], dtype=np.float32)
    if model_config.get("_num_std") is not None:
        model._num_std = np.array(model_config["_num_std"], dtype=np.float32)

    if model.cat_cardinalities is None:
        raise ValueError(
            "FT checkpoint is missing cat_cardinalities in model_config."
        )

    core = FTTransformerCore(
        num_numeric=len(model.num_cols),
        cat_cardinalities=model.cat_cardinalities,
        d_model=model.d_model,
        n_heads=model.n_heads,
        n_layers=model.n_layers,
        dropout=model.dropout,
        task_type=model.task_type,
        num_geo=model.num_geo,
        num_numeric_tokens=model.num_numeric_tokens,
    )
    model.ft = core
    model.ft.load_state_dict(state_dict)

    return model


__all__ = [
    "serialize_ft_model_config",
    "rebuild_ft_model_from_checkpoint",
]
