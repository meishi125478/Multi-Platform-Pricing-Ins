import types

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from ins_pricing.modelling.bayesopt.checkpoints import (
    rebuild_ft_model_from_checkpoint,
    serialize_ft_model_config,
)
from ins_pricing.modelling.bayesopt.models.model_ft_components import FTTransformerCore


def test_serialize_ft_model_config_converts_arrays_and_categories():
    model = types.SimpleNamespace(
        model_nme="demo",
        num_cols=["num_1", "num_2"],
        cat_cols=["cat_1"],
        d_model=32,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        task_type="regression",
        loss_name="poisson",
        tw_power=1.0,
        num_geo=0,
        num_numeric_tokens=2,
        cat_cardinalities=[3],
        cat_categories={"cat_1": pd.Index(["a", "b"])},
        _num_mean=np.array([1.0, 2.0], dtype=np.float32),
        _num_std=np.array([0.5, 0.25], dtype=np.float32),
    )

    payload = serialize_ft_model_config(model)
    assert payload["_num_mean"] == [1.0, 2.0]
    assert payload["_num_std"] == [0.5, 0.25]
    assert payload["cat_categories"]["cat_1"] == ["a", "b"]


def test_rebuild_ft_model_from_checkpoint_restores_state_dict():
    core = FTTransformerCore(
        num_numeric=1,
        cat_cardinalities=[3],
        d_model=16,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        task_type="regression",
        num_geo=0,
        num_numeric_tokens=1,
    )
    state_dict = core.state_dict()
    model_config = {
        "model_nme": "demo",
        "num_cols": ["num_1"],
        "cat_cols": ["cat_1"],
        "d_model": 16,
        "n_heads": 4,
        "n_layers": 2,
        "dropout": 0.1,
        "task_type": "regression",
        "loss_name": "poisson",
        "tw_power": 1.0,
        "num_geo": 0,
        "num_numeric_tokens": 1,
        "cat_cardinalities": [3],
        "cat_categories": {"cat_1": ["a", "b"]},
        "_num_mean": [0.0],
        "_num_std": [1.0],
    }

    rebuilt = rebuild_ft_model_from_checkpoint(
        state_dict=state_dict,
        model_config=model_config,
    )
    assert rebuilt.ft is not None
    assert rebuilt.cat_cardinalities == [3]
    assert set(rebuilt.ft.state_dict().keys()) == set(state_dict.keys())

