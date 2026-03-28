from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import ins_pricing.frontend.workflows_predict as workflows_predict


class _FakeFTModel:
    def predict(self, frame: pd.DataFrame, return_embedding: bool = False):
        assert return_embedding is True
        return np.full((len(frame), 1), 0.5, dtype=float)


class _FakeXGBModel:
    def __init__(self):
        self.used_predict = False
        self.used_predict_proba = False

    def predict(self, frame: pd.DataFrame):
        self.used_predict = True
        return np.zeros(len(frame), dtype=float)

    def predict_proba(self, frame: pd.DataFrame):
        self.used_predict_proba = True
        proba = np.full(len(frame), 0.8, dtype=float)
        return np.column_stack([1.0 - proba, proba])


def test_run_predict_uses_predict_proba_for_binary_xgb(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ft_cfg_path = tmp_path / "ft_cfg.json"
    xgb_cfg_path = tmp_path / "xgb_cfg.json"
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    ft_cfg_path.write_text(
        json.dumps(
            {
                "model_list": ["od"],
                "model_categories": ["bc"],
                "output_dir": "./ResultsFT",
                "ft_feature_prefix": "ft_emb",
            }
        ),
        encoding="utf-8",
    )
    xgb_cfg_path.write_text(
        json.dumps(
            {
                "task_type": "binary",
                "output_dir": "./ResultsXGB",
                "feature_list": ["pred_ft_emb_0"],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"f1": [1, 2, 3]}).to_csv(input_path, index=False)

    fake_xgb = _FakeXGBModel()

    def _fake_resolve(**kwargs):
        return tmp_path / f"{kwargs['model_key']}.artifact"

    monkeypatch.setattr(
        workflows_predict,
        "_resolve_model_file_for_prediction",
        _fake_resolve,
    )
    monkeypatch.setattr(
        workflows_predict,
        "_load_ft_embedding_model",
        lambda _path: _FakeFTModel(),
    )
    monkeypatch.setattr(
        workflows_predict,
        "_load_pickled_model_payload",
        lambda _path: {
            "model": fake_xgb,
            "preprocess_artifacts": {"factor_nmes": ["pred_ft_emb_0"]},
        },
    )

    workflows_predict.run_predict_ft_embed(
        ft_cfg_path=str(ft_cfg_path),
        xgb_cfg_path=str(xgb_cfg_path),
        resn_cfg_path=None,
        input_path=str(input_path),
        output_path=str(output_path),
        model_name="od_bc",
        model_keys="xgb",
    )

    pred = pd.read_csv(output_path)
    assert fake_xgb.used_predict_proba is True
    assert fake_xgb.used_predict is False
    assert pred["pred_xgb"].tolist() == [0.8, 0.8, 0.8]
