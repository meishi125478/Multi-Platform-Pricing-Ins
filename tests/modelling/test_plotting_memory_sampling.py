from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

import ins_pricing.modelling.bayesopt.model_plotting_mixin as plotting_mixin_mod
from ins_pricing.modelling.bayesopt.model_plotting_mixin import BayesOptPlottingMixin


class _DummyOutputManager:
    def __init__(self, root: Path) -> None:
        self.root = root

    def plot_path(self, filename: str) -> str:
        target = self.root / "plot" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        return str(target)


class _DummyModel(BayesOptPlottingMixin):
    pass


def _build_dummy_model(tmp_path: Path, rows: int = 500) -> _DummyModel:
    rng = np.random.default_rng(13)
    train = pd.DataFrame(
        {
            "x1": rng.normal(size=rows),
            "x2": rng.normal(size=rows),
            "y": rng.normal(size=rows),
            "y_bin": rng.integers(0, 2, size=rows),
            "w": rng.uniform(0.5, 1.5, size=rows),
        }
    )
    train["w_act"] = train["y"] * train["w"]
    train["w_binary_act"] = train["y_bin"] * train["w"]
    train["pred_xgb"] = rng.normal(size=rows)
    train["pred_resn"] = rng.normal(size=rows)
    train["w_pred_xgb"] = train["pred_xgb"] * train["w"]
    train["w_pred_resn"] = train["pred_resn"] * train["w"]

    test = pd.DataFrame(
        {
            "x1": rng.normal(size=rows),
            "x2": rng.normal(size=rows),
            "y": rng.normal(size=rows),
            "y_bin": rng.integers(0, 2, size=rows),
            "w": rng.uniform(0.5, 1.5, size=rows),
        }
    )
    test["w_act"] = test["y"] * test["w"]
    test["w_binary_act"] = test["y_bin"] * test["w"]
    test["pred_xgb"] = rng.normal(size=rows)
    test["pred_resn"] = rng.normal(size=rows)
    test["w_pred_xgb"] = test["pred_xgb"] * test["w"]
    test["w_pred_resn"] = test["pred_resn"] * test["w"]

    model = _DummyModel()
    model.model_nme = "demo_sampling"
    model.task_type = "regression"
    model.config = SimpleNamespace(
        ft_role="model",
        plot_path_style="nested",
        classification_plot_prediction="score",
        plot_max_rows=120,
        plot_oneway_max_rows=80,
        plot_curve_max_rows=60,
        plot_sampling_seed=42,
    )
    model.factor_nmes = ["x1", "x2"]
    model.cate_list = []
    model.weight_nme = "w"
    model.binary_resp_nme = "y_bin"
    model.train_data = train
    model.test_data = test
    model.output_manager = _DummyOutputManager(tmp_path)
    return model


def test_plot_oneway_applies_row_sampling(tmp_path, monkeypatch):
    if plotting_mixin_mod.plot_diagnostics is None:
        pytest.skip("plot_diagnostics unavailable")
    model = _build_dummy_model(tmp_path, rows=500)
    sampled_lengths: List[int] = []

    def _fake_plot_oneway(df, **kwargs):
        sampled_lengths.append(int(len(df)))
        return None

    monkeypatch.setattr(plotting_mixin_mod.plot_diagnostics, "plot_oneway", _fake_plot_oneway)
    model.plot_oneway(n_bins=8, pred_col="pred_xgb")
    assert sampled_lengths
    assert all(length <= 80 for length in sampled_lengths)


def test_plot_lift_and_dlift_apply_row_sampling(tmp_path, monkeypatch):
    if plotting_mixin_mod.plot_curves is None:
        pytest.skip("plot_curves unavailable")
    model = _build_dummy_model(tmp_path, rows=500)
    lift_lengths: List[Tuple[int, int, int]] = []
    dlift_lengths: List[Tuple[int, int, int, int]] = []

    def _fake_plot_lift_curve(pred, actual, weight, **kwargs):
        lift_lengths.append((len(pred), len(actual), len(weight)))
        ax = kwargs.get("ax")
        if ax is not None:
            ax.plot([0, 1], [0, 1], color="black")
            return ax.figure
        return None

    def _fake_plot_double_lift_curve(pred1, pred2, actual, weight, **kwargs):
        dlift_lengths.append((len(pred1), len(pred2), len(actual), len(weight)))
        ax = kwargs.get("ax")
        if ax is not None:
            ax.plot([0, 1], [0, 1], color="black")
            return ax.figure
        return None

    monkeypatch.setattr(plotting_mixin_mod.plot_curves, "plot_lift_curve", _fake_plot_lift_curve)
    monkeypatch.setattr(
        plotting_mixin_mod.plot_curves,
        "plot_double_lift_curve",
        _fake_plot_double_lift_curve,
    )

    model.plot_lift("Xgboost", "pred_xgb", n_bins=8)
    model.plot_dlift(["xgb", "resn"], n_bins=8)

    assert lift_lengths
    assert all(max(lengths) <= 60 for lengths in lift_lengths)
    assert dlift_lengths
    assert all(max(lengths) <= 60 for lengths in dlift_lengths)


def test_plot_conversion_lift_applies_row_sampling(tmp_path, monkeypatch):
    if plotting_mixin_mod.plot_curves is None:
        pytest.skip("plot_curves unavailable")
    model = _build_dummy_model(tmp_path, rows=500)
    conversion_lengths: List[Tuple[int, int, int]] = []

    def _fake_plot_conversion_lift(pred, actual_binary, weight, **kwargs):
        conversion_lengths.append((len(pred), len(actual_binary), len(weight)))
        ax = kwargs.get("ax")
        if ax is not None:
            ax.plot([0, 1], [0.1, 0.9], color="black")
            return ax.figure
        return None

    monkeypatch.setattr(
        plotting_mixin_mod.plot_curves,
        "plot_conversion_lift",
        _fake_plot_conversion_lift,
    )
    monkeypatch.setattr(plotting_mixin_mod.plt, "show", lambda *args, **kwargs: None)

    model.plot_conversion_lift("pred_xgb", n_bins=10)
    assert conversion_lengths
    assert all(max(lengths) <= 60 for lengths in conversion_lengths)
