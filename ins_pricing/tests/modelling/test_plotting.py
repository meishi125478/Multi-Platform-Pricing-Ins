import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("xgboost")
pytest.importorskip("optuna")
pytest.importorskip("statsmodels")
pytest.importorskip("shap")


def test_plotting_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    mpl_cfg = tmp_path / ".mplconfig"
    cache_dir = tmp_path / ".cache"
    (cache_dir / "fontconfig").mkdir(parents=True, exist_ok=True)
    mpl_cfg.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MPLCONFIGDIR", str(mpl_cfg))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_dir))

    from ins_pricing.BayesOpt import BayesOptModel
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    rng = np.random.default_rng(0)
    train = pd.DataFrame(
        {
            "x1": rng.normal(size=30),
            "y": rng.normal(size=30),
            "w": rng.uniform(0.5, 1.5, size=30),
        }
    )
    test = pd.DataFrame({"x1": rng.normal(size=20)})

    model = BayesOptModel(
        train,
        test,
        model_nme="demo",
        resp_nme="y",
        weight_nme="w",
        factor_nmes=["x1"],
        task_type="regression",
        use_gpu=False,
        output_dir=str(tmp_path),
    )

    for df in (model.train_data, model.test_data):
        df["pred_xgb"] = rng.normal(size=len(df))
        df["pred_resn"] = rng.normal(size=len(df))
        df["w_pred_xgb"] = df["pred_xgb"] * df[model.weight_nme]
        df["w_pred_resn"] = df["pred_resn"] * df[model.weight_nme]

    model.plot_lift("Xgboost", "pred_xgb", n_bins=5)
    model.plot_dlift(["xgb", "resn"], n_bins=5)

    lift_path = tmp_path / "plot" / "demo" / "lift" / "01_demo_Xgboost_lift.png"
    dlift_path = tmp_path / "plot" / "demo" / "double_lift" / "02_demo_dlift_xgb_vs_resn.png"

    assert lift_path.exists()
    assert dlift_path.exists()
