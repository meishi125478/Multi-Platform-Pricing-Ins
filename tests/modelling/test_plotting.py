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

    from ins_pricing.modelling.bayesopt import BayesOptConfig, BayesOptModel
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    rng = np.random.default_rng(0)
    train = pd.DataFrame(
        {
            "x1": rng.normal(size=30),
            "y": rng.normal(size=30),
            "y_bin": rng.integers(0, 2, size=30),
            "w": rng.uniform(0.5, 1.5, size=30),
        }
    )
    test = pd.DataFrame(
        {
            "x1": rng.normal(size=20),
            "y_bin": rng.integers(0, 2, size=20),
        }
    )

    config = BayesOptConfig(
        model_nme="demo",
        resp_nme="y",
        weight_nme="w",
        factor_nmes=["x1"],
        task_type="regression",
        use_gpu=False,
        output_dir=str(tmp_path),
    )
    model = BayesOptModel(train, test, config=config)
    model.binary_resp_nme = "y_bin"

    for df in (model.train_data, model.test_data):
        df["pred_xgb"] = rng.normal(size=len(df))
        df["pred_resn"] = rng.normal(size=len(df))
        df["w_pred_xgb"] = df["pred_xgb"] * df[model.weight_nme]
        df["w_pred_resn"] = df["pred_resn"] * df[model.weight_nme]

    model.plot_lift("Xgboost", "pred_xgb", n_bins=5)
    model.plot_dlift(["xgb", "resn"], n_bins=5)
    model.plot_conversion_lift("pred_xgb", n_bins=5)

    lift_path = tmp_path / "plot" / "demo" / "lift" / "01_demo_Xgboost_lift.png"
    dlift_path = tmp_path / "plot" / "demo" / "double_lift" / "02_demo_dlift_xgb_vs_resn.png"
    conversion_lift_path = (
        tmp_path / "plot" / "demo" / "conversion_lift" / "03_demo_pred_xgb_conversion_lift.png"
    )

    assert lift_path.exists()
    assert dlift_path.exists()
    assert conversion_lift_path.exists()


def test_use_gpu_enables_mps_backend(tmp_path, monkeypatch):
    from ins_pricing.modelling.bayesopt import BayesOptConfig, BayesOptModel
    import ins_pricing.modelling.bayesopt.core as core_mod

    monkeypatch.setattr(core_mod.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        core_mod.DeviceManager,
        "is_mps_available",
        classmethod(lambda cls: True),
    )

    train = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "w": [1.0, 1.0, 1.0]})
    test = pd.DataFrame({"x1": [4.0, 5.0]})
    config = BayesOptConfig(
        model_nme="demo_mps",
        resp_nme="y",
        weight_nme="w",
        factor_nmes=["x1"],
        task_type="regression",
        use_gpu=True,
        output_dir=str(tmp_path),
    )
    model = BayesOptModel(train, test, config=config)

    assert model.use_gpu is True
