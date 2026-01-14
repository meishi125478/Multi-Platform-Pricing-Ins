import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("sklearn")


def _configure_matplotlib(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    mpl_cfg = tmp_path / ".mplconfig"
    cache_dir = tmp_path / ".cache"
    (cache_dir / "fontconfig").mkdir(parents=True, exist_ok=True)
    mpl_cfg.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MPLCONFIGDIR", str(mpl_cfg))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_dir))


def test_plotting_library_outputs(tmp_path, monkeypatch):
    _configure_matplotlib(tmp_path, monkeypatch)

    from ins_pricing.plotting import curves, diagnostics, geo, importance

    rng = np.random.default_rng(42)
    n = 80
    pred1 = rng.normal(loc=0.2, scale=1.0, size=n)
    pred2 = rng.normal(loc=0.1, scale=1.0, size=n)
    actual = np.abs(rng.normal(loc=1.0, scale=0.5, size=n))
    weight = rng.uniform(0.5, 2.0, size=n)

    curves.plot_lift_curve(
        pred1,
        actual * weight,
        weight,
        n_bins=8,
        save_path=str(tmp_path / "lift.png"),
    )
    curves.plot_double_lift_curve(
        pred1,
        pred2,
        actual * weight,
        weight,
        n_bins=8,
        save_path=str(tmp_path / "dlift.png"),
    )

    y_true = rng.integers(0, 2, size=n)
    curves.plot_roc_curves(
        y_true,
        {"m1": pred1, "m2": pred2},
        save_path=str(tmp_path / "roc.png"),
    )

    importance.plot_feature_importance(
        {"x1": 0.3, "x2": 0.1, "x3": 0.05},
        save_path=str(tmp_path / "importance.png"),
    )

    diagnostics.plot_loss_curve(
        history={"train": [1.0, 0.7, 0.5], "val": [1.2, 0.8, 0.6]},
        save_path=str(tmp_path / "loss.png"),
    )
    diagnostics.plot_oneway(
        pd.DataFrame(
            {
                "x1": rng.normal(size=n),
                "w_act": actual * weight,
                "w": weight,
            }
        ),
        feature="x1",
        weight_col="w",
        target_col="w_act",
        n_bins=6,
        save_path=str(tmp_path / "oneway.png"),
    )

    df_geo = pd.DataFrame(
        {
            "lon": rng.uniform(100, 120, size=n),
            "lat": rng.uniform(20, 40, size=n),
            "loss": actual,
        }
    )
    geo.plot_geo_heatmap(
        df_geo,
        x_col="lon",
        y_col="lat",
        value_col="loss",
        bins=10,
        save_path=str(tmp_path / "geo_heat.png"),
    )
    geo.plot_geo_contour(
        df_geo,
        x_col="lon",
        y_col="lat",
        value_col="loss",
        max_points=40,
        levels=6,
        save_path=str(tmp_path / "geo_contour.png"),
    )

    assert (tmp_path / "lift.png").exists()
    assert (tmp_path / "dlift.png").exists()
    assert (tmp_path / "roc.png").exists()
    assert (tmp_path / "importance.png").exists()
    assert (tmp_path / "loss.png").exists()
    assert (tmp_path / "oneway.png").exists()
    assert (tmp_path / "geo_heat.png").exists()
    assert (tmp_path / "geo_contour.png").exists()


def test_geo_plotting_on_map_optional(tmp_path, monkeypatch):
    _configure_matplotlib(tmp_path, monkeypatch)
    pytest.importorskip("contextily")

    from ins_pricing.plotting import geo

    rng = np.random.default_rng(7)
    n = 60
    df_geo = pd.DataFrame(
        {
            "lon": rng.uniform(105, 115, size=n),
            "lat": rng.uniform(25, 35, size=n),
            "loss": np.abs(rng.normal(loc=1.0, scale=0.4, size=n)),
        }
    )

    geo.plot_geo_heatmap_on_map(
        df_geo,
        lon_col="lon",
        lat_col="lat",
        value_col="loss",
        bins=12,
        basemap=None,
        save_path=str(tmp_path / "geo_heat_map.png"),
    )
    geo.plot_geo_contour_on_map(
        df_geo,
        lon_col="lon",
        lat_col="lat",
        value_col="loss",
        max_points=30,
        levels=5,
        basemap=None,
        save_path=str(tmp_path / "geo_contour_map.png"),
    )

    assert (tmp_path / "geo_heat_map.png").exists()
    assert (tmp_path / "geo_contour_map.png").exists()
