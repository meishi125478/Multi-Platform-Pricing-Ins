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

    from ins_pricing.modelling.plotting import curves, diagnostics, geo, importance

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
        target_weighted=True,
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


def test_plot_oneway_weighted_target_mean(tmp_path, monkeypatch):
    _configure_matplotlib(tmp_path, monkeypatch)

    from ins_pricing.modelling.plotting import diagnostics
    from matplotlib import pyplot as plt

    df = pd.DataFrame(
        {
            "seg": ["A", "A", "B", "B"],
            "target": [1.0, 3.0, 2.0, 4.0],
            "w": [1.0, 3.0, 1.0, 1.0],
        }
    )

    fig, ax = plt.subplots()
    diagnostics.plot_oneway(
        df,
        feature="seg",
        weight_col="w",
        target_col="target",
        target_weighted=False,
        is_categorical=True,
        ax=ax,
        show=False,
    )
    labels = [tick.get_text() for tick in ax.get_xticklabels()]
    values = dict(zip(labels, ax.lines[0].get_ydata()))
    assert values["A"] == pytest.approx(2.5)
    assert values["B"] == pytest.approx(3.0)
    plt.close(fig)

    df["w_act"] = df["target"] * df["w"]
    fig, ax = plt.subplots()
    diagnostics.plot_oneway(
        df,
        feature="seg",
        weight_col="w",
        target_col="w_act",
        target_weighted=True,
        is_categorical=True,
        ax=ax,
        show=False,
    )
    labels = [tick.get_text() for tick in ax.get_xticklabels()]
    values = dict(zip(labels, ax.lines[0].get_ydata()))
    assert values["A"] == pytest.approx(2.5)
    assert values["B"] == pytest.approx(3.0)
    plt.close(fig)


def test_geo_plotting_on_map_optional(tmp_path, monkeypatch):
    _configure_matplotlib(tmp_path, monkeypatch)
    pytest.importorskip("contextily")

    from ins_pricing.modelling.plotting import geo

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


def test_double_lift_table_shared_mask_alignment():
    from ins_pricing.modelling.plotting import curves

    pred1 = np.array([0.1, np.nan, 0.3, 0.4, np.inf], dtype=float)
    pred2 = np.array([0.2, 0.2, 0.3, 0.8, 1.0], dtype=float)
    actual = np.array([1.0, 1.0, np.nan, 2.0, 3.0], dtype=float)
    weight = np.array([1.0, 1.0, 1.0, 2.0, 1.0], dtype=float)

    # Regression: this used to fail with a length mismatch in a two-stage align flow.
    out = curves.double_lift_table(
        pred1,
        pred2,
        actual,
        weight,
        n_bins=2,
        pred1_weighted=False,
        pred2_weighted=False,
        actual_weighted=False,
    )
    # Only rows 0 and 3 are finite across all four arrays => total weight = 3.
    assert out["weight"].sum() == pytest.approx(3.0)


def test_aggregate_weight_bins_starts_from_zero_bin():
    from ins_pricing.modelling.plotting import curves

    out = curves._aggregate_by_weight_bins(
        sort_key=np.array([0.0, 1.0, 2.0], dtype=float),
        weight=np.array([0.6, 0.2, 0.2], dtype=float),
        values={"v": np.array([1.0, 1.0, 1.0], dtype=float)},
        n_bins=10,
    )
    assert 0 in set(out["bins"].tolist())


def test_double_lift_sort_ratio_preserves_near_zero_negative_denominator_sign():
    from ins_pricing.modelling.plotting import curves

    out = curves.double_lift_table(
        pred1=np.array([2.0, 1.0], dtype=float),
        pred2=np.array([-1e-9, 0.1], dtype=float),
        actual=np.array([1.0, 1.0], dtype=float),
        weight=np.array([1.0, 1.0], dtype=float),
        n_bins=10,
        pred1_weighted=False,
        pred2_weighted=False,
        actual_weighted=False,
    )
    assert len(out) >= 2
    assert float(out["exp_v1"].iloc[0]) == pytest.approx(2.0)
    assert float(out["exp_v1"].iloc[-1]) == pytest.approx(1.0)


def test_oneway_fallback_groups_only_required_columns(tmp_path, monkeypatch):
    _configure_matplotlib(tmp_path, monkeypatch)

    from ins_pricing.modelling.bayesopt.model_plotting_mixin import BayesOptPlottingMixin
    from matplotlib import pyplot as plt

    class _Dummy(BayesOptPlottingMixin):
        weight_nme = "w"

        @staticmethod
        def _load_plot_diagnostics_module():
            return None

    rng = np.random.default_rng(123)
    n = 30
    plot_source = pd.DataFrame(
        {
            "feat": rng.normal(size=n),
            "pred_demo": rng.normal(size=n),
            "w": rng.uniform(0.5, 2.0, size=n),
            "w_act": rng.uniform(0.0, 3.0, size=n),
            # Wide-table noise columns that should not be aggregated by fallback.
            "noise_1": rng.normal(size=n),
            "noise_2": rng.normal(size=n),
            "noise_3": rng.normal(size=n),
        }
    )

    original_groupby = pd.DataFrame.groupby
    captured_groupby_columns = []

    def _spy_groupby(self, by=None, *args, **kwargs):
        captured_groupby_columns.append(list(self.columns))
        return original_groupby(self, by=by, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "groupby", _spy_groupby)

    dummy = _Dummy()
    save_path = tmp_path / "oneway_fallback.png"
    dummy._plot_oneway_fallback(
        plot_source=plot_source,
        feature="feat",
        group_col="feat_grp",
        is_categorical=False,
        n_bins=5,
        pred_col="pred_demo",
        pred_label="demo",
        pred_weighted=False,
        title="oneway-fallback",
        save_path=str(save_path),
    )

    assert save_path.exists()
    assert captured_groupby_columns, "Expected fallback groupby to be executed."
    assert captured_groupby_columns[-1] == ["feat_grp", "_weight", "_act_w", "_pred_w"]
    plt.close("all")
