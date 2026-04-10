from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ins_pricing.modelling.bayesopt.core_geo_preprocess_mixin import (
    BayesOptGeoPreprocessMixin,
)


class _DummyBundleModel(BayesOptGeoPreprocessMixin):
    pass


def _build_dummy_model(
    *,
    tmp_path: Path,
    include_raw: bool,
    keep_unscaled_oht: bool,
) -> _DummyBundleModel:
    model = _DummyBundleModel()
    model.model_nme = "demo"
    model.resp_nme = "y"
    model.weight_nme = "w"
    model.binary_resp_nme = "y_bin"
    model.config = SimpleNamespace(
        preprocess_bundle_path=None,
        preprocess_bundle_include_raw=include_raw,
        keep_unscaled_oht=keep_unscaled_oht,
        cv_group_col="grp",
        cv_time_col="ts",
        region=SimpleNamespace(province_col="prov", city_col="city"),
        geo_token=SimpleNamespace(feature_nmes=["geo1"]),
    )
    model.output_manager = SimpleNamespace(result_dir=str(tmp_path))
    model.train_data = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0],
            "y": [10.0, 20.0, 30.0],
            "w": [1.0, 1.0, 1.0],
            "y_bin": [0, 1, 0],
            "grp": ["a", "a", "b"],
            "ts": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "prov": ["p1", "p1", "p2"],
            "city": ["c1", "c2", "c3"],
            "geo1": [100.0, 101.0, 102.0],
        }
    )
    model.test_data = pd.DataFrame(
        {
            "x1": [4.0, 5.0],
            "y": [40.0, 50.0],
            "w": [1.0, 1.0],
            "y_bin": [1, 0],
            "grp": ["b", "b"],
            "ts": pd.to_datetime(["2024-01-04", "2024-01-05"]),
            "prov": ["p2", "p2"],
            "city": ["c4", "c5"],
            "geo1": [103.0, 104.0],
        }
    )
    model.train_oht_data = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0],
            "y": [10.0, 20.0, 30.0],
            "w": [1.0, 1.0, 1.0],
        }
    )
    model.test_oht_data = pd.DataFrame(
        {
            "x1": [4.0, 5.0],
            "y": [40.0, 50.0],
            "w": [1.0, 1.0],
        }
    )
    model.train_oht_scl_data = pd.DataFrame(
        {
            "x1": [-1.0, 0.0, 1.0],
            "y": [10.0, 20.0, 30.0],
            "w": [1.0, 1.0, 1.0],
        }
    )
    model.test_oht_scl_data = pd.DataFrame(
        {
            "x1": [2.0, 3.0],
            "y": [40.0, 50.0],
            "w": [1.0, 1.0],
        }
    )
    model.var_nmes = ["x1"]
    model.num_features = ["x1"]
    model.cat_categories_for_shap = {}
    model.numeric_scalers = {"x1": {"mean": 2.0, "scale": 1.0}}
    model.ohe_feature_names = []
    model.oht_sparse_csr = False
    return model


def test_preprocess_bundle_can_drop_full_raw_frame(tmp_path: Path):
    bundle_path = tmp_path / "bundle.pkl"
    source = _build_dummy_model(
        tmp_path=tmp_path,
        include_raw=False,
        keep_unscaled_oht=False,
    )
    source._save_preprocess_bundle(bundle_path)

    loaded = _build_dummy_model(
        tmp_path=tmp_path,
        include_raw=False,
        keep_unscaled_oht=False,
    )
    loaded._load_preprocess_bundle(bundle_path)

    assert "x1" not in loaded.train_data.columns
    assert {"y", "w", "y_bin", "grp", "ts", "prov", "city", "geo1"} <= set(
        loaded.train_data.columns
    )
    assert loaded.train_oht_data is None
    assert loaded.test_oht_data is None


def test_preprocess_bundle_keeps_full_raw_and_unscaled_when_requested(tmp_path: Path):
    bundle_path = tmp_path / "bundle.pkl"
    source = _build_dummy_model(
        tmp_path=tmp_path,
        include_raw=True,
        keep_unscaled_oht=True,
    )
    source._save_preprocess_bundle(bundle_path)

    loaded = _build_dummy_model(
        tmp_path=tmp_path,
        include_raw=True,
        keep_unscaled_oht=True,
    )
    loaded._load_preprocess_bundle(bundle_path)

    assert "x1" in loaded.train_data.columns
    assert loaded.train_oht_data is not None
    assert loaded.test_oht_data is not None


def test_preprocess_bundle_can_rebuild_raw_columns_when_absent(tmp_path: Path):
    bundle_path = tmp_path / "bundle.pkl"
    source = _build_dummy_model(
        tmp_path=tmp_path,
        include_raw=False,
        keep_unscaled_oht=False,
    )
    source._save_preprocess_bundle(bundle_path)

    with bundle_path.open("rb") as fh:
        payload = pickle.load(fh)
    payload.pop("train_data", None)
    payload.pop("test_data", None)
    with bundle_path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    loaded = _build_dummy_model(
        tmp_path=tmp_path,
        include_raw=False,
        keep_unscaled_oht=False,
    )
    loaded._load_preprocess_bundle(bundle_path)

    assert "y" in loaded.train_data.columns
    assert "w" in loaded.train_data.columns
    assert loaded.train_data["y"].isna().sum() == 0
