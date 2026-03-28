from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from ins_pricing.exceptions import ModelLoadError
from ins_pricing.utils.model_loading import load_pickle_artifact


def test_load_pickle_artifact_allows_safe_payload(tmp_path):
    path = tmp_path / "safe.pkl"
    with path.open("wb") as fh:
        pickle.dump({"values": [1, 2, 3]}, fh, protocol=pickle.HIGHEST_PROTOCOL)

    payload = load_pickle_artifact(path)
    assert payload == {"values": [1, 2, 3]}


def test_load_pickle_artifact_allows_bytearray_payload(tmp_path):
    path = tmp_path / "bytearray.pkl"
    payload = {"blob": bytearray(b"\x01\x02\x03")}
    with path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    loaded = load_pickle_artifact(path)
    assert isinstance(loaded, dict)
    assert loaded["blob"] == payload["blob"]
    assert isinstance(loaded["blob"], bytearray)


def test_load_pickle_artifact_blocks_untrusted_global_by_default(tmp_path):
    path = tmp_path / "unsafe.pkl"
    with path.open("wb") as fh:
        pickle.dump(Path("abc"), fh, protocol=pickle.HIGHEST_PROTOCOL)

    with pytest.raises(ModelLoadError):
        load_pickle_artifact(path, allow_unsafe=False)


def test_load_pickle_artifact_allows_xgb_wrapper_payload(tmp_path):
    pytest.importorskip("xgboost")
    pytest.importorskip("optuna")
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")

    from ins_pricing.modelling.bayesopt.trainers.trainer_xgb import _XGBDMatrixWrapper

    path = tmp_path / "xgb_wrapper.pkl"
    payload = {
        "model": _XGBDMatrixWrapper(
            params={"n_estimators": 1},
            task_type="regression",
            use_gpu=False,
        )
    }
    with path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    loaded = load_pickle_artifact(path, allow_unsafe=False)
    assert isinstance(loaded, dict)
    assert isinstance(loaded.get("model"), _XGBDMatrixWrapper)
