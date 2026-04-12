from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from ins_pricing.exceptions import ModelLoadError
from ins_pricing.utils.model_loading import (
    load_model_artifact_payload,
    load_pickle_artifact,
    load_pickle_artifact_with_optional_unsafe_retry,
    load_torch_payload,
)


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
        load_pickle_artifact(path)


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

    loaded = load_pickle_artifact(path)
    assert isinstance(loaded, dict)
    assert isinstance(loaded.get("model"), _XGBDMatrixWrapper)


def test_load_torch_payload_falls_back_to_legacy_when_weights_only_is_unsupported(
    tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    from ins_pricing.utils import model_loading
    from ins_pricing.utils import torch_compat

    path = tmp_path / "weights.pt"
    torch.save({"state_dict": {"x": torch.tensor([1.0])}}, path)

    monkeypatch.delenv("INS_PRICING_ALLOW_LEGACY_TORCH_LOAD", raising=False)
    monkeypatch.setattr(model_loading, "supports_weights_only", lambda: False)
    captured = {}

    def _fake_load(path_obj, *args, **kwargs):
        captured["path"] = path_obj
        captured["kwargs"] = dict(kwargs)
        return {"state_dict": {"x": torch.tensor([1.0])}}

    monkeypatch.setattr(torch_compat.torch, "load", _fake_load)
    out = load_torch_payload(path, map_location="cpu", weights_only=True)
    assert "state_dict" in out
    assert captured["path"] == path
    assert "weights_only" not in captured["kwargs"]


def test_load_torch_payload_blocks_legacy_fallback_when_disabled(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from ins_pricing.utils import model_loading

    path = tmp_path / "weights.pt"
    torch.save({"state_dict": {"x": torch.tensor([1.0])}}, path)

    monkeypatch.setenv("INS_PRICING_ALLOW_LEGACY_TORCH_LOAD", "0")
    monkeypatch.setattr(model_loading, "supports_weights_only", lambda: False)

    with pytest.raises(ModelLoadError, match="legacy fallback is disabled"):
        load_torch_payload(path, map_location="cpu", weights_only=True)


def test_torch_load_allows_legacy_trusted_path_when_weights_only_false(monkeypatch):
    pytest.importorskip("torch")
    from ins_pricing.utils import torch_compat

    monkeypatch.setattr(torch_compat, "_supports_weights_only", lambda: False)
    captured = {}

    def _fake_load(path, *args, **kwargs):
        captured["path"] = path
        captured["kwargs"] = dict(kwargs)
        return {"ok": True}

    monkeypatch.setattr(torch_compat.torch, "load", _fake_load)
    out = torch_compat.torch_load("dummy.pt", weights_only=False)
    assert out == {"ok": True}
    assert captured["path"] == "dummy.pt"
    assert "weights_only" not in captured["kwargs"]


def test_load_model_artifact_payload_dispatches_pickle(monkeypatch):
    from ins_pricing.utils import model_loading

    captured = {}

    def _fake_pickle_loader(path, *, allow_unsafe_retry=False):
        captured["path"] = path
        captured["allow_unsafe_retry"] = allow_unsafe_retry
        return {"ok": "pickle"}

    monkeypatch.setattr(
        model_loading,
        "load_pickle_artifact_with_optional_unsafe_retry",
        _fake_pickle_loader,
    )

    out = load_model_artifact_payload(
        "demo.pkl",
        model_key="xgb",
        allow_unsafe_pickle_retry=True,
    )
    assert out == {"ok": "pickle"}
    assert captured["allow_unsafe_retry"] is True


def test_load_model_artifact_payload_dispatches_torch(monkeypatch):
    from ins_pricing.utils import model_loading

    captured = {}

    def _fake_torch_loader(path, *, map_location="cpu", weights_only=True):
        captured["path"] = path
        captured["map_location"] = map_location
        captured["weights_only"] = weights_only
        return {"ok": "torch"}

    monkeypatch.setattr(model_loading, "load_torch_payload", _fake_torch_loader)

    out = load_model_artifact_payload(
        "demo.pth",
        model_key="resn",
        map_location="cpu",
    )
    assert out == {"ok": "torch"}
    assert captured["weights_only"] is True


def test_load_pickle_artifact_with_optional_unsafe_retry_skips_io_errors(
    tmp_path,
    monkeypatch,
):
    from ins_pricing.utils import model_loading

    path = tmp_path / "model.pkl"
    path.write_bytes(b"placeholder")
    captured = {"unsafe_called": False}

    def _raise_io(_path):
        raise ModelLoadError("Cannot read model artifact: model.pkl")

    def _unsafe(_fh):
        captured["unsafe_called"] = True
        return {"ok": True}

    monkeypatch.setattr(model_loading, "load_pickle_artifact", _raise_io)
    monkeypatch.setattr(model_loading.pickle, "load", _unsafe)

    with pytest.raises(ModelLoadError, match="Cannot read model artifact"):
        load_pickle_artifact_with_optional_unsafe_retry(path, allow_unsafe_retry=True)
    assert captured["unsafe_called"] is False


def test_load_pickle_artifact_with_optional_unsafe_retry_allows_restricted_errors(
    tmp_path,
    monkeypatch,
):
    from ins_pricing.utils import model_loading

    path = tmp_path / "model.pkl"
    path.write_bytes(b"placeholder")

    def _raise_restricted(_path):
        raise ModelLoadError("Failed secure pickle load for artifact: blocked allowlist")

    monkeypatch.setattr(model_loading, "load_pickle_artifact", _raise_restricted)
    monkeypatch.setattr(model_loading.pickle, "load", lambda _fh: {"model": "unsafe-loaded"})

    out = load_pickle_artifact_with_optional_unsafe_retry(path, allow_unsafe_retry=True)
    assert out == {"model": "unsafe-loaded"}


def test_load_pickle_artifact_with_optional_unsafe_retry_allows_non_io_secure_errors(
    tmp_path,
    monkeypatch,
):
    from ins_pricing.utils import model_loading

    path = tmp_path / "model.pkl"
    path.write_bytes(b"placeholder")

    def _raise_non_io_secure_error(_path):
        raise ModelLoadError("Failed secure pickle load for artifact: ModuleNotFoundError")

    monkeypatch.setattr(model_loading, "load_pickle_artifact", _raise_non_io_secure_error)
    monkeypatch.setattr(model_loading.pickle, "load", lambda _fh: {"model": "unsafe-loaded"})

    out = load_pickle_artifact_with_optional_unsafe_retry(path, allow_unsafe_retry=True)
    assert out == {"model": "unsafe-loaded"}


def test_torch1_runtime_smoke_load_torch_payload(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if not str(torch.__version__).startswith("1."):
        pytest.skip("requires torch 1.x runtime")

    path = tmp_path / "torch1_weights.pt"
    torch.save({"state_dict": {"x": torch.tensor([1.0])}}, path)

    monkeypatch.delenv("INS_PRICING_ALLOW_LEGACY_TORCH_LOAD", raising=False)
    loaded = load_torch_payload(path, map_location="cpu", weights_only=True)
    assert isinstance(loaded, dict)
    assert "state_dict" in loaded


def test_torch1_runtime_smoke_blocks_when_legacy_disabled(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if not str(torch.__version__).startswith("1."):
        pytest.skip("requires torch 1.x runtime")

    path = tmp_path / "torch1_weights.pt"
    torch.save({"state_dict": {"x": torch.tensor([1.0])}}, path)

    monkeypatch.setenv("INS_PRICING_ALLOW_LEGACY_TORCH_LOAD", "0")
    with pytest.raises(ModelLoadError, match="legacy fallback is disabled"):
        load_torch_payload(path, map_location="cpu", weights_only=True)
