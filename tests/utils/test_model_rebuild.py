from __future__ import annotations

import pytest

from ins_pricing.utils import model_rebuild


def test_rebuild_ft_payload_delegates_to_checkpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def _fake_rebuild_ft_model_from_payload(
        *,
        payload,
        model_config_overrides=None,
        fill_missing_model_config=True,
    ):
        captured["payload"] = payload
        captured["model_config_overrides"] = model_config_overrides
        captured["fill_missing_model_config"] = fill_missing_model_config
        return "model", {"hidden_dim": 64}, "state_dict"

    monkeypatch.setattr(
        model_rebuild,
        "rebuild_ft_model_from_payload",
        _fake_rebuild_ft_model_from_payload,
    )

    out = model_rebuild.rebuild_ft_payload(
        payload={"state_dict": {}},
        model_config_overrides={"loss_name": "mse"},
        fill_missing_model_config=False,
    )

    assert out == ("model", {"hidden_dim": 64}, "state_dict")
    assert captured == {
        "payload": {"state_dict": {}},
        "model_config_overrides": {"loss_name": "mse"},
        "fill_missing_model_config": False,
    }


def test_rebuild_resn_payload_requires_callable_builder() -> None:
    with pytest.raises(ValueError, match="model_builder"):
        model_rebuild.rebuild_resn_payload(  # type: ignore[arg-type]
            payload={},
            model_builder=None,
        )


def test_rebuild_gnn_payload_requires_callable_builder() -> None:
    with pytest.raises(ValueError, match="model_builder"):
        model_rebuild.rebuild_gnn_payload(  # type: ignore[arg-type]
            payload={},
            model_builder=None,
        )


def test_rebuild_model_artifact_payload_dispatches_pickle_model() -> None:
    payload = {"model": "xgb-model", "meta": 1}
    out = model_rebuild.rebuild_model_artifact_payload(
        payload=payload,
        model_key="xgb",
    )
    assert out == "xgb-model"


def test_rebuild_model_artifact_payload_dispatches_resn(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def _fake_resn(*, payload, model_builder, params_fallback=None, require_params=True):
        captured["payload"] = payload
        captured["model_builder"] = model_builder
        captured["params_fallback"] = params_fallback
        captured["require_params"] = require_params
        return "resn-model", {"hidden_dim": 32}

    monkeypatch.setattr(model_rebuild, "rebuild_resn_payload", _fake_resn)
    builder = lambda p: p
    out = model_rebuild.rebuild_model_artifact_payload(
        payload={"state_dict": {}},
        model_key="resn",
        model_builder=builder,
        params_fallback={"hidden_dim": 64},
        require_params=False,
    )

    assert out == ("resn-model", {"hidden_dim": 32})
    assert captured["payload"] == {"state_dict": {}}
    assert captured["model_builder"] is builder
    assert captured["params_fallback"] == {"hidden_dim": 64}
    assert captured["require_params"] is False


def test_rebuild_model_artifact_payload_rejects_unknown_model_key() -> None:
    with pytest.raises(ValueError, match="Unsupported model key"):
        model_rebuild.rebuild_model_artifact_payload(
            payload={},
            model_key="unknown",
        )
