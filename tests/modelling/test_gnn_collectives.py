from __future__ import annotations

import pytest

import ins_pricing.modelling.bayesopt.models.model_gnn as gnn_mod


pytest.importorskip("torch")


def test_use_distributed_collectives_requires_ddp_flag(monkeypatch):
    monkeypatch.setattr(gnn_mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(gnn_mod.dist, "is_initialized", lambda: True)
    assert gnn_mod._use_distributed_collectives(ddp_enabled=False) is False


def test_use_distributed_collectives_requires_initialized_dist(monkeypatch):
    monkeypatch.setattr(gnn_mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(gnn_mod.dist, "is_initialized", lambda: False)
    assert gnn_mod._use_distributed_collectives(ddp_enabled=True) is False
