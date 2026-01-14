import pytest

pytest.importorskip("torch")

from ins_pricing.bayesopt.utils import DistributedUtils


def test_setup_ddp_without_env(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    ok, local_rank, rank, world_size = DistributedUtils.setup_ddp()

    assert ok is False
    assert local_rank == 0
    assert rank == 0
    assert world_size == 1
