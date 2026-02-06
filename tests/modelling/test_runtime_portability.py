import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

import ins_pricing.modelling.bayesopt.utils.torch_trainer_mixin as mixin_mod
import ins_pricing.modelling.bayesopt.utils.torch_runtime as runtime_mod
from ins_pricing.modelling.bayesopt.trainers.trainer_gnn import GNNTrainer
from ins_pricing.modelling.bayesopt.utils import distributed_utils
from ins_pricing.modelling.bayesopt.utils.torch_trainer_mixin import TorchTrainerMixin
from ins_pricing.utils.device import DeviceManager


def test_device_manager_respects_local_rank(monkeypatch):
    DeviceManager.reset_cache()
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    dev1 = DeviceManager.get_best_device()
    assert str(dev1) == "cuda:1"

    monkeypatch.setenv("LOCAL_RANK", "3")
    dev2 = DeviceManager.get_best_device()
    assert str(dev2) == "cuda:3"


def test_free_cuda_calls_lightweight_manager(monkeypatch):
    calls = {}

    def _fake_clean(*, synchronize, empty_cache):
        calls["synchronize"] = synchronize
        calls["empty_cache"] = empty_cache

    monkeypatch.setattr(distributed_utils.GPUMemoryManager, "clean", _fake_clean)
    monkeypatch.setattr(distributed_utils.torch.cuda, "is_available", lambda: False)

    distributed_utils.TrainingUtils.free_cuda(synchronize=False, empty_cache=False)
    assert calls == {"synchronize": False, "empty_cache": False}


def test_setup_ddp_if_not_requested_returns_single_process():
    ok, local_rank, rank, world_size = runtime_mod.setup_ddp_if_requested(False)
    assert ok is False
    assert local_rank == 0
    assert rank == 0
    assert world_size == 1


def test_wrap_model_for_parallel_uses_data_parallel(monkeypatch):
    class _FakeDataParallel(torch.nn.Module):
        def __init__(self, module, device_ids=None):
            super().__init__()
            self.module = module
            self.device_ids = device_ids

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    monkeypatch.setattr(runtime_mod.nn, "DataParallel", _FakeDataParallel)
    monkeypatch.setattr(runtime_mod.torch.cuda, "device_count", lambda: 2)

    core = torch.nn.Linear(2, 1)
    wrapped, use_dp, device = runtime_mod.wrap_model_for_parallel(
        core,
        device=torch.device("cuda"),
        use_data_parallel=True,
        use_ddp_requested=False,
        is_ddp_enabled=False,
        local_rank=0,
    )
    assert use_dp is True
    assert device.type == "cuda"
    assert isinstance(wrapped, _FakeDataParallel)


class _DummyTorchTrainer(TorchTrainerMixin):
    pass


def test_ddp_num_workers_scales_by_world_size(monkeypatch):
    dummy = _DummyTorchTrainer()
    dummy.device = torch.device("cuda:0")
    dummy.is_ddp_enabled = True
    dummy.world_size = 4

    monkeypatch.setattr(mixin_mod.os, "name", "posix", raising=False)
    monkeypatch.setattr(mixin_mod.os, "cpu_count", lambda: 16)

    workers = dummy._resolve_num_workers(8, profile="throughput")
    assert workers == 2


def test_gnn_trainer_disables_distributed_optuna_when_unsupported(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    ctx = types.SimpleNamespace(
        config=types.SimpleNamespace(use_gnn_ddp=True)
    )
    trainer = GNNTrainer(ctx)
    assert trainer.enable_distributed_optuna is False
    assert trainer._runtime_use_ddp is False
