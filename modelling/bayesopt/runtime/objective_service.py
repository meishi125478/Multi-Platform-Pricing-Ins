from __future__ import annotations

from typing import Optional

from ins_pricing.modelling.bayesopt.runtime.data_registry import DataRegistry
from ins_pricing.modelling.bayesopt.runtime.trial_executor import TrialExecutor


class ObjectiveService:
    """Objective orchestration scaffolding."""

    def __init__(self, ctx) -> None:
        self.ctx = ctx
        self._trial_executor: Optional[TrialExecutor] = None

    @property
    def trial_executor(self) -> TrialExecutor:
        executor = self._trial_executor
        if isinstance(executor, TrialExecutor):
            return executor
        executor = TrialExecutor()
        self._trial_executor = executor
        return executor

    def ensure_data_registry(self) -> DataRegistry:
        registry = getattr(self.ctx, "_data_registry", None)
        if isinstance(registry, DataRegistry):
            return registry
        registry = DataRegistry.from_context(self.ctx)
        setattr(self.ctx, "_data_registry", registry)
        return registry
