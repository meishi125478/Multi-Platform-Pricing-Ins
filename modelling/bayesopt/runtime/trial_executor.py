from __future__ import annotations

from typing import Any, Callable, Protocol, Sequence

import numpy as np

from ins_pricing.modelling.bayesopt.runtime.types import FoldResult, FoldSlice


class BackendKernel(Protocol):
    model_key: str

    def run_fold(
        self,
        *,
        params: dict[str, Any],
        fold: FoldSlice,
        trial: Any,
    ) -> FoldResult:
        ...


class TrialExecutor:
    """Minimal fold executor for objective workflows."""

    def evaluate_trial(
        self,
        *,
        params: dict[str, Any],
        trial: Any,
        folds: Sequence[FoldSlice],
        backend: BackendKernel,
        aggregate_fn: Callable[[list[float]], float] | None = None,
    ) -> float:
        losses: list[float] = []
        for fold in folds:
            result = backend.run_fold(params=params, fold=fold, trial=trial)
            losses.append(float(result.loss))
        if not losses:
            raise ValueError("No fold losses available for trial execution.")
        if aggregate_fn is not None:
            return float(aggregate_fn(losses))
        return float(np.mean(losses))
