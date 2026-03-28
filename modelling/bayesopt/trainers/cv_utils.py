from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    ShuffleSplit,
    TimeSeriesSplit,
)


class _OrderSplitter:
    def __init__(self, splitter, order: np.ndarray) -> None:
        self._splitter = splitter
        self._order = np.asarray(order)

    def split(self, X, y=None, groups=None):
        order = self._order
        X_ord = X.iloc[order] if hasattr(X, "iloc") else X[order]
        for tr_idx, val_idx in self._splitter.split(X_ord, y=y, groups=groups):
            yield order[tr_idx], order[val_idx]


class CVStrategyResolver:
    """Resolve CV splitters for random/time/group strategies."""

    TIME_STRATEGIES = {"time", "timeseries", "temporal"}
    GROUP_STRATEGIES = {"group", "grouped"}

    def __init__(self, config, train_data: pd.DataFrame, rand_seed: Optional[int] = None):
        self.config = config
        self.train_data = train_data
        self.rand_seed = rand_seed
        self._strategy = self._normalize_strategy()

    def _normalize_strategy(self) -> str:
        raw = str(getattr(self.config, "cv_strategy", "random") or "random")
        return raw.strip().lower()

    @property
    def strategy(self) -> str:
        return self._strategy

    def is_time_strategy(self) -> bool:
        return self._strategy in self.TIME_STRATEGIES

    def is_group_strategy(self) -> bool:
        return self._strategy in self.GROUP_STRATEGIES

    def get_time_col(self) -> str:
        time_col = getattr(self.config, "cv_time_col", None)
        if not time_col:
            raise ValueError("cv_time_col is required for time cv_strategy.")
        if time_col not in self.train_data.columns:
            raise KeyError(f"cv_time_col '{time_col}' not in train_data.")
        return time_col

    def get_time_ascending(self) -> bool:
        return bool(getattr(self.config, "cv_time_ascending", True))

    def get_group_col(self) -> str:
        group_col = getattr(self.config, "cv_group_col", None)
        if not group_col:
            raise ValueError("cv_group_col is required for group cv_strategy.")
        if group_col not in self.train_data.columns:
            raise KeyError(f"cv_group_col '{group_col}' not in train_data.")
        return group_col

    def get_time_ordered_indices(self, X_all: pd.DataFrame) -> np.ndarray:
        time_col = self.get_time_col()
        ascending = self.get_time_ascending()
        order_index = self.train_data[time_col].sort_values(ascending=ascending).index
        index_set = set(X_all.index)
        order_index = [idx for idx in order_index if idx in index_set]
        order = X_all.index.get_indexer(order_index)
        return order[order >= 0]

    def get_groups(self, X_all: pd.DataFrame) -> pd.Series:
        group_col = self.get_group_col()
        return self.train_data.reindex(X_all.index)[group_col]

    def create_train_val_splitter(
        self,
        X_all: pd.DataFrame,
        val_ratio: float,
    ) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray]], Optional[pd.Series]]:
        if self.is_time_strategy():
            order = self.get_time_ordered_indices(X_all)
            cutoff = int(len(order) * (1.0 - val_ratio))
            if cutoff <= 0 or cutoff >= len(order):
                raise ValueError(f"val_ratio={val_ratio} leaves no data for train/val split.")
            return (order[:cutoff], order[cutoff:]), None

        if self.is_group_strategy():
            groups = self.get_groups(X_all)
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=val_ratio, random_state=self.rand_seed
            )
            train_idx, val_idx = next(splitter.split(X_all, groups=groups))
            return (train_idx, val_idx), groups

        splitter = ShuffleSplit(
            n_splits=1, test_size=val_ratio, random_state=self.rand_seed
        )
        train_idx, val_idx = next(splitter.split(X_all))
        return (train_idx, val_idx), None

    def create_cv_splitter(
        self,
        X_all: pd.DataFrame,
        y_all: Optional[pd.Series],
        n_splits: int,
        val_ratio: float,
    ) -> Tuple[Iterable[Tuple[np.ndarray, np.ndarray]], int]:
        n_splits = max(2, int(n_splits))

        if self.is_group_strategy():
            groups = self.get_groups(X_all)
            n_groups = int(groups.nunique(dropna=False))
            if n_groups < 2:
                return iter([]), 0
            n_splits = min(n_splits, n_groups)
            if n_splits < 2:
                return iter([]), 0
            splitter = GroupKFold(n_splits=n_splits)
            return splitter.split(X_all, y_all, groups=groups), n_splits

        if self.is_time_strategy():
            order = self.get_time_ordered_indices(X_all)
            if len(order) < 2:
                return iter([]), 0
            n_splits = min(n_splits, max(2, len(order) - 1))
            if n_splits < 2:
                return iter([]), 0
            splitter = TimeSeriesSplit(n_splits=n_splits)
            return _OrderSplitter(splitter, order).split(X_all), n_splits

        if len(X_all) < n_splits:
            n_splits = len(X_all)
        if n_splits < 2:
            return iter([]), 0
        splitter = ShuffleSplit(
            n_splits=n_splits, test_size=val_ratio, random_state=self.rand_seed
        )
        return splitter.split(X_all), n_splits

    def create_kfold_splitter(
        self,
        X_all: pd.DataFrame,
        k: int,
    ) -> Tuple[Optional[Iterable[Tuple[np.ndarray, np.ndarray]]], int]:
        k = max(2, int(k))
        n_samples = len(X_all)
        if n_samples < 2:
            return None, 0

        if self.is_group_strategy():
            groups = self.get_groups(X_all)
            n_groups = int(groups.nunique(dropna=False))
            if n_groups < 2:
                return None, 0
            k = min(k, n_groups)
            if k < 2:
                return None, 0
            splitter = GroupKFold(n_splits=k)
            return splitter.split(X_all, y=None, groups=groups), k

        if self.is_time_strategy():
            order = self.get_time_ordered_indices(X_all)
            if len(order) < 2:
                return None, 0
            k = min(k, max(2, len(order) - 1))
            if k < 2:
                return None, 0
            splitter = TimeSeriesSplit(n_splits=k)
            return _OrderSplitter(splitter, order).split(X_all), k

        k = min(k, n_samples)
        if k < 2:
            return None, 0
        splitter = KFold(n_splits=k, shuffle=True, random_state=self.rand_seed)
        return splitter.split(X_all), k


__all__ = ["CVStrategyResolver", "_OrderSplitter"]
