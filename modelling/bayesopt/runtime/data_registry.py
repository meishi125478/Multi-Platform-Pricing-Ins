from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, MutableMapping, Tuple

import numpy as np
import pandas as pd

from ins_pricing.modelling.bayesopt.runtime.types import RowStore


@dataclass
class DataRegistry:
    """Cached row store and lightweight matrix views for dispatch."""

    row_store: RowStore
    max_cache_entries: int = 8
    _train_cache: MutableMapping[Tuple[str, ...], np.ndarray] = field(default_factory=OrderedDict)
    _test_cache: MutableMapping[Tuple[str, ...], np.ndarray] = field(default_factory=OrderedDict)

    @classmethod
    def from_context(cls, ctx) -> "DataRegistry":
        train_raw = getattr(ctx, "train_data", None)
        test_raw = getattr(ctx, "test_data", None)
        if not isinstance(train_raw, pd.DataFrame) or not isinstance(test_raw, pd.DataFrame):
            raise TypeError("DataRegistry requires train_data/test_data as pandas DataFrame.")

        train_row_id = np.arange(len(train_raw), dtype=np.int64)
        test_row_id = np.arange(len(test_raw), dtype=np.int64)
        row_store = RowStore(
            train_raw=train_raw,
            test_raw=test_raw,
            train_row_id=train_row_id,
            test_row_id=test_row_id,
            train_source_index=train_raw.index.to_numpy(copy=True),
            test_source_index=test_raw.index.to_numpy(copy=True),
        )
        return cls(row_store=row_store)

    @staticmethod
    def _cache_key(columns: Iterable[str], dtype) -> Tuple[str, ...]:
        cols = tuple(str(col) for col in columns)
        dtype_tag = np.dtype(dtype).str
        return cols + (f"__dtype__:{dtype_tag}",)

    @staticmethod
    def _cache_columns_from_key(key: Tuple[str, ...]) -> list[str]:
        return [col for col in key if not col.startswith("__dtype__:")]

    def _cache_put(
        self,
        cache: MutableMapping[Tuple[str, ...], np.ndarray],
        key: Tuple[str, ...],
        matrix: np.ndarray,
    ) -> np.ndarray:
        cache[key] = matrix
        if hasattr(cache, "move_to_end"):
            cache.move_to_end(key)  # type: ignore[attr-defined]
        while len(cache) > max(int(self.max_cache_entries), 1):
            if isinstance(cache, OrderedDict):
                cache.popitem(last=False)
            else:
                oldest_key = next(iter(cache))
                cache.pop(oldest_key, None)
        return matrix

    def train_matrix(self, columns: Iterable[str], *, dtype=np.float32) -> np.ndarray:
        key = self._cache_key(columns, dtype)
        cached = self._train_cache.get(key)
        if cached is not None:
            if hasattr(self._train_cache, "move_to_end"):
                self._train_cache.move_to_end(key)  # type: ignore[attr-defined]
            return cached
        cols = self._cache_columns_from_key(key)
        matrix = self.row_store.train_raw.loc[:, cols].to_numpy(dtype=dtype, copy=False)
        return self._cache_put(self._train_cache, key, matrix)

    def test_matrix(self, columns: Iterable[str], *, dtype=np.float32) -> np.ndarray:
        key = self._cache_key(columns, dtype)
        cached = self._test_cache.get(key)
        if cached is not None:
            if hasattr(self._test_cache, "move_to_end"):
                self._test_cache.move_to_end(key)  # type: ignore[attr-defined]
            return cached
        cols = self._cache_columns_from_key(key)
        matrix = self.row_store.test_raw.loc[:, cols].to_numpy(dtype=dtype, copy=False)
        return self._cache_put(self._test_cache, key, matrix)
