from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


def resolve_model_scoped_path(
    value: Any,
    *,
    model_name: str,
    base_dir: Path,
    resolve_path: Optional[Callable[[str, Path], Optional[Path]]] = None,
) -> Optional[Path]:
    if value is None:
        return None

    candidate: Any = value
    if isinstance(value, dict):
        candidate = value.get(model_name)
        if candidate is None:
            candidate = value.get("*")
    if candidate is None:
        return None

    candidate_str = str(candidate).strip()
    if not candidate_str:
        return None
    try:
        candidate_str = candidate_str.format(model_name=model_name)
    except Exception:
        pass

    if resolve_path is not None:
        return resolve_path(candidate_str, base_dir)

    path = Path(candidate_str)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def write_split_cache(
    path: Path,
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    row_count: int,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload_meta = json.dumps(meta or {}, ensure_ascii=True, sort_keys=True)
    with tmp_path.open("wb") as f:
        np.savez_compressed(
            f,
            train_idx=np.asarray(train_idx, dtype=np.int64),
            test_idx=np.asarray(test_idx, dtype=np.int64),
            row_count=np.asarray([int(row_count)], dtype=np.int64),
            meta_json=np.asarray([payload_meta]),
        )
    tmp_path.replace(path)


def load_split_cache(path: Path) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        if "train_idx" not in payload or "test_idx" not in payload:
            raise ValueError(
                f"split cache missing required arrays train_idx/test_idx: {path}"
            )
        train_idx = np.asarray(payload["train_idx"], dtype=np.int64).reshape(-1)
        test_idx = np.asarray(payload["test_idx"], dtype=np.int64).reshape(-1)
        row_count = -1
        if "row_count" in payload:
            row_count_arr = np.asarray(payload["row_count"], dtype=np.int64).reshape(-1)
            if row_count_arr.size > 0:
                row_count = int(row_count_arr[0])
        meta: Dict[str, Any] = {}
        if "meta_json" in payload:
            try:
                meta_arr = np.asarray(payload["meta_json"]).reshape(-1)
                if meta_arr.size > 0:
                    raw_json = str(meta_arr[0]).strip()
                    if raw_json:
                        meta = json.loads(raw_json)
            except Exception:
                meta = {}
    return train_idx, test_idx, row_count, meta


def validate_split_indices(
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    row_count: int,
    cache_path: Path,
) -> None:
    if row_count <= 0:
        raise ValueError(f"Invalid row_count={row_count} for split cache: {cache_path}")
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError(f"split cache has empty train/test indices: {cache_path}")
    if np.any(train_idx < 0) or np.any(test_idx < 0):
        raise ValueError(f"split cache contains negative row indices: {cache_path}")
    if np.any(train_idx >= row_count) or np.any(test_idx >= row_count):
        raise ValueError(
            f"split cache row indices exceed dataset bounds row_count={row_count}: {cache_path}"
        )
    if np.intersect1d(train_idx, test_idx).size > 0:
        raise ValueError(f"split cache contains overlapping train/test rows: {cache_path}")


def _normalize_seed(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except Exception:
        return text


def validate_split_cache_metadata(
    *,
    cache_path: Path,
    cached_row_count: int,
    current_row_count: int,
    cache_meta: Dict[str, Any],
    split_strategy: str,
    holdout_ratio: float,
    rand_seed: Any,
    data_path: Optional[str | Path] = None,
    rebuild_hint: bool = False,
) -> None:
    hint = " Set split_cache_force_rebuild=true to regenerate." if rebuild_hint else ""

    if cached_row_count > 0 and int(cached_row_count) != int(current_row_count):
        raise ValueError(
            f"split cache row_count mismatch for {cache_path}: "
            f"cached={cached_row_count}, current={current_row_count}.{hint}"
        )

    strategy_cfg = str(split_strategy or "random").strip().lower()
    strategy_cache = cache_meta.get("split_strategy")
    if strategy_cache is not None and str(strategy_cache).strip().lower() != strategy_cfg:
        raise ValueError(
            f"split cache strategy mismatch for {cache_path}: "
            f"cached={strategy_cache}, current={strategy_cfg}.{hint}"
        )

    holdout_cache = cache_meta.get("holdout_ratio")
    if holdout_cache is not None and not np.isclose(float(holdout_cache), float(holdout_ratio)):
        raise ValueError(
            f"split cache holdout_ratio mismatch for {cache_path}: "
            f"cached={holdout_cache}, current={holdout_ratio}.{hint}"
        )

    seed_cache = _normalize_seed(cache_meta.get("rand_seed"))
    seed_cfg = _normalize_seed(rand_seed)
    if seed_cache is not None and seed_cache != seed_cfg:
        raise ValueError(
            f"split cache rand_seed mismatch for {cache_path}: "
            f"cached={cache_meta.get('rand_seed')}, current={rand_seed}.{hint}"
        )

    data_path_cache = cache_meta.get("data_path")
    if data_path is not None and data_path_cache is not None:
        try:
            data_path_cfg = str(Path(str(data_path)).resolve())
        except Exception:
            data_path_cfg = str(data_path)
        try:
            data_path_cached = str(Path(str(data_path_cache)).resolve())
        except Exception:
            data_path_cached = str(data_path_cache)
        if data_path_cached != data_path_cfg:
            raise ValueError(
                f"split cache data_path mismatch for {cache_path}: "
                f"cached={data_path_cached}, current={data_path_cfg}.{hint}"
            )


__all__ = [
    "load_split_cache",
    "resolve_model_scoped_path",
    "validate_split_cache_metadata",
    "validate_split_indices",
    "write_split_cache",
]
