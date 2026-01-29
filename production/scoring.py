from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd


def batch_score(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    data: pd.DataFrame,
    *,
    output_col: str = "prediction",
    batch_size: int = 10000,
    output_path: Optional[str | Path] = None,
    keep_input: bool = True,
) -> pd.DataFrame:
    """Batch scoring for large datasets."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    n_rows = len(data)
    prediction = np.empty(n_rows, dtype=float)
    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        chunk = data.iloc[start:end]
        pred = np.asarray(predict_fn(chunk)).reshape(-1)
        if pred.shape[0] != (end - start):
            raise ValueError("predict_fn output length must match batch size.")
        prediction[start:end] = pred
    result = data.copy() if keep_input else pd.DataFrame(index=data.index)
    result[output_col] = prediction
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() in {".parquet", ".pq"}:
            result.to_parquet(output_path, index=False)
        else:
            result.to_csv(output_path, index=False)
    return result
