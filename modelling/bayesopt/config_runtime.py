from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ins_pricing.utils.io import IOUtils
from ins_pricing.utils import get_logger, log_print

_logger = get_logger("ins_pricing.modelling.bayesopt.config_runtime")


def _log(*args, **kwargs) -> None:
    log_print(_logger, *args, **kwargs)
@dataclass
class PreprocessArtifacts:
    factor_nmes: List[str]
    cate_list: List[str]
    num_features: List[str]
    var_nmes: List[str]
    cat_categories: Dict[str, List[Any]]
    ohe_feature_names: List[str]
    dummy_columns: List[str]
    numeric_scalers: Dict[str, Dict[str, float]]
    weight_nme: str
    resp_nme: str
    binary_resp_nme: Optional[str] = None
    drop_first: bool = True
    oht_sparse_csr: bool = True


class OutputManager:
    # Centralize output paths for plots, results, and models.

    def __init__(self, root: Optional[str] = None, model_name: str = "model") -> None:
        self.root = Path(root or os.getcwd())
        self.model_name = model_name
        self.plot_dir = self.root / 'plot'
        self.result_dir = self.root / 'Results'
        self.model_dir = self.root / 'model'

    def _prepare(self, path: Path) -> str:
        IOUtils.ensure_parent_dir(str(path))
        return str(path)

    def plot_path(self, filename: str) -> str:
        return self._prepare(self.plot_dir / filename)

    def result_path(self, filename: str) -> str:
        return self._prepare(self.result_dir / filename)

    def model_path(self, filename: str) -> str:
        return self._prepare(self.model_dir / filename)


class VersionManager:
    """Lightweight versioning: save config and best-params snapshots for traceability."""

    def __init__(self, output: OutputManager) -> None:
        self.output = output
        self.version_dir = Path(self.output.result_dir) / "versions"
        IOUtils.ensure_parent_dir(str(self.version_dir))

    def save(self, tag: str, payload: Dict[str, Any]) -> str:
        safe_tag = tag.replace(" ", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.version_dir / f"{ts}_{safe_tag}.json"
        IOUtils.ensure_parent_dir(str(path))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        _log(f"[Version] Saved snapshot: {path}")
        return str(path)

    def load_latest(self, tag: str) -> Optional[Dict[str, Any]]:
        """Load the latest snapshot for a tag (sorted by timestamp prefix)."""
        safe_tag = tag.replace(" ", "_")
        pattern = f"*_{safe_tag}.json"
        candidates = sorted(self.version_dir.glob(pattern))
        if not candidates:
            return None
        path = candidates[-1]
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            _log(f"[Version] Failed to load snapshot {path}: {exc}")
            return None


