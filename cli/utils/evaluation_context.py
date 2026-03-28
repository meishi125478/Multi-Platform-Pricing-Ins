"""Data classes for evaluation and reporting context.

These data classes group related parameters together to reduce function signatures
and improve code readability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ModelIdentity:
    """Identifies a model within the evaluation pipeline."""

    model_name: str
    model_key: str
    version: str
    task_type: str = "regression"

    @property
    def full_name(self) -> str:
        """Return the full model name with key."""
        return f"{self.model_name}/{self.model_key}"


@dataclass
class DataFingerprint:
    """Fingerprint information for data provenance tracking."""

    path: str
    sha256_prefix: str = ""
    size: str = ""
    mtime: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataFingerprint":
        """Create from a dictionary."""
        return cls(
            path=str(d.get("path", "")),
            sha256_prefix=str(d.get("sha256_prefix", "")),
            size=str(d.get("size", "")),
            mtime=str(d.get("mtime", "")),
        )

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "sha256_prefix": self.sha256_prefix,
            "size": self.size,
            "mtime": self.mtime,
        }


@dataclass
class CalibrationConfig:
    """Configuration for prediction calibration."""

    enable: bool = False
    method: str = "sigmoid"
    max_rows: Optional[int] = None
    seed: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "CalibrationConfig":
        """Create from a dictionary."""
        if not d:
            return cls()
        return cls(
            enable=bool(d.get("enable", False) or d.get("method")),
            method=str(d.get("method", "sigmoid")),
            max_rows=d.get("max_rows"),
            seed=d.get("seed"),
        )


@dataclass
class ThresholdConfig:
    """Configuration for classification threshold selection."""

    enable: bool = False
    metric: str = "f1"
    value: Optional[float] = None
    min_positive_rate: Optional[float] = None
    grid: int = 99
    max_rows: Optional[int] = None
    seed: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "ThresholdConfig":
        """Create from a dictionary."""
        if not d:
            return cls()
        return cls(
            enable=bool(
                d.get("enable", False)
                or d.get("metric")
                or d.get("value") is not None
            ),
            metric=str(d.get("metric", "f1")),
            value=float(d["value"]) if d.get("value") is not None else None,
            min_positive_rate=d.get("min_positive_rate"),
            grid=int(d.get("grid", 99)),
            max_rows=d.get("max_rows"),
            seed=d.get("seed"),
        )


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap confidence intervals."""

    enable: bool = False
    metrics: Optional[List[str]] = None
    n_samples: int = 200
    ci: float = 0.95
    seed: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "BootstrapConfig":
        """Create from a dictionary."""
        if not d:
            return cls()
        return cls(
            enable=bool(d.get("enable", False)),
            metrics=d.get("metrics"),
            n_samples=int(d.get("n_samples", 200)),
            ci=float(d.get("ci", 0.95)),
            seed=d.get("seed"),
        )


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    output_dir: Optional[str] = None
    group_cols: Optional[List[str]] = None
    time_col: Optional[str] = None
    time_freq: str = "M"
    time_ascending: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReportConfig":
        """Create from a dictionary."""
        return cls(
            output_dir=d.get("report_output_dir"),
            group_cols=d.get("report_group_cols"),
            time_col=d.get("report_time_col"),
            time_freq=str(d.get("report_time_freq", "M")),
            time_ascending=bool(d.get("report_time_ascending", True)),
        )


@dataclass
class RegistryConfig:
    """Configuration for model registry."""

    register: bool = False
    path: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    status: str = "candidate"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RegistryConfig":
        """Create from a dictionary."""
        return cls(
            register=bool(d.get("register_model", False)),
            path=d.get("registry_path"),
            tags=dict(d.get("registry_tags") or {}),
            status=str(d.get("registry_status", "candidate")),
        )


@dataclass
class MetricsResult:
    """Results from metrics computation."""

    metrics: Dict[str, float] = field(default_factory=dict)
    threshold_info: Optional[Dict[str, Any]] = None
    calibration_info: Optional[Dict[str, Any]] = None
    bootstrap_results: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class EvaluationContext:
    """Complete context for model evaluation and reporting.

    This groups all the parameters needed for _evaluate_and_report into a single
    object, reducing the function signature from 19+ parameters to 1.
    """

    # Model identification
    identity: ModelIdentity

    # Data info
    data_path: Path
    data_fingerprint: DataFingerprint
    config_sha: str
    run_id: str

    # Prediction column
    pred_col: str

    # Configuration
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)

    # Pre-computed reports
    psi_report_df: Optional[pd.DataFrame] = None

    # Full config dict (for artifact collection)
    cfg: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrainingContext:
    """Context for distributed training orchestration."""

    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    is_distributed: bool = False

    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return not self.is_distributed or self.rank == 0

    @classmethod
    def from_env(cls) -> "TrainingContext":
        """Create from environment variables."""
        import os

        def _safe_int_env(key: str, default: int) -> int:
            try:
                return int(os.environ.get(key, default))
            except (TypeError, ValueError):
                return default

        world_size = _safe_int_env("WORLD_SIZE", 1)
        rank = _safe_int_env("RANK", 0)
        local_rank = _safe_int_env("LOCAL_RANK", 0)

        return cls(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            is_distributed=world_size > 1,
        )
