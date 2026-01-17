"""Unified import resolver for CLI modules.

This module provides a single source of truth for all import fallback chains,
eliminating the need for nested try/except blocks in multiple CLI files.

Usage:
    from ins_pricing.cli.utils.import_resolver import resolve_imports
    imports = resolve_imports()
    ropt = imports.bayesopt
    PLOT_MODEL_LABELS = imports.PLOT_MODEL_LABELS
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type


@dataclass
class ResolvedImports:
    """Container for resolved imports from the bayesopt ecosystem."""

    # Core bayesopt module
    bayesopt: Any = None

    # CLI common utilities
    PLOT_MODEL_LABELS: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    PYTORCH_TRAINERS: List[str] = field(default_factory=list)
    build_model_names: Optional[Callable] = None
    dedupe_preserve_order: Optional[Callable] = None
    load_dataset: Optional[Callable] = None
    parse_model_pairs: Optional[Callable] = None
    resolve_data_path: Optional[Callable] = None
    resolve_path: Optional[Callable] = None
    fingerprint_file: Optional[Callable] = None
    coerce_dataset_types: Optional[Callable] = None
    split_train_test: Optional[Callable] = None

    # CLI config utilities
    add_config_json_arg: Optional[Callable] = None
    add_output_dir_arg: Optional[Callable] = None
    resolve_and_load_config: Optional[Callable] = None
    resolve_data_config: Optional[Callable] = None
    resolve_report_config: Optional[Callable] = None
    resolve_split_config: Optional[Callable] = None
    resolve_runtime_config: Optional[Callable] = None
    resolve_output_dirs: Optional[Callable] = None

    # Evaluation utilities
    bootstrap_ci: Optional[Callable] = None
    calibrate_predictions: Optional[Callable] = None
    metrics_report: Optional[Callable] = None
    select_threshold: Optional[Callable] = None

    # Governance and reporting
    ModelArtifact: Optional[Type] = None
    ModelRegistry: Optional[Type] = None
    drift_psi_report: Optional[Callable] = None
    group_metrics: Optional[Callable] = None
    ReportPayload: Optional[Type] = None
    write_report: Optional[Callable] = None

    # Logging
    configure_run_logging: Optional[Callable] = None

    # Plotting
    plot_loss_curve: Optional[Callable] = None


def _debug_imports_enabled() -> bool:
    value = os.environ.get("BAYESOPT_DEBUG_IMPORTS")
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _try_import(module_path: str, attr_name: Optional[str] = None) -> Optional[Any]:
    """Attempt to import a module or attribute, returning None on failure."""
    try:
        module = importlib.import_module(module_path)
        if attr_name:
            result = getattr(module, attr_name, None)
        else:
            result = module
        if _debug_imports_enabled():
            origin = getattr(module, "__file__", None)
            origin = origin or getattr(module, "__path__", None)
            print(
                f"[BAYESOPT_DEBUG_IMPORTS] imported {module_path}"
                f"{'::' + attr_name if attr_name else ''} from {origin}",
                file=sys.stderr,
                flush=True,
            )
        return result
    except Exception as exc:
        if _debug_imports_enabled():
            print(
                f"[BAYESOPT_DEBUG_IMPORTS] failed import {module_path}"
                f"{'::' + attr_name if attr_name else ''}: {exc.__class__.__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
        return None


def _try_import_from_paths(
    paths: List[str],
    attr_name: Optional[str] = None
) -> Optional[Any]:
    """Try importing from multiple module paths, return first success."""
    for path in paths:
        result = _try_import(path, attr_name)
        if result is not None:
            return result
    return None


def _resolve_bayesopt() -> Optional[Any]:
    """Resolve the bayesopt module from multiple possible locations."""
    paths = [
        "ins_pricing.modelling.core.bayesopt",
        "bayesopt",
        "BayesOpt",
    ]
    return _try_import_from_paths(paths)


def _resolve_cli_common() -> Dict[str, Any]:
    """Resolve CLI common utilities."""
    paths = [
        "ins_pricing.cli.utils.cli_common",
        "cli.utils.cli_common",
        "utils.cli_common",
    ]

    attrs = [
        "PLOT_MODEL_LABELS",
        "PYTORCH_TRAINERS",
        "build_model_names",
        "dedupe_preserve_order",
        "load_dataset",
        "parse_model_pairs",
        "resolve_data_path",
        "resolve_path",
        "fingerprint_file",
        "coerce_dataset_types",
        "split_train_test",
    ]

    results = {}
    for path in paths:
        module = _try_import(path)
        if module is not None:
            for attr in attrs:
                if attr not in results or results[attr] is None:
                    results[attr] = getattr(module, attr, None)
            # If we got most attributes, break
            if sum(1 for v in results.values() if v is not None) >= len(attrs) // 2:
                break

    return results


def _resolve_cli_config() -> Dict[str, Any]:
    """Resolve CLI config utilities."""
    paths = [
        "ins_pricing.cli.utils.cli_config",
        "cli.utils.cli_config",
        "utils.cli_config",
    ]

    attrs = [
        "add_config_json_arg",
        "add_output_dir_arg",
        "resolve_and_load_config",
        "resolve_data_config",
        "resolve_report_config",
        "resolve_split_config",
        "resolve_runtime_config",
        "resolve_output_dirs",
    ]

    results = {}
    for path in paths:
        module = _try_import(path)
        if module is not None:
            for attr in attrs:
                if attr not in results or results[attr] is None:
                    results[attr] = getattr(module, attr, None)
            if sum(1 for v in results.values() if v is not None) >= len(attrs) // 2:
                break

    return results


def _resolve_evaluation() -> Dict[str, Any]:
    """Resolve evaluation utilities."""
    paths = [
        "ins_pricing.modelling.core.evaluation",
        "evaluation",
    ]

    results = {}
    for path in paths:
        module = _try_import(path)
        if module is not None:
            results["bootstrap_ci"] = getattr(module, "bootstrap_ci", None)
            results["calibrate_predictions"] = getattr(module, "calibrate_predictions", None)
            results["metrics_report"] = getattr(module, "metrics_report", None)
            results["select_threshold"] = getattr(module, "select_threshold", None)
            if any(v is not None for v in results.values()):
                break

    return results


def _resolve_governance() -> Dict[str, Any]:
    """Resolve governance and reporting utilities."""
    results = {}

    # ModelRegistry and ModelArtifact
    registry_paths = [
        "ins_pricing.governance.registry",
    ]
    for path in registry_paths:
        module = _try_import(path)
        if module is not None:
            results["ModelArtifact"] = getattr(module, "ModelArtifact", None)
            results["ModelRegistry"] = getattr(module, "ModelRegistry", None)
            break

    # PSI report
    psi_paths = [
        "ins_pricing.production",
    ]
    for path in psi_paths:
        module = _try_import(path)
        if module is not None:
            results["drift_psi_report"] = getattr(module, "psi_report", None)
            break

    # Group metrics
    monitoring_paths = [
        "ins_pricing.production.monitoring",
    ]
    for path in monitoring_paths:
        module = _try_import(path)
        if module is not None:
            results["group_metrics"] = getattr(module, "group_metrics", None)
            break

    # Report builder
    report_paths = [
        "ins_pricing.reporting.report_builder",
    ]
    for path in report_paths:
        module = _try_import(path)
        if module is not None:
            results["ReportPayload"] = getattr(module, "ReportPayload", None)
            results["write_report"] = getattr(module, "write_report", None)
            break

    return results


def _resolve_logging() -> Dict[str, Any]:
    """Resolve logging utilities."""
    paths = [
        "ins_pricing.cli.utils.run_logging",
        "cli.utils.run_logging",
        "utils.run_logging",
    ]

    results = {}
    for path in paths:
        module = _try_import(path)
        if module is not None:
            results["configure_run_logging"] = getattr(module, "configure_run_logging", None)
            break

    return results


def _resolve_plotting() -> Dict[str, Any]:
    """Resolve plotting utilities."""
    paths = [
        "ins_pricing.modelling.plotting.diagnostics",
        "ins_pricing.plotting.diagnostics",
    ]

    results = {}
    for path in paths:
        module = _try_import(path)
        if module is not None:
            results["plot_loss_curve"] = getattr(module, "plot_loss_curve", None)
            break

    return results


def resolve_imports() -> ResolvedImports:
    """Resolve all imports from the bayesopt ecosystem.

    This function attempts to import modules from multiple possible locations,
    handling the various ways the package might be installed or run.

    Returns:
        ResolvedImports object containing all resolved imports.
    """
    imports = ResolvedImports()

    # Resolve bayesopt core
    imports.bayesopt = _resolve_bayesopt()

    # Resolve CLI common utilities
    cli_common = _resolve_cli_common()
    imports.PLOT_MODEL_LABELS = cli_common.get("PLOT_MODEL_LABELS", {})
    imports.PYTORCH_TRAINERS = cli_common.get("PYTORCH_TRAINERS", [])
    imports.build_model_names = cli_common.get("build_model_names")
    imports.dedupe_preserve_order = cli_common.get("dedupe_preserve_order")
    imports.load_dataset = cli_common.get("load_dataset")
    imports.parse_model_pairs = cli_common.get("parse_model_pairs")
    imports.resolve_data_path = cli_common.get("resolve_data_path")
    imports.resolve_path = cli_common.get("resolve_path")
    imports.fingerprint_file = cli_common.get("fingerprint_file")
    imports.coerce_dataset_types = cli_common.get("coerce_dataset_types")
    imports.split_train_test = cli_common.get("split_train_test")

    # Resolve CLI config utilities
    cli_config = _resolve_cli_config()
    imports.add_config_json_arg = cli_config.get("add_config_json_arg")
    imports.add_output_dir_arg = cli_config.get("add_output_dir_arg")
    imports.resolve_and_load_config = cli_config.get("resolve_and_load_config")
    imports.resolve_data_config = cli_config.get("resolve_data_config")
    imports.resolve_report_config = cli_config.get("resolve_report_config")
    imports.resolve_split_config = cli_config.get("resolve_split_config")
    imports.resolve_runtime_config = cli_config.get("resolve_runtime_config")
    imports.resolve_output_dirs = cli_config.get("resolve_output_dirs")

    # Resolve evaluation utilities
    evaluation = _resolve_evaluation()
    imports.bootstrap_ci = evaluation.get("bootstrap_ci")
    imports.calibrate_predictions = evaluation.get("calibrate_predictions")
    imports.metrics_report = evaluation.get("metrics_report")
    imports.select_threshold = evaluation.get("select_threshold")

    # Resolve governance and reporting
    governance = _resolve_governance()
    imports.ModelArtifact = governance.get("ModelArtifact")
    imports.ModelRegistry = governance.get("ModelRegistry")
    imports.drift_psi_report = governance.get("drift_psi_report")
    imports.group_metrics = governance.get("group_metrics")
    imports.ReportPayload = governance.get("ReportPayload")
    imports.write_report = governance.get("write_report")

    # Resolve logging
    logging_utils = _resolve_logging()
    imports.configure_run_logging = logging_utils.get("configure_run_logging")

    # Resolve plotting
    plotting = _resolve_plotting()
    imports.plot_loss_curve = plotting.get("plot_loss_curve")

    return imports


# Convenience function for backward compatibility
def setup_sys_path() -> None:
    """Ensure the repository root is in sys.path for imports."""
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
