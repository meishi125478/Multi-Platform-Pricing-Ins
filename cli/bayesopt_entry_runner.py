"""
CLI entry point generated from BayesOpt_AutoPricing.ipynb so the workflow can
run non-interactively (e.g., via torchrun).

Example:
    python -m torch.distributed.run --standalone --nproc_per_node=2 \\
        ins_pricing/cli/BayesOpt_entry.py \\
        --config-json examples/config_template.json \\
        --model-keys ft --max-evals 50 --use-ft-ddp
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys


def _ensure_repo_root() -> None:
    if __package__ not in {None, ""}:
        return
    if importlib.util.find_spec("ins_pricing") is not None:
        return
    bootstrap_path = Path(__file__).resolve().parents[1] / "utils" / "bootstrap.py"
    spec = importlib.util.spec_from_file_location("ins_pricing.cli.utils.bootstrap", bootstrap_path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ensure_repo_root()


_ensure_repo_root()

import argparse
from typing import Any, Dict, List, Optional

# Use unified import resolver to eliminate nested try/except chains
from ins_pricing.cli.utils import bayesopt_runner_reporting as reporting_utils
from ins_pricing.cli.utils.bayesopt_runner_ui import (
    parse_bayesopt_args,
    plot_curves_for_model,
    plot_loss_curve_for_trainer,
)
from ins_pricing.cli.utils.import_resolver import resolve_imports
from ins_pricing.cli.utils.evaluation_context import TrainingContext
from ins_pricing.modelling.bayesopt.runtime import (
    BayesOptRunnerDeps,
    BayesOptRunnerHooks,
    run_bayesopt_entry_training,
)
from ins_pricing.modelling.bayesopt.utils.distributed_utils import DistributedUtils

# Resolve all imports from a single location
_imports = resolve_imports()

ropt = _imports.bayesopt
PLOT_MODEL_LABELS = _imports.PLOT_MODEL_LABELS
PYTORCH_TRAINERS = _imports.PYTORCH_TRAINERS
build_model_names = _imports.build_model_names
dedupe_preserve_order = _imports.dedupe_preserve_order
load_dataset = _imports.load_dataset
parse_model_pairs = _imports.parse_model_pairs
resolve_data_path = _imports.resolve_data_path
resolve_path = _imports.resolve_path
fingerprint_file = _imports.fingerprint_file
coerce_dataset_types = _imports.coerce_dataset_types
split_train_test = _imports.split_train_test

add_config_json_arg = _imports.add_config_json_arg
add_output_dir_arg = _imports.add_output_dir_arg
resolve_and_load_config = _imports.resolve_and_load_config
resolve_data_config = _imports.resolve_data_config
resolve_report_config = _imports.resolve_report_config
resolve_split_config = _imports.resolve_split_config
resolve_runtime_config = _imports.resolve_runtime_config
resolve_output_dirs = _imports.resolve_output_dirs

configure_run_logging = _imports.configure_run_logging
plot_loss_curve_common = _imports.plot_loss_curve

reporting_utils.configure_reporting_dependencies(
    plot_model_labels=PLOT_MODEL_LABELS,
    bootstrap_ci=_imports.bootstrap_ci,
    calibrate_predictions=_imports.calibrate_predictions,
    metrics_report=_imports.metrics_report,
    select_threshold=_imports.select_threshold,
    model_artifact_cls=_imports.ModelArtifact,
    model_registry_cls=_imports.ModelRegistry,
    drift_psi_report=_imports.drift_psi_report,
    group_metrics=_imports.group_metrics,
    report_payload_cls=_imports.ReportPayload,
    write_report=_imports.write_report,
)


def _parse_args() -> argparse.Namespace:
    return parse_bayesopt_args(
        add_config_json_arg=add_config_json_arg,
        add_output_dir_arg=add_output_dir_arg,
    )


def _plot_curves_for_model(model: ropt.BayesOptModel, trained_keys: List[str], cfg: Dict) -> None:
    plot_curves_for_model(
        model,
        trained_keys,
        cfg,
        dedupe_preserve_order=dedupe_preserve_order,
        plot_model_labels=PLOT_MODEL_LABELS,
        parse_model_pairs=parse_model_pairs,
    )


def _plot_loss_curve_for_trainer(model_name: str, trainer) -> None:
    plot_loss_curve_for_trainer(
        model_name,
        trainer,
        plot_loss_curve_common=plot_loss_curve_common,
    )


def _compute_psi_report(
    model: ropt.BayesOptModel,
    *,
    features: Optional[List[str]],
    bins: int,
    strategy: str,
) -> Any:
    return reporting_utils.compute_psi_report(
        model,
        features=features,
        bins=bins,
        strategy=strategy,
    )


def _evaluate_and_report(
    model: ropt.BayesOptModel,
    *,
    model_name: str,
    model_key: str,
    cfg: Dict[str, Any],
    data_path: Path,
    data_fingerprint: Dict[str, Any],
    report_output_dir: Optional[str],
    report_group_cols: Optional[List[str]],
    report_time_col: Optional[str],
    report_time_freq: str,
    report_time_ascending: bool,
    psi_report_df: Any,
    calibration_cfg: Dict[str, Any],
    threshold_cfg: Dict[str, Any],
    bootstrap_cfg: Dict[str, Any],
    register_model: bool,
    registry_path: Optional[str],
    registry_tags: Dict[str, Any],
    registry_status: str,
    run_id: str,
    config_sha: str,
) -> None:
    reporting_utils.evaluate_and_report(
        model,
        model_name=model_name,
        model_key=model_key,
        cfg=cfg,
        data_path=data_path,
        data_fingerprint=data_fingerprint,
        report_output_dir=report_output_dir,
        report_group_cols=report_group_cols,
        report_time_col=report_time_col,
        report_time_freq=report_time_freq,
        report_time_ascending=report_time_ascending,
        psi_report_df=psi_report_df,
        calibration_cfg=calibration_cfg,
        threshold_cfg=threshold_cfg,
        bootstrap_cfg=bootstrap_cfg,
        register_model=register_model,
        registry_path=registry_path,
        registry_tags=registry_tags,
        registry_status=registry_status,
        run_id=run_id,
        config_sha=config_sha,
    )


def _evaluate_with_context(
    model: ropt.BayesOptModel,
    ctx: Any,
) -> None:
    reporting_utils.evaluate_with_context(model, ctx)

def train_from_config(args: argparse.Namespace) -> None:
    deps = BayesOptRunnerDeps(
        ropt=ropt,
        pytorch_trainers=PYTORCH_TRAINERS,
        build_model_names=build_model_names,
        dedupe_preserve_order=dedupe_preserve_order,
        load_dataset=load_dataset,
        resolve_data_path=resolve_data_path,
        resolve_path=resolve_path,
        fingerprint_file=fingerprint_file,
        coerce_dataset_types=coerce_dataset_types,
        split_train_test=split_train_test,
        resolve_and_load_config=resolve_and_load_config,
        resolve_data_config=resolve_data_config,
        resolve_report_config=resolve_report_config,
        resolve_split_config=resolve_split_config,
        resolve_runtime_config=resolve_runtime_config,
        resolve_output_dirs=resolve_output_dirs,
    )
    hooks = BayesOptRunnerHooks(
        plot_loss_curve_for_trainer=_plot_loss_curve_for_trainer,
        compute_psi_report=_compute_psi_report,
        evaluate_and_report=_evaluate_and_report,
        plot_curves_for_model=_plot_curves_for_model,
    )
    run_bayesopt_entry_training(
        args,
        script_dir=Path(__file__).resolve().parents[1],
        deps=deps,
        hooks=hooks,
        training_context_from_env=TrainingContext.from_env,
    )

def main() -> None:
    if configure_run_logging:
        configure_run_logging(prefix="bayesopt_entry")
    args = _parse_args()
    try:
        train_from_config(args)
    finally:
        DistributedUtils.cleanup_ddp()


if __name__ == "__main__":
    main()
