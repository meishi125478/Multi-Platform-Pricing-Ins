"""Metrics computation, GPU management, device selection, CV utilities, and plotting.

This module contains:
- get_logger() for package-level logging
- MetricFactory for consistent metric computation
- GPUMemoryManager for CUDA memory management (imported from package utils)
- DeviceManager for device selection and model placement (imported from package utils)
- CVStrategyResolver for cross-validation strategy selection
- PlotUtils for lift chart plotting
- Backward compatibility wrappers for plotting functions
"""

from __future__ import annotations

import gc
import logging
import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, mean_tweedie_deviance
from sklearn.model_selection import KFold, GroupKFold, TimeSeriesSplit, StratifiedKFold

# Try to import plotting dependencies
try:
    import matplotlib
    if os.name != "nt" and not os.environ.get("DISPLAY") and not os.environ.get("MPLBACKEND"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:
    matplotlib = None
    plt = None
    _MPL_IMPORT_ERROR = exc

try:
    from ....plotting import curves as plot_curves_common
except Exception:
    try:
        from ins_pricing.plotting import curves as plot_curves_common
    except Exception:
        plot_curves_common = None

from .constants import EPS

# Import DeviceManager and GPUMemoryManager from package-level utils
# (Eliminates ~230 lines of code duplication)
from ins_pricing.utils import DeviceManager, GPUMemoryManager
from .io_utils import IOUtils


# =============================================================================
# Logging System
# =============================================================================

@lru_cache(maxsize=1)
def _get_package_logger() -> logging.Logger:
    """Get or create the package-level logger with consistent formatting."""
    logger = logging.getLogger("ins_pricing")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(levelname)s][%(name)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # Default to INFO, can be changed via environment variable
        level = os.environ.get("INS_PRICING_LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, level, logging.INFO))
    return logger


def get_logger(name: str = "ins_pricing") -> logging.Logger:
    """Get a logger with the given name, inheriting package-level settings.

    Args:
        name: Logger name, typically module name like 'ins_pricing.trainer'

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger("ins_pricing.trainer.ft")
        >>> logger.info("Training started")
    """
    _get_package_logger()
    return logging.getLogger(name)


# =============================================================================
# Metric Computation Factory
# =============================================================================

class MetricFactory:
    """Factory for computing evaluation metrics consistently across all trainers.

    This class centralizes metric computation logic that was previously duplicated
    across FTTrainer, ResNetTrainer, GNNTrainer, XGBTrainer, and GLMTrainer.

    Example:
        >>> factory = MetricFactory(task_type='regression', tweedie_power=1.5)
        >>> score = factory.compute(y_true, y_pred, sample_weight)
    """

    def __init__(
        self,
        task_type: str = "regression",
        tweedie_power: float = 1.5,
        loss_name: str = "tweedie",
        clip_min: float = 1e-8,
        clip_max: float = 1 - 1e-8,
    ):
        """Initialize the metric factory.

        Args:
            task_type: Either 'regression' or 'classification'
            tweedie_power: Power parameter for Tweedie deviance (1.0-2.0)
            loss_name: Regression loss name ('tweedie', 'poisson', 'gamma', 'mse', 'mae')
            clip_min: Minimum value for clipping predictions
            clip_max: Maximum value for clipping predictions (for classification)
        """
        self.task_type = task_type
        self.tweedie_power = tweedie_power
        self.loss_name = loss_name
        self.clip_min = clip_min
        self.clip_max = clip_max

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """Compute the appropriate metric based on task type.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            sample_weight: Optional sample weights

        Returns:
            Computed metric value (lower is better)
        """
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)

        if self.task_type == "classification":
            y_pred_clipped = np.clip(y_pred, self.clip_min, self.clip_max)
            return float(log_loss(y_true, y_pred_clipped, sample_weight=sample_weight))

        loss_name = str(self.loss_name or "tweedie").strip().lower()
        if loss_name in {"mse", "mae"}:
            if loss_name == "mse":
                return float(mean_squared_error(
                    y_true, y_pred, sample_weight=sample_weight))
            return float(mean_absolute_error(
                y_true, y_pred, sample_weight=sample_weight))

        y_pred_safe = np.maximum(y_pred, self.clip_min)
        power = self.tweedie_power
        if loss_name == "poisson":
            power = 1.0
        elif loss_name == "gamma":
            power = 2.0
        return float(mean_tweedie_deviance(
            y_true,
            y_pred_safe,
            sample_weight=sample_weight,
            power=power,
        ))

    def update_power(self, power: float) -> None:
        """Update the Tweedie power parameter.

        Args:
            power: New power value (1.0-2.0)
        """
        self.tweedie_power = power


# =============================================================================
# GPU Memory Manager and Device Manager
# =============================================================================
# NOTE: These classes are imported from ins_pricing.utils (see top of file)
# This eliminates ~230 lines of duplicate code while maintaining backward compatibility


# =============================================================================
# Cross-Validation Strategy Resolver
# =============================================================================

class CVStrategyResolver:
    """Resolver for cross-validation splitting strategies.

    This class consolidates CV strategy resolution logic that was previously
    duplicated across trainer_base.py and trainer_ft.py.

    Supported strategies:
        - 'random': Standard KFold
        - 'stratified': StratifiedKFold (for classification)
        - 'group': GroupKFold (requires group column)
        - 'time': TimeSeriesSplit (requires time column)

    Example:
        >>> resolver = CVStrategyResolver(
        ...     strategy='group',
        ...     n_splits=5,
        ...     group_col='policy_id',
        ...     data=train_df,
        ... )
        >>> splitter, groups = resolver.get_splitter()
        >>> for train_idx, val_idx in splitter.split(X, y, groups):
        ...     pass
    """

    VALID_STRATEGIES = {"random", "stratified", "group", "grouped", "time", "timeseries", "temporal"}

    def __init__(
        self,
        strategy: str = "random",
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        group_col: Optional[str] = None,
        time_col: Optional[str] = None,
        time_ascending: bool = True,
        data: Optional[pd.DataFrame] = None,
    ):
        """Initialize the CV strategy resolver.

        Args:
            strategy: CV strategy name
            n_splits: Number of CV folds
            shuffle: Whether to shuffle for random/stratified
            random_state: Random seed for reproducibility
            group_col: Column name for group-based splitting
            time_col: Column name for time-based splitting
            time_ascending: Sort order for time-based splitting
            data: DataFrame containing group/time columns
        """
        self.strategy = strategy.strip().lower()
        self.n_splits = max(2, int(n_splits))
        self.shuffle = shuffle
        self.random_state = random_state
        self.group_col = group_col
        self.time_col = time_col
        self.time_ascending = time_ascending
        self.data = data

        if self.strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{strategy}'. "
                f"Valid options: {sorted(self.VALID_STRATEGIES)}"
            )

    def get_splitter(self) -> Tuple[Any, Optional[pd.Series]]:
        """Get the appropriate splitter and groups.

        Returns:
            Tuple of (splitter, groups) where groups may be None

        Raises:
            ValueError: If required columns are missing
        """
        if self.strategy in {"group", "grouped"}:
            return self._get_group_splitter()
        elif self.strategy in {"time", "timeseries", "temporal"}:
            return self._get_time_splitter()
        elif self.strategy == "stratified":
            return self._get_stratified_splitter()
        else:
            return self._get_random_splitter()

    def _get_random_splitter(self) -> Tuple[KFold, None]:
        """Get a random KFold splitter."""
        splitter = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state if self.shuffle else None,
        )
        return splitter, None

    def _get_stratified_splitter(self) -> Tuple[StratifiedKFold, None]:
        """Get a stratified KFold splitter."""
        splitter = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state if self.shuffle else None,
        )
        return splitter, None

    def _get_group_splitter(self) -> Tuple[GroupKFold, pd.Series]:
        """Get a group-based KFold splitter."""
        if not self.group_col:
            raise ValueError("group_col is required for group strategy")
        if self.data is None:
            raise ValueError("data DataFrame is required for group strategy")
        if self.group_col not in self.data.columns:
            raise KeyError(f"group_col '{self.group_col}' not found in data")

        groups = self.data[self.group_col]
        splitter = GroupKFold(n_splits=self.n_splits)
        return splitter, groups

    def _get_time_splitter(self) -> Tuple[Any, None]:
        """Get a time-series splitter."""
        if not self.time_col:
            raise ValueError("time_col is required for time strategy")
        if self.data is None:
            raise ValueError("data DataFrame is required for time strategy")
        if self.time_col not in self.data.columns:
            raise KeyError(f"time_col '{self.time_col}' not found in data")

        splitter = TimeSeriesSplit(n_splits=self.n_splits)

        # Create an ordered wrapper that sorts by time column
        order_index = self.data[self.time_col].sort_values(
            ascending=self.time_ascending
        ).index
        order = self.data.index.get_indexer(order_index)

        return _OrderedSplitter(splitter, order), None


class _OrderedSplitter:
    """Wrapper for splitters that need to respect a specific ordering."""

    def __init__(self, base_splitter, order: np.ndarray):
        self.base_splitter = base_splitter
        self.order = order

    def split(self, X, y=None, groups=None):
        """Split with ordering applied."""
        n = len(X)
        X_ordered = np.arange(n)[self.order]
        for train_idx, val_idx in self.base_splitter.split(X_ordered):
            yield self.order[train_idx], self.order[val_idx]

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.base_splitter.get_n_splits()


# =============================================================================
# Plot Utils
# =============================================================================

def _plot_skip(label: str) -> None:
    """Print message when plot is skipped due to missing matplotlib."""
    if _MPL_IMPORT_ERROR is not None:
        print(f"[Plot] Skip {label}: matplotlib unavailable ({_MPL_IMPORT_ERROR}).", flush=True)
    else:
        print(f"[Plot] Skip {label}: matplotlib unavailable.", flush=True)


class PlotUtils:
    """Plotting utilities for lift charts."""

    @staticmethod
    def split_data(data: pd.DataFrame, col_nme: str, wgt_nme: str, n_bins: int = 10) -> pd.DataFrame:
        """Split data into bins by cumulative weight."""
        data_sorted = data.sort_values(by=col_nme, ascending=True).copy()
        data_sorted['cum_weight'] = data_sorted[wgt_nme].cumsum()
        w_sum = data_sorted[wgt_nme].sum()
        if w_sum <= EPS:
            data_sorted['bins'] = 0
        else:
            data_sorted['bins'] = np.floor(
                data_sorted['cum_weight'] * float(n_bins) / w_sum
            )
            data_sorted.loc[(data_sorted['bins'] == n_bins),
                            'bins'] = n_bins - 1
        return data_sorted.groupby(['bins'], observed=True).sum(numeric_only=True)

    @staticmethod
    def plot_lift_ax(ax, plot_data, title, pred_label='Predicted', act_label='Actual', weight_label='Earned Exposure'):
        """Plot lift chart on given axes."""
        ax.plot(plot_data.index, plot_data['act_v'],
                label=act_label, color='red')
        ax.plot(plot_data.index, plot_data['exp_v'],
                label=pred_label, color='blue')
        ax.set_title(title, fontsize=8)
        ax.set_xticks(plot_data.index)
        ax.set_xticklabels(plot_data.index, rotation=90, fontsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.legend(loc='upper left', fontsize=5, frameon=False)
        ax.margins(0.05)
        ax2 = ax.twinx()
        ax2.bar(plot_data.index, plot_data['weight'],
                alpha=0.5, color='seagreen',
                label=weight_label)
        ax2.tick_params(axis='y', labelsize=6)
        ax2.legend(loc='upper right', fontsize=5, frameon=False)

    @staticmethod
    def plot_dlift_ax(ax, plot_data, title, label1, label2, act_label='Actual', weight_label='Earned Exposure'):
        """Plot double lift chart on given axes."""
        ax.plot(plot_data.index, plot_data['act_v'],
                label=act_label, color='red')
        ax.plot(plot_data.index, plot_data['exp_v1'],
                label=label1, color='blue')
        ax.plot(plot_data.index, plot_data['exp_v2'],
                label=label2, color='black')
        ax.set_title(title, fontsize=8)
        ax.set_xticks(plot_data.index)
        ax.set_xticklabels(plot_data.index, rotation=90, fontsize=6)
        ax.set_xlabel(f'{label1} / {label2}', fontsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.legend(loc='upper left', fontsize=5, frameon=False)
        ax.margins(0.1)
        ax2 = ax.twinx()
        ax2.bar(plot_data.index, plot_data['weight'],
                alpha=0.5, color='seagreen',
                label=weight_label)
        ax2.tick_params(axis='y', labelsize=6)
        ax2.legend(loc='upper right', fontsize=5, frameon=False)

    @staticmethod
    def plot_lift_list(pred_model, w_pred_list, w_act_list,
                       weight_list, tgt_nme, n_bins: int = 10,
                       fig_nme: str = 'Lift Chart'):
        """Plot lift chart for model predictions."""
        if plot_curves_common is not None:
            save_path = os.path.join(
                os.getcwd(), 'plot', f'05_{tgt_nme}_{fig_nme}.png')
            plot_curves_common.plot_lift_curve(
                pred_model,
                w_act_list,
                weight_list,
                n_bins=n_bins,
                title=f'Lift Chart of {tgt_nme}',
                pred_label='Predicted',
                act_label='Actual',
                weight_label='Earned Exposure',
                pred_weighted=False,
                actual_weighted=True,
                save_path=save_path,
                show=False,
            )
            return
        if plt is None:
            _plot_skip("lift plot")
            return
        lift_data = pd.DataFrame({
            'pred': pred_model,
            'w_pred': w_pred_list,
            'act': w_act_list,
            'weight': weight_list
        })
        plot_data = PlotUtils.split_data(lift_data, 'pred', 'weight', n_bins)
        plot_data['exp_v'] = plot_data['w_pred'] / plot_data['weight']
        plot_data['act_v'] = plot_data['act'] / plot_data['weight']
        plot_data.reset_index(inplace=True)

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)
        PlotUtils.plot_lift_ax(ax, plot_data, f'Lift Chart of {tgt_nme}')
        plt.subplots_adjust(wspace=0.3)

        save_path = os.path.join(
            os.getcwd(), 'plot', f'05_{tgt_nme}_{fig_nme}.png')
        IOUtils.ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

    @staticmethod
    def plot_dlift_list(pred_model_1, pred_model_2,
                        model_nme_1, model_nme_2,
                        tgt_nme,
                        w_list, w_act_list, n_bins: int = 10,
                        fig_nme: str = 'Double Lift Chart'):
        """Plot double lift chart comparing two models."""
        if plot_curves_common is not None:
            save_path = os.path.join(
                os.getcwd(), 'plot', f'06_{tgt_nme}_{fig_nme}.png')
            plot_curves_common.plot_double_lift_curve(
                pred_model_1,
                pred_model_2,
                w_act_list,
                w_list,
                n_bins=n_bins,
                title=f'Double Lift Chart of {tgt_nme}',
                label1=model_nme_1,
                label2=model_nme_2,
                pred1_weighted=False,
                pred2_weighted=False,
                actual_weighted=True,
                save_path=save_path,
                show=False,
            )
            return
        if plt is None:
            _plot_skip("double lift plot")
            return
        lift_data = pd.DataFrame({
            'pred1': pred_model_1,
            'pred2': pred_model_2,
            'act': w_act_list,
            'weight': w_list
        })
        lift_data['diff_ly'] = lift_data['pred1'] / lift_data['pred2']
        lift_data['w_pred1'] = lift_data['pred1'] * lift_data['weight']
        lift_data['w_pred2'] = lift_data['pred2'] * lift_data['weight']
        plot_data = PlotUtils.split_data(
            lift_data, 'diff_ly', 'weight', n_bins)
        plot_data['exp_v1'] = plot_data['w_pred1'] / plot_data['act']
        plot_data['exp_v2'] = plot_data['w_pred2'] / plot_data['act']
        plot_data['act_v'] = plot_data['act']/plot_data['act']
        plot_data.reset_index(inplace=True)

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)
        PlotUtils.plot_dlift_ax(
            ax, plot_data, f'Double Lift Chart of {tgt_nme}', model_nme_1, model_nme_2)
        plt.subplots_adjust(bottom=0.25, top=0.95, right=0.8)

        save_path = os.path.join(
            os.getcwd(), 'plot', f'06_{tgt_nme}_{fig_nme}.png')
        IOUtils.ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)


# =============================================================================
# Backward Compatibility Wrappers
# =============================================================================

def split_data(data, col_nme, wgt_nme, n_bins=10):
    """Legacy function wrapper for PlotUtils.split_data()."""
    return PlotUtils.split_data(data, col_nme, wgt_nme, n_bins)


def plot_lift_list(pred_model, w_pred_list, w_act_list,
                   weight_list, tgt_nme, n_bins=10,
                   fig_nme='Lift Chart'):
    """Legacy function wrapper for PlotUtils.plot_lift_list()."""
    return PlotUtils.plot_lift_list(pred_model, w_pred_list, w_act_list,
                                    weight_list, tgt_nme, n_bins, fig_nme)


def plot_dlift_list(pred_model_1, pred_model_2,
                    model_nme_1, model_nme_2,
                    tgt_nme,
                    w_list, w_act_list, n_bins=10,
                    fig_nme='Double Lift Chart'):
    """Legacy function wrapper for PlotUtils.plot_dlift_list()."""
    return PlotUtils.plot_dlift_list(pred_model_1, pred_model_2,
                                     model_nme_1, model_nme_2,
                                     tgt_nme, w_list, w_act_list,
                                     n_bins, fig_nme)
