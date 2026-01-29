from __future__ import annotations

import os
from typing import List, Optional

try:  # matplotlib is optional; avoid hard import failures in headless/minimal envs
    import matplotlib
    if os.name != "nt" and not os.environ.get("DISPLAY") and not os.environ.get("MPLBACKEND"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]
    _MPL_IMPORT_ERROR = exc

import numpy as np
import pandas as pd

from .utils import EPS, PlotUtils

try:
    from ...plotting import curves as plot_curves
    from ...plotting import diagnostics as plot_diagnostics
    from ...plotting.common import PlotStyle, finalize_figure
except Exception:  # pragma: no cover - optional for legacy imports
    try:  # best-effort for non-package imports
        from ins_pricing.plotting import curves as plot_curves
        from ins_pricing.plotting import diagnostics as plot_diagnostics
        from ins_pricing.plotting.common import PlotStyle, finalize_figure
    except Exception:  # pragma: no cover
        plot_curves = None
        plot_diagnostics = None
        PlotStyle = None
        finalize_figure = None


def _plot_skip(label: str) -> None:
    if _MPL_IMPORT_ERROR is not None:
        print(f"[Plot] Skip {label}: matplotlib unavailable ({_MPL_IMPORT_ERROR}).", flush=True)
    else:
        print(f"[Plot] Skip {label}: matplotlib unavailable.", flush=True)


class BayesOptPlottingMixin:
    def plot_oneway(
        self,
        n_bins=10,
        pred_col: Optional[str] = None,
        pred_label: Optional[str] = None,
        pred_weighted: Optional[bool] = None,
        plot_subdir: Optional[str] = None,
    ):
        if plt is None and plot_diagnostics is None:
            _plot_skip("oneway plot")
            return
        if pred_col is not None and pred_col not in self.train_data.columns:
            print(
                f"[Oneway] Missing prediction column '{pred_col}'; skip predicted line.",
                flush=True,
            )
            pred_col = None
        if pred_weighted is None and pred_col is not None:
            pred_weighted = pred_col.startswith("w_pred_")
        if pred_weighted is None:
            pred_weighted = False
        plot_subdir = plot_subdir.strip("/\\") if plot_subdir else "oneway"
        plot_prefix = f"{self.model_nme}/{plot_subdir}"

        def _safe_tag(value: str) -> str:
            return (
                value.strip()
                .replace(" ", "_")
                .replace("/", "_")
                .replace("\\", "_")
                .replace(":", "_")
            )

        if plot_diagnostics is None:
            for c in self.factor_nmes:
                fig = plt.figure(figsize=(7, 5))
                if c in self.cate_list:
                    group_col = c
                    plot_source = self.train_data
                else:
                    group_col = f'{c}_bins'
                    bins = pd.qcut(
                        self.train_data[c],
                        n_bins,
                        duplicates='drop'  # Drop duplicate quantiles to avoid errors.
                    )
                    plot_source = self.train_data.assign(**{group_col: bins})
                if pred_col is not None and pred_col in plot_source.columns:
                    if pred_weighted:
                        plot_source = plot_source.assign(
                            _pred_w=plot_source[pred_col]
                        )
                    else:
                        plot_source = plot_source.assign(
                            _pred_w=plot_source[pred_col] * plot_source[self.weight_nme]
                        )
                plot_data = plot_source.groupby(
                    [group_col], observed=True).sum(numeric_only=True)
                plot_data.reset_index(inplace=True)
                plot_data['act_v'] = plot_data['w_act'] / \
                    plot_data[self.weight_nme]
                if pred_col is not None and "_pred_w" in plot_data.columns:
                    plot_data["pred_v"] = plot_data["_pred_w"] / plot_data[self.weight_nme]
                ax = fig.add_subplot(111)
                ax.plot(plot_data.index, plot_data['act_v'],
                        label='Actual', color='red')
                if pred_col is not None and "pred_v" in plot_data.columns:
                    ax.plot(
                        plot_data.index,
                        plot_data["pred_v"],
                        label=pred_label or "Predicted",
                        color="tab:blue",
                    )
                ax.set_title(
                    'Analysis of  %s : Train Data' % group_col,
                    fontsize=8)
                plt.xticks(plot_data.index,
                           list(plot_data[group_col].astype(str)),
                           rotation=90)
                if len(list(plot_data[group_col].astype(str))) > 50:
                    plt.xticks(fontsize=3)
                else:
                    plt.xticks(fontsize=6)
                plt.yticks(fontsize=6)
                ax2 = ax.twinx()
                ax2.bar(plot_data.index,
                        plot_data[self.weight_nme],
                        alpha=0.5, color='seagreen')
                plt.yticks(fontsize=6)
                plt.margins(0.05)
                plt.subplots_adjust(wspace=0.3)
                if pred_col is not None and "pred_v" in plot_data.columns:
                    ax.legend(fontsize=6)
                pred_tag = _safe_tag(pred_label or pred_col) if pred_col else None
                if pred_tag:
                    filename = f'00_{self.model_nme}_{group_col}_oneway_{pred_tag}.png'
                else:
                    filename = f'00_{self.model_nme}_{group_col}_oneway.png'
                save_path = self._resolve_plot_path(plot_prefix, filename)
                plt.savefig(save_path, dpi=300)
                plt.close(fig)
            return

        if "w_act" not in self.train_data.columns:
            print("[Oneway] Missing w_act column; skip plotting.", flush=True)
            return

        for c in self.factor_nmes:
            is_cat = c in (self.cate_list or [])
            group_col = c if is_cat else f"{c}_bins"
            title = f"Analysis of {group_col} : Train Data"
            pred_tag = _safe_tag(pred_label or pred_col) if pred_col else None
            if pred_tag:
                filename = f"00_{self.model_nme}_{group_col}_oneway_{pred_tag}.png"
            else:
                filename = f"00_{self.model_nme}_{group_col}_oneway.png"
            save_path = self._resolve_plot_path(plot_prefix, filename)
            plot_diagnostics.plot_oneway(
                self.train_data,
                feature=c,
                weight_col=self.weight_nme,
                target_col="w_act",
                pred_col=pred_col,
                pred_weighted=pred_weighted,
                pred_label=pred_label,
                n_bins=n_bins,
                is_categorical=is_cat,
                title=title,
                save_path=save_path,
                show=False,
            )


    def _resolve_plot_path(self, subdir: Optional[str], filename: str) -> str:
        style = str(getattr(self.config, "plot_path_style", "nested") or "nested").strip().lower()
        if style in {"flat", "root"}:
            return self.output_manager.plot_path(filename)
        if subdir:
            return self.output_manager.plot_path(f"{subdir}/{filename}")
        return self.output_manager.plot_path(filename)


    def plot_lift(self, model_label, pred_nme, n_bins=10):
        if plt is None:
            _plot_skip("lift plot")
            return
        model_map = {
            'Xgboost': 'pred_xgb',
            'ResNet': 'pred_resn',
            'ResNetClassifier': 'pred_resn',
            'GLM': 'pred_glm',
            'GNN': 'pred_gnn',
        }
        if str(self.config.ft_role) == "model":
            model_map.update({
                'FTTransformer': 'pred_ft',
                'FTTransformerClassifier': 'pred_ft',
            })
        for k, v in model_map.items():
            if model_label.startswith(k):
                pred_nme = v
                break
        safe_label = (
            str(model_label)
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
        )
        plot_prefix = f"{self.model_nme}/lift"
        filename = f"01_{self.model_nme}_{safe_label}_lift.png"

        datasets = []
        for title, data in [
            ('Lift Chart on Train Data', self.train_data),
            ('Lift Chart on Test Data', self.test_data),
        ]:
            if 'w_act' not in data.columns or data['w_act'].isna().all():
                print(
                    f"[Lift] Missing labels for {title}; skip.",
                    flush=True,
                )
                continue
            datasets.append((title, data))

        if not datasets:
            print("[Lift] No labeled data available; skip plotting.", flush=True)
            return

        if plot_curves is None:
            fig = plt.figure(figsize=(11, 5))
            positions = [111] if len(datasets) == 1 else [121, 122]
            for pos, (title, data) in zip(positions, datasets):
                if pred_nme not in data.columns or f'w_{pred_nme}' not in data.columns:
                    print(
                        f"[Lift] Missing prediction columns in {title}; skip.",
                        flush=True,
                    )
                    continue
                lift_df = pd.DataFrame({
                    'pred': data[pred_nme].values,
                    'w_pred': data[f'w_{pred_nme}'].values,
                    'act': data['w_act'].values,
                    'weight': data[self.weight_nme].values
                })
                plot_data = PlotUtils.split_data(lift_df, 'pred', 'weight', n_bins)
                denom = np.maximum(plot_data['weight'], EPS)
                plot_data['exp_v'] = plot_data['w_pred'] / denom
                plot_data['act_v'] = plot_data['act'] / denom
                plot_data = plot_data.reset_index()

                ax = fig.add_subplot(pos)
                PlotUtils.plot_lift_ax(ax, plot_data, title)

            plt.subplots_adjust(wspace=0.3)
            save_path = self._resolve_plot_path(plot_prefix, filename)
            plt.savefig(save_path, dpi=300)
            plt.show()
            plt.close(fig)
            return

        style = PlotStyle() if PlotStyle else None
        fig, axes = plt.subplots(1, len(datasets), figsize=(11, 5))
        if len(datasets) == 1:
            axes = [axes]

        for ax, (title, data) in zip(axes, datasets):
            pred_vals = None
            if pred_nme in data.columns:
                pred_vals = data[pred_nme].values
            else:
                w_pred_col = f"w_{pred_nme}"
                if w_pred_col in data.columns:
                    denom = np.maximum(data[self.weight_nme].values, EPS)
                    pred_vals = data[w_pred_col].values / denom
            if pred_vals is None:
                print(
                    f"[Lift] Missing prediction columns in {title}; skip.",
                    flush=True,
                )
                continue

            plot_curves.plot_lift_curve(
                pred_vals,
                data['w_act'].values,
                data[self.weight_nme].values,
                n_bins=n_bins,
                title=title,
                pred_label="Predicted",
                act_label="Actual",
                weight_label="Earned Exposure",
                pred_weighted=False,
                actual_weighted=True,
                ax=ax,
                show=False,
                style=style,
            )

        plt.subplots_adjust(wspace=0.3)
        save_path = self._resolve_plot_path(plot_prefix, filename)
        if finalize_figure:
            finalize_figure(fig, save_path=save_path, show=True, style=style)
        else:
            plt.savefig(save_path, dpi=300)
            plt.show()
            plt.close(fig)

    # Double lift curve plot.

    def plot_dlift(self, model_comp: List[str] = ['xgb', 'resn'], n_bins: int = 10) -> None:
        # Compare two models across bins.
        # Args:
        #   model_comp: model keys to compare (e.g., ['xgb', 'resn']).
        #   n_bins: number of bins for lift curves.
        if plt is None:
            _plot_skip("double lift plot")
            return
        if len(model_comp) != 2:
            raise ValueError("`model_comp` must contain two models to compare.")

        model_name_map = {
            'xgb': 'Xgboost',
            'resn': 'ResNet',
            'glm': 'GLM',
            'gnn': 'GNN',
        }
        if str(self.config.ft_role) == "model":
            model_name_map['ft'] = 'FTTransformer'

        name1, name2 = model_comp
        if name1 not in model_name_map or name2 not in model_name_map:
            raise ValueError(f"Unsupported model key. Choose from {list(model_name_map.keys())}.")
        plot_prefix = f"{self.model_nme}/double_lift"
        filename = f"02_{self.model_nme}_dlift_{name1}_vs_{name2}.png"

        datasets = []
        for data_name, data in [('Train Data', self.train_data),
                                ('Test Data', self.test_data)]:
            if 'w_act' not in data.columns or data['w_act'].isna().all():
                print(
                    f"[Double Lift] Missing labels for {data_name}; skip.",
                    flush=True,
                )
                continue
            datasets.append((data_name, data))

        if not datasets:
            print("[Double Lift] No labeled data available; skip plotting.", flush=True)
            return

        if plot_curves is None:
            fig, axes = plt.subplots(1, len(datasets), figsize=(11, 5))
            if len(datasets) == 1:
                axes = [axes]

            for ax, (data_name, data) in zip(axes, datasets):
                pred1_col = f'w_pred_{name1}'
                pred2_col = f'w_pred_{name2}'

                if pred1_col not in data.columns or pred2_col not in data.columns:
                    print(
                        f"Warning: missing prediction columns {pred1_col} or {pred2_col} in {data_name}. Skip plot.")
                    continue

                lift_data = pd.DataFrame({
                    'pred1': data[pred1_col].values,
                    'pred2': data[pred2_col].values,
                    'diff_ly': data[pred1_col].values / np.maximum(data[pred2_col].values, EPS),
                    'act': data['w_act'].values,
                    'weight': data[self.weight_nme].values
                })
                plot_data = PlotUtils.split_data(
                    lift_data, 'diff_ly', 'weight', n_bins)
                denom = np.maximum(plot_data['act'], EPS)
                plot_data['exp_v1'] = plot_data['pred1'] / denom
                plot_data['exp_v2'] = plot_data['pred2'] / denom
                plot_data['act_v'] = plot_data['act'] / denom
                plot_data.reset_index(inplace=True)

                label1 = model_name_map[name1]
                label2 = model_name_map[name2]

                PlotUtils.plot_dlift_ax(
                    ax, plot_data, f'Double Lift Chart on {data_name}', label1, label2)

            plt.subplots_adjust(bottom=0.25, top=0.95, right=0.8, wspace=0.3)
            save_path = self._resolve_plot_path(plot_prefix, filename)
            plt.savefig(save_path, dpi=300)
            plt.show()
            plt.close(fig)
            return

        style = PlotStyle() if PlotStyle else None
        fig, axes = plt.subplots(1, len(datasets), figsize=(11, 5))
        if len(datasets) == 1:
            axes = [axes]

        label1 = model_name_map[name1]
        label2 = model_name_map[name2]

        for ax, (data_name, data) in zip(axes, datasets):
            weight_vals = data[self.weight_nme].values
            pred1 = None
            pred2 = None

            pred1_col = f"pred_{name1}"
            pred2_col = f"pred_{name2}"
            if pred1_col in data.columns:
                pred1 = data[pred1_col].values
            else:
                w_pred1_col = f"w_pred_{name1}"
                if w_pred1_col in data.columns:
                    pred1 = data[w_pred1_col].values / np.maximum(weight_vals, EPS)

            if pred2_col in data.columns:
                pred2 = data[pred2_col].values
            else:
                w_pred2_col = f"w_pred_{name2}"
                if w_pred2_col in data.columns:
                    pred2 = data[w_pred2_col].values / np.maximum(weight_vals, EPS)

            if pred1 is None or pred2 is None:
                print(
                    f"Warning: missing pred_{name1}/pred_{name2} or w_pred columns in {data_name}. Skip plot.")
                continue

            plot_curves.plot_double_lift_curve(
                pred1,
                pred2,
                data['w_act'].values,
                weight_vals,
                n_bins=n_bins,
                title=f"Double Lift Chart on {data_name}",
                label1=label1,
                label2=label2,
                pred1_weighted=False,
                pred2_weighted=False,
                actual_weighted=True,
                ax=ax,
                show=False,
                style=style,
            )

        plt.subplots_adjust(bottom=0.25, top=0.95, right=0.8, wspace=0.3)
        save_path = self._resolve_plot_path(plot_prefix, filename)
        if finalize_figure:
            finalize_figure(fig, save_path=save_path, show=True, style=style)
        else:
            plt.savefig(save_path, dpi=300)
            plt.show()
            plt.close(fig)

    # Conversion lift curve plot.

    def plot_conversion_lift(self, model_pred_col: str, n_bins: int = 20):
        if plt is None:
            _plot_skip("conversion lift plot")
            return
        if not self.binary_resp_nme:
            print("Error: `binary_resp_nme` not provided at BayesOptModel init; cannot plot conversion lift.")
            return

        if plot_curves is None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
            datasets = {
                'Train Data': self.train_data,
                'Test Data': self.test_data
            }

            for ax, (data_name, data) in zip(axes, datasets.items()):
                if model_pred_col not in data.columns:
                    print(f"Warning: missing prediction column '{model_pred_col}' in {data_name}. Skip plot.")
                    continue

                # Sort by model prediction and compute bins.
                plot_data = data.sort_values(by=model_pred_col).copy()
                plot_data['cum_weight'] = plot_data[self.weight_nme].cumsum()
                total_weight = plot_data[self.weight_nme].sum()

                if total_weight > EPS:
                    plot_data['bin'] = pd.cut(
                        plot_data['cum_weight'],
                        bins=n_bins,
                        labels=False,
                        right=False
                    )
                else:
                    plot_data['bin'] = 0

                # Aggregate by bins.
                lift_agg = plot_data.groupby('bin').agg(
                    total_weight=(self.weight_nme, 'sum'),
                    actual_conversions=(self.binary_resp_nme, 'sum'),
                    weighted_conversions=('w_binary_act', 'sum'),
                    avg_pred=(model_pred_col, 'mean')
                ).reset_index()

                # Compute conversion rate.
                lift_agg['conversion_rate'] = lift_agg['weighted_conversions'] / \
                    lift_agg['total_weight']

                # Compute overall average conversion rate.
                overall_conversion_rate = data['w_binary_act'].sum(
                ) / data[self.weight_nme].sum()
                ax.axhline(y=overall_conversion_rate, color='gray', linestyle='--',
                           label=f'Overall Avg Rate ({overall_conversion_rate:.2%})')

                ax.plot(lift_agg['bin'], lift_agg['conversion_rate'],
                        marker='o', linestyle='-', label='Actual Conversion Rate')
                ax.set_title(f'Conversion Rate Lift Chart on {data_name}')
                ax.set_xlabel(f'Model Score Decile (based on {model_pred_col})')
                ax.set_ylabel('Conversion Rate')
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend()

            plt.tight_layout()
            plt.show()
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        datasets = {
            'Train Data': self.train_data,
            'Test Data': self.test_data
        }

        for ax, (data_name, data) in zip(axes, datasets.items()):
            if model_pred_col not in data.columns:
                print(f"Warning: missing prediction column '{model_pred_col}' in {data_name}. Skip plot.")
                continue

            plot_curves.plot_conversion_lift(
                data[model_pred_col].values,
                data[self.binary_resp_nme].values,
                data[self.weight_nme].values,
                n_bins=n_bins,
                title=f'Conversion Rate Lift Chart on {data_name}',
                ax=ax,
                show=False,
            )

        plt.tight_layout()
        plt.show()

    # ========= Lightweight explainability: Permutation Importance =========
