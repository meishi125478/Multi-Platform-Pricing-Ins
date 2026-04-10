from __future__ import annotations

import os
import zlib
from typing import Dict, List, Optional, Tuple

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

from ins_pricing.utils import EPS, get_logger, log_print

_logger = get_logger("ins_pricing.modelling.bayesopt.model_plotting_mixin")


def _log(*args, **kwargs) -> None:
    log_print(_logger, *args, **kwargs)

try:
    from ins_pricing.modelling.plotting import curves as plot_curves
    from ins_pricing.modelling.plotting import diagnostics as plot_diagnostics
    from ins_pricing.modelling.plotting.common import PlotStyle, finalize_figure
except Exception:  # pragma: no cover
    plot_curves = None
    plot_diagnostics = None
    PlotStyle = None
    finalize_figure = None


def _plot_skip(label: str) -> None:
    if _MPL_IMPORT_ERROR is not None:
        _log(f"[Plot] Skip {label}: matplotlib unavailable ({_MPL_IMPORT_ERROR}).", flush=True)
    else:
        _log(f"[Plot] Skip {label}: matplotlib unavailable.", flush=True)


class BayesOptPlottingMixin:
    def _classification_plot_prediction_mode(self) -> str:
        cfg = getattr(self, "config", None)
        raw = getattr(cfg, "classification_plot_prediction", "score")
        mode = str(raw or "score").strip().lower()
        return mode if mode in {"score", "label"} else "score"

    def resolve_plot_prediction_column(self, base_col: Optional[str]) -> Optional[str]:
        if not base_col:
            return base_col
        if str(getattr(self, "task_type", "")).lower() != "classification":
            return base_col
        if not str(base_col).startswith("pred_"):
            return base_col
        if self._classification_plot_prediction_mode() != "label":
            return base_col
        prefix = str(base_col)[len("pred_"):]
        label_col = f"pred_label_{prefix}"
        if label_col in self.train_data.columns:
            return label_col
        _log(
            f"[Plot] classification_plot_prediction=label but '{label_col}' is missing; "
            f"fallback to '{base_col}'.",
            flush=True,
        )
        return base_col

    @staticmethod
    def _parse_plot_limit(raw_value: object) -> Optional[int]:
        if raw_value is None:
            return None
        text = str(raw_value).strip().lower()
        if text in {"", "none", "null", "off", "disable", "disabled", "0"}:
            return None
        try:
            limit = int(raw_value)
        except (TypeError, ValueError):
            return None
        if limit <= 0:
            return None
        return limit

    def _resolve_plot_row_limit(self, kind: str) -> Optional[int]:
        cfg = getattr(self, "config", None)
        global_cfg_limit = self._parse_plot_limit(getattr(cfg, "plot_max_rows", None))
        if kind == "oneway":
            kind_cfg_limit = self._parse_plot_limit(
                getattr(cfg, "plot_oneway_max_rows", None)
            )
            kind_env_limit = self._parse_plot_limit(
                os.environ.get("BAYESOPT_PLOT_ONEWAY_MAX_ROWS")
            )
        else:
            kind_cfg_limit = self._parse_plot_limit(
                getattr(cfg, "plot_curve_max_rows", None)
            )
            kind_env_limit = self._parse_plot_limit(
                os.environ.get("BAYESOPT_PLOT_CURVE_MAX_ROWS")
            )
        global_env_limit = self._parse_plot_limit(os.environ.get("BAYESOPT_PLOT_MAX_ROWS"))
        return (
            kind_env_limit
            or kind_cfg_limit
            or global_env_limit
            or global_cfg_limit
            or None
        )

    def _resolve_plot_sampling_seed(self) -> int:
        cfg = getattr(self, "config", None)
        raw_seed = os.environ.get("BAYESOPT_PLOT_SAMPLING_SEED")
        if raw_seed is None:
            raw_seed = getattr(cfg, "plot_sampling_seed", 13)
        try:
            return int(raw_seed)
        except (TypeError, ValueError):
            return 13

    def _log_plot_sampling(
        self,
        *,
        kind: str,
        dataset_name: str,
        original_rows: int,
        sampled_rows: int,
    ) -> None:
        cache = getattr(self, "_plot_sampling_log_cache", None)
        if not isinstance(cache, set):
            cache = set()
            setattr(self, "_plot_sampling_log_cache", cache)
        key = (kind, dataset_name, original_rows, sampled_rows)
        if key in cache:
            return
        cache.add(key)
        _log(
            f"[Plot] downsample {kind}/{dataset_name}: "
            f"{original_rows} -> {sampled_rows} rows.",
            flush=True,
        )

    def _resolve_plot_sample_positions(
        self,
        *,
        data: pd.DataFrame,
        kind: str,
        dataset_name: str,
    ) -> Optional[np.ndarray]:
        limit = self._resolve_plot_row_limit(kind)
        total_rows = int(len(data))
        if limit is None or total_rows <= int(limit):
            return None
        seed = self._resolve_plot_sampling_seed()
        salt = int(
            zlib.crc32(
                f"{getattr(self, 'model_nme', 'model')}|{kind}|{dataset_name}".encode(
                    "utf-8"
                )
            )
        )
        rng = np.random.default_rng((seed + salt) % (2 ** 32))
        positions = np.sort(
            rng.choice(total_rows, size=int(limit), replace=False).astype(np.int64)
        )
        self._log_plot_sampling(
            kind=kind,
            dataset_name=dataset_name,
            original_rows=total_rows,
            sampled_rows=int(limit),
        )
        return positions

    @staticmethod
    def _column_values(
        data: pd.DataFrame,
        column: str,
        sample_positions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        values = data[column].to_numpy(copy=False)
        if sample_positions is None:
            return np.asarray(values)
        return np.take(values, sample_positions, axis=0)

    @classmethod
    def _numeric_column_values(
        cls,
        data: pd.DataFrame,
        column: str,
        sample_positions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        raw = cls._column_values(data, column, sample_positions)
        arr = np.asarray(raw)
        if np.issubdtype(arr.dtype, np.number):
            return arr.astype(float, copy=False)
        coerced = pd.to_numeric(arr, errors="coerce")
        return np.asarray(coerced, dtype=float)

    def _build_oneway_source(
        self,
        *,
        feature: str,
        pred_col: Optional[str],
        sample_positions: Optional[np.ndarray],
    ) -> pd.DataFrame:
        if sample_positions is None:
            return self.train_data
        required_cols: List[str] = [feature, self.weight_nme, "w_act"]
        if pred_col:
            required_cols.append(pred_col)
        payload: Dict[str, np.ndarray] = {}
        for col in required_cols:
            if col in self.train_data.columns:
                payload[col] = self._column_values(
                    self.train_data,
                    col,
                    sample_positions=sample_positions,
                )
        index = self.train_data.index.take(sample_positions)
        return pd.DataFrame(payload, index=index)

    @staticmethod
    def _safe_plot_token(value: object) -> str:
        return (
            str(value)
            .strip()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
        )

    @staticmethod
    def _as_axes_list(axes) -> List:
        if isinstance(axes, np.ndarray):
            return list(axes.ravel())
        return [axes]

    @staticmethod
    def _dataset_split_name(label: str) -> str:
        return "train" if "Train" in label else "test"

    def _resolve_curve_sample_positions(
        self,
        *,
        data: pd.DataFrame,
        curve_tag: str,
        dataset_label: str,
    ) -> Optional[np.ndarray]:
        return self._resolve_plot_sample_positions(
            data=data,
            kind="curve",
            dataset_name=f"{curve_tag}_{self._dataset_split_name(dataset_label)}",
        )

    def _resolve_prediction_values(
        self,
        *,
        data: pd.DataFrame,
        pred_col: str,
        weight_vals: np.ndarray,
        sample_positions: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if pred_col in data.columns:
            return self._numeric_column_values(
                data,
                pred_col,
                sample_positions=sample_positions,
            )
        weighted_col = f"w_{pred_col}"
        if weighted_col not in data.columns:
            return None
        weighted_pred = self._numeric_column_values(
            data,
            weighted_col,
            sample_positions=sample_positions,
        )
        return weighted_pred / np.maximum(weight_vals, EPS)

    def _collect_labeled_curve_datasets(
        self,
        *,
        log_prefix: str,
        title_prefix: Optional[str] = None,
    ) -> List[Tuple[str, pd.DataFrame]]:
        datasets: List[Tuple[str, pd.DataFrame]] = []
        for data_name, data in [("Train Data", self.train_data), ("Test Data", self.test_data)]:
            title = f"{title_prefix} on {data_name}" if title_prefix else data_name
            if "w_act" not in data.columns or data["w_act"].isna().all():
                _log(f"[{log_prefix}] Missing labels for {title}; skip.", flush=True)
                continue
            datasets.append((title, data))
        if not datasets:
            _log(f"[{log_prefix}] No labeled data available; skip plotting.", flush=True)
        return datasets

    def _save_plot_figure(
        self,
        *,
        fig,
        plot_prefix: str,
        filename: str,
        style: Optional[object] = None,
    ) -> None:
        save_path = self._resolve_plot_path(plot_prefix, filename)
        if finalize_figure:
            finalize_figure(fig, save_path=save_path, show=False, style=style)
        else:
            plt.savefig(save_path, dpi=300)
            plt.close(fig)

    def _oneway_filename(
        self,
        *,
        group_col: str,
        pred_col: Optional[str],
        pred_label: Optional[str],
    ) -> str:
        pred_tag = (
            self._safe_plot_token(pred_label or pred_col)
            if pred_col
            else None
        )
        if pred_tag:
            return f"00_{self.model_nme}_{group_col}_oneway_{pred_tag}.png"
        return f"00_{self.model_nme}_{group_col}_oneway.png"

    def _plot_oneway_fallback(
        self,
        *,
        plot_source: pd.DataFrame,
        feature: str,
        group_col: str,
        is_categorical: bool,
        n_bins: int,
        pred_col: Optional[str],
        pred_label: Optional[str],
        pred_weighted: bool,
        title: str,
        save_path: str,
    ) -> None:
        fig = plt.figure(figsize=(7, 5))
        if not is_categorical:
            bins = pd.qcut(
                pd.to_numeric(plot_source[feature], errors="coerce"),
                n_bins,
                duplicates="drop",
            )
            plot_source = plot_source.assign(**{group_col: bins})
        if pred_col is not None and pred_col in plot_source.columns:
            pred_values = pd.to_numeric(plot_source[pred_col], errors="coerce")
            if pred_weighted:
                plot_source = plot_source.assign(_pred_w=pred_values)
            else:
                plot_source = plot_source.assign(
                    _pred_w=pred_values
                    * pd.to_numeric(plot_source[self.weight_nme], errors="coerce")
                )
        plot_data = plot_source.groupby([group_col], observed=True).sum(numeric_only=True)
        plot_data.reset_index(inplace=True)
        plot_data["act_v"] = (
            pd.to_numeric(plot_data["w_act"], errors="coerce")
            / np.maximum(
                pd.to_numeric(plot_data[self.weight_nme], errors="coerce"),
                EPS,
            )
        )
        if pred_col is not None and "_pred_w" in plot_data.columns:
            plot_data["pred_v"] = (
                pd.to_numeric(plot_data["_pred_w"], errors="coerce")
                / np.maximum(
                    pd.to_numeric(plot_data[self.weight_nme], errors="coerce"),
                    EPS,
                )
            )
        ax = fig.add_subplot(111)
        ax.plot(plot_data.index, plot_data["act_v"], label="Actual", color="red")
        if pred_col is not None and "pred_v" in plot_data.columns:
            ax.plot(
                plot_data.index,
                plot_data["pred_v"],
                label=pred_label or "Predicted",
                color="tab:blue",
            )
        ax.set_title(title, fontsize=8)
        labels = list(plot_data[group_col].astype(str))
        plt.xticks(plot_data.index, labels, rotation=90)
        plt.xticks(fontsize=3 if len(labels) > 50 else 6)
        plt.yticks(fontsize=6)
        ax2 = ax.twinx()
        ax2.bar(
            plot_data.index,
            plot_data[self.weight_nme],
            alpha=0.5,
            color="seagreen",
        )
        plt.yticks(fontsize=6)
        plt.margins(0.05)
        plt.subplots_adjust(wspace=0.3)
        if pred_col is not None and "pred_v" in plot_data.columns:
            ax.legend(fontsize=6)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

    def _plot_conversion_lift_fallback(
        self,
        *,
        ax,
        pred_vals: np.ndarray,
        binary_vals: np.ndarray,
        weight_vals: np.ndarray,
        model_pred_col: str,
        title: str,
        n_bins: int,
    ) -> None:
        weighted_binary = binary_vals * weight_vals
        plot_data = pd.DataFrame(
            {
                model_pred_col: pred_vals,
                self.weight_nme: weight_vals,
                "w_binary_act": weighted_binary,
            }
        )
        plot_data = plot_data.sort_values(by=model_pred_col).copy()
        plot_data["cum_weight"] = plot_data[self.weight_nme].cumsum()
        total_weight = plot_data[self.weight_nme].sum()
        if total_weight > EPS:
            plot_data["bin"] = pd.cut(
                plot_data["cum_weight"],
                bins=n_bins,
                labels=False,
                right=False,
            )
        else:
            plot_data["bin"] = 0
        lift_agg = plot_data.groupby("bin").agg(
            total_weight=(self.weight_nme, "sum"),
            weighted_conversions=("w_binary_act", "sum"),
        ).reset_index()
        lift_agg["conversion_rate"] = (
            lift_agg["weighted_conversions"] / lift_agg["total_weight"]
        )
        overall_conversion_rate = float(plot_data["w_binary_act"].sum()) / max(
            float(plot_data[self.weight_nme].sum()),
            EPS,
        )
        ax.axhline(
            y=overall_conversion_rate,
            color="gray",
            linestyle="--",
            label=f"Overall Avg Rate ({overall_conversion_rate:.2%})",
        )
        ax.plot(
            lift_agg["bin"],
            lift_agg["conversion_rate"],
            marker="o",
            linestyle="-",
            label="Actual Conversion Rate",
        )
        ax.set_title(title)
        ax.set_xlabel(f"Model Score Decile (based on {model_pred_col})")
        ax.set_ylabel("Conversion Rate")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

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
        pred_col = self.resolve_plot_prediction_column(pred_col)
        if pred_col is not None and pred_col not in self.train_data.columns:
            _log(
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
        sample_positions = self._resolve_plot_sample_positions(
            data=self.train_data,
            kind="oneway",
            dataset_name="train",
        )

        if "w_act" not in self.train_data.columns:
            _log("[Oneway] Missing w_act column; skip plotting.", flush=True)
            return

        for c in self.factor_nmes:
            if c not in self.train_data.columns:
                continue
            is_cat = c in (self.cate_list or [])
            group_col = c if is_cat else f"{c}_bins"
            title = f"Analysis of {group_col} : Train Data"
            filename = self._oneway_filename(
                group_col=group_col,
                pred_col=pred_col,
                pred_label=pred_label,
            )
            save_path = self._resolve_plot_path(plot_prefix, filename)
            plot_source = self._build_oneway_source(
                feature=c,
                pred_col=pred_col,
                sample_positions=sample_positions,
            )
            if plot_diagnostics is None:
                self._plot_oneway_fallback(
                    plot_source=plot_source,
                    feature=c,
                    group_col=group_col,
                    is_categorical=is_cat,
                    n_bins=n_bins,
                    pred_col=pred_col,
                    pred_label=pred_label,
                    pred_weighted=pred_weighted,
                    title=title,
                    save_path=save_path,
                )
                continue
            plot_diagnostics.plot_oneway(
                plot_source,
                feature=c,
                weight_col=self.weight_nme,
                target_col="w_act",
                target_weighted=True,
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
        pred_nme = self.resolve_plot_prediction_column(pred_nme)
        safe_label = self._safe_plot_token(model_label)
        plot_prefix = f"{self.model_nme}/lift"
        filename = f"01_{self.model_nme}_{safe_label}_lift.png"

        datasets = self._collect_labeled_curve_datasets(
            log_prefix="Lift",
            title_prefix="Lift Chart",
        )
        if not datasets:
            return

        if plot_curves is None:
            _plot_skip("lift plot")
            return

        style = PlotStyle() if PlotStyle else None
        fig, axes = plt.subplots(1, len(datasets), figsize=(11, 5))
        axes = self._as_axes_list(axes)

        for ax, (title, data) in zip(axes, datasets):
            sample_positions = self._resolve_curve_sample_positions(
                data=data,
                curve_tag="lift",
                dataset_label=title,
            )
            weight_vals = self._numeric_column_values(
                data,
                self.weight_nme,
                sample_positions=sample_positions,
            )
            actual_vals = self._numeric_column_values(
                data,
                "w_act",
                sample_positions=sample_positions,
            )
            pred_vals = self._resolve_prediction_values(
                data=data,
                pred_col=pred_nme,
                weight_vals=weight_vals,
                sample_positions=sample_positions,
            )
            if pred_vals is None:
                _log(
                    f"[Lift] Missing prediction columns in {title}; skip.",
                    flush=True,
                )
                continue

            plot_curves.plot_lift_curve(
                pred_vals,
                actual_vals,
                weight_vals,
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
        self._save_plot_figure(
            fig=fig,
            plot_prefix=plot_prefix,
            filename=filename,
            style=style,
        )

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

        datasets = self._collect_labeled_curve_datasets(log_prefix="Double Lift")
        if not datasets:
            return

        if plot_curves is None:
            _plot_skip("double lift plot")
            return

        style = PlotStyle() if PlotStyle else None
        fig, axes = plt.subplots(1, len(datasets), figsize=(11, 5))
        axes = self._as_axes_list(axes)

        label1 = model_name_map[name1]
        label2 = model_name_map[name2]

        for ax, (data_name, data) in zip(axes, datasets):
            sample_positions = self._resolve_curve_sample_positions(
                data=data,
                curve_tag="dlift",
                dataset_label=data_name,
            )
            weight_vals = self._numeric_column_values(
                data,
                self.weight_nme,
                sample_positions=sample_positions,
            )
            actual_vals = self._numeric_column_values(
                data,
                "w_act",
                sample_positions=sample_positions,
            )
            pred1 = None
            pred2 = None

            pred1_col = self.resolve_plot_prediction_column(f"pred_{name1}")
            pred2_col = self.resolve_plot_prediction_column(f"pred_{name2}")
            pred1 = self._resolve_prediction_values(
                data=data,
                pred_col=pred1_col,
                weight_vals=weight_vals,
                sample_positions=sample_positions,
            )
            pred2 = self._resolve_prediction_values(
                data=data,
                pred_col=pred2_col,
                weight_vals=weight_vals,
                sample_positions=sample_positions,
            )

            if pred1 is None or pred2 is None:
                _log(
                    f"Warning: missing '{pred1_col}'/'{pred2_col}' (or weighted variants) in "
                    f"{data_name}. Skip plot.")
                continue

            plot_curves.plot_double_lift_curve(
                pred1,
                pred2,
                actual_vals,
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
        self._save_plot_figure(
            fig=fig,
            plot_prefix=plot_prefix,
            filename=filename,
            style=style,
        )

    # Conversion lift curve plot.

    def plot_conversion_lift(self, model_pred_col: str, n_bins: int = 20):
        if plt is None:
            _plot_skip("conversion lift plot")
            return
        if not self.binary_resp_nme:
            _log("Error: `binary_resp_nme` not provided at BayesOptModel init; cannot plot conversion lift.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        axes = self._as_axes_list(axes)
        datasets = [("Train Data", self.train_data), ("Test Data", self.test_data)]

        for ax, (data_name, data) in zip(axes, datasets):
            if model_pred_col not in data.columns:
                _log(f"Warning: missing prediction column '{model_pred_col}' in {data_name}. Skip plot.")
                continue
            sample_positions = self._resolve_curve_sample_positions(
                data=data,
                curve_tag="conversion",
                dataset_label=data_name,
            )
            pred_vals = self._numeric_column_values(
                data,
                model_pred_col,
                sample_positions=sample_positions,
            )
            binary_vals = self._numeric_column_values(
                data,
                self.binary_resp_nme,
                sample_positions=sample_positions,
            )
            weight_vals = self._numeric_column_values(
                data,
                self.weight_nme,
                sample_positions=sample_positions,
            )
            title = f"Conversion Rate Lift Chart on {data_name}"
            if plot_curves is None:
                self._plot_conversion_lift_fallback(
                    ax=ax,
                    pred_vals=pred_vals,
                    binary_vals=binary_vals,
                    weight_vals=weight_vals,
                    model_pred_col=model_pred_col,
                    title=title,
                    n_bins=n_bins,
                )
                continue
            plot_curves.plot_conversion_lift(
                pred_vals,
                binary_vals,
                weight_vals,
                n_bins=n_bins,
                title=title,
                ax=ax,
                show=False,
            )

        plt.tight_layout()
        plt.show()
        plt.close(fig)

    # ========= Lightweight explainability: Permutation Importance =========
