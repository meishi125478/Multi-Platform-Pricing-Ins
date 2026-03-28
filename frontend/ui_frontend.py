"""
NiceGUI-based frontend for Insurance Pricing Model Training.

Launch:
    python -m ins_pricing.frontend.app
"""

import json
import os
import queue
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from nicegui import ui

from ins_pricing.frontend.app_controller import PricingApp


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _dump(obj, indent=2):
    return json.dumps(obj, indent=indent, ensure_ascii=False)


class _StreamRunner:
    """Consume a blocking generator in a background thread, stream to UI."""

    def __init__(self, status_el, log_el, gallery_el=None):
        self._status = status_el
        self._log = log_el
        self._gallery = gallery_el
        self._q: queue.Queue = queue.Queue()
        self._timer = None
        self._log_max_chars = self._resolve_log_limit()

    @staticmethod
    def _resolve_log_limit() -> int:
        raw = os.environ.get("INS_PRICING_FRONTEND_LOG_MAX_CHARS", "").strip()
        if not raw:
            return 200_000
        try:
            return max(200, int(raw))
        except ValueError:
            return 200_000

    def _clip_log(self, text: str) -> str:
        if len(text) <= self._log_max_chars:
            return text
        clipped = text[-self._log_max_chars:]
        return f"[...log truncated to last {self._log_max_chars} chars...]\n{clipped}"

    def run(self, gen_fn, *a, **kw):
        self._log.value = ""
        self._status.value = "Starting..."
        if self._gallery is not None:
            self._gallery.clear()

        def _worker():
            try:
                for item in gen_fn(*a, **kw):
                    self._q.put(item)
            except Exception as exc:
                self._q.put((f"Error: {exc}", str(exc)))
            self._q.put(None)

        threading.Thread(target=_worker, daemon=True).start()
        self._timer = ui.timer(0.15, self._drain)

    def _drain(self):
        while not self._q.empty():
            item = self._q.get_nowait()
            if item is None:
                if self._timer:
                    self._timer.deactivate()
                return
            if isinstance(item, tuple) and len(item) >= 2:
                self._status.value = str(item[0])
                self._log.value = self._clip_log(str(item[1]))
            if isinstance(item, tuple) and len(item) >= 3 and self._gallery is not None:
                images = item[2]
                if images:
                    self._show_images(images)

    def _show_images(self, paths):
        self._gallery.clear()
        with self._gallery:
            with ui.row().classes("flex-wrap gap-2 p-2"):
                for p in paths:
                    if Path(p).exists():
                        ui.image(Path(p)).classes(
                            "max-w-[260px] max-h-[220px] object-contain rounded shadow"
                        )


# ═══════════════════════════════════════════════════════════════════════
#  Main Frontend
# ═══════════════════════════════════════════════════════════════════════

class PricingFrontend:
    """Build the complete NiceGUI interface."""

    def __init__(self, pricing_app: PricingApp):
        self.app = pricing_app
        self.cfg: Dict[str, Any] = {}  # config‑form components
        self.ui: Dict[str, Any] = {}   # non‑config UI elements

        cb = pricing_app.config_builder
        self._ss = {
            "xgb": _dump(cb._default_xgb_search_space()),
            "resn": _dump(cb._default_resn_search_space()),
            "ft": _dump(cb._default_ft_search_space()),
            "ft_unsup": _dump(cb._default_ft_unsupervised_search_space()),
        }
        self._xgb_step2_tpl = _dump({
            "output_dir": "./ResultsXGBFromFTUnsupervised",
            "optuna_storage": "./ResultsXGBFromFTUnsupervised/optuna/bayesopt.sqlite3",
            "optuna_study_prefix": "pricing_ft_unsup_xgb",
            "loss_name": "mse", "build_oht": False, "final_refit": False,
            "runner": {"model_keys": ["xgb"], "nproc_per_node": 1, "plot_curves": False},
            "plot_curves": False, "plot": {"enable": False},
        })
        self._resn_step2_tpl = _dump({
            "use_resn_ddp": True,
            "output_dir": "./ResultsResNFromFTUnsupervised",
            "optuna_storage": "./ResultsResNFromFTUnsupervised/optuna/bayesopt.sqlite3",
            "optuna_study_prefix": "pricing_ft_unsup_resn_ddp",
            "loss_name": "mse", "build_oht": True,
            "runner": {"model_keys": ["resn"], "nproc_per_node": 2, "plot_curves": False},
            "plot_curves": False, "plot": {"enable": False},
        })

    # ── UI helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _tip(text: str, icon: str = "lightbulb"):
        """Render an inline tip banner."""
        with ui.element("div").classes(
            "w-full rounded-lg bg-amber-50 border border-amber-200 "
            "px-4 py-2 flex items-start gap-2 my-1"
        ):
            ui.icon(icon).classes("text-amber-600 mt-0.5 text-base")
            ui.label(text).classes("text-xs text-amber-900 leading-relaxed")

    @staticmethod
    def _guide(title: str, steps: list):
        """Render a numbered step guide."""
        with ui.element("div").classes(
            "w-full rounded-lg bg-blue-50 border border-blue-200 "
            "px-4 py-3 my-1"
        ):
            ui.label(title).classes("text-sm font-semibold text-blue-800 mb-1")
            for i, step in enumerate(steps, 1):
                ui.label(f"{i}. {step}").classes("text-xs text-blue-700 leading-relaxed ml-2")

    @staticmethod
    def _info(text: str, icon: str = "info"):
        """Render an info note."""
        with ui.element("div").classes(
            "w-full rounded-lg bg-gray-50 border border-gray-200 "
            "px-4 py-2 flex items-start gap-2 my-1"
        ):
            ui.icon(icon).classes("text-gray-500 mt-0.5 text-base")
            ui.label(text).classes("text-xs text-gray-600 leading-relaxed")

    # ── component helpers (register in self.cfg) ──────────────────────

    def _inp(self, key, label, value="", **kw):
        c = ui.input(label, value=str(value), **kw).classes("w-full")
        self.cfg[key] = c
        return c

    def _num(self, key, label, value=0, **kw):
        c = ui.number(label, value=value, **kw).classes("w-full")
        self.cfg[key] = c
        return c

    def _sel(self, key, label, options, value=None, **kw):
        c = ui.select(options, label=label, value=value, **kw).classes("w-full")
        self.cfg[key] = c
        return c

    def _chk(self, key, label, value=False):
        c = ui.checkbox(label, value=value)
        self.cfg[key] = c
        return c

    def _txt(self, key, label, value="", **kw):
        c = ui.textarea(label, value=str(value), **kw).classes("w-full font-mono text-xs")
        self.cfg[key] = c
        return c

    def _collect(self) -> Dict[str, Any]:
        """Collect all config component values into a dict matching build_config_from_ui params."""
        return {k: c.value for k, c in self.cfg.items()}

    # ── file upload helper ────────────────────────────────────────────

    @staticmethod
    def _save_upload(content_bytes: bytes, suffix: str = ".json") -> str:
        fd, tmp = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(content_bytes)
        return tmp

    # ══════════════════════════════════════════════════════════════════
    #  BUILD
    # ══════════════════════════════════════════════════════════════════

    def build(self):
        ui.query("body").classes("bg-gray-50")

        # ── header ────────────────────────────────────────────────────
        with ui.header().classes(
            "items-center justify-between px-6 bg-blue-800 text-white shadow"
        ):
            ui.label("Insurance Pricing Model Training").classes(
                "text-xl font-bold tracking-wide"
            )
            with ui.row().classes("items-center gap-2"):
                dark = ui.dark_mode(value=False)
                ui.button(icon="dark_mode", on_click=dark.toggle).props(
                    "flat round dense color=white"
                )

        # ── working directory (expansion) ─────────────────────────────
        with ui.column().classes("w-full max-w-7xl mx-auto px-4 pt-4"):
            with ui.expansion("Working Directory", icon="folder").classes(
                "w-full bg-white shadow-sm rounded"
            ):
                self._section_working_dir()

        # ── main tabs ─────────────────────────────────────────────────
        with ui.column().classes("w-full max-w-7xl mx-auto px-4 pb-8"):
            with ui.card().classes("w-full shadow-sm"):
                with ui.tabs().classes("w-full") as tabs:
                    t_cfg = ui.tab("Configuration", icon="settings")
                    t_wf = ui.tab("Workflow", icon="play_arrow")
                    t_ft = ui.tab("FT Two-Step", icon="layers")
                    t_plot = ui.tab("Plotting", icon="bar_chart")
                    t_pred = ui.tab("Prediction", icon="analytics")

                with ui.tab_panels(tabs, value=t_cfg).classes("w-full"):
                    with ui.tab_panel(t_cfg):
                        self._tab_config()
                    with ui.tab_panel(t_wf):
                        self._tab_workflow()
                    with ui.tab_panel(t_ft):
                        self._tab_ft_workflow()
                    with ui.tab_panel(t_plot):
                        self._tab_plotting()
                    with ui.tab_panel(t_pred):
                        self._tab_prediction()

    # ══════════════════════════════════════════════════════════════════
    #  WORKING DIRECTORY
    # ══════════════════════════════════════════════════════════════════

    def _section_working_dir(self):
        self._info(
            "Working Directory 是所有相对路径的基准目录。"
            "配置中的 ./Data、./Results 等路径都相对于此目录解析。"
            "默认为启动命令时的当前目录（cwd），也可以在此处随时修改。"
            "建议设置为包含数据文件和配置文件的项目目录。"
        )
        _, choices, selected = self.app.list_directory_candidates(str(self.app.working_dir))

        with ui.row().classes("w-full items-end gap-2"):
            wd = ui.input("Working Directory", value=str(self.app.working_dir)).classes("flex-grow")
            ui.button("Set", on_click=lambda: self._set_wd(wd.value)).props("flat")
        with ui.row().classes("w-full items-end gap-2"):
            browse = ui.input("Browse Root", value=str(self.app.working_dir)).classes("flex-grow")
            ui.button("Refresh", on_click=lambda: self._refresh_wd(browse.value)).props("flat")
        with ui.row().classes("w-full items-end gap-2"):
            picker = ui.select(choices, label="Select Folder", value=selected).classes("flex-grow")
            ui.button("Use Selected", on_click=lambda: self._set_wd(picker.value)).props("flat")
        wd_status = ui.label(f"Current: {self.app.working_dir}").classes("text-xs text-gray-500")

        self.ui["wd_input"] = wd
        self.ui["wd_browse"] = browse
        self.ui["wd_picker"] = picker
        self.ui["wd_status"] = wd_status

    def _set_wd(self, path: str):
        status, resolved = self.app.set_working_dir(path)
        self.ui["wd_input"].value = resolved
        self.ui["wd_browse"].value = resolved
        self.ui["wd_status"].text = status
        self._refresh_wd(resolved)
        ui.notify(status, type="positive" if "set to" in status else "warning")

    def _refresh_wd(self, root: str):
        status, choices, selected = self.app.list_directory_candidates(root)
        self.ui["wd_picker"].options = choices
        self.ui["wd_picker"].value = selected
        self.ui["wd_status"].text = status

    # ══════════════════════════════════════════════════════════════════
    #  TAB: CONFIGURATION
    # ══════════════════════════════════════════════════════════════════

    def _tab_config(self):
        self._guide("Configuration 使用流程（二选一）", [
            "方式 A（上传）：上传已有 JSON 配置文件 → 直接切到 Workflow 标签页运行，无需 Build",
            "方式 B（手动）：填写下方各项参数 → 点击 \"Build Configuration\" 生成 JSON → 再去运行",
            "（可选）点击 \"Save Configuration\" 将配置保存为文件，方便下次上传复用",
        ])
        self._tip(
            "上传 JSON 后会自动填入底部的 \"Generated Config\" 文本框，可直接运行。"
            "如需微调，可修改文本框中的 JSON 后再运行，无需重新 Build。"
            "手动填写时，核心必填项：Data Directory、Target、Weight、Feature List、Model Keys。"
        )

        # ── Load config ───────────────────────────────────────────────
        with ui.expansion("Load JSON Config", icon="upload_file").classes("w-full"):
            with ui.row().classes("w-full items-end gap-4"):
                ui.upload(
                    label="Upload Config (.json)",
                    auto_upload=True,
                    on_upload=self._on_upload_config,
                ).props("accept=.json").classes("max-w-xs")
                self.ui["load_status"] = ui.label("").classes("text-sm")
            self.ui["config_display"] = ui.textarea(
                "Current Configuration (read-only)",
            ).classes("w-full font-mono text-xs").props("readonly outlined")

        ui.separator()

        # ── Core: Data Settings ───────────────────────────────────────
        with ui.expansion("Data Settings", icon="table_chart", value=True).classes("w-full"):
            self._info(
                "Data Directory 应包含 {model}_{category}.csv 格式的文件，"
                "如 od_bc.csv。Feature List 为空时将自动推断。"
            )
            with ui.row().classes("w-full gap-2"):
                self._inp("data_dir", "Data Directory", "./Data")
                self._inp("output_dir", "Output Directory", "./Results")
            with ui.row().classes("w-full gap-2"):
                self._inp("model_list", "Model List (comma-separated)", "od")
                self._inp("model_categories", "Model Categories (comma-separated)", "bc")
            with ui.row().classes("w-full gap-2"):
                self._inp("target", "Target Column", "response")
                self._inp("weight", "Weight Column", "weights")
            self._inp("feature_list", "Feature List (comma-separated)", "").props(
                "type=textarea rows=3"
            )
            self._inp("categorical_features", "Categorical Features (comma-separated)", "").props(
                "type=textarea rows=2"
            )

        # ── Task & Training ───────────────────────────────────────────
        with ui.expansion("Task & Training", icon="psychology", value=True).classes("w-full"):
            self._info(
                "Model Keys 决定训练哪些模型，用逗号分隔：xgb=XGBoost, resn=ResNet, "
                "ft=FT-Transformer, gnn=GNN。Distribution 会覆盖 loss_name。"
            )
            with ui.row().classes("w-full gap-2"):
                self._sel("task_type", "Task Type",
                          ["regression", "binary", "multiclass"], "regression")
                self._sel("distribution", "Distribution",
                          ["tweedie", "poisson", "gamma", "gaussian", "mse", "mae"],
                          value=None, clearable=True, with_input=True)
            with ui.row().classes("w-full gap-2"):
                self._inp("model_keys", "Model Keys (comma-separated)", "xgb, resn")
                self._inp("binary_resp_nme", "Binary Response Column (optional)", "")
            with ui.row().classes("w-full gap-2"):
                self._sel("split_strategy", "Split Strategy",
                          ["random", "stratified", "time", "group"], "random")
                self._num("rand_seed", "Random Seed", 13)
            with ui.row().classes("w-full gap-2"):
                self._num("epochs", "Epochs", 50)
                self._num("max_evals", "Max Evaluations", 50)
            with ui.row().classes("w-full gap-2"):
                self._num("prop_test", "Test Proportion", 0.25, min=0.05, max=0.5, step=0.05)
                self._num("holdout_ratio", "Holdout Ratio", 0.25, min=0.05, max=0.5, step=0.05)
                self._num("val_ratio", "Validation Ratio", 0.25, min=0.05, max=0.5, step=0.05)
            with ui.row().classes("w-full gap-4"):
                self._chk("use_gpu", "Use GPU", True)
                self._chk("plot_curves", "Plot Curves", False)

        # ── Accordion sections ────────────────────────────────────────
        self._section_split()
        self._section_cv()
        self._section_xgb()
        self._section_resn()
        self._section_ft()
        self._section_gnn()
        self._section_distributed()
        self._section_preprocess()
        self._section_geo()
        self._section_ensemble()
        self._section_output()
        self._section_plot()
        self._section_calibration()
        self._section_threshold()
        self._section_bootstrap()

        # ── Advanced JSON overrides ───────────────────────────────────
        with ui.expansion("Advanced Manual Overrides (JSON)", icon="code").classes("w-full"):
            self._tip(
                "在此输入任意 JSON 覆盖配置项，会与上方 Build 的结果深度合并。"
                "适用于 UI 中没有对应控件的参数，如 runner、report_*、psi_*、registry_* 等。"
                "示例：{\"runner\": {\"mode\": \"explain\"}} 可将任务模式切换为解释性分析。"
            )
            self._txt("config_overrides_json", "Config Overrides JSON", "{}")

        ui.separator()

        # ── Build & Save ──────────────────────────────────────────────
        with ui.row().classes("w-full items-end gap-4"):
            ui.button("Build Configuration", icon="build",
                       on_click=self._on_build_config).props("color=primary")
            ui.button("Save Configuration", icon="save",
                       on_click=self._on_save_config).props("color=secondary")
        self.ui["build_status"] = ui.label("").classes("text-sm")
        self.ui["config_json"] = ui.textarea(
            "Generated Config (JSON)",
        ).classes("w-full font-mono text-xs").props("outlined rows=14")
        with ui.row().classes("w-full items-end gap-4"):
            self.ui["save_filename"] = ui.input("Save Filename", value="my_config.json").classes(
                "flex-grow"
            )
            self.ui["save_status"] = ui.label("").classes("text-sm")

    # ── config accordion sections ─────────────────────────────────────

    def _section_split(self):
        with ui.expansion("Split & Pre-split Data", icon="call_split").classes("w-full"):
            self._tip(
                "默认随机拆分。如需分组拆分填写 Group Column；时间拆分填写 Time Column。"
                "Pre-split：如果已有训练/测试集文件，填写 Train/Test Data Path 即可跳过拆分。"
            )
            with ui.row().classes("w-full gap-2"):
                self._inp("split_group_col", "Split Group Column", "")
                self._inp("split_time_col", "Split Time Column", "")
                self._chk("split_time_ascending", "Time Ascending", True)
            with ui.row().classes("w-full gap-2"):
                self._inp("train_data_path", "Train Data Path (pre-split)", "")
                self._inp("test_data_path", "Test Data Path (pre-split)", "")
            with ui.row().classes("w-full gap-2"):
                self._inp("split_cache_path", "Split Cache Path (.npz)", "")
                self._chk("split_cache_force_rebuild", "Force Rebuild", False)

    def _section_cv(self):
        with ui.expansion("Cross-Validation", icon="grid_view").classes("w-full"):
            self._tip("CV Splits=0 表示不使用交叉验证。group/time 策略需要指定对应的列名。")
            with ui.row().classes("w-full gap-2"):
                self._sel("cv_strategy", "CV Strategy",
                          ["random", "group", "grouped", "time", "stratified",
                           "timeseries", "temporal"],
                          value=None, clearable=True, with_input=True)
                self._num("cv_splits", "CV Splits", 0)
            with ui.row().classes("w-full gap-2"):
                self._inp("cv_group_col", "CV Group Column", "")
                self._inp("cv_time_col", "CV Time Column", "")
                self._chk("cv_time_ascending", "CV Time Ascending", True)

    def _section_xgb(self):
        with ui.expansion("XGBoost Settings", icon="forest").classes("w-full"):
            with ui.row().classes("w-full gap-2"):
                self._num("xgb_max_depth_max", "Max Depth", 25)
                self._num("xgb_n_estimators_max", "Max Estimators", 500)
                self._num("xgb_gpu_id", "GPU ID", 0)
            with ui.row().classes("w-full gap-2"):
                self._chk("xgb_use_dmatrix", "Use DMatrix", True)
                self._num("xgb_chunk_size", "Chunk Size (0=off)", 0)
                self._chk("xgb_cleanup_per_fold", "Cleanup Per Fold", False)
                self._chk("xgb_cleanup_synchronize", "Cleanup Sync", False)
            self._txt("xgb_search_space_json", "Search Space (JSON)", self._ss["xgb"])

    def _section_resn(self):
        with ui.expansion("ResNet Settings", icon="hub").classes("w-full"):
            with ui.row().classes("w-full gap-2"):
                self._chk("resn_use_lazy_dataset", "Lazy Dataset", True)
                self._num("resn_predict_batch_size", "Predict Batch (0=auto)", 0)
                self._num("resn_weight_decay", "Weight Decay", 0.0001, step=0.0001, format="%.4f")
            with ui.row().classes("w-full gap-2"):
                self._chk("resn_cleanup_per_fold", "Cleanup Per Fold", False)
                self._chk("resn_cleanup_synchronize", "Cleanup Sync", False)
            self._txt("resn_search_space_json", "Search Space (JSON)", self._ss["resn"])

    def _section_ft(self):
        with ui.expansion("FT-Transformer Settings", icon="transform").classes("w-full"):
            self._tip(
                "FT Role：model=直接预测，embedding=有监督嵌入，"
                "unsupervised_embedding=无监督嵌入（用于 FT Two-Step 流程）。"
                "OOF Folds > 0 启用 Out-of-Fold 嵌入以避免过拟合。"
            )
            with ui.row().classes("w-full gap-2"):
                self._sel("ft_role", "FT Role",
                          ["model", "embedding", "unsupervised_embedding"], "model")
                self._inp("ft_feature_prefix", "Feature Prefix", "ft_emb")
                self._num("ft_num_numeric_tokens", "Num Numeric Tokens (0=auto)", 0)
            with ui.row().classes("w-full gap-2"):
                self._chk("ft_use_lazy_dataset", "Lazy Dataset", True)
                self._num("ft_predict_batch_size", "Predict Batch (0=auto)", 0)
            with ui.row().classes("w-full gap-2"):
                self._num("ft_oof_folds", "OOF Folds (0=off)", 0)
                self._sel("ft_oof_strategy", "OOF Strategy",
                          ["random", "group", "time", "stratified"],
                          value=None, clearable=True, with_input=True)
                self._chk("ft_oof_shuffle", "OOF Shuffle", True)
            with ui.row().classes("w-full gap-2"):
                self._chk("ft_cleanup_per_fold", "Cleanup Per Fold", False)
                self._chk("ft_cleanup_synchronize", "Cleanup Sync", False)
            with ui.row().classes("w-full gap-4"):
                with ui.column().classes("flex-1"):
                    self._txt("ft_search_space_json",
                              "Supervised Search Space", self._ss["ft"])
                with ui.column().classes("flex-1"):
                    self._txt("ft_unsupervised_search_space_json",
                              "Unsupervised Search Space", self._ss["ft_unsup"])

    def _section_gnn(self):
        with ui.expansion("GNN Settings", icon="scatter_plot").classes("w-full"):
            with ui.row().classes("w-full gap-2"):
                self._chk("gnn_use_approx_knn", "Approx KNN", True)
                self._num("gnn_approx_knn_threshold", "Approx KNN Threshold", 50000)
                self._num("gnn_max_gpu_knn_nodes", "Max GPU KNN Nodes", 200000)
            with ui.row().classes("w-full gap-2"):
                self._num("gnn_knn_gpu_mem_ratio", "GPU Mem Ratio", 0.9, min=0.1, max=1.0, step=0.05)
                self._num("gnn_knn_gpu_mem_overhead", "GPU Mem Overhead (GB)", 2.0)
            with ui.row().classes("w-full gap-2"):
                self._num("gnn_max_fit_rows", "Max Fit Rows (0=all)", 0)
                self._num("gnn_max_predict_rows", "Max Predict Rows (0=all)", 0)
                self._num("gnn_predict_chunk_rows", "Predict Chunk (0=auto)", 0)
            with ui.row().classes("w-full gap-2"):
                self._inp("gnn_graph_cache", "Graph Cache Path", "")
                self._chk("gnn_cleanup_per_fold", "Cleanup Per Fold", False)
                self._chk("gnn_cleanup_synchronize", "Cleanup Sync", False)

    def _section_distributed(self):
        with ui.expansion("Distributed Training", icon="device_hub").classes("w-full"):
            self._tip(
                "DDP（分布式数据并行）适合多 GPU 训练 ResNet/FT，数据量小于 ddp_min_rows 时自动跳过。"
                "DataParallel 为单机多卡的替代方案，一般推荐 DDP。"
            )
            with ui.row().classes("w-full gap-2"):
                self._num("nproc_per_node", "Processes Per Node", 2)
                self._num("ddp_min_rows", "DDP Min Rows", 50000)
            with ui.row().classes("w-full gap-4"):
                self._chk("use_resn_ddp", "ResNet DDP", True)
                self._chk("use_ft_ddp", "FT DDP", True)
            with ui.row().classes("w-full gap-4"):
                self._chk("use_resn_data_parallel", "ResNet DataParallel", False)
                self._chk("use_ft_data_parallel", "FT DataParallel", False)
                self._chk("use_gnn_data_parallel", "GNN DataParallel", False)

    def _section_preprocess(self):
        with ui.expansion("Preprocessing", icon="tune").classes("w-full"):
            with ui.row().classes("w-full gap-4"):
                self._chk("build_oht", "Build OHT", True)
                self._chk("oht_sparse_csr", "OHT Sparse CSR", True)
                self._chk("keep_unscaled_oht", "Keep Unscaled OHT", False)
            with ui.row().classes("w-full gap-2"):
                self._num("infer_categorical_max_unique", "Infer Cat. Max Unique", 50)
                self._num("infer_categorical_max_ratio", "Infer Cat. Max Ratio", 0.05, step=0.01)

    def _section_geo(self):
        with ui.expansion("Geographic / Regional", icon="public").classes("w-full"):
            with ui.row().classes("w-full gap-2"):
                self._inp("geo_feature_nmes", "Geo Feature Names (comma-sep.)", "")
                self._inp("region_province_col", "Province Column", "")
                self._inp("region_city_col", "City Column", "")
                self._num("region_effect_alpha", "Region Effect Alpha", 0.0)
            with ui.row().classes("w-full gap-2"):
                self._num("geo_token_hidden_dim", "Hidden Dim", 32)
                self._num("geo_token_layers", "Layers", 2)
                self._num("geo_token_dropout", "Dropout", 0.1, step=0.05)
            with ui.row().classes("w-full gap-2"):
                self._num("geo_token_k_neighbors", "K Neighbors", 10)
                self._num("geo_token_learning_rate", "Learning Rate", 0.001, step=0.0001, format="%.4f")
                self._num("geo_token_epochs", "Epochs", 50)

    def _section_ensemble(self):
        with ui.expansion("Ensemble & Refit", icon="merge_type").classes("w-full"):
            with ui.row().classes("w-full gap-4"):
                self._chk("final_ensemble", "Final Ensemble", False)
                self._num("final_ensemble_k", "Ensemble K", 3)
                self._chk("final_refit", "Final Refit", True)
                self._chk("reuse_best_params", "Reuse Best Params", False)

    def _section_output(self):
        with ui.expansion("Output & Caching", icon="storage").classes("w-full"):
            with ui.row().classes("w-full gap-2"):
                self._inp("optuna_study_prefix", "Optuna Study Prefix", "pricing")
                self._chk("optuna_cleanup_synchronize", "Optuna Cleanup Sync", False)
            with ui.row().classes("w-full gap-2"):
                self._chk("cache_predictions", "Cache Predictions", False)
                self._sel("prediction_cache_format", "Cache Format",
                          ["parquet", "csv"], "parquet")
            with ui.row().classes("w-full gap-2"):
                self._num("dataloader_workers", "DataLoader Workers", 0)
                self._num("bo_sample_limit", "BO Sample Limit (0=all)", 0)

    def _section_plot(self):
        with ui.expansion("Plot Settings", icon="insert_chart").classes("w-full"):
            with ui.row().classes("w-full gap-4"):
                self._chk("plot_enable", "Enable Plot", False)
                self._num("plot_n_bins", "Plot Bins", 10)
            with ui.row().classes("w-full gap-4"):
                self._chk("plot_oneway", "Oneway", False)
                self._chk("plot_oneway_pred", "Oneway Pred", False)
                self._chk("plot_pre_oneway", "Pre Oneway", False)
                self._chk("plot_double_lift", "Double Lift", False)

    def _section_calibration(self):
        with ui.expansion("Calibration", icon="straighten").classes("w-full"):
            with ui.row().classes("w-full gap-2"):
                self._chk("calibration_enable", "Enable", False)
                self._sel("calibration_method", "Method",
                          ["sigmoid", "isotonic"], "sigmoid")
                self._num("calibration_max_rows", "Max Rows (0=all)", 0)
                self._num("calibration_seed", "Seed", 13)

    def _section_threshold(self):
        with ui.expansion("Threshold", icon="tune").classes("w-full"):
            with ui.row().classes("w-full gap-2"):
                self._chk("threshold_enable", "Enable", False)
                self._sel("threshold_metric", "Metric",
                          ["f1", "precision", "recall", "balanced_accuracy"], "f1",
                          with_input=True)
                self._num("threshold_grid", "Grid", 99)
            with ui.row().classes("w-full gap-2"):
                self._num("threshold_max_rows", "Max Rows (0=all)", 0)
                self._num("threshold_seed", "Seed", 13)

    def _section_bootstrap(self):
        with ui.expansion("Bootstrap", icon="loop").classes("w-full"):
            with ui.row().classes("w-full gap-2"):
                self._chk("bootstrap_enable", "Enable", False)
                self._num("bootstrap_n_samples", "N Samples", 200)
                self._num("bootstrap_ci", "CI", 0.95, step=0.01)
                self._num("bootstrap_seed", "Seed", 13)

    # ── config event handlers ─────────────────────────────────────────

    async def _on_upload_config(self, e):
        try:
            content = await e.file.read()
            tmp = self._save_upload(content, ".json")
            status, config, config_json = self.app.load_json_config(tmp)
            self.ui["load_status"].text = status
            self.ui["config_display"].value = config_json if config_json else ""
            self.ui["config_json"].value = config_json if config_json else ""
            ui.notify(status, type="positive" if "success" in status.lower() else "warning")
        except Exception as exc:
            ui.notify(f"Upload error: {exc}", type="negative")

    def _on_build_config(self):
        try:
            params = self._collect()
            # Ensure None for cleared selects (NiceGUI returns None, need "" for some)
            for k in ("distribution", "cv_strategy", "ft_oof_strategy"):
                if params.get(k) is None:
                    params[k] = ""
            status, config_json = self.app.build_config_from_ui(**params)
            self.ui["build_status"].text = status
            self.ui["config_json"].value = config_json or ""
            ui.notify(status, type="positive" if "success" in status.lower() else "negative")
        except Exception as exc:
            ui.notify(f"Build error: {exc}", type="negative")

    def _on_save_config(self):
        config_json = self.ui["config_json"].value
        filename = self.ui["save_filename"].value
        status = self.app.save_config(config_json, filename)
        self.ui["save_status"].text = status
        ui.notify(status, type="positive" if "saved" in status.lower() else "negative")

    # ══════════════════════════════════════════════════════════════════
    #  TAB: FT TWO-STEP
    # ══════════════════════════════════════════════════════════════════

    def _tab_ft_workflow(self):
        ui.label("FT-Transformer Two-Step Training").classes("text-lg font-semibold")

        self._guide("FT Two-Step 完整操作流程", [
            "前提：先在 Configuration 标签页配好基础参数并 Build Configuration",
            "Step 1：点击 \"Prepare Config\" 生成 FT 无监督嵌入配置 → 点击 \"Run Step 1\" 训练 FT 嵌入模型",
            "等待 Step 1 完成（日志区域显示 completed）",
            "Step 2：点击 \"Prepare Configs\" 生成 XGB/ResN 配置（自动引用 Step 1 的嵌入结果）",
            "点击 \"Run Step 2 (XGB)\" 或 \"Run Step 2 (ResN)\" 训练最终模型",
        ])
        self._tip(
            "此流程适用于：先用 FT-Transformer 学习无监督特征嵌入，再将嵌入作为增强特征输入 XGB/ResN。"
            "相比直接训练，这种两步法通常能显著提升模型效果。"
        )

        # ── Step 1 ──
        with ui.expansion("Step 1: FT Embedding", icon="looks_one", value=True).classes("w-full"):
            self._info(
                "基于 Configuration 标签页的配置，自动设置 FT role=unsupervised_embedding。"
                "DDP 推荐在多 GPU 环境下开启，Processes 设为 GPU 数量。"
            )
            with ui.row().classes("w-full gap-2 items-end"):
                ft_ddp = ui.checkbox("Use DDP", value=True)
                ft_nproc = ui.number("Processes", value=2).classes("w-32")
            with ui.row().classes("w-full gap-2"):
                ui.button("Prepare Config", icon="settings",
                           on_click=lambda: self._on_ft_step1(ft_ddp.value, ft_nproc.value)
                           ).props("color=primary")
                ui.button("Run Step 1", icon="play_arrow",
                           on_click=self._on_run_ft_step1).props("color=positive")
            self.ui["step1_status"] = ui.label("").classes("text-sm")
            self.ui["step1_config"] = ui.textarea("Step 1 Config").classes(
                "w-full font-mono text-xs"
            ).props("outlined rows=10")
            self.ui["step1_log"] = ui.textarea("Step 1 Logs").classes(
                "w-full font-mono text-xs"
            ).props("readonly outlined rows=12")

        ui.separator()

        # ── Step 2 ──
        with ui.expansion("Step 2: XGB/ResN with Embeddings", icon="looks_two").classes("w-full"):
            self._info(
                "Step 1 完成后，嵌入数据自动保存到 Augmented Data Dir。"
                "Overrides JSON 可自定义输出路径、损失函数等。"
                "XGB 和 ResN 可以分别运行，互不影响。"
            )
            with ui.row().classes("w-full gap-2 items-end"):
                tgt = ui.input("Target Models", value="xgb, resn").classes("flex-grow")
                aug_dir = ui.input("Augmented Data Dir", value="./DataFTUnsupervised").classes(
                    "flex-grow"
                )
            xgb_ov = ui.textarea("XGB Step 2 Overrides", value=self._xgb_step2_tpl).classes(
                "w-full font-mono text-xs"
            ).props("outlined rows=5")
            resn_ov = ui.textarea("ResN Step 2 Overrides", value=self._resn_step2_tpl).classes(
                "w-full font-mono text-xs"
            ).props("outlined rows=5")
            with ui.row().classes("w-full gap-2"):
                ui.button(
                    "Prepare Configs", icon="settings",
                    on_click=lambda: self._on_ft_step2(
                        tgt.value, aug_dir.value, xgb_ov.value, resn_ov.value
                    ),
                ).props("color=primary")
                ui.button("Run Step 2 (XGB)", icon="play_arrow",
                           on_click=lambda: self._on_run_ft_step2("xgb")).props("color=positive")
                ui.button("Run Step 2 (ResN)", icon="play_arrow",
                           on_click=lambda: self._on_run_ft_step2("resn")).props("color=positive")
            self.ui["step2_status"] = ui.label("").classes("text-sm")
            with ui.tabs().classes("w-full") as s2_tabs:
                ui.tab("XGB Config")
                ui.tab("ResN Config")
            with ui.tab_panels(s2_tabs).classes("w-full"):
                with ui.tab_panel("XGB Config"):
                    self.ui["step2_xgb"] = ui.textarea().classes(
                        "w-full font-mono text-xs"
                    ).props("outlined rows=10")
                with ui.tab_panel("ResN Config"):
                    self.ui["step2_resn"] = ui.textarea().classes(
                        "w-full font-mono text-xs"
                    ).props("outlined rows=10")
            self.ui["step2_log"] = ui.textarea("Step 2 Logs").classes(
                "w-full font-mono text-xs"
            ).props("readonly outlined rows=12")

    def _on_ft_step1(self, use_ddp, nproc):
        config_json = self.ui["config_json"].value
        status, step1_json = self.app.prepare_ft_step1(config_json, use_ddp, int(nproc or 2))
        self.ui["step1_status"].text = status
        self.ui["step1_config"].value = step1_json
        ui.notify(status, type="positive" if "prepared" in status.lower() else "warning")

    def _on_run_ft_step1(self):
        step1_json = self.ui["step1_config"].value
        if not step1_json.strip():
            ui.notify("Please prepare Step 1 config first", type="warning")
            return
        runner = _StreamRunner(self.ui["step1_status"], self.ui["step1_log"])
        runner.run(self.app.run_training, step1_json)

    def _on_ft_step2(self, target_models, aug_dir, xgb_ov, resn_ov):
        step1_path = self.app.current_step1_config or "temp_ft_step1_config.json"
        status, xgb_json, resn_json = self.app.prepare_ft_step2(
            step1_path, target_models, aug_dir, xgb_ov, resn_ov
        )
        self.ui["step2_status"].text = status
        self.ui["step2_xgb"].value = xgb_json
        self.ui["step2_resn"].value = resn_json
        ui.notify(status, type="positive" if "prepared" in status.lower() else "warning")

    def _on_run_ft_step2(self, model_type: str):
        key = "step2_xgb" if model_type == "xgb" else "step2_resn"
        config_json = self.ui[key].value
        if not config_json.strip():
            ui.notify(f"Please prepare Step 2 {model_type.upper()} config first", type="warning")
            return
        runner = _StreamRunner(self.ui["step2_status"], self.ui["step2_log"])
        runner.run(self.app.run_training, config_json)

    # ══════════════════════════════════════════════════════════════════
    #  TAB: WORKFLOW
    # ══════════════════════════════════════════════════════════════════

    def _tab_workflow(self):
        self._guide("Workflow 标签页说明", [
            "Run from Config：运行 Configuration 标签页生成的 JSON 配置（训练、解释、增量训练等）",
            "Config-Driven Workflow：上传独立的 workflow JSON，运行绘图、预测、模型对比等后处理任务",
        ])
        self._tip(
            "常规模型训练用 \"Run from Config\"。"
            "如需批量绘图、模型对比等后处理，用 \"Config-Driven Workflow\" 并上传 workflow 配置文件。"
        )

        # ── Section 1: Run from Configuration tab ──
        with ui.expansion("Run from Config", icon="play_arrow", value=True).classes("w-full"):
            self._info(
                "直接读取 Configuration 标签页底部 \"Generated Config\" 文本框中的 JSON 并运行。"
                "无论是上传的还是 Build 生成的，只要文本框中有有效 JSON 即可。"
                "任务类型由 runner.mode 字段决定：entry=训练, explain=解释性分析, "
                "incremental=增量训练, watchdog=监控模式。"
            )

            with ui.row().classes("w-full items-center gap-4"):
                ui.button("Run Task", icon="play_arrow",
                           on_click=self._on_run_task).props("color=primary size=lg")
                self.ui["run_status"] = ui.label("").classes("text-sm")

            self.ui["run_log"] = ui.textarea("Task Logs").classes(
                "w-full font-mono text-xs"
            ).props("readonly outlined rows=18")

            with ui.row().classes("w-full items-center gap-4"):
                ui.button("Open Results Folder", icon="folder_open",
                           on_click=self._on_open_results).props("flat")
                self.ui["folder_status"] = ui.label("").classes("text-sm")

        ui.separator()

        # ── Section 2: Config-driven workflow ──
        with ui.expansion("Config-Driven Workflow", icon="account_tree").classes("w-full"):
            self._info(
                "Workflow 模式说明：pre_oneway=单因素分析, plot_direct=直接模型绘图, "
                "plot_embed=嵌入模型绘图, predict_ft_embed=FT嵌入预测, "
                "compare=模型对比, double_lift=双提升图。"
                "上传或编辑下方 JSON，设置 workflow.mode 和相关配置路径后运行。"
            )

            with ui.row().classes("w-full gap-4"):
                with ui.column().classes("w-64"):
                    ui.upload(
                        label="Upload Workflow Config",
                        auto_upload=True,
                        on_upload=self._on_upload_workflow,
                    ).props("accept=.json").classes("w-full")
                    self.ui["wf_load_status"] = ui.label("").classes("text-sm")
                with ui.column().classes("flex-grow"):
                    wf_tpl = _dump({
                        "workflow": {
                            "mode": "plot_direct",
                            "cfg_path": "config_plot.json",
                            "xgb_cfg_path": "config_xgb_direct.json",
                            "resn_cfg_path": "config_resn_direct.json",
                        }
                    })
                    self.ui["wf_json"] = ui.textarea(
                        "Workflow Config (JSON)", value=wf_tpl,
                    ).classes("w-full font-mono text-xs").props("outlined rows=12")

            ui.button("Run Workflow", icon="play_arrow",
                       on_click=self._on_run_workflow).props("color=primary size=lg")
            self.ui["wf_status"] = ui.label("").classes("text-sm")
            self.ui["wf_log"] = ui.textarea("Workflow Logs").classes(
                "w-full font-mono text-xs"
            ).props("readonly outlined rows=16")

    async def _on_upload_workflow(self, e):
        try:
            content = await e.file.read()
            tmp = self._save_upload(content, ".json")
            status, wf_json = self.app.load_workflow_config(tmp)
            self.ui["wf_load_status"].text = status
            self.ui["wf_json"].value = wf_json
            ui.notify(status, type="positive" if "loaded" in status.lower() else "warning")
        except Exception as exc:
            ui.notify(f"Upload error: {exc}", type="negative")

    def _on_run_workflow(self):
        wf_json = self.ui["wf_json"].value
        runner = _StreamRunner(self.ui["wf_status"], self.ui["wf_log"])
        runner.run(self.app.run_workflow_config_ui, wf_json)

    def _on_run_task(self):
        config_json = self.ui["config_json"].value
        runner = _StreamRunner(self.ui["run_status"], self.ui["run_log"])
        runner.run(self.app.run_training, config_json)

    def _on_open_results(self):
        config_json = self.ui["config_json"].value
        status = self.app.open_results_folder(config_json)
        self.ui["folder_status"].text = status
        ui.notify(status)

    # ══════════════════════════════════════════════════════════════════
    #  TAB: PLOTTING
    # ══════════════════════════════════════════════════════════════════

    def _tab_plotting(self):
        self._guide("Plotting 标签页使用说明", [
            "Pre Oneway：训练前的单因素分析，查看各特征与目标变量的关系",
            "Direct Plot：对直接训练的 XGB/ResN 模型生成诊断图表",
            "Embed Plot：对 FT 嵌入训练的模型生成诊断图表",
            "Double Lift：比较两个模型预测结果的提升图",
            "FT-Embed Compare：对比直接训练 vs FT 嵌入训练的同一模型",
        ])
        self._tip(
            "绘图功能需要对应的训练结果目录和配置文件。"
            "config 路径支持相对路径（相对于 Working Directory）。"
            "点击 \"Load Factors\" 可从配置文件中自动加载可用的单因素分析因子。"
        )
        with ui.tabs().classes("w-full") as plot_tabs:
            ui.tab("Pre Oneway")
            ui.tab("Direct Plot")
            ui.tab("Embed Plot")
            ui.tab("Double Lift")
            ui.tab("FT-Embed Compare")

        with ui.tab_panels(plot_tabs).classes("w-full"):
            with ui.tab_panel("Pre Oneway"):
                self._subtab_pre_oneway()
            with ui.tab_panel("Direct Plot"):
                self._subtab_direct_plot()
            with ui.tab_panel("Embed Plot"):
                self._subtab_embed_plot()
            with ui.tab_panel("Double Lift"):
                self._subtab_double_lift()
            with ui.tab_panel("FT-Embed Compare"):
                self._subtab_compare()

    def _oneway_factor_loader(self, cfg_input, factors_select, status_label):
        """Load oneway factors from a config file into a select component."""
        def load():
            status, choices, selected = self.app.suggest_oneway_factors(cfg_input.value)
            factors_select.options = choices
            factors_select.value = selected
            status_label.text = status
        return load

    def _subtab_pre_oneway(self):
        self._info(
            "训练前的特征探索：为每个因子生成 actual vs weight 的分箱图。"
            "Data Path 指向原始数据文件。Plot Config 用于读取特征列表和绘图参数。"
        )
        with ui.row().classes("w-full gap-4"):
            with ui.column().classes("flex-[3] gap-1"):
                with ui.row().classes("w-full gap-2"):
                    pre_data = ui.input("Data Path", value="./Data/od_bc.csv").classes("flex-grow")
                    pre_out = ui.input("Output Dir (optional)", value="").classes("flex-grow")
                with ui.row().classes("w-full gap-2"):
                    pre_model = ui.input("Model Name", value="od_bc").classes("flex-1")
                    pre_tgt = ui.input("Target", value="response").classes("flex-1")
                    pre_wgt = ui.input("Weight", value="weights").classes("flex-1")
                with ui.row().classes("w-full gap-2"):
                    pre_cfg = ui.input("Plot Config", value="config_plot.json").classes("flex-grow")
                    pre_fac_status = ui.label("").classes("text-xs text-gray-500")
                pre_factors = ui.select(
                    [], label="Oneway Factors", multiple=True,
                ).classes("w-full")
                ui.button("Load Factors", icon="refresh",
                           on_click=self._oneway_factor_loader(pre_cfg, pre_factors, pre_fac_status)
                           ).props("flat dense")
                with ui.expansion("Advanced: Split Data Override", icon="settings").classes("w-full"):
                    with ui.row().classes("w-full gap-2"):
                        pre_train = ui.input("Train Data Path", value="").classes("flex-1")
                        pre_test = ui.input("Test Data Path", value="").classes("flex-1")
            with ui.column().classes("flex-[2] gap-1"):
                pre_feat = ui.input("Fallback Feature List", value="").classes("w-full")
                pre_cat = ui.input("Categorical Features", value="").classes("w-full")
                with ui.row().classes("w-full gap-2"):
                    pre_bins = ui.number("Bins", value=10)
                    pre_hold = ui.number("Holdout", value=0.25, min=0, max=0.5, step=0.05)
                    pre_seed = ui.number("Seed", value=13)

        pre_status = ui.label("").classes("text-sm")
        pre_log = ui.textarea("Logs").classes("w-full font-mono text-xs").props("readonly outlined rows=10")
        pre_gallery = ui.row().classes("w-full flex-wrap gap-2")

        ui.button("Run Pre Oneway", icon="play_arrow", on_click=lambda: _StreamRunner(
            pre_status, pre_log, pre_gallery
        ).run(
            self.app.run_pre_oneway_ui,
            pre_data.value, pre_model.value, pre_tgt.value, pre_wgt.value,
            pre_feat.value, pre_factors.value, pre_cat.value, int(pre_bins.value or 10),
            float(pre_hold.value or 0.25), int(pre_seed.value or 13),
            pre_out.value, pre_train.value, pre_test.value,
        )).props("color=primary")

    def _subtab_direct_plot(self):
        self._info(
            "对直接训练的模型绘制诊断图：需提供 Plot Config 和各模型的训练配置文件。"
        )
        with ui.row().classes("w-full gap-2"):
            d_cfg = ui.input("Plot Config", value="config_plot.json").classes("flex-1")
            d_xgb = ui.input("XGB Config", value="config_xgb_direct.json").classes("flex-1")
            d_resn = ui.input("ResN Config", value="config_resn_direct.json").classes("flex-1")
        with ui.row().classes("w-full gap-2"):
            d_fac_status = ui.label("").classes("text-xs text-gray-500")
            d_factors = ui.select([], label="Oneway Factors", multiple=True).classes("flex-grow")
            ui.button("Load Factors", icon="refresh",
                       on_click=self._oneway_factor_loader(d_cfg, d_factors, d_fac_status)
                       ).props("flat dense")
        with ui.expansion("Advanced: Data/Model Overrides", icon="settings").classes("w-full"):
            with ui.row().classes("w-full gap-2"):
                d_train = ui.input("Train Data Path", value="").classes("flex-1")
                d_test = ui.input("Test Data Path", value="").classes("flex-1")

        d_status = ui.label("").classes("text-sm")
        d_log = ui.textarea("Logs").classes("w-full font-mono text-xs").props("readonly outlined rows=10")
        d_gallery = ui.row().classes("w-full flex-wrap gap-2")

        ui.button("Run Direct Plot", icon="play_arrow", on_click=lambda: _StreamRunner(
            d_status, d_log, d_gallery
        ).run(
            self.app.run_plot_direct_ui,
            d_cfg.value, d_xgb.value, d_resn.value, d_factors.value,
            d_train.value, d_test.value, None, None,
        )).props("color=primary")

    def _subtab_embed_plot(self):
        self._info(
            "对 FT Two-Step 流程训练的模型绘制诊断图。"
            "需要 FT 嵌入配置和对应的 XGB/ResN 嵌入配置。"
            "Runtime FT Embedding：勾选后在绘图时实时计算嵌入（较慢但无需预计算）。"
        )
        with ui.row().classes("w-full gap-2"):
            e_cfg = ui.input("Plot Config", value="config_plot.json").classes("flex-1")
            e_ft = ui.input("FT Config", value="config_ft_unsupervised_ddp_embed.json").classes("flex-1")
            e_rt = ui.checkbox("Runtime FT Embedding", value=False)
        with ui.row().classes("w-full gap-2"):
            e_xgb = ui.input("XGB Embed Config", value="config_xgb_from_ft_unsupervised.json").classes("flex-1")
            e_resn = ui.input("ResN Embed Config", value="config_resn_from_ft_unsupervised.json").classes("flex-1")
        with ui.row().classes("w-full gap-2"):
            e_fac_status = ui.label("").classes("text-xs text-gray-500")
            e_factors = ui.select([], label="Oneway Factors", multiple=True).classes("flex-grow")
            ui.button("Load Factors", icon="refresh",
                       on_click=self._oneway_factor_loader(e_cfg, e_factors, e_fac_status)
                       ).props("flat dense")
        with ui.expansion("Advanced: Data/Model Overrides", icon="settings").classes("w-full"):
            with ui.row().classes("w-full gap-2"):
                e_train = ui.input("Train Data Path", value="").classes("flex-1")
                e_test = ui.input("Test Data Path", value="").classes("flex-1")

        e_status = ui.label("").classes("text-sm")
        e_log = ui.textarea("Logs").classes("w-full font-mono text-xs").props("readonly outlined rows=10")
        e_gallery = ui.row().classes("w-full flex-wrap gap-2")

        ui.button("Run Embed Plot", icon="play_arrow", on_click=lambda: _StreamRunner(
            e_status, e_log, e_gallery
        ).run(
            self.app.run_plot_embed_ui,
            e_cfg.value, e_xgb.value, e_resn.value, e_ft.value, e_rt.value,
            e_factors.value, e_train.value, e_test.value, None, None, None,
        )).props("color=primary")

    def _subtab_double_lift(self):
        self._info(
            "双提升图：比较两个模型的预测排序能力。"
            "需要一个包含两列预测值的 CSV 文件。"
            "Holdout=0 表示使用全部数据，>0 则拆分测试集绘图。"
        )
        with ui.row().classes("w-full gap-2"):
            dl_data = ui.input("Data Path (CSV)", value="./Data/od_bc.csv").classes("flex-[3]")
            dl_out = ui.input("Output Image Path (optional)", value="").classes("flex-[2]")
        with ui.row().classes("w-full gap-2"):
            dl_p1 = ui.input("Pred Column 1", value="pred_xgb").classes("flex-1")
            dl_p2 = ui.input("Pred Column 2", value="pred_resn").classes("flex-1")
            dl_tgt = ui.input("Target", value="response").classes("flex-1")
            dl_wgt = ui.input("Weight", value="weights").classes("flex-1")
        with ui.row().classes("w-full gap-2"):
            dl_l1 = ui.input("Label 1", value="Model 1").classes("flex-1")
            dl_l2 = ui.input("Label 2", value="Model 2").classes("flex-1")
            dl_bins = ui.number("Bins", value=10).classes("flex-1")
            dl_seed = ui.number("Seed", value=13).classes("flex-1")
        with ui.row().classes("w-full gap-2"):
            dl_hold = ui.number("Holdout (0=all)", value=0.0, min=0, max=0.5, step=0.05)
            dl_split = ui.select(["random", "stratified", "time", "group"],
                                  label="Split Strategy", value="random")
            dl_gcol = ui.input("Group Col", value="").classes("flex-1")
            dl_tcol = ui.input("Time Col", value="").classes("flex-1")
            dl_tasc = ui.checkbox("Time Ascending", value=True)
        with ui.row().classes("w-full gap-4"):
            dl_pw1 = ui.checkbox("Pred 1 Weighted", value=False)
            dl_pw2 = ui.checkbox("Pred 2 Weighted", value=False)
            dl_aw = ui.checkbox("Actual Weighted", value=False)

        dl_status = ui.label("").classes("text-sm")
        dl_log = ui.textarea("Logs").classes("w-full font-mono text-xs").props("readonly outlined rows=10")
        dl_gallery = ui.row().classes("w-full flex-wrap gap-2")

        ui.button("Run Double Lift", icon="play_arrow", on_click=lambda: _StreamRunner(
            dl_status, dl_log, dl_gallery
        ).run(
            self.app.run_double_lift_ui,
            dl_data.value, dl_p1.value, dl_p2.value, dl_tgt.value, dl_wgt.value,
            int(dl_bins.value or 10), dl_l1.value, dl_l2.value,
            dl_pw1.value, dl_pw2.value, dl_aw.value,
            float(dl_hold.value or 0), dl_split.value, dl_gcol.value, dl_tcol.value,
            dl_tasc.value, int(dl_seed.value or 13), dl_out.value,
        )).props("color=primary")

    def _subtab_compare(self):
        self._info(
            "对比直接训练的模型与 FT 嵌入增强模型的效果差异。"
            "选择 Model Key 后会自动填充默认配置路径和标签。"
        )
        with ui.row().classes("w-full gap-2"):
            c_key = ui.select(["xgb", "resn"], label="Model Key", value="xgb").classes("w-32")
            c_direct = ui.input("Direct Config", value="config_xgb_direct.json").classes("flex-1")
            c_ft = ui.input("FT Config", value="config_ft_unsupervised_ddp_embed.json").classes("flex-1")
            c_embed = ui.input("FT-Embed Config", value="config_xgb_from_ft_unsupervised.json").classes("flex-1")
        with ui.row().classes("w-full gap-2"):
            c_ld = ui.input("Direct Label", value="XGB_raw").classes("flex-1")
            c_lf = ui.input("FT Label", value="XGB_ft_embed").classes("flex-1")
            c_rt = ui.checkbox("Runtime FT Embedding", value=False)
            c_bins = ui.number("Bins", value=10)

        def _suggest_defaults():
            key = str(c_key.value or "").lower()
            if key == "resn":
                c_direct.value = "config_resn_direct.json"
                c_embed.value = "config_resn_from_ft_unsupervised.json"
                c_ld.value = "ResN_raw"
                c_lf.value = "ResN_ft_embed"
            else:
                c_direct.value = "config_xgb_direct.json"
                c_embed.value = "config_xgb_from_ft_unsupervised.json"
                c_ld.value = "XGB_raw"
                c_lf.value = "XGB_ft_embed"
        c_key.on_value_change(lambda _: _suggest_defaults())

        with ui.expansion("Advanced: Data/Model Overrides", icon="settings").classes("w-full"):
            with ui.row().classes("w-full gap-2"):
                c_train = ui.input("Train Data Path", value="").classes("flex-1")
                c_test = ui.input("Test Data Path", value="").classes("flex-1")

        c_status = ui.label("").classes("text-sm")
        c_log = ui.textarea("Logs").classes("w-full font-mono text-xs").props("readonly outlined rows=10")
        c_gallery = ui.row().classes("w-full flex-wrap gap-2")

        ui.button("Run Compare", icon="play_arrow", on_click=lambda: _StreamRunner(
            c_status, c_log, c_gallery
        ).run(
            self.app.run_compare_ui,
            c_key.value, c_direct.value, c_ft.value, c_embed.value,
            c_ld.value, c_lf.value, c_rt.value, int(c_bins.value or 10),
            c_train.value, c_test.value, None, None, None,
        )).props("color=primary")

    # ══════════════════════════════════════════════════════════════════
    #  TAB: PREDICTION
    # ══════════════════════════════════════════════════════════════════

    def _tab_prediction(self):
        ui.label("FT Embed Prediction").classes("text-lg font-semibold")

        self._guide("Prediction 使用说明", [
            "用 FT Two-Step 训练好的模型对新数据进行预测",
            "FT Config 指向 FT 嵌入模型配置（必填）",
            "XGB/ResN Config 指向对应的预测模型配置（至少填一个）",
            "Input Data 为待预测数据（CSV），Output CSV 为预测结果输出路径",
        ])
        self._tip(
            "Model Keys 决定使用哪些模型预测（xgb/resn），用逗号分隔。"
            "Model Name 为空时会自动从配置中推断。"
        )

        with ui.row().classes("w-full gap-2"):
            p_ft = ui.input("FT Config", value="config_ft_unsupervised_ddp_embed.json").classes("flex-1")
            p_xgb = ui.input("XGB Config (optional)",
                              value="config_xgb_from_ft_unsupervised.json").classes("flex-1")
            p_resn = ui.input("ResN Config (optional)",
                               value="config_resn_from_ft_unsupervised.json").classes("flex-1")
        with ui.row().classes("w-full gap-2"):
            p_name = ui.input("Model Name (optional)", value="").classes("flex-1")
            p_keys = ui.input("Model Keys", value="xgb, resn").classes("flex-1")
            p_in = ui.input("Input Data", value="./Data/od_bc_new.csv").classes("flex-1")
            p_out = ui.input("Output CSV", value="./Results/predictions_ft_xgb.csv").classes("flex-1")

        p_status = ui.label("").classes("text-sm")
        p_log = ui.textarea("Logs").classes("w-full font-mono text-xs").props("readonly outlined rows=12")

        ui.button("Run Prediction", icon="play_arrow", on_click=lambda: _StreamRunner(
            p_status, p_log
        ).run(
            self.app.run_predict_ui,
            p_ft.value, p_xgb.value, p_resn.value, p_in.value,
            p_out.value, p_name.value, p_keys.value,
            None, None, None,
        )).props("color=primary")
