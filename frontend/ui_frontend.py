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

from ins_pricing.frontend.access_control import AuthorizationError
from ins_pricing.frontend.app_controller import PricingApp
from ins_pricing.frontend.config_comments_default import DEFAULT_CONFIG_COMMENTS


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?#  Helpers
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
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


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?#  Main Frontend
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
class PricingFrontend:
    """Build the complete NiceGUI interface."""

    def __init__(self, pricing_app: PricingApp):
        self.app = pricing_app
        self.cfg: Dict[str, Any] = {}  # config鈥慺orm components
        self.ui: Dict[str, Any] = {}   # non鈥慶onfig UI elements
        self.current_user: Optional[Dict[str, Any]] = None
        self._permission_controls: List[tuple[Any, str]] = []
        self._header_menu_visible = False
        self._account_page_open = False
        self._ui_scale = 1.0

        cb = pricing_app.config_builder
        self._ss = {
            "xgb": _dump(cb._default_xgb_search_space()),
            "resn": _dump(cb._default_resn_search_space()),
            "ft": _dump(cb._default_ft_search_space()),
            "ft_unsup": _dump(cb._default_ft_unsupervised_search_space()),
        }
        self._xgb_step2_tpl = _dump({
            "output_dir": "./ResultsXGBFromFTEmbed",
            "optuna_storage": "./ResultsXGBFromFTEmbed/optuna/bayesopt.sqlite3",
            "distribution": "tweedie",
            "build_oht": False,
            "final_refit": True,
            "cache_predictions": True,
            "prediction_cache_format": "csv",
            "runner": {"model_keys": ["xgb"], "nproc_per_node": 1, "plot_curves": False},
            "plot_curves": False, "plot": {"enable": False},
        })
        self._resn_step2_tpl = _dump({
            "output_dir": "./ResultsResNFromFTEmbed",
            "optuna_storage": "./ResultsResNFromFTEmbed/optuna/bayesopt.sqlite3",
            "distribution": "tweedie",
            "build_oht": False,
            "use_resn_ddp": False,
            "cache_predictions": True,
            "prediction_cache_format": "csv",
            "runner": {"model_keys": ["resn"], "nproc_per_node": 1, "plot_curves": False},
            "plot_curves": False, "plot": {"enable": False},
        })
        self._config_comments: Dict[str, str] = self._load_config_comments()

    @staticmethod
    def _inject_theme() -> None:
        ui.add_head_html("""
        <style>
          :root {
            --ui-scale: 1.0;
            --page-bg:
              radial-gradient(circle at top left, rgba(210, 228, 223, 0.9), transparent 28%),
              radial-gradient(circle at top right, rgba(232, 220, 204, 0.78), transparent 26%),
              linear-gradient(180deg, #f6f2ea 0%, #f1ede5 52%, #ece7de 100%);
            --panel: rgba(255, 252, 246, 0.84);
            --panel-strong: rgba(255, 252, 246, 0.96);
            --panel-border: rgba(106, 92, 73, 0.12);
            --ink: #201a16;
            --muted: #6a5c4d;
            --accent: #355c55;
            --accent-soft: #dce8e3;
            --warm: #9a6a3a;
            --shadow: 0 20px 60px rgba(44, 34, 24, 0.08);
          }

          body {
            background: var(--page-bg);
            color: var(--ink);
            font-size: calc(16px * var(--ui-scale));
            line-height: 1.55;
          }

          .app-shell {
            position: relative;
          }

          .app-shell::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background-image:
              linear-gradient(rgba(53, 92, 85, 0.04) 1px, transparent 1px),
              linear-gradient(90deg, rgba(53, 92, 85, 0.04) 1px, transparent 1px);
            background-size: 36px 36px;
            mask-image: linear-gradient(180deg, rgba(0, 0, 0, 0.32), transparent 82%);
          }

          .hero-panel,
          .main-panel,
          .section-panel,
          .tone-panel {
            background: var(--panel);
            border: 1px solid var(--panel-border);
            box-shadow: var(--shadow);
            backdrop-filter: blur(10px);
          }

          .hero-panel {
            background:
              linear-gradient(135deg, rgba(255, 250, 242, 0.94), rgba(247, 243, 235, 0.82)),
              linear-gradient(120deg, rgba(53, 92, 85, 0.06), rgba(154, 106, 58, 0.03));
          }

          .main-panel {
            background: var(--panel-strong);
          }

          .section-panel .q-expansion__content {
            padding-top: calc(14px * var(--ui-scale));
          }

          .soft-tabs {
            gap: calc(12px * var(--ui-scale));
            padding: calc(12px * var(--ui-scale));
            border-bottom: 1px solid rgba(106, 92, 73, 0.1);
            background: rgba(249, 245, 237, 0.78);
          }

          .soft-tabs .q-tab {
            min-height: calc(48px * var(--ui-scale));
            border-radius: 999px;
            padding: 0 calc(20px * var(--ui-scale));
            color: var(--muted);
            transition: all 0.18s ease;
            font-size: calc(0.98rem * var(--ui-scale));
          }

          .soft-tabs .q-tab.q-tab--active {
            background: #f7f3eb;
            color: var(--accent);
            box-shadow: inset 0 0 0 1px rgba(53, 92, 85, 0.14);
          }

          .soft-input .q-field__control,
          .soft-textarea .q-field__control,
          .soft-select .q-field__control {
            background: rgba(255, 252, 246, 0.94);
            border-radius: 9px;
            border: 1px solid rgba(106, 92, 73, 0.12);
            box-shadow: none;
            min-height: calc(44px * var(--ui-scale));
          }

          .soft-input .q-field__native,
          .soft-textarea textarea,
          .soft-select .q-field__native,
          .soft-select .q-field__input {
            color: var(--ink);
            font-size: calc(14px * var(--ui-scale));
            line-height: 1.35;
            font-weight: 400;
          }

          .soft-input .q-field__label,
          .soft-textarea .q-field__label,
          .soft-select .q-field__label,
          .soft-input .q-field__marginal,
          .soft-textarea .q-field__marginal,
          .soft-select .q-field__marginal {
            font-size: calc(13px * var(--ui-scale));
            line-height: 1.2;
            color: rgba(106, 92, 73, 0.88);
          }

          .soft-input.q-field--focused .q-field__control,
          .soft-textarea.q-field--focused .q-field__control,
          .soft-select.q-field--focused .q-field__control {
            border-color: rgba(53, 92, 85, 0.52);
            box-shadow: none;
            background: rgba(255, 253, 249, 0.98);
          }

          .soft-input.q-field--focused .q-field__label,
          .soft-textarea.q-field--focused .q-field__label,
          .soft-select.q-field--focused .q-field__label {
            color: var(--accent);
          }

          .soft-textarea textarea {
            font-size: calc(13px * var(--ui-scale));
            line-height: 1.45;
          }

          .q-checkbox__label {
            font-size: calc(14px * var(--ui-scale));
            color: var(--muted);
          }

          .q-field--dense .q-field__control,
          .q-field--dense .q-field__native,
          .q-field--dense .q-field__label {
            font-size: calc(13px * var(--ui-scale));
          }

          .q-btn {
            font-size: calc(0.96rem * var(--ui-scale));
            min-height: calc(38px * var(--ui-scale));
            padding: 0 calc(16px * var(--ui-scale));
          }

          .q-btn--dense {
            min-height: calc(34px * var(--ui-scale));
            padding: 0 calc(12px * var(--ui-scale));
            font-size: calc(0.9rem * var(--ui-scale));
          }

          .q-tab__label {
            font-size: calc(0.95rem * var(--ui-scale));
          }

          .text-xs {
            font-size: calc(0.875rem * var(--ui-scale)) !important;
            line-height: 1.45 !important;
          }

          .text-sm {
            font-size: calc(0.98rem * var(--ui-scale)) !important;
            line-height: 1.5 !important;
          }

          .text-base {
            font-size: calc(1.08rem * var(--ui-scale)) !important;
            line-height: 1.55 !important;
          }

          .text-lg {
            font-size: calc(1.22rem * var(--ui-scale)) !important;
            line-height: 1.45 !important;
          }

          .text-xl {
            font-size: calc(1.38rem * var(--ui-scale)) !important;
            line-height: 1.4 !important;
          }

          .text-2xl {
            font-size: calc(1.7rem * var(--ui-scale)) !important;
            line-height: 1.35 !important;
          }

          .muted-copy {
            color: var(--muted);
          }

          .section-title {
            letter-spacing: 0.08em;
            text-transform: uppercase;
          }

          .resource-progress .q-linear-progress__track,
          .resource-progress .q-linear-progress__model {
            border-radius: 999px;
          }

          .resource-progress .q-linear-progress__label,
          .resource-progress .q-linear-progress__text,
          .resource-progress .q-linear-progress__content {
            display: none !important;
          }
        </style>
        """)

    # 鈹€鈹€ UI helpers 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    @staticmethod
    def _tip(text: str, icon: str = "lightbulb"):
        """Render an inline tip banner."""
        with ui.element("div").classes(
            "tone-panel w-full rounded-2xl px-4 py-3 flex items-start gap-3 my-2"
        ):
            ui.icon(icon).classes("text-[var(--warm)] mt-0.5 text-base")
            ui.label(text).classes("text-xs leading-relaxed text-[var(--ink)]")

    @staticmethod
    def _guide(title: str, steps: list):
        """Render a numbered step guide."""
        with ui.element("div").classes(
            "tone-panel w-full rounded-2xl px-4 py-3 my-2"
        ):
            ui.label(title).classes("text-sm font-semibold text-[var(--accent)] mb-1")
            for i, step in enumerate(steps, 1):
                ui.label(f"{i}. {step}").classes("text-xs leading-relaxed ml-2 text-[var(--muted)]")

    @staticmethod
    def _info(text: str, icon: str = "info"):
        """Render an info note."""
        with ui.element("div").classes(
            "tone-panel w-full rounded-2xl px-4 py-3 flex items-start gap-3 my-2"
        ):
            ui.icon(icon).classes("text-[var(--accent)] mt-0.5 text-base")
            ui.label(text).classes("text-xs leading-relaxed text-[var(--muted)]")

    # 鈹€鈹€ component helpers (register in self.cfg) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    @staticmethod
    def _join_classes(*parts: Optional[str]) -> str:
        return " ".join(part for part in parts if part)

    def _inp(self, key, label, value="", **kw):
        c = ui.input(label, value=str(value), **kw).classes("w-full soft-input").props("dense")
        self.cfg[key] = c
        self._bind_config_help(key=key, label=label, control=c)
        return c

    def _num(self, key, label, value=0, **kw):
        c = ui.number(label, value=value, **kw).classes("w-full soft-input").props("dense")
        self.cfg[key] = c
        self._bind_config_help(key=key, label=label, control=c)
        return c

    def _sel(self, key, label, options, value=None, **kw):
        c = ui.select(options, label=label, value=value, **kw).classes("w-full soft-select").props("dense")
        self.cfg[key] = c
        self._bind_config_help(key=key, label=label, control=c)
        return c

    def _chk(self, key, label, value=False):
        c = ui.checkbox(label, value=value)
        self.cfg[key] = c
        self._bind_config_help(key=key, label=label, control=c)
        return c

    def _txt(self, key, label, value="", **kw):
        c = ui.textarea(label, value=str(value), **kw).classes("w-full font-mono text-xs soft-textarea").props("dense")
        self.cfg[key] = c
        self._bind_config_help(key=key, label=label, control=c)
        return c

    def _field_inp(self, label, value="", classes="w-full", **kw):
        return ui.input(label, value=str(value), **kw).classes(
            self._join_classes(classes, "soft-input")
        ).props("dense")

    def _field_num(self, label, value=0, classes="w-full", **kw):
        return ui.number(label, value=value, **kw).classes(
            self._join_classes(classes, "soft-input")
        ).props("dense")

    def _field_sel(self, options, label, value=None, classes="w-full", **kw):
        return ui.select(options, label=label, value=value, **kw).classes(
            self._join_classes(classes, "soft-select")
        ).props("dense")

    def _field_txt(self, label="", value="", classes="w-full", mono=False, **kw):
        mono_classes = "font-mono text-xs" if mono else ""
        return ui.textarea(label, value=str(value), **kw).classes(
            self._join_classes(classes, mono_classes, "soft-textarea")
        ).props("dense")

    def _collect(self) -> Dict[str, Any]:
        """Collect all config component values into a dict matching build_config_from_ui params."""
        return {k: c.value for k, c in self.cfg.items()}

    @staticmethod
    def _default_config_comments() -> Dict[str, str]:
        # Packaged default comments (generated from examples/config_template.json::__comments).
        return dict(DEFAULT_CONFIG_COMMENTS)

    def _load_config_comments(self) -> Dict[str, str]:
        comments = dict(self._default_config_comments())
        template_path = Path(__file__).resolve().parents[2] / "examples" / "config_template.json"
        if not template_path.exists():
            return comments
        try:
            data = json.loads(template_path.read_text(encoding="utf-8"))
            template_comments = data.get("__comments", {})
            if isinstance(template_comments, dict):
                for k, v in template_comments.items():
                    if not isinstance(k, str):
                        continue
                    if not isinstance(v, str):
                        continue
                    if k.startswith("__"):
                        continue
                    comments[k] = v.strip()
        except Exception:
            pass
        return comments

    def _show_config_help(self, key: str, label: str) -> None:
        panel = self.ui.get("cfg_help_panel")
        title = self.ui.get("cfg_help_title")
        body = self.ui.get("cfg_help_body")
        if panel is None or title is None or body is None:
            return
        text = self._resolve_config_help_text(str(key), str(label))
        title.text = f"{label} ({key})"
        body.text = text
        panel.set_visibility(True)

    def _hide_config_help(self) -> None:
        panel = self.ui.get("cfg_help_panel")
        if panel is not None:
            panel.set_visibility(False)

    def _bind_config_help(self, *, key: str, label: str, control: Any) -> None:
        if not isinstance(key, str) or not key.strip():
            return
        try:
            control.on("focus", lambda _e, k=key, lb=label: self._show_config_help(k, lb))
            control.on("blur", lambda _e: self._hide_config_help())
            control.on("click", lambda _e, k=key, lb=label: self._show_config_help(k, lb))
        except Exception:
            return

    def _resolve_config_help_text(self, key: str, label: str) -> str:
        key_norm = str(key or "").strip()
        label_norm = str(label or "").strip()
        candidates = [key_norm]
        if key_norm.endswith("_json"):
            candidates.append(key_norm[:-5])
        candidates.append(key_norm.replace("__", "."))
        candidates.append(key_norm.replace("_", "."))

        alias_map = {
            "max_evals": "runner.max_evals",
            "nproc_per_node": "runner.nproc_per_node",
            "xgb_search_space_json": "xgb_search_space",
            "resn_search_space_json": "resn_search_space",
            "ft_search_space_json": "ft_search_space",
            "ft_unsupervised_search_space_json": "ft_unsupervised_search_space",
            "config_overrides_json": "runner",
        }
        mapped = alias_map.get(key_norm)
        if mapped:
            candidates.append(mapped)

        for cand in candidates:
            text = str(self._config_comments.get(cand, "")).strip()
            if text:
                return text

        if label_norm:
            return f"No template comment found for this field. It controls: {label_norm}."
        return "No template comment found for this field."

    # 鈹€鈹€ file upload helper 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def _current_username(self) -> Optional[str]:
        if not self.current_user:
            return None
        username = str(self.current_user.get("username", "")).strip()
        return username or None

    def _actor_for_runtime(self) -> str:
        return self._current_username() or ""

    def _has_permission(self, permission: str) -> bool:
        return self.app.has_permission(self._current_username(), permission)

    def _require_ui_permission(self, permission: str) -> bool:
        if self._has_permission(permission):
            return True
        if self._current_username():
            ui.notify(f"Permission denied: {permission}", type="negative")
        else:
            ui.notify("Please log in first.", type="warning")
        return False

    def _register_permission_control(self, control: Any, permission: str) -> Any:
        self._permission_controls.append((control, permission))
        return control

    def _refresh_workspace_visibility(self) -> None:
        is_logged_in = bool(self._current_username())
        show_main = is_logged_in and (not self._account_page_open)
        for key in ("workspace_top_container", "workspace_wd_container", "workspace_main_container"):
            container = self.ui.get(key)
            if container is not None:
                container.set_visibility(show_main)
        back_to_top = self.ui.get("back_to_top_btn")
        if back_to_top is not None:
            back_to_top.set_visibility(show_main)
        cfg_help_panel = self.ui.get("cfg_help_panel")
        if cfg_help_panel is not None and (not show_main):
            cfg_help_panel.set_visibility(False)
        guard = self.ui.get("workspace_guard")
        if guard is not None:
            guard.set_visibility(False)
        login_page = self.ui.get("login_page")
        if login_page is not None:
            login_page.set_visibility(not is_logged_in)
        account_page = self.ui.get("account_page")
        if account_page is not None:
            account_page.set_visibility(is_logged_in and self._account_page_open)

    def _set_header_menu_visible(self, visible: bool) -> None:
        self._header_menu_visible = bool(visible)
        for key in ("header_menu_backdrop", "header_menu_panel"):
            panel = self.ui.get(key)
            if panel is not None:
                panel.set_visibility(self._header_menu_visible)

    def _toggle_header_menu(self) -> None:
        self._set_header_menu_visible(not self._header_menu_visible)

    def _apply_ui_scale(self, scale: float) -> None:
        try:
            value = float(scale)
        except Exception:
            value = 1.0
        value = min(1.2, max(0.9, value))
        self._ui_scale = value
        ui.run_javascript(
            f"document.documentElement.style.setProperty('--ui-scale', '{value:.2f}')"
        )

    def _on_ui_scale_change(self, e: Any) -> None:
        raw = str(getattr(e, "value", "") or "").strip()
        if raw.endswith("%"):
            raw = raw[:-1].strip()
        try:
            value = float(raw) / 100.0 if float(raw) > 3 else float(raw)
        except Exception:
            value = 1.0
        self._apply_ui_scale(value)

    def _set_account_page_visible(self, visible: bool) -> None:
        self._account_page_open = bool(visible and self._current_username())
        page = self.ui.get("account_page")
        if page is not None:
            page.set_visibility(self._account_page_open)
        back_btn = self.ui.get("account_back_btn")
        if back_btn is not None:
            back_btn.set_visibility(bool(self._current_username()) and self._account_page_open)
        self._refresh_workspace_visibility()

    def _open_account_page_from_menu(self) -> None:
        self._set_header_menu_visible(False)
        if not self._current_username():
            ui.notify("Please sign in first.", type="warning")
            return
        self._set_account_page_visible(True)

    def _close_account_page(self) -> None:
        self._set_account_page_visible(False)

    def _refresh_permission_controls(self) -> None:
        self._refresh_workspace_visibility()
        for control, permission in self._permission_controls:
            allowed = self._has_permission(permission)
            try:
                if allowed:
                    control.enable()
                else:
                    control.disable()
            except Exception:
                continue
        self._refresh_account_admin_visibility()
        self._update_auth_summary()
        self._refresh_system_status_panel()
        self._refresh_accounts_table()

    def _refresh_auth_controls(self) -> None:
        logged_in = bool(self._current_username())
        login_row = self.ui.get("auth_login_row")
        if login_row is not None:
            login_row.set_visibility(not logged_in)
        session_row = self.ui.get("auth_session_row")
        if session_row is not None:
            session_row.set_visibility(logged_in)
        header_logout = self.ui.get("header_logout_btn")
        if header_logout is not None:
            header_logout.set_visibility(logged_in)

    def _update_auth_summary(self) -> None:
        self._refresh_auth_controls()
        summary = self.ui.get("auth_summary")
        session_user = self.ui.get("auth_session_user")
        header_user = self.ui.get("header_user_badge")
        if summary is not None:
            if self.current_user:
                username = str(self.current_user.get("username", "")).strip()
                roles = ", ".join(self.current_user.get("roles", []))
                summary.text = f"Signed in as {username} [{roles}]"
                if session_user is not None:
                    session_user.text = f"{username} [{roles}]"
                if header_user is not None:
                    header_user.text = username
            else:
                summary.text = "Not signed in"
                if session_user is not None:
                    session_user.text = ""
                if header_user is not None:
                    header_user.text = "Guest"

        note = self.ui.get("auth_default_pwd_note")
        if note is not None:
            meta = self.app.auth_metadata()
            if bool(meta.get("default_admin_password_in_use", False)):
                note.text = (
                    "Security notice: using default admin password. "
                    "Set INS_PRICING_FRONTEND_ADMIN_PASSWORD before deploying."
                )
            else:
                note.text = ""

    def _on_login(self) -> None:
        username = str(self.ui.get("auth_username").value or "").strip()
        password = str(self.ui.get("auth_password").value or "")
        if not username or not password:
            ui.notify("Username and password are required.", type="warning")
            return
        try:
            self._set_header_menu_visible(False)
            self.current_user = self.app.authenticate_user(username, password)
            self.ui["auth_password"].value = ""
            workspace_dir = str(self.current_user.get("workspace_dir", "") or "").strip()
            if workspace_dir:
                self.ui["wd_input"].value = workspace_dir
                self.ui["wd_browse"].value = workspace_dir
                self._refresh_wd(workspace_dir)
            self._reset_and_refresh_workdir_files()
            self._set_account_page_visible(False)
            self._refresh_permission_controls()
            ui.notify(f"Signed in as {username}", type="positive")
        except AuthorizationError as exc:
            self.current_user = None
            self._refresh_permission_controls()
            ui.notify(str(exc), type="negative")
        except Exception as exc:
            self.current_user = None
            self._refresh_permission_controls()
            ui.notify(f"Sign-in error: {exc}", type="negative")

    def _on_logout(self) -> None:
        self._set_header_menu_visible(False)
        self.app.clear_active_user_workspace()
        self.current_user = None
        if "auth_password" in self.ui:
            self.ui["auth_password"].value = ""
        for key in (
            "acct_self_current_password",
            "acct_self_new_password",
            "acct_self_confirm_password",
        ):
            if key in self.ui:
                self.ui[key].value = ""
        default_dir = str(self.app.working_dir)
        if "wd_input" in self.ui:
            self.ui["wd_input"].value = default_dir
        if "wd_browse" in self.ui:
            self.ui["wd_browse"].value = default_dir
        if "wd_status" in self.ui:
            self.ui["wd_status"].text = f"Current: {default_dir}"
        self._reset_and_refresh_workdir_files()
        self._set_account_page_visible(False)
        self._refresh_permission_controls()
        ui.notify("Signed out", type="positive")

    def _refresh_system_status_panel(self) -> None:
        decision = self.ui.get("sys_decision")
        if decision is None:
            return
        if not self._has_permission("system:view"):
            self.ui["sys_cpu_meta"].text = "-"
            self.ui["sys_mem_meta"].text = "-"
            self.ui["sys_disk_meta"].text = "-"
            self.ui["sys_proc_text"].text = "-"
            self.ui["sys_cpu_bar"].value = 0
            self.ui["sys_mem_bar"].value = 0
            self.ui["sys_disk_bar"].value = 0
            gpu_panel = self.ui.get("sys_gpu_panel")
            if gpu_panel is not None:
                gpu_panel.set_visibility(False)
            decision.text = "Login as viewer/operator/admin to view server status."
            self.ui["sys_reason"].text = ""
            return

        try:
            snapshot = self.app.get_system_status(actor=self._current_username())
        except Exception as exc:
            decision.text = f"Resource probe error: {exc}"
            self.ui["sys_reason"].text = ""
            return

        cpu = snapshot.get("cpu_percent")
        cpu_cores = snapshot.get("cpu_logical_cores")
        mem = snapshot.get("memory_percent")
        mem_used_mb = snapshot.get("memory_used_mb")
        mem_total_mb = snapshot.get("memory_total_mb")
        disk = snapshot.get("disk_percent")
        disk_used_gb = snapshot.get("disk_used_gb")
        disk_total_gb = snapshot.get("disk_total_gb")
        proc_mem = snapshot.get("process_memory_mb")

        if cpu is None:
            self.ui["sys_cpu_meta"].text = "-"
        else:
            if isinstance(cpu_cores, (int, float)) and int(cpu_cores) > 0:
                self.ui["sys_cpu_meta"].text = f"{float(cpu):.1f}% / {int(cpu_cores)} logical cores"
            else:
                self.ui["sys_cpu_meta"].text = f"{float(cpu):.1f}%"

        if mem is None or mem_used_mb is None or mem_total_mb is None:
            self.ui["sys_mem_meta"].text = "-"
        else:
            self.ui["sys_mem_meta"].text = (
                f"{float(mem):.1f}% / {float(mem_used_mb):.0f}/{float(mem_total_mb):.0f} MB"
            )

        if disk is None or disk_used_gb is None or disk_total_gb is None:
            self.ui["sys_disk_meta"].text = "-"
        else:
            self.ui["sys_disk_meta"].text = (
                f"{float(disk):.1f}% / {float(disk_used_gb):.1f}/{float(disk_total_gb):.1f} GB"
            )

        self.ui["sys_proc_text"].text = "-" if proc_mem is None else f"{float(proc_mem):.1f} MB (RSS)"
        self.ui["sys_cpu_bar"].value = 0 if cpu is None else min(max(float(cpu) / 100.0, 0), 1)
        self.ui["sys_mem_bar"].value = 0 if mem is None else min(max(float(mem) / 100.0, 0), 1)
        self.ui["sys_disk_bar"].value = 0 if disk is None else min(max(float(disk) / 100.0, 0), 1)

        gpu_info = snapshot.get("gpu", {})
        gpu_available = isinstance(gpu_info, dict) and bool(gpu_info.get("available", False))
        gpu_panel = self.ui.get("sys_gpu_panel")
        if gpu_panel is not None:
            gpu_panel.set_visibility(gpu_available)
        if gpu_available:
            max_util = float(gpu_info.get("max_utilization_percent", 0.0))
            max_mem = float(gpu_info.get("max_memory_percent", 0.0))
            self.ui["sys_gpu_util_bar"].value = min(max(max_util / 100.0, 0), 1)
            self.ui["sys_gpu_mem_bar"].value = min(max(max_mem / 100.0, 0), 1)
            self.ui["sys_gpu_util_meta"].text = f"{max_util:.1f}% / max utilization"

            total_gpu_used_mb = 0.0
            total_gpu_capacity_mb = 0.0
            detail_lines: list[str] = []
            for device in gpu_info.get("devices", []):
                idx = int(device.get("index", 0))
                name = str(device.get("name", "GPU")).strip()
                util = float(device.get("utilization_percent", 0.0))
                mem_used = float(device.get("memory_used_mb", 0.0))
                mem_total = float(device.get("memory_total_mb", 0.0))
                mem_pct = float(device.get("memory_percent", 0.0))
                temp = device.get("temperature_c")
                temp_text = "" if temp is None else f", temp {float(temp):.0f}C"
                total_gpu_used_mb += mem_used
                total_gpu_capacity_mb += mem_total
                detail_lines.append(
                    f"GPU {idx} ({name}): util {util:.1f}%, "
                    f"mem {mem_used:.0f}/{mem_total:.0f} MB ({mem_pct:.1f}%){temp_text}"
                )
            if total_gpu_capacity_mb > 0:
                self.ui["sys_gpu_mem_meta"].text = (
                    f"{max_mem:.1f}% / {total_gpu_used_mb:.0f}/{total_gpu_capacity_mb:.0f} MB"
                )
            else:
                self.ui["sys_gpu_mem_meta"].text = f"{max_mem:.1f}%"
            self.ui["sys_gpu_detail"].text = "\n".join(detail_lines)
        else:
            self.ui["sys_gpu_util_bar"].value = 0
            self.ui["sys_gpu_mem_bar"].value = 0
            self.ui["sys_gpu_util_meta"].text = "-"
            self.ui["sys_gpu_mem_meta"].text = "-"
            self.ui["sys_gpu_detail"].text = ""

        decision.text = str(snapshot.get("decision", "Unknown"))
        reasons = snapshot.get("reasons", [])
        self.ui["sys_reason"].text = "; ".join(str(item) for item in reasons if str(item).strip())

    def _refresh_accounts_table(self) -> None:
        table = self.ui.get("acct_table")
        if table is None:
            return
        if not self._has_permission("account:manage"):
            table.rows = []
            table.update()
            self.ui["acct_status"].text = "Admin permission required."
            return

        actor = self._current_username()
        try:
            users = self.app.list_accounts(actor=actor)
            rows = [
                {
                    "username": item["username"],
                    "roles": ",".join(item.get("roles", [])),
                    "active": bool(item.get("active", True)),
                    "workspace_dir": str(item.get("workspace_dir", "")),
                    "updated_at": str(item.get("updated_at") or ""),
                }
                for item in users
            ]
            table.rows = rows
            table.update()
            roles_map = self.app.list_account_roles(actor=actor)
            options = list(roles_map.keys())
            self.ui["acct_new_role"].options = options
            self.ui["acct_edit_roles"].options = options
            if options and self.ui["acct_new_role"].value not in options:
                self.ui["acct_new_role"].value = options[0]
            self.ui["acct_status"].text = f"{len(rows)} account(s) loaded."
        except Exception as exc:
            self.ui["acct_status"].text = f"Account refresh failed: {exc}"

    def _refresh_account_admin_visibility(self) -> None:
        admin_only = self.ui.get("acct_admin_container")
        if admin_only is None:
            return
        admin_only.set_visibility(self._has_permission("account:manage"))

    def _on_create_account(self) -> None:
        if not self._require_ui_permission("account:manage"):
            return
        username = str(self.ui["acct_new_username"].value or "").strip()
        password = str(self.ui["acct_new_password"].value or "")
        role_value = self.ui["acct_new_role"].value
        roles = [str(role_value).strip()] if str(role_value or "").strip() else []
        if not username or not password or not roles:
            ui.notify("Username, password, and role are required.", type="warning")
            return
        try:
            created = self.app.create_account(
                actor=self._current_username(),
                username=username,
                password=password,
                roles=roles,
            )
            self.ui["acct_new_password"].value = ""
            self._refresh_accounts_table()
            workspace_dir = str(created.get("workspace_dir", ""))
            ui.notify(f"Account created: {username} (workspace: {workspace_dir})", type="positive")
        except Exception as exc:
            ui.notify(f"Create account failed: {exc}", type="negative")

    def _on_set_account_roles(self) -> None:
        if not self._require_ui_permission("account:manage"):
            return
        username = str(self.ui["acct_edit_username"].value or "").strip()
        selected = self.ui["acct_edit_roles"].value
        if isinstance(selected, str):
            roles = [selected]
        else:
            roles = [str(item).strip() for item in (selected or []) if str(item).strip()]
        if not username or not roles:
            ui.notify("Username and at least one role are required.", type="warning")
            return
        try:
            self.app.set_account_roles(
                actor=self._current_username(),
                username=username,
                roles=roles,
            )
            self._refresh_accounts_table()
            ui.notify(f"Updated roles for {username}", type="positive")
        except Exception as exc:
            ui.notify(f"Update roles failed: {exc}", type="negative")

    def _on_set_account_active(self) -> None:
        if not self._require_ui_permission("account:manage"):
            return
        username = str(self.ui["acct_edit_username"].value or "").strip()
        active = bool(self.ui["acct_edit_active"].value)
        if not username:
            ui.notify("Username is required.", type="warning")
            return
        try:
            self.app.set_account_active(
                actor=self._current_username(),
                username=username,
                active=active,
            )
            self._refresh_accounts_table()
            ui.notify(f"Updated active status for {username}", type="positive")
        except Exception as exc:
            ui.notify(f"Update active flag failed: {exc}", type="negative")

    def _on_change_my_password(self) -> None:
        actor = self._current_username()
        if not actor:
            ui.notify("Please sign in first.", type="warning")
            return
        current_password = str(self.ui["acct_self_current_password"].value or "")
        new_password = str(self.ui["acct_self_new_password"].value or "")
        confirm_password = str(self.ui["acct_self_confirm_password"].value or "")
        if not current_password or not new_password:
            ui.notify("Current password and new password are required.", type="warning")
            return
        if new_password != confirm_password:
            ui.notify("New password and confirm password do not match.", type="warning")
            return
        try:
            self.app.change_own_password(
                actor=actor,
                current_password=current_password,
                new_password=new_password,
            )
            self.ui["acct_self_current_password"].value = ""
            self.ui["acct_self_new_password"].value = ""
            self.ui["acct_self_confirm_password"].value = ""
            ui.notify("Password updated.", type="positive")
        except Exception as exc:
            ui.notify(f"Change password failed: {exc}", type="negative")

    def _on_set_account_password(self) -> None:
        if not self._require_ui_permission("account:manage"):
            return
        username = str(self.ui["acct_pwd_username"].value or "").strip()
        new_password = str(self.ui["acct_pwd_new_password"].value or "")
        if not username or not new_password:
            ui.notify("Username and new password are required.", type="warning")
            return
        try:
            self.app.set_account_password(
                actor=self._current_username(),
                username=username,
                new_password=new_password,
            )
            self.ui["acct_pwd_new_password"].value = ""
            ui.notify(f"Password updated for {username}", type="positive")
            self._refresh_accounts_table()
        except Exception as exc:
            ui.notify(f"Set password failed: {exc}", type="negative")

    def _section_login_page(self) -> None:
        with ui.card().classes("main-panel w-full rounded-[26px] px-5 py-5"):
            ui.label("Sign In").classes("text-xl font-semibold")
            ui.label(
                "Please sign in to access configuration, workflow, plotting, prediction, and workdir operations."
            ).classes("text-sm muted-copy")
            with ui.row().classes("w-full items-end gap-2") as auth_login_row:
                self.ui["auth_username"] = self._field_inp("Username", "", classes="w-72")
                self.ui["auth_password"] = ui.input(
                    "Password",
                    password=True,
                    password_toggle_button=True,
                ).classes("w-72 soft-input").props("dense")
                self.ui["auth_username"].on("keydown.enter", lambda _e: self._on_login())
                self.ui["auth_password"].on("keydown.enter", lambda _e: self._on_login())
                ui.button("Sign In", icon="login", on_click=self._on_login).props("color=primary")
            self.ui["auth_login_row"] = auth_login_row
            with ui.row().classes("w-full items-center gap-2") as auth_session_row:
                self.ui["auth_session_user"] = ui.label("").classes("text-sm font-medium")
                ui.button("Sign Out", icon="logout", on_click=self._on_logout).props("flat")
            self.ui["auth_session_row"] = auth_session_row
            self.ui["auth_summary"] = ui.label("Not signed in").classes("text-sm")
            self.ui["auth_default_pwd_note"] = ui.label("").classes("text-xs muted-copy")

    def _section_runtime_access(self) -> None:
        with ui.expansion("Access Control", icon="manage_accounts", value=True).classes(
            "section-panel w-full rounded-[22px] px-2"
        ):
            self._info(
                "Roles: viewer (monitor only), operator (monitor + run), admin (plus account management). "
                "System monitor is shown as a persistent panel at bottom-left."
            )

            with ui.expansion("Account Management", icon="manage_accounts").classes("w-full"):
                self._info(
                    "All logged-in users can change their own password. "
                    "Only admin can create/disable accounts, change roles, and reset other users' passwords."
                )

                with ui.card().classes("tone-panel w-full rounded-xl px-4 py-3"):
                    ui.label("My Password").classes("text-sm font-semibold text-[var(--accent)]")
                    with ui.row().classes("w-full items-end gap-2"):
                        self.ui["acct_self_current_password"] = ui.input(
                            "Current Password",
                            password=True,
                            password_toggle_button=True,
                        ).classes("flex-1 soft-input").props("dense")
                        self.ui["acct_self_new_password"] = ui.input(
                            "New Password",
                            password=True,
                            password_toggle_button=True,
                        ).classes("flex-1 soft-input").props("dense")
                        self.ui["acct_self_confirm_password"] = ui.input(
                            "Confirm Password",
                            password=True,
                            password_toggle_button=True,
                        ).classes("flex-1 soft-input").props("dense")
                        ui.button(
                            "Change My Password",
                            icon="lock_reset",
                            on_click=self._on_change_my_password,
                        ).props("color=primary")

                with ui.column().classes("w-full gap-2") as acct_admin_container:
                    ui.label("Admin Controls").classes("text-sm font-semibold text-[var(--accent)]")
                    columns = [
                        {"name": "username", "label": "Username", "field": "username", "align": "left"},
                        {"name": "roles", "label": "Roles", "field": "roles", "align": "left"},
                        {"name": "active", "label": "Active", "field": "active", "align": "left"},
                        {"name": "workspace_dir", "label": "Workspace", "field": "workspace_dir", "align": "left"},
                        {"name": "updated_at", "label": "Updated At", "field": "updated_at", "align": "left"},
                    ]
                    self.ui["acct_table"] = ui.table(columns=columns, rows=[]).classes("w-full")
                    self.ui["acct_status"] = ui.label("Admin permission required.").classes("text-xs muted-copy")

                    with ui.row().classes("w-full items-end gap-2"):
                        self.ui["acct_new_username"] = self._field_inp("New Username", "", classes="flex-1")
                        self.ui["acct_new_password"] = ui.input(
                            "New Password",
                            password=True,
                            password_toggle_button=True,
                        ).classes("flex-1 soft-input").props("dense")
                        self.ui["acct_new_role"] = self._field_sel(
                            ["viewer", "operator", "admin"],
                            "Role",
                            value="viewer",
                            classes="w-44",
                        )
                        create_btn = ui.button("Create", icon="person_add", on_click=self._on_create_account).props(
                            "color=primary"
                        )
                        self._register_permission_control(create_btn, "account:manage")

                    with ui.row().classes("w-full items-end gap-2"):
                        self.ui["acct_edit_username"] = self._field_inp("Edit Username", "", classes="flex-1")
                        self.ui["acct_edit_roles"] = self._field_sel(
                            ["viewer", "operator", "admin"],
                            "Roles",
                            value=["viewer"],
                            classes="flex-1",
                            multiple=True,
                        )
                        self.ui["acct_edit_active"] = ui.checkbox("Active", value=True)
                        set_roles_btn = ui.button(
                            "Set Roles",
                            icon="verified_user",
                            on_click=self._on_set_account_roles,
                        ).props("color=secondary")
                        self._register_permission_control(set_roles_btn, "account:manage")
                        set_active_btn = ui.button(
                            "Set Active",
                            icon="toggle_on",
                            on_click=self._on_set_account_active,
                        ).props("color=secondary")
                        self._register_permission_control(set_active_btn, "account:manage")
                        refresh_btn = ui.button(
                            "Refresh Accounts",
                            icon="refresh",
                            on_click=self._refresh_accounts_table,
                        ).props("flat")
                        self._register_permission_control(refresh_btn, "account:manage")

                    with ui.row().classes("w-full items-end gap-2"):
                        self.ui["acct_pwd_username"] = self._field_inp("Reset Password For", "", classes="flex-1")
                        self.ui["acct_pwd_new_password"] = ui.input(
                            "New Password",
                            password=True,
                            password_toggle_button=True,
                        ).classes("flex-1 soft-input").props("dense")
                        set_pwd_btn = ui.button(
                            "Set User Password",
                            icon="password",
                            on_click=self._on_set_account_password,
                        ).props("color=secondary")
                        self._register_permission_control(set_pwd_btn, "account:manage")
                self.ui["acct_admin_container"] = acct_admin_container

    def _section_system_monitor(self) -> None:
        with ui.card().classes("tone-panel w-full rounded-lg px-3 py-2"):
            ui.label("Server Resource Snapshot").classes("text-sm font-semibold text-[var(--accent)]")

            with ui.column().classes("w-full gap-2"):
                with ui.column().classes("w-full gap-1"):
                    with ui.row().classes("w-full items-center justify-between gap-3"):
                        ui.label("CPU").classes("text-xs font-medium")
                        self.ui["sys_cpu_meta"] = ui.label("-").classes("text-xs text-right")
                    self.ui["sys_cpu_bar"] = ui.linear_progress(value=0, show_value=False).classes("w-full resource-progress")

                with ui.column().classes("w-full gap-1"):
                    with ui.row().classes("w-full items-center justify-between gap-3"):
                        ui.label("Memory").classes("text-xs font-medium")
                        self.ui["sys_mem_meta"] = ui.label("-").classes("text-xs text-right")
                    self.ui["sys_mem_bar"] = ui.linear_progress(value=0, show_value=False).classes("w-full resource-progress")

                with ui.column().classes("w-full gap-1"):
                    with ui.row().classes("w-full items-center justify-between gap-3"):
                        ui.label("Disk").classes("text-xs font-medium")
                        self.ui["sys_disk_meta"] = ui.label("-").classes("text-xs text-right")
                    self.ui["sys_disk_bar"] = ui.linear_progress(value=0, show_value=False).classes("w-full resource-progress")

                with ui.row().classes("w-full items-center justify-between gap-3"):
                    ui.label("Process RSS").classes("text-xs font-medium")
                    self.ui["sys_proc_text"] = ui.label("-").classes("text-xs text-right")

            with ui.column().classes("w-full gap-2") as gpu_panel:
                with ui.column().classes("w-full gap-1"):
                    with ui.row().classes("w-full items-center justify-between gap-3"):
                        ui.label("GPU Util").classes("text-xs font-medium")
                        self.ui["sys_gpu_util_meta"] = ui.label("-").classes("text-xs text-right")
                    self.ui["sys_gpu_util_bar"] = ui.linear_progress(value=0, show_value=False).classes("w-full resource-progress")

                with ui.column().classes("w-full gap-1"):
                    with ui.row().classes("w-full items-center justify-between gap-3"):
                        ui.label("GPU Memory").classes("text-xs font-medium")
                        self.ui["sys_gpu_mem_meta"] = ui.label("-").classes("text-xs text-right")
                    self.ui["sys_gpu_mem_bar"] = ui.linear_progress(value=0, show_value=False).classes("w-full resource-progress")

                self.ui["sys_gpu_detail"] = ui.label("").classes("text-xs muted-copy whitespace-pre-line")
            self.ui["sys_gpu_panel"] = gpu_panel
            gpu_panel.set_visibility(False)

            self.ui["sys_decision"] = ui.label("Login required").classes("text-sm font-medium")
            self.ui["sys_reason"] = ui.label("").classes("text-xs muted-copy")

    def _save_upload(self, content_bytes: bytes, suffix: str = ".json") -> str:
        upload_dir = (Path(self.app.working_dir).resolve() / ".uploads").resolve()
        upload_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=suffix, dir=str(upload_dir))
        with os.fdopen(fd, "wb") as f:
            f.write(content_bytes)
        return tmp

    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
    #  BUILD
    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲

    def build(self):
        self._inject_theme()
        ui.query("body").classes("app-shell")

        # 鈹€鈹€ header 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        with ui.header().classes(
            "items-center justify-between px-6 py-4 bg-transparent text-[var(--ink)]"
        ):
            with ui.column().classes("gap-0"):
                ui.label("Insurance Pricing Studio").classes(
                    "text-[1.45rem] font-semibold tracking-[0.08em] uppercase"
                )
                ui.label("Model configuration, workflow orchestration, and diagnostics in one surface").classes(
                    "text-xs muted-copy"
                )
            with ui.row().classes("items-center gap-2"):
                dark = ui.dark_mode(value=False)
                ui.button(icon="dark_mode", on_click=dark.toggle).props(
                    "flat round dense"
                )
                self.ui["ui_scale_select"] = ui.select(
                    ["90%", "100%", "110%", "120%"],
                    label="UI Size",
                    value="100%",
                    on_change=self._on_ui_scale_change,
                ).classes("w-28 soft-select").props("dense options-dense")
                self.ui["header_user_badge"] = ui.label("Guest").classes(
                    "text-sm font-medium px-2 py-1 rounded bg-[rgba(53,92,85,0.08)]"
                )
                self.ui["header_logout_btn"] = ui.button(
                    "Logout",
                    icon="logout",
                    on_click=self._on_logout,
                ).props("flat dense")
                ui.button(icon="more_vert", on_click=self._toggle_header_menu).props("flat round dense")

        with ui.element("div").classes("fixed inset-0 z-[1800]") as header_menu_backdrop:
            header_menu_backdrop.on("click", lambda _e: self._set_header_menu_visible(False))
        self.ui["header_menu_backdrop"] = header_menu_backdrop
        header_menu_backdrop.set_visibility(False)

        with ui.element("div").classes("fixed top-16 right-6 z-[1810]") as header_menu_panel:
            with ui.card().classes("rounded-xl px-2 py-2 min-w-[220px] border border-[rgba(53,92,85,0.14)]"):
                ui.button(
                    "Account Management",
                    icon="manage_accounts",
                    on_click=self._open_account_page_from_menu,
                ).props("flat align=left")
        self.ui["header_menu_panel"] = header_menu_panel
        header_menu_panel.set_visibility(False)

        with ui.column().classes("w-full max-w-3xl mx-auto px-4 pt-8 pb-6") as login_page:
            self._section_login_page()
        self.ui["login_page"] = login_page

        with ui.column().classes("w-full max-w-7xl mx-auto px-4 pt-3 pb-4") as account_page:
            with ui.card().classes("main-panel w-full rounded-[26px] px-4 py-4"):
                with ui.row().classes("w-full items-center justify-between gap-2"):
                    ui.label("Account Management").classes("text-xl font-semibold")
                    self.ui["account_back_btn"] = ui.button(
                        "Back",
                        icon="arrow_back",
                        on_click=self._close_account_page,
                    ).props("flat")
                self._section_runtime_access()
        self.ui["account_page"] = account_page

        # 鈹€鈹€ working directory (expansion) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        with ui.column().classes("w-full max-w-7xl mx-auto px-4 pt-4 pb-2"):
            with ui.card().classes("hero-panel w-full rounded-[28px] px-6 py-5") as workspace_hero:
                ui.label("Operational Workspace").classes("section-title text-[11px] text-[var(--warm)]")
                with ui.row().classes("w-full items-start justify-between gap-6"):
                    with ui.column().classes("gap-1 max-w-2xl"):
                        ui.label("Keep configuration-heavy work readable and calm.").classes(
                            "text-2xl font-semibold leading-tight"
                        )
                        ui.label(
                            "Set a stable working directory first, then move through configuration, execution, and analysis without losing path context."
                        ).classes("text-sm muted-copy leading-relaxed")
                    with ui.element("div").classes(
                        "hidden md:block rounded-[22px] px-4 py-3 bg-[rgba(53,92,85,0.08)] border border-[rgba(53,92,85,0.12)]"
                    ):
                        ui.label("Current mode").classes("text-[10px] uppercase tracking-[0.12em] text-[var(--accent)]")
                        ui.label("Research workflow").classes("text-sm font-medium")
            with ui.expansion("Working Directory", icon="folder").classes(
                "section-panel w-full rounded-[22px] px-2"
            ) as workspace_wd:
                self._section_working_dir()
            self.ui["workspace_top_container"] = workspace_hero
            self.ui["workspace_wd_container"] = workspace_wd
            self.ui["workspace_guard"] = ui.card().classes("tone-panel w-full rounded-[20px] px-5 py-4 my-3")
            with self.ui["workspace_guard"]:
                ui.label("Access Required").classes("text-sm font-semibold text-[var(--accent)]")
                ui.label(
                    "Please sign in first. After login, configuration/workflow/plotting/prediction interfaces will be unlocked."
                ).classes("text-xs muted-copy")

        # 鈹€鈹€ main tabs 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        with ui.column().classes("w-full max-w-7xl mx-auto px-4 pb-8") as workspace_main:
            with ui.card().classes("main-panel w-full rounded-[26px] overflow-hidden"):
                with ui.tabs().classes("w-full soft-tabs") as tabs:
                    t_cfg = ui.tab("Configuration", icon="settings")
                    t_wf = ui.tab("Workflow", icon="play_arrow")
                    t_ft = ui.tab("FT Two-Step", icon="layers")
                    t_plot = ui.tab("Plotting", icon="bar_chart")
                    t_pred = ui.tab("Prediction", icon="analytics")
                    t_files = ui.tab("WorkDir Files", icon="folder")

                with ui.tab_panels(tabs, value=t_cfg).classes("w-full bg-transparent px-2 pb-4"):
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
                    with ui.tab_panel(t_files):
                        self._tab_workdir_files()
        self.ui["workspace_main_container"] = workspace_main
        with ui.page_sticky(position="top-right", x_offset=18, y_offset=92):
            with ui.card().classes("tone-panel w-[420px] max-w-[44vw] rounded-xl px-5 py-4") as cfg_help_panel:
                self.ui["cfg_help_title"] = ui.label("Parameter Help").classes(
                    "text-sm font-semibold text-[var(--accent)]"
                )
                self.ui["cfg_help_body"] = ui.label("").classes(
                    "text-xs muted-copy whitespace-pre-line leading-relaxed"
                )
        self.ui["cfg_help_panel"] = cfg_help_panel
        cfg_help_panel.set_visibility(False)
        with ui.page_sticky(position="bottom-left", x_offset=16, y_offset=16):
            with ui.column().classes("w-[390px] max-h-[74vh] overflow-auto gap-2"):
                ui.label("System Monitor").classes("text-sm font-semibold")
                self._section_system_monitor()
        with ui.page_sticky(position="bottom-right", x_offset=20, y_offset=20):
            self.ui["back_to_top_btn"] = ui.button(
                icon="keyboard_arrow_up",
                on_click=lambda: ui.run_javascript(
                    "window.scrollTo({top: 0, behavior: 'smooth'});"
                ),
            ).props("round color=primary")

        self._apply_ui_scale(self._ui_scale)
        self._refresh_permission_controls()
        ui.timer(2.0, self._refresh_system_status_panel)

    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
    #  WORKING DIRECTORY
    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲

    def _section_working_dir(self):
        self._info(
            "The Working Directory is the base path for all relative paths. "
            "Entries such as ./Data and ./Results are resolved from this location. "
            "It defaults to the current launch directory (cwd), and you can change it at any time. "
            "Use a project folder that contains both data files and configuration files."
        )
        _, choices, selected = self.app.list_directory_candidates(str(self.app.working_dir))

        with ui.row().classes("w-full items-end gap-2"):
            wd = self._field_inp("Working Directory", str(self.app.working_dir), classes="flex-grow")
            set_btn = ui.button("Set", on_click=lambda: self._set_wd(wd.value)).props("flat")
            self._register_permission_control(set_btn, "config:edit")
        with ui.row().classes("w-full items-end gap-2"):
            browse = self._field_inp("Browse Root", str(self.app.working_dir), classes="flex-grow")
            ui.button("Refresh", on_click=lambda: self._refresh_wd(browse.value)).props("flat")
        with ui.row().classes("w-full items-end gap-2"):
            picker = self._field_sel(choices, "Select Folder", selected, classes="flex-grow")
            use_btn = ui.button("Use Selected", on_click=lambda: self._set_wd(picker.value)).props("flat")
            self._register_permission_control(use_btn, "config:edit")
        wd_status = ui.label(f"Current: {self.app.working_dir}").classes("text-xs muted-copy")

        self.ui["wd_input"] = wd
        self.ui["wd_browse"] = browse
        self.ui["wd_picker"] = picker
        self.ui["wd_status"] = wd_status

    def _set_wd(self, path: str):
        if not self._require_ui_permission("config:edit"):
            return
        status, resolved = self.app.set_working_dir(path)
        self.ui["wd_input"].value = resolved
        self.ui["wd_browse"].value = resolved
        self.ui["wd_status"].text = status
        self._refresh_wd(resolved)
        if "workdir_subdir" in self.ui and "workdir_include_hidden" in self.ui:
            self._refresh_workdir_files(
                str(self.ui["workdir_subdir"].value or ""),
                bool(self.ui["workdir_include_hidden"].value),
            )
        ui.notify(status, type="positive" if "set to" in status else "warning")

    def _refresh_wd(self, root: str):
        status, choices, selected = self.app.list_directory_candidates(root)
        self.ui["wd_picker"].options = choices
        self.ui["wd_picker"].value = selected
        self.ui["wd_status"].text = status

    def _reset_and_refresh_workdir_files(self) -> None:
        subdir_input = self.ui.get("workdir_subdir")
        include_hidden = self.ui.get("workdir_include_hidden")
        if subdir_input is None or include_hidden is None:
            return
        subdir_input.value = ""
        self._refresh_workdir_files("", bool(include_hidden.value))

    def _refresh_workdir_files(self, subdir: str, include_hidden: bool):
        status, rows = self.app.list_workdir_entries(
            subdir=subdir,
            include_hidden=bool(include_hidden),
        )
        table = self.ui.get("workdir_table")
        if table is not None:
            table.rows = rows
            table.update()
        status_label = self.ui.get("workdir_files_status")
        if status_label is not None:
            status_label.text = status
        folder_links = self.ui.get("workdir_folder_links")
        if folder_links is not None:
            folder_links.clear()
            dir_rows = [row for row in rows if str(row.get("type")) == "dir"]
            with folder_links:
                with ui.row().classes("w-full flex-wrap gap-2"):
                    if not dir_rows:
                        ui.label("No subfolders in current level.").classes("text-xs muted-copy")
                    for row in dir_rows:
                        path_val = str(row.get("path", "")).strip()
                        name_val = str(row.get("name", "")).strip() or path_val
                        ui.button(
                            f"Open {name_val}",
                            icon="folder_open",
                            on_click=lambda p=path_val: self._open_workdir_subdir(p),
                        ).props("flat dense")

    def _open_workdir_subdir(self, relative_path: str):
        subdir_input = self.ui.get("workdir_subdir")
        include_hidden = self.ui.get("workdir_include_hidden")
        if subdir_input is None or include_hidden is None:
            return
        subdir_input.value = str(relative_path or "").strip()
        self._refresh_workdir_files(subdir_input.value, include_hidden.value)

    def _go_up_workdir_subdir(self):
        subdir_input = self.ui.get("workdir_subdir")
        include_hidden = self.ui.get("workdir_include_hidden")
        if subdir_input is None or include_hidden is None:
            return
        current = str(subdir_input.value or "").strip().replace("\\", "/")
        if not current:
            return
        parent = str(Path(current).parent).replace("\\", "/")
        if parent == ".":
            parent = ""
        subdir_input.value = parent
        self._refresh_workdir_files(parent, include_hidden.value)

    async def _on_upload_workdir_file(self, e, subdir_input):
        if not self._require_ui_permission("config:edit"):
            return
        try:
            content = await e.file.read()
            raw_name = getattr(e, "name", "") or getattr(e.file, "name", "") or "upload.bin"
            status = self.app.save_workdir_upload(
                file_name=str(raw_name),
                content_bytes=content,
                subdir=str(subdir_input.value or ""),
            )
            status_label = self.ui.get("workdir_files_status")
            if status_label is not None:
                status_label.text = status
            ui.notify(status, type="positive" if "Uploaded to" in status else "warning")
            self._refresh_workdir_files(
                str(self.ui["workdir_subdir"].value or ""),
                bool(self.ui["workdir_include_hidden"].value),
            )
        except Exception as exc:
            ui.notify(f"Upload error: {exc}", type="negative")

    def _on_delete_workdir_file(self, relative_path: str, recursive: bool, confirm_delete: bool):
        if not self._require_ui_permission("config:edit"):
            return
        if not bool(confirm_delete):
            ui.notify("Please check Confirm Delete first.", type="warning")
            return
        status = self.app.delete_workdir_entry(relative_path, recursive=bool(recursive))
        status_label = self.ui.get("workdir_files_status")
        if status_label is not None:
            status_label.text = status
        ui.notify(status, type="positive" if status.startswith("Deleted") else "warning")
        self._refresh_workdir_files(
            str(self.ui["workdir_subdir"].value or ""),
            bool(self.ui["workdir_include_hidden"].value),
        )

    def _on_create_workdir_folder(self, relative_path: str):
        if not self._require_ui_permission("config:edit"):
            return
        status = self.app.create_workdir_folder(relative_path)
        status_label = self.ui.get("workdir_files_status")
        if status_label is not None:
            status_label.text = status
        ui.notify(status, type="positive" if status.startswith("Folder ready") else "warning")
        self._refresh_workdir_files(
            str(self.ui["workdir_subdir"].value or ""),
            bool(self.ui["workdir_include_hidden"].value),
        )

    def _tab_workdir_files(self):
        self._guide("WorkDir File Operations", [
            "Browse files under the current work_dir (optionally in a subdir).",
            "Upload files directly into work_dir/subdir.",
            "Delete selected files or folders when needed.",
        ])
        self._tip(
            "All operations are restricted to the current work_dir to avoid accidental changes outside the project workspace."
        )
        if self._has_permission("account:manage"):
            self._info("Admin scope: you can browse the whole workspaces root and open any user folder.")
        else:
            self._info("User scope: you can browse only your own workspace folder.")

        with ui.row().classes("w-full gap-2 items-end"):
            subdir = self._field_inp("Subdir (relative to work_dir)", "", classes="flex-1")
            include_hidden = ui.checkbox("Include Hidden", value=False)
            up_btn = ui.button(
                "Up One Level",
                icon="arrow_upward",
                on_click=self._go_up_workdir_subdir,
            ).props("flat")
            self._register_permission_control(up_btn, "config:edit")
            refresh_btn = ui.button(
                "Refresh",
                icon="refresh",
                on_click=lambda: self._refresh_workdir_files(subdir.value, include_hidden.value),
            ).props("flat")
            self._register_permission_control(refresh_btn, "config:edit")

        status = ui.label("").classes("text-sm muted-copy")
        columns = [
            {"name": "path", "label": "Path", "field": "path", "align": "left"},
            {"name": "type", "label": "Type", "field": "type", "align": "left"},
            {"name": "size", "label": "Size", "field": "size", "align": "right"},
            {"name": "modified", "label": "Modified", "field": "modified", "align": "left"},
        ]
        table = ui.table(columns=columns, rows=[]).classes("w-full")
        table.props("dense wrap-cells row-key=path")
        folder_links = ui.column().classes("w-full gap-1")

        with ui.expansion("Upload File", icon="upload_file", value=True).classes("w-full"):
            upload_subdir = self._field_inp("Upload Subdir", "", classes="flex-1")
            uploader = ui.upload(
                label="Upload to work_dir",
                auto_upload=True,
                on_upload=lambda e: self._on_upload_workdir_file(e, upload_subdir),
            ).props("max-files=1")
            uploader.classes("w-full")
            self._register_permission_control(uploader, "config:edit")

        with ui.expansion("Create Folder", icon="create_new_folder").classes("w-full"):
            with ui.row().classes("w-full gap-2 items-end"):
                new_folder_path = self._field_inp("New Folder Path (relative)", "", classes="flex-1")
                create_folder_btn = ui.button(
                    "Create Folder",
                    icon="create_new_folder",
                    on_click=lambda: self._on_create_workdir_folder(new_folder_path.value),
                ).props("color=primary")
                self._register_permission_control(create_folder_btn, "config:edit")

        with ui.expansion("Delete File/Folder", icon="delete").classes("w-full"):
            with ui.row().classes("w-full gap-2 items-end"):
                delete_path = self._field_inp("Relative Path", "", classes="flex-1")
                delete_recursive = ui.checkbox("Recursive Folder Delete", value=True)
                delete_confirm = ui.checkbox("Confirm Delete", value=False)
                delete_btn = ui.button(
                    "Delete",
                    icon="delete_forever",
                    on_click=lambda: self._on_delete_workdir_file(
                        delete_path.value, delete_recursive.value, delete_confirm.value
                    ),
                ).props("color=negative")
                self._register_permission_control(delete_btn, "config:edit")

        self.ui["workdir_subdir"] = subdir
        self.ui["workdir_include_hidden"] = include_hidden
        self.ui["workdir_files_status"] = status
        self.ui["workdir_table"] = table
        self.ui["workdir_folder_links"] = folder_links
        self._refresh_workdir_files("", False)

    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
    #  TAB: CONFIGURATION
    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲

    def _tab_config(self):
        self._guide("Configuration Workflow (choose one)", [
            "Option A: upload an existing JSON config and run it directly from the Workflow tab. No build step is required.",
            "Option B: fill in the parameters below, click \"Build Configuration\" to generate JSON, then run it.",
            "Optional: click \"Save Configuration\" to persist the generated config for reuse.",
        ])
        self._tip(
            "Uploaded JSON is automatically copied into the \"Generated Config\" panel and can be run immediately. "
            "If you need minor adjustments, edit the JSON there and run it without rebuilding. "
            "For manual entry, the key required fields are Data Directory, Target, Weight, Feature List, and Model Keys."
        )

        # 鈹€鈹€ Load config 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        with ui.expansion("Load JSON Config", icon="upload_file").classes("w-full"):
            with ui.row().classes("w-full items-end gap-4"):
                ui.upload(
                    label="Upload Config (.json)",
                    auto_upload=True,
                    on_upload=self._on_upload_config,
                ).props("accept=.json").classes("max-w-xs")
                self.ui["load_status"] = ui.label("").classes("text-sm")
            self.ui["config_display"] = self._field_txt(
                "Current Configuration (read-only)",
                mono=True,
            ).props("readonly outlined")

        ui.separator()

        # 鈹€鈹€ Core: Data Settings 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        with ui.expansion("Data Settings", icon="table_chart", value=True).classes("w-full"):
            self._info(
                "Data Directory should contain files named like {model}_{category}.csv, "
                "for example od_bc.csv. If Feature List is empty, it will be inferred automatically."
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

        # 鈹€鈹€ Task & Training 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        with ui.expansion("Task & Training", icon="psychology", value=True).classes("w-full"):
            self._info(
                "Model Keys determines which models to train. Use comma-separated values: "
                "xgb=XGBoost, resn=ResNet, ft=FT-Transformer, gnn=GNN. "
                "Distribution overrides loss_name."
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

        # 鈹€鈹€ Accordion sections 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
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

        # 鈹€鈹€ Advanced JSON overrides 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        with ui.expansion("Advanced Manual Overrides (JSON)", icon="code").classes("w-full"):
            self._tip(
                "Enter any JSON overrides here and they will be deep-merged into the built configuration. "
                "This is useful for parameters that do not have dedicated UI controls, such as runner, report_*, psi_*, or registry_*. "
                "Example: {\"runner\": {\"mode\": \"explain\"}} switches the task mode to explain."
            )
            self._txt("config_overrides_json", "Config Overrides JSON", "{}")

        ui.separator()

        # 鈹€鈹€ Build & Save 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        with ui.row().classes("w-full items-end gap-4"):
            build_btn = ui.button("Build Configuration", icon="build",
                                  on_click=self._on_build_config).props("color=primary")
            self._register_permission_control(build_btn, "config:edit")
            save_btn = ui.button("Save Configuration", icon="save",
                                 on_click=self._on_save_config).props("color=secondary")
            self._register_permission_control(save_btn, "config:edit")
        self.ui["build_status"] = ui.label("").classes("text-sm")
        self.ui["config_json"] = self._field_txt(
            "Generated Config (JSON)",
            mono=True,
        ).props("outlined rows=14")
        with ui.row().classes("w-full items-end gap-4"):
            self.ui["save_filename"] = self._field_inp(
                "Save Filename", "my_config.json", classes="flex-grow"
            )
            self.ui["save_status"] = ui.label("").classes("text-sm")

    # 鈹€鈹€ config accordion sections 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def _section_split(self):
        with ui.expansion("Split & Pre-split Data", icon="call_split").classes("w-full"):
            self._tip(
                "Random split is used by default. For grouped splits, set Group Column; for temporal splits, set Time Column. "
                "If you already have train and test files, provide Train/Test Data Path to skip splitting."
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
            self._tip("Set CV Splits to 0 to disable cross-validation. Group and time strategies require the corresponding column names.")
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
                "FT Role: model=direct prediction, embedding=FT two-step embedding workflow, "
                "unsupervised_embedding=legacy alias for backward compatibility. "
                "Set OOF Folds > 0 to enable out-of-fold embeddings and reduce leakage."
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
                "DDP (Distributed Data Parallel) is recommended for multi-GPU ResNet and FT training. "
                "It is skipped automatically when the dataset is smaller than ddp_min_rows. "
                "DataParallel is a fallback for single-machine multi-GPU setups, but DDP is usually the better choice."
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

    # 鈹€鈹€ config event handlers 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

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
        if not self._require_ui_permission("config:edit"):
            return
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
        if not self._require_ui_permission("config:edit"):
            return
        config_json = self.ui["config_json"].value
        filename = self.ui["save_filename"].value
        status = self.app.save_config(config_json, filename)
        self.ui["save_status"].text = status
        ui.notify(status, type="positive" if "saved" in status.lower() else "negative")

    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
    #  TAB: FT TWO-STEP
    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲

    def _tab_ft_workflow(self):
        ui.label("FT-Transformer Two-Step Training").classes("text-lg font-semibold")

        self._guide("FT Two-Step Workflow", [
            "Start by building the base configuration in the Configuration tab.",
            "Step 1: click \"Prepare Config\" to generate the FT embedding config, then run Step 1.",
            "Wait until Step 1 finishes and the log shows completion.",
            "Step 2: click \"Prepare Configs\" to generate XGB and ResN configs that reference the Step 1 embedding outputs.",
            "Run either \"Run Step 2 (XGB)\" or \"Run Step 2 (ResN)\" to train the final model.",
        ])
        self._tip(
            "This workflow first learns FT feature embeddings, "
            "then feeds those embeddings into XGB or ResN as augmented features. "
            "It is often stronger than direct training when representation quality matters."
        )

        # 鈹€鈹€ Step 1 鈹€鈹€
        with ui.expansion("Step 1: FT Embedding", icon="looks_one", value=True).classes("w-full"):
            self._info(
                "This step starts from the Configuration tab settings and automatically sets FT role=embedding. "
                "Enable DDP for multi-GPU training and set Processes to the number of GPUs."
            )
            with ui.row().classes("w-full gap-2 items-end"):
                ft_ddp = ui.checkbox("Use DDP", value=True)
                ft_nproc = self._field_num("Processes", 2, classes="w-32")
            with ui.row().classes("w-full gap-2"):
                prep_step1_btn = ui.button(
                    "Prepare Config",
                    icon="settings",
                    on_click=lambda: self._on_ft_step1(ft_ddp.value, ft_nproc.value),
                ).props("color=primary")
                self._register_permission_control(prep_step1_btn, "config:edit")
                run_step1_btn = ui.button(
                    "Run Step 1",
                    icon="play_arrow",
                    on_click=self._on_run_ft_step1,
                ).props("color=positive")
                self._register_permission_control(run_step1_btn, "task:run")
            self.ui["step1_status"] = ui.label("").classes("text-sm")
            self.ui["step1_config"] = self._field_txt("Step 1 Config", mono=True).props(
                "outlined rows=10"
            )
            self.ui["step1_log"] = self._field_txt("Step 1 Logs", mono=True).props(
                "readonly outlined rows=12"
            )

        ui.separator()

        # 鈹€鈹€ Step 2 鈹€鈹€
        with ui.expansion("Step 2: XGB/ResN with Embeddings", icon="looks_two").classes("w-full"):
            self._info(
                "After Step 1 completes, the embedding data is saved automatically to Augmented Data Dir. "
                "Use the Overrides JSON fields to customize output paths, loss settings, and other details. "
                "XGB and ResN can be prepared and run independently."
            )
            with ui.row().classes("w-full gap-2 items-end"):
                tgt = self._field_inp("Target Models", "xgb, resn", classes="flex-grow")
                aug_dir = self._field_inp("Augmented Data Dir", "./DataFTEmbed", classes=(
                    "flex-grow"
                ))
            xgb_ov = self._field_txt("XGB Step 2 Overrides", self._xgb_step2_tpl, mono=True).props(
                "outlined rows=5"
            )
            resn_ov = self._field_txt("ResN Step 2 Overrides", self._resn_step2_tpl, mono=True).props(
                "outlined rows=5"
            )
            with ui.row().classes("w-full gap-2"):
                prep_step2_btn = ui.button(
                    "Prepare Configs", icon="settings",
                    on_click=lambda: self._on_ft_step2(
                        tgt.value, aug_dir.value, xgb_ov.value, resn_ov.value
                    ),
                ).props("color=primary")
                self._register_permission_control(prep_step2_btn, "config:edit")
                run_step2_xgb_btn = ui.button(
                    "Run Step 2 (XGB)",
                    icon="play_arrow",
                    on_click=lambda: self._on_run_ft_step2("xgb"),
                ).props("color=positive")
                self._register_permission_control(run_step2_xgb_btn, "task:run")
                run_step2_resn_btn = ui.button(
                    "Run Step 2 (ResN)",
                    icon="play_arrow",
                    on_click=lambda: self._on_run_ft_step2("resn"),
                ).props("color=positive")
                self._register_permission_control(run_step2_resn_btn, "task:run")
            self.ui["step2_status"] = ui.label("").classes("text-sm")
            with ui.tabs().classes("w-full") as s2_tabs:
                ui.tab("XGB Config")
                ui.tab("ResN Config")
            with ui.tab_panels(s2_tabs).classes("w-full"):
                with ui.tab_panel("XGB Config"):
                    self.ui["step2_xgb"] = self._field_txt(mono=True).props("outlined rows=10")
                with ui.tab_panel("ResN Config"):
                    self.ui["step2_resn"] = self._field_txt(mono=True).props("outlined rows=10")
            self.ui["step2_log"] = self._field_txt("Step 2 Logs", mono=True).props(
                "readonly outlined rows=12"
            )

    def _on_ft_step1(self, use_ddp, nproc):
        if not self._require_ui_permission("config:edit"):
            return
        config_json = self.ui["config_json"].value
        status, step1_json = self.app.prepare_ft_step1(config_json, use_ddp, int(nproc or 2))
        self.ui["step1_status"].text = status
        self.ui["step1_config"].value = step1_json
        ui.notify(status, type="positive" if "prepared" in status.lower() else "warning")

    def _on_run_ft_step1(self):
        if not self._require_ui_permission("task:run"):
            return
        step1_json = self.ui["step1_config"].value
        if not step1_json.strip():
            ui.notify("Please prepare Step 1 config first", type="warning")
            return
        runner = _StreamRunner(self.ui["step1_status"], self.ui["step1_log"])
        runner.run(self.app.run_training, step1_json, self._actor_for_runtime())

    def _on_ft_step2(self, target_models, aug_dir, xgb_ov, resn_ov):
        if not self._require_ui_permission("config:edit"):
            return
        step1_path = self.app.current_step1_config or "temp_ft_step1_config.json"
        status, xgb_json, resn_json = self.app.prepare_ft_step2(
            step1_path, target_models, aug_dir, xgb_ov, resn_ov
        )
        self.ui["step2_status"].text = status
        self.ui["step2_xgb"].value = xgb_json
        self.ui["step2_resn"].value = resn_json
        ui.notify(status, type="positive" if "prepared" in status.lower() else "warning")

    def _on_run_ft_step2(self, model_type: str):
        if not self._require_ui_permission("task:run"):
            return
        key = "step2_xgb" if model_type == "xgb" else "step2_resn"
        config_json = self.ui[key].value
        if not config_json.strip():
            ui.notify(f"Please prepare Step 2 {model_type.upper()} config first", type="warning")
            return
        runner = _StreamRunner(self.ui["step2_status"], self.ui["step2_log"])
        runner.run(self.app.run_training, config_json, self._actor_for_runtime())

    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
    #  TAB: WORKFLOW
    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲

    def _tab_workflow(self):
        self._guide("Workflow Tab Overview", [
            "Run from Config executes the JSON generated in the Configuration tab for training, explanation, incremental jobs, and related tasks.",
            "Config-Driven Workflow runs post-processing tasks such as plotting, prediction, and model comparison from a standalone workflow JSON file.",
        ])
        self._tip(
            "Use \"Run from Config\" for standard model training and analysis tasks. "
            "Use \"Config-Driven Workflow\" for post-processing jobs such as plotting batches, prediction, and comparisons."
        )

        # 鈹€鈹€ Section 1: Run from Configuration tab 鈹€鈹€
        with ui.expansion("Run from Config", icon="play_arrow", value=True).classes("w-full"):
            self._info(
                "This runs the JSON currently shown in the \"Generated Config\" editor at the bottom of the Configuration tab. "
                "It works for both uploaded configs and manually built configs as long as the JSON is valid. "
                "Task type is determined by runner.mode: entry=training, explain=explainability, incremental=incremental training, watchdog=monitoring."
            )

            with ui.row().classes("w-full items-center gap-4"):
                run_task_btn = ui.button("Run Task", icon="play_arrow",
                                         on_click=self._on_run_task).props("color=primary size=lg")
                self._register_permission_control(run_task_btn, "task:run")
                self.ui["run_status"] = ui.label("").classes("text-sm")

            self.ui["run_log"] = self._field_txt("Task Logs", mono=True).props(
                "readonly outlined rows=18"
            )

            with ui.row().classes("w-full items-center gap-4"):
                open_btn = ui.button("Open Results Folder", icon="folder_open",
                                     on_click=self._on_open_results).props("flat")
                self._register_permission_control(open_btn, "config:edit")
                self.ui["folder_status"] = ui.label("").classes("text-sm")

        ui.separator()

        # 鈹€鈹€ Section 2: Config-driven workflow 鈹€鈹€
        with ui.expansion("Config-Driven Workflow", icon="account_tree").classes("w-full"):
            self._info(
                "Supported workflow modes include pre_oneway, plot_direct, plot_embed, predict_ft_embed, compare, and double_lift. "
                "Upload or edit the JSON below, set workflow.mode and the required config paths, then run it."
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
                    self.ui["wf_json"] = self._field_txt(
                        "Workflow Config (JSON)", value=wf_tpl,
                        mono=True,
                    ).props("outlined rows=12")

            run_workflow_btn = ui.button("Run Workflow", icon="play_arrow",
                                         on_click=self._on_run_workflow).props("color=primary size=lg")
            self._register_permission_control(run_workflow_btn, "task:run")
            self.ui["wf_status"] = ui.label("").classes("text-sm")
            self.ui["wf_log"] = self._field_txt("Workflow Logs", mono=True).props(
                "readonly outlined rows=16"
            )

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
        if not self._require_ui_permission("task:run"):
            return
        wf_json = self.ui["wf_json"].value
        runner = _StreamRunner(self.ui["wf_status"], self.ui["wf_log"])
        runner.run(self.app.run_workflow_config_ui, wf_json, self._actor_for_runtime())

    def _on_run_task(self):
        if not self._require_ui_permission("task:run"):
            return
        config_json = self.ui["config_json"].value
        runner = _StreamRunner(self.ui["run_status"], self.ui["run_log"])
        runner.run(self.app.run_training, config_json, self._actor_for_runtime())

    def _on_open_results(self):
        if not self._require_ui_permission("config:edit"):
            return
        config_json = self.ui["config_json"].value
        status = self.app.open_results_folder(config_json)
        self.ui["folder_status"].text = status
        ui.notify(status)

    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
    #  TAB: PLOTTING
    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲

    def _tab_plotting(self):
        self._guide("Plotting Tab Overview", [
            "Pre Oneway performs pre-model single-factor analysis to inspect how features relate to the target.",
            "Direct Plot generates diagnostic plots for directly trained XGB and ResN models.",
            "Embed Plot generates diagnostic plots for FT-embedding-based models.",
            "Double Lift compares the ranking power of two prediction columns.",
            "FT-Embed Compare compares a direct model against its FT-embedding-enhanced counterpart.",
        ])
        self._tip(
            "Plotting requires the corresponding training result directories and configuration files. "
            "Config paths may be relative to the Working Directory. "
            "Click \"Load Factors\" to populate available oneway factors from the selected config file."
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
            "Use this for pre-model feature exploration. It generates actual-vs-weight binned charts for each selected factor. "
            "Data Path points to the raw dataset, and Plot Config provides feature and plotting settings."
        )
        with ui.row().classes("w-full gap-4"):
            with ui.column().classes("flex-[3] gap-1"):
                with ui.row().classes("w-full gap-2"):
                    pre_data = self._field_inp("Data Path", "./Data/od_bc.csv", classes="flex-grow")
                    pre_out = self._field_inp("Output Dir (optional)", "", classes="flex-grow")
                with ui.row().classes("w-full gap-2"):
                    pre_model = self._field_inp("Model Name", "od_bc", classes="flex-1")
                    pre_tgt = self._field_inp("Target", "response", classes="flex-1")
                    pre_wgt = self._field_inp("Weight", "weights", classes="flex-1")
                with ui.row().classes("w-full gap-2"):
                    pre_cfg = self._field_inp("Plot Config", "config_plot.json", classes="flex-grow")
                    pre_fac_status = ui.label("").classes("text-xs text-gray-500")
                pre_factors = self._field_sel([], "Oneway Factors", multiple=True)
                ui.button("Load Factors", icon="refresh",
                           on_click=self._oneway_factor_loader(pre_cfg, pre_factors, pre_fac_status)
                           ).props("flat dense")
                with ui.expansion("Advanced: Split Data Override", icon="settings").classes("w-full"):
                    with ui.row().classes("w-full gap-2"):
                        pre_train = self._field_inp("Train Data Path", "", classes="flex-1")
                        pre_test = self._field_inp("Test Data Path", "", classes="flex-1")
            with ui.column().classes("flex-[2] gap-1"):
                pre_feat = self._field_inp("Fallback Feature List", "")
                pre_cat = self._field_inp("Categorical Features", "")
                with ui.row().classes("w-full gap-2"):
                    pre_bins = self._field_num("Bins", 10)
                    pre_hold = self._field_num("Holdout", 0.25, min=0, max=0.5, step=0.05)
                    pre_seed = self._field_num("Seed", 13)

        pre_status = ui.label("").classes("text-sm")
        pre_log = self._field_txt("Logs", mono=True).props("readonly outlined rows=10")
        pre_gallery = ui.row().classes("w-full flex-wrap gap-2")

        def _run_pre_oneway():
            if not self._require_ui_permission("task:run"):
                return
            _StreamRunner(pre_status, pre_log, pre_gallery).run(
                self.app.run_pre_oneway_ui,
                pre_data.value, pre_model.value, pre_tgt.value, pre_wgt.value,
                pre_feat.value, pre_factors.value, pre_cat.value, int(pre_bins.value or 10),
                float(pre_hold.value or 0.25), int(pre_seed.value or 13),
                pre_out.value, pre_train.value, pre_test.value,
                actor=self._actor_for_runtime(),
            )

        run_pre_btn = ui.button("Run Pre Oneway", icon="play_arrow", on_click=_run_pre_oneway).props("color=primary")
        self._register_permission_control(run_pre_btn, "task:run")

    def _subtab_direct_plot(self):
        self._info(
            "Generate diagnostic plots for directly trained models. Provide the Plot Config and the training configs for each model."
        )
        with ui.row().classes("w-full gap-2"):
            d_cfg = self._field_inp("Plot Config", "config_plot.json", classes="flex-1")
            d_xgb = self._field_inp("XGB Config", "config_xgb_direct.json", classes="flex-1")
            d_resn = self._field_inp("ResN Config", "config_resn_direct.json", classes="flex-1")
        with ui.row().classes("w-full gap-2"):
            d_fac_status = ui.label("").classes("text-xs text-gray-500")
            d_factors = self._field_sel([], "Oneway Factors", classes="flex-grow", multiple=True)
            ui.button("Load Factors", icon="refresh",
                       on_click=self._oneway_factor_loader(d_cfg, d_factors, d_fac_status)
                       ).props("flat dense")
        with ui.expansion("Advanced: Data/Model Overrides", icon="settings").classes("w-full"):
            with ui.row().classes("w-full gap-2"):
                d_train = self._field_inp("Train Data Path", "", classes="flex-1")
                d_test = self._field_inp("Test Data Path", "", classes="flex-1")

        d_status = ui.label("").classes("text-sm")
        d_log = self._field_txt("Logs", mono=True).props("readonly outlined rows=10")
        d_gallery = ui.row().classes("w-full flex-wrap gap-2")

        def _run_direct_plot():
            if not self._require_ui_permission("task:run"):
                return
            _StreamRunner(d_status, d_log, d_gallery).run(
                self.app.run_plot_direct_ui,
                d_cfg.value, d_xgb.value, d_resn.value, d_factors.value,
                d_train.value, d_test.value, None, None,
                actor=self._actor_for_runtime(),
            )

        run_direct_btn = ui.button("Run Direct Plot", icon="play_arrow", on_click=_run_direct_plot).props("color=primary")
        self._register_permission_control(run_direct_btn, "task:run")

    def _subtab_embed_plot(self):
        self._info(
            "Generate diagnostic plots for models trained with the FT Two-Step pipeline. "
            "You need the FT embedding config and the corresponding XGB or ResN embedding configs. "
            "Enable Runtime FT Embedding to compute embeddings on the fly during plotting instead of using cached outputs."
        )
        with ui.row().classes("w-full gap-2"):
            e_cfg = self._field_inp("Plot Config", "config_plot.json", classes="flex-1")
            e_ft = self._field_inp("FT Config", "config_ft_ddp_embed.json", classes="flex-1")
            e_rt = ui.checkbox("Runtime FT Embedding", value=False)
        with ui.row().classes("w-full gap-2"):
            e_xgb = self._field_inp("XGB Embed Config", "config_xgb_from_ft_embed.json", classes="flex-1")
            e_resn = self._field_inp("ResN Embed Config", "config_resn_from_ft_embed.json", classes="flex-1")
        with ui.row().classes("w-full gap-2"):
            e_fac_status = ui.label("").classes("text-xs text-gray-500")
            e_factors = self._field_sel([], "Oneway Factors", classes="flex-grow", multiple=True)
            ui.button("Load Factors", icon="refresh",
                       on_click=self._oneway_factor_loader(e_cfg, e_factors, e_fac_status)
                       ).props("flat dense")
        with ui.expansion("Advanced: Data/Model Overrides", icon="settings").classes("w-full"):
            with ui.row().classes("w-full gap-2"):
                e_train = self._field_inp("Train Data Path", "", classes="flex-1")
                e_test = self._field_inp("Test Data Path", "", classes="flex-1")

        e_status = ui.label("").classes("text-sm")
        e_log = self._field_txt("Logs", mono=True).props("readonly outlined rows=10")
        e_gallery = ui.row().classes("w-full flex-wrap gap-2")

        def _run_embed_plot():
            if not self._require_ui_permission("task:run"):
                return
            _StreamRunner(e_status, e_log, e_gallery).run(
                self.app.run_plot_embed_ui,
                e_cfg.value, e_xgb.value, e_resn.value, e_ft.value, e_rt.value,
                e_factors.value, e_train.value, e_test.value, None, None, None,
                actor=self._actor_for_runtime(),
            )

        run_embed_btn = ui.button("Run Embed Plot", icon="play_arrow", on_click=_run_embed_plot).props("color=primary")
        self._register_permission_control(run_embed_btn, "task:run")

    def _subtab_double_lift(self):
        self._info(
            "Double Lift compares the ranking quality of two prediction columns. "
            "Provide a CSV file that contains both prediction columns. "
            "Use Holdout=0 to plot on the full dataset, or a positive value to split out a test set. "
            "When split cache path is configured, existing cache is reused and missing cache is created."
        )
        with ui.row().classes("w-full gap-2"):
            dl_data = self._field_inp("Data Path (CSV)", "./Data/od_bc.csv", classes="flex-[3]")
            dl_out = self._field_inp("Output Image Path (optional)", "", classes="flex-[2]")
        with ui.row().classes("w-full gap-2"):
            dl_p1 = self._field_inp("Pred Column 1", "pred_xgb", classes="flex-1")
            dl_p2 = self._field_inp("Pred Column 2", "pred_resn", classes="flex-1")
            dl_tgt = self._field_inp("Target", "reponse", classes="flex-1")
            dl_wgt = self._field_inp("Weight", "weights", classes="flex-1")
        with ui.row().classes("w-full gap-2"):
            dl_l1 = self._field_inp("Label 1", "Model 1", classes="flex-1")
            dl_l2 = self._field_inp("Label 2", "Model 2", classes="flex-1")
            dl_bins = self._field_num("Bins", 10, classes="flex-1")
            dl_seed = self._field_num("Seed", 13, classes="flex-1")
        with ui.row().classes("w-full gap-2"):
            dl_hold = self._field_num("Holdout (0=all)", 0.0, min=0, max=0.5, step=0.05)
            dl_split = self._field_sel(["random", "stratified", "time", "group"],
                                       "Split Strategy", "random")
            dl_gcol = self._field_inp("Group Col", "", classes="flex-1")
            dl_tcol = self._field_inp("Time Col", "", classes="flex-1")
            dl_tasc = ui.checkbox("Time Ascending", value=True)
        with ui.expansion("Advanced: Split Override / Cache", icon="settings").classes("w-full"):
            with ui.row().classes("w-full gap-2"):
                dl_train = self._field_inp("Train Data Path", "", classes="flex-1")
                dl_test = self._field_inp("Validation Data Path", "", classes="flex-1")
            with ui.row().classes("w-full gap-2"):
                dl_cache = self._field_inp("Split Cache Path (.npz)", "", classes="flex-1")
                dl_cache_force = ui.checkbox("Force Rebuild Cache", value=False)
        with ui.row().classes("w-full gap-4"):
            dl_pw1 = ui.checkbox("Pred 1 Weighted", value=False)
            dl_pw2 = ui.checkbox("Pred 2 Weighted", value=False)
            dl_aw = ui.checkbox("Actual Weighted", value=False)

        dl_status = ui.label("").classes("text-sm")
        dl_log = self._field_txt("Logs", mono=True).props("readonly outlined rows=10")
        dl_gallery = ui.row().classes("w-full flex-wrap gap-2")

        def _run_double_lift():
            if not self._require_ui_permission("task:run"):
                return
            _StreamRunner(dl_status, dl_log, dl_gallery).run(
                self.app.run_double_lift_ui,
                dl_data.value, dl_train.value, dl_test.value,
                dl_p1.value, dl_p2.value, dl_tgt.value, dl_wgt.value,
                int(dl_bins.value or 10), dl_l1.value, dl_l2.value,
                dl_pw1.value, dl_pw2.value, dl_aw.value,
                float(dl_hold.value or 0), dl_split.value, dl_gcol.value, dl_tcol.value,
                dl_tasc.value, int(dl_seed.value or 13),
                dl_cache.value, dl_cache_force.value, dl_out.value,
                actor=self._actor_for_runtime(),
            )

        run_double_lift_btn = ui.button("Run Double Lift", icon="play_arrow", on_click=_run_double_lift).props("color=primary")
        self._register_permission_control(run_double_lift_btn, "task:run")

    def _subtab_compare(self):
        self._info(
            "Compare a directly trained model against its FT-embedding-enhanced version. "
            "Selecting Model Key updates the default config paths and labels automatically."
        )
        with ui.row().classes("w-full gap-2"):
            c_key = self._field_sel(["xgb", "resn"], "Model Key", "xgb", classes="w-32")
            c_direct = self._field_inp("Direct Config", "config_xgb_direct.json", classes="flex-1")
            c_ft = self._field_inp("FT Config", "config_ft_ddp_embed.json", classes="flex-1")
            c_embed = self._field_inp("FT-Embed Config", "config_xgb_from_ft_embed.json", classes="flex-1")
        with ui.row().classes("w-full gap-2"):
            c_ld = self._field_inp("Direct Label", "XGB_raw", classes="flex-1")
            c_lf = self._field_inp("FT Label", "XGB_ft_embed", classes="flex-1")
            c_rt = ui.checkbox("Runtime FT Embedding", value=False)
            c_bins = self._field_num("Bins", 10)

        def _suggest_defaults():
            key = str(c_key.value or "").lower()
            if key == "resn":
                c_direct.value = "config_resn_direct.json"
                c_embed.value = "config_resn_from_ft_embed.json"
                c_ld.value = "ResN_raw"
                c_lf.value = "ResN_ft_embed"
            else:
                c_direct.value = "config_xgb_direct.json"
                c_embed.value = "config_xgb_from_ft_embed.json"
                c_ld.value = "XGB_raw"
                c_lf.value = "XGB_ft_embed"

        c_key.on_value_change(lambda _e: _suggest_defaults())

        with ui.expansion("Advanced: Split Data Override", icon="settings").classes("w-full"):
            with ui.row().classes("w-full gap-2"):
                c_train = self._field_inp("Train Data Path", "", classes="flex-1")
                c_test = self._field_inp("Test Data Path", "", classes="flex-1")

        c_status = ui.label("").classes("text-sm")
        c_log = self._field_txt("Logs", mono=True).props("readonly outlined rows=10")
        c_gallery = ui.row().classes("w-full flex-wrap gap-2")

        def _run_compare():
            if not self._require_ui_permission("task:run"):
                return
            _StreamRunner(c_status, c_log, c_gallery).run(
                self.app.run_compare_ui,
                c_key.value,
                c_direct.value,
                c_ft.value,
                c_embed.value,
                c_ld.value,
                c_lf.value,
                bool(c_rt.value),
                int(c_bins.value or 10),
                c_train.value,
                c_test.value,
                None,
                None,
                None,
                None,
                None,
                None,
                actor=self._actor_for_runtime(),
            )

        run_compare_btn = ui.button(
            "Run FT-Embed Compare",
            icon="play_arrow",
            on_click=_run_compare,
        ).props("color=primary")
        self._register_permission_control(run_compare_btn, "task:run")

    def _tab_prediction(self):
        self._guide("Prediction Workflow", [
            "Use models trained with FT Two-Step to score a new dataset.",
            "Provide FT config and downstream model configs (XGB/ResN) as needed.",
            "Input Data is the file to score; Output CSV is where predictions are saved.",
        ])
        self._tip(
            "Model Keys controls which downstream models are used for scoring. "
            "Typical value is xgb,resn. Model Name can be left empty."
        )

        with ui.row().classes("w-full gap-2"):
            p_ft = self._field_inp("FT Config", "config_ft_ddp_embed.json", classes="flex-1")
            p_xgb = self._field_inp("XGB Config (optional)", "config_xgb_from_ft_embed.json", classes="flex-1")
            p_resn = self._field_inp("ResN Config (optional)", "config_resn_from_ft_embed.json", classes="flex-1")
        with ui.row().classes("w-full gap-2"):
            p_name = self._field_inp("Model Name (optional)", "", classes="flex-1")
            p_keys = self._field_inp("Model Keys", "xgb, resn", classes="flex-1")
            p_in = self._field_inp("Input Data", "./Data/od_bc_new.csv", classes="flex-1")
            p_out = self._field_inp("Output CSV", "./Results/predictions_ft_xgb.csv", classes="flex-1")

        p_status = ui.label("").classes("text-sm")
        p_log = self._field_txt("Logs", mono=True).props("readonly outlined rows=12")

        def _run_prediction():
            if not self._require_ui_permission("task:run"):
                return
            _StreamRunner(p_status, p_log).run(
                self.app.run_predict_ui,
                p_ft.value,
                p_xgb.value,
                p_resn.value,
                p_in.value,
                p_out.value,
                p_name.value,
                p_keys.value,
                None,
                None,
                None,
                None,
                None,
                None,
                actor=self._actor_for_runtime(),
            )

        run_prediction_btn = ui.button(
            "Run Prediction",
            icon="play_arrow",
            on_click=_run_prediction,
        ).props("color=primary")
        self._register_permission_control(run_prediction_btn, "task:run")
