"""Runtime workflow methods for PricingApp."""

from __future__ import annotations

from collections import deque
import json
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterable, Optional, Sequence


class _MissingTrainingConfigError(ValueError):
    """Raised when no runnable training config can be resolved."""


class _LogBuffer:
    """Bounded log buffer that keeps the most recent characters."""

    def __init__(self, max_chars: int):
        self._max_chars = max(200, int(max_chars))
        self._chunks: deque[str] = deque()
        self._total_chars = 0
        self._version = 0

    @property
    def version(self) -> int:
        return self._version

    def extend(self, lines: Sequence[str]) -> None:
        for raw in lines:
            chunk = str(raw or "")
            if not chunk:
                continue
            if not chunk.endswith("\n"):
                chunk = f"{chunk}\n"
            self._chunks.append(chunk)
            self._total_chars += len(chunk)
            self._version += 1
            while self._total_chars > self._max_chars and self._chunks:
                removed = self._chunks.popleft()
                self._total_chars -= len(removed)

    def text(self) -> str:
        return "".join(self._chunks)


class AppControllerRuntimeMixin:
    @staticmethod
    def _dump_json(payload: Dict[str, Any]) -> str:
        return json.dumps(payload, indent=2, ensure_ascii=False)

    @staticmethod
    def _extract_task_mode(config: Dict[str, Any]) -> str:
        runner = config.get("runner", {})
        if not isinstance(runner, dict):
            return "entry"
        return str(runner.get("mode", "entry") or "entry")

    def _write_temp_config(
        self,
        config: Dict[str, Any],
        *,
        base_dir: Path,
        prefix: str = "temp_config_",
        suffix: str = ".json",
    ) -> Path:
        fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=base_dir)
        temp_config_path = Path(temp_path)
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(config, handle, indent=2)
        return temp_config_path

    def _resolve_training_config(
        self,
        config_json: str,
    ) -> tuple[Dict[str, Any], str, Path, Optional[Path]]:
        temp_config_path: Optional[Path] = None
        if config_json:
            config = json.loads(config_json)
            task_mode = self._extract_task_mode(config)
            base_dir = self._default_base_dir(self.current_config_dir)
            temp_config_path = self._write_temp_config(config, base_dir=base_dir)
            return config, task_mode, temp_config_path, temp_config_path
        if self.current_config_path and self.current_config_path.exists():
            config_path = self.current_config_path
            config = json.loads(config_path.read_text(encoding="utf-8"))
            task_mode = self._extract_task_mode(config)
            return config, task_mode, config_path, None
        if self.current_config:
            config = self.current_config
            task_mode = self._extract_task_mode(config)
            base_dir = self._default_base_dir(self.current_config_dir)
            temp_config_path = self._write_temp_config(config, base_dir=base_dir)
            return config, task_mode, temp_config_path, temp_config_path
        raise _MissingTrainingConfigError("No configuration provided")

    def _resolve_existing_config_path(
        self,
        path_value: str,
        *,
        candidate_dirs: Iterable[Optional[Path]],
    ) -> Path:
        path_obj = Path(path_value).expanduser()
        if path_obj.is_absolute():
            resolved = path_obj.resolve()
            if resolved.exists():
                return resolved
            raise FileNotFoundError(path_value)

        for root in candidate_dirs:
            if root is None:
                continue
            candidate = (Path(root).resolve() / path_obj).resolve()
            if candidate.exists():
                return candidate
        raise FileNotFoundError(path_value)

    def _resolve_model_override_paths(
        self,
        **path_specs: tuple[Optional[str], Optional[Any]],
    ) -> Dict[str, Optional[str]]:
        resolved: Dict[str, Optional[str]] = {}
        for key, (manual_path, uploaded_file) in path_specs.items():
            resolved[key] = self._resolve_override_path(manual_path, uploaded_file)
        return resolved

    @staticmethod
    def _log_char_limit() -> int:
        raw = os.environ.get("INS_PRICING_FRONTEND_LOG_MAX_CHARS", "").strip()
        if not raw:
            return 200_000
        try:
            return max(200, int(raw))
        except ValueError:
            return 200_000

    def _iter_log_snapshots(self, log_generator: Iterable[str]) -> Generator[str, None, None]:
        """Yield bounded log snapshots with batched updates."""
        buffer = _LogBuffer(self._log_char_limit())
        pending: list[str] = []
        last_emit = time.monotonic()
        last_sent_version = -1
        for log_line in log_generator:
            pending.append(str(log_line))
            now = time.monotonic()
            if len(pending) >= 24 or (now - last_emit) >= 0.2:
                buffer.extend(pending)
                pending = []
                last_emit = now
                if buffer.version != last_sent_version:
                    last_sent_version = buffer.version
                    yield buffer.text()

        if pending:
            buffer.extend(pending)
        if buffer.version != last_sent_version:
            yield buffer.text()

    def run_workflow_config_ui(self, workflow_config_json: str) -> Generator[tuple[str, str], None, None]:
        """Run plotting/prediction/compare/pre-oneway from workflow JSON config."""
        try:
            if workflow_config_json:
                workflow_config = json.loads(workflow_config_json)
                base_dir = self._default_base_dir(self.current_workflow_config_dir)
                self.current_workflow_config = workflow_config
            elif self.current_workflow_config:
                workflow_config = self.current_workflow_config
                base_dir = self._default_base_dir(self.current_workflow_config_dir)
            else:
                yield "No workflow configuration provided", ""
                return

            mode = str(workflow_config.get("workflow", workflow_config).get("mode", "unknown")).strip().lower()
            runner = self._get_runner()
            log_generator = runner.run_callable(
                self._run_workflow_from_config,
                workflow_config,
                base_dir,
            )

            full_log = ""
            for full_log in self._iter_log_snapshots(log_generator):
                yield f"Workflow [{mode}] in progress...", full_log

            yield f"Workflow [{mode}] completed!", full_log

        except Exception as e:
            error_msg = f"Workflow config execution error: {str(e)}"
            yield error_msg, error_msg

    def run_training(self, config_json: str) -> Generator[tuple[str, str], None, None]:
        """
        Run task (training, explain, plotting, etc.) with the current configuration.

        The task type is automatically detected from config.runner.mode.
        Supported modes: entry (training), explain, incremental, watchdog, etc.
        """
        temp_config_path: Optional[Path] = None
        try:
            _, task_mode, config_path, temp_config_path = self._resolve_training_config(
                config_json
            )

            runner = self._get_runner()
            log_generator = runner.run_task(str(config_path))

            full_log = ""
            for full_log in self._iter_log_snapshots(log_generator):
                yield f"Task [{task_mode}] in progress...", full_log

            yield f"Task [{task_mode}] completed!", full_log

        except _MissingTrainingConfigError:
            yield "No configuration provided", ""
            return
        except Exception as e:
            error_msg = f"Error during task execution: {str(e)}"
            yield error_msg, error_msg
        finally:
            if temp_config_path is not None:
                try:
                    temp_config_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def prepare_ft_step1(self, config_json: str, use_ddp: bool, nproc: int) -> tuple[str, str]:
        """Prepare FT Step 1 configuration."""
        if not config_json:
            return "No configuration provided", ""

        try:
            config = json.loads(config_json)
            step1_config = self.ft_workflow.prepare_step1_config(
                base_config=config,
                use_ddp=use_ddp,
                nproc_per_node=int(nproc)
            )

            # Save to temp file
            base_dir = self._default_base_dir(self.current_config_dir)
            temp_path = (base_dir / "temp_ft_step1_config.json").resolve()
            temp_path.write_text(self._dump_json(step1_config), encoding="utf-8")

            self.current_step1_config = str(temp_path)
            step1_json = self._dump_json(step1_config)

            return "Step 1 config prepared. Click 'Run Step 1' to train FT embeddings.", step1_json

        except Exception as e:
            return f"Error preparing Step 1 config: {str(e)}", ""

    def prepare_ft_step2(
        self,
        step1_config_path: str,
        target_models: str,
        augmented_data_dir: str,
        xgb_overrides_json: str,
        resn_overrides_json: str,
    ) -> tuple[str, str, str]:
        """Prepare FT Step 2 configurations."""
        if not step1_config_path:
            return "Step 1 config not found. Run Step 1 first.", "", ""

        try:
            step1_path = self._resolve_existing_config_path(
                step1_config_path,
                candidate_dirs=(
                    self.current_config_dir,
                    self.working_dir,
                    Path.cwd(),
                ),
            )

            models = [m.strip() for m in target_models.split(',') if m.strip()]
            data_dir_value = str(augmented_data_dir or "").strip() or "./DataFTUnsupervised"
            xgb_overrides = self._parse_json_dict(
                xgb_overrides_json,
                "xgb_overrides_json",
            )
            resn_overrides = self._parse_json_dict(
                resn_overrides_json,
                "resn_overrides_json",
            )
            xgb_cfg, resn_cfg = self.ft_workflow.generate_step2_configs(
                step1_config_path=str(step1_path),
                target_models=models,
                augmented_data_dir=data_dir_value,
                xgb_overrides=xgb_overrides,
                resn_overrides=resn_overrides,
            )
            save_dir = step1_path.parent
            saved_paths: Dict[str, str] = {}

            if xgb_cfg:
                xgb_cfg_path = save_dir / "config_xgb_from_ft_unsupervised.json"
                xgb_cfg_path.write_text(
                    self._dump_json(xgb_cfg),
                    encoding="utf-8",
                )
                saved_paths["xgb"] = str(xgb_cfg_path)

            if resn_cfg:
                resn_cfg_path = save_dir / "config_resn_from_ft_unsupervised.json"
                resn_cfg_path.write_text(
                    self._dump_json(resn_cfg),
                    encoding="utf-8",
                )
                saved_paths["resn"] = str(resn_cfg_path)

            self.current_step2_config_paths = saved_paths

            status_lines = [
                f"Step 2 configs prepared for: {', '.join(models)}",
                f"Augmented data dir: {data_dir_value}",
            ]
            if "xgb" in saved_paths:
                status_lines.append(f"Saved XGB config: {saved_paths['xgb']}")
            if "resn" in saved_paths:
                status_lines.append(f"Saved ResN config: {saved_paths['resn']}")
            status_msg = "\n".join(status_lines)
            xgb_json = self._dump_json(xgb_cfg) if xgb_cfg else ""
            resn_json = self._dump_json(resn_cfg) if resn_cfg else ""

            return status_msg, xgb_json, resn_json

        except FileNotFoundError as e:
            return f"Error: {str(e)}\n\nMake sure Step 1 completed successfully.", "", ""
        except Exception as e:
            return f"Error preparing Step 2 configs: {str(e)}", "", ""

    def open_results_folder(self, config_json: str) -> str:
        """Open the results folder in file explorer."""
        try:
            if config_json:
                config = json.loads(config_json)
                output_dir = config.get('output_dir', './Results')
                out = Path(str(output_dir))
                if out.is_absolute():
                    results_path = out.resolve()
                else:
                    results_path = (self._default_base_dir(self.current_config_dir) / out).resolve()
            elif self.current_config_path and self.current_config_path.exists():
                config = json.loads(
                    self.current_config_path.read_text(encoding="utf-8"))
                output_dir = config.get('output_dir', './Results')
                results_path = (
                    self.current_config_path.parent / output_dir).resolve()
            elif self.current_config:
                output_dir = self.current_config.get('output_dir', './Results')
                out = Path(str(output_dir))
                if out.is_absolute():
                    results_path = out.resolve()
                else:
                    results_path = (self._default_base_dir(self.current_config_dir) / out).resolve()
            else:
                return "No configuration loaded"

            if not results_path.exists():
                return f"Results folder does not exist yet: {results_path}"

            # Open folder based on OS
            system = platform.system()
            if system == "Windows":
                os.startfile(results_path)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", str(results_path)])
            else:  # Linux
                subprocess.run(["xdg-open", str(results_path)])

            return f"Opened folder: {results_path}"

        except Exception as e:
            return f"Error opening folder: {str(e)}"

    def _run_workflow(self, label: str, func: Callable, *args, **kwargs):
        """Run a workflow function and stream logs."""
        try:
            runner = self._get_runner()
            log_generator = runner.run_callable(func, *args, **kwargs)
            full_log = ""
            for full_log in self._iter_log_snapshots(log_generator):
                yield f"{label} in progress...", full_log
            yield f"{label} completed!", full_log
        except Exception as e:
            error_msg = f"{label} error: {str(e)}"
            yield error_msg, error_msg

    def _run_workflow_with_preview(
        self,
        label: str,
        func: Callable,
        preview_collector: Callable[[float], list[str]],
        *args,
        **kwargs,
    ):
        """Run a workflow and return status/logs plus generated image previews."""
        try:
            runner = self._get_runner()
            started_at = time.time()
            log_generator = runner.run_callable(func, *args, **kwargs)
            full_log = ""
            for full_log in self._iter_log_snapshots(log_generator):
                yield f"{label} in progress...", full_log, []

            previews: list[str] = []
            try:
                previews = preview_collector(started_at)
            except Exception as preview_exc:
                full_log += f"[Warn] Failed to collect image previews: {preview_exc}\n"

            yield f"{label} completed!", full_log, previews
        except Exception as e:
            error_msg = f"{label} error: {str(e)}"
            yield error_msg, error_msg, []

    def _load_json_file(self, path_value: str) -> tuple[Path, Dict[str, Any]]:
        path_obj = self._resolve_user_path(path_value, base_dir=self.working_dir)
        if not path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {path_obj}")
        payload = json.loads(path_obj.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Config must be a JSON object: {path_obj}")
        return path_obj, payload

    @staticmethod
    def _resolve_cfg_output_dir(cfg_obj: Dict[str, Any], cfg_path: Path) -> Path:
        raw_output = str(cfg_obj.get("output_dir", "./Results") or "./Results").strip()
        out_path = Path(raw_output).expanduser()
        if not out_path.is_absolute():
            out_path = (cfg_path.parent / out_path).resolve()
        else:
            out_path = out_path.resolve()
        return out_path

    @staticmethod
    def _collect_png_paths(
        candidates: Iterable[Path],
        *,
        min_mtime: Optional[float] = None,
        limit: int = 40,
    ) -> list[str]:
        ranked: list[tuple[float, str]] = []
        seen: set[str] = set()
        for candidate in candidates:
            path_obj = Path(candidate).resolve()
            if not path_obj.exists():
                continue
            if path_obj.is_file():
                file_iter = [path_obj] if path_obj.suffix.lower() == ".png" else []
            else:
                file_iter = path_obj.rglob("*.png")
            for file_path in file_iter:
                if not file_path.is_file():
                    continue
                resolved = str(file_path.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                try:
                    mtime = file_path.stat().st_mtime
                except OSError:
                    continue
                ranked.append((mtime, resolved))

        ranked.sort(key=lambda x: x[0], reverse=True)
        if min_mtime is not None:
            recent = [path for mtime, path in ranked if mtime >= min_mtime]
            if recent:
                return recent[:limit]
        return [path for _, path in ranked[:limit]]

    def _collect_prediction_plot_images(
        self,
        *,
        cfg_path: str,
        xgb_cfg_path: str,
        resn_cfg_path: str,
        started_at: float,
    ) -> list[str]:
        cfg_file, cfg_obj = self._load_json_file(cfg_path)
        xgb_file, xgb_obj = self._load_json_file(xgb_cfg_path)
        resn_file, resn_obj = self._load_json_file(resn_cfg_path)

        model_list = cfg_obj.get("model_list") or []
        model_categories = cfg_obj.get("model_categories") or []
        model_name = ""
        if model_list and model_categories:
            model_name = f"{model_list[0]}_{model_categories[0]}"

        candidate_roots: list[Path] = []
        for file_path, cfg_item in (
            (cfg_file, cfg_obj),
            (xgb_file, xgb_obj),
            (resn_file, resn_obj),
        ):
            output_root = self._resolve_cfg_output_dir(cfg_item, file_path)
            plot_root = output_root / "plot"
            candidate_roots.append(plot_root)
            if model_name:
                candidate_roots.append(plot_root / model_name)

        return self._collect_png_paths(
            candidate_roots,
            min_mtime=started_at - 1.0,
        )

    def _collect_pre_oneway_images(
        self,
        *,
        data_path: str,
        train_data_path: str,
        test_data_path: str,
        model_name: str,
        output_dir: str,
        started_at: float,
    ) -> list[str]:
        if str(output_dir or "").strip():
            out_dir = self._resolve_user_path(output_dir, base_dir=self.working_dir)
        else:
            base_path: Optional[Path] = None
            if str(train_data_path or "").strip():
                base_path = self._resolve_user_path(train_data_path, base_dir=self.working_dir)
            elif str(test_data_path or "").strip():
                base_path = self._resolve_user_path(test_data_path, base_dir=self.working_dir)
            elif str(data_path or "").strip():
                base_path = self._resolve_user_path(data_path, base_dir=self.working_dir)
            root_dir = base_path.parent if base_path is not None else self.working_dir
            out_dir = (root_dir / "Results" / "plot" / model_name / "oneway" / "pre").resolve()
        return self._collect_png_paths([out_dir], min_mtime=started_at - 1.0)

    def _collect_compare_images(
        self,
        *,
        direct_cfg_path: str,
        started_at: float,
    ) -> list[str]:
        cfg_file, cfg_obj = self._load_json_file(direct_cfg_path)
        output_root = self._resolve_cfg_output_dir(cfg_obj, cfg_file)
        plot_root = output_root / "plot"

        model_list = cfg_obj.get("model_list") or []
        model_categories = cfg_obj.get("model_categories") or []
        candidates: list[Path] = [plot_root]
        if model_list and model_categories:
            model_name = f"{model_list[0]}_{model_categories[0]}"
            candidates.append(plot_root / model_name / "double_lift")

        return self._collect_png_paths(candidates, min_mtime=started_at - 1.0)

    def _collect_double_lift_images(
        self,
        *,
        data_path: str,
        output_path: str,
        started_at: float,
    ) -> list[str]:
        raw_output_path = str(output_path or "").strip()
        if raw_output_path:
            target_path = self._resolve_user_path(raw_output_path, base_dir=self.working_dir)
            candidates = [target_path]
        else:
            data_obj = self._resolve_user_path(data_path, base_dir=self.working_dir)
            candidates = [(data_obj.parent / "Results" / "plot").resolve()]
        return self._collect_png_paths(candidates, min_mtime=started_at - 1.0)

    def run_pre_oneway_ui(
        self,
        data_path: str,
        model_name: str,
        target_col: str,
        weight_col: str,
        feature_list: str,
        oneway_factors: Optional[Sequence[str]],
        categorical_features: str,
        n_bins: int,
        holdout_ratio: float,
        rand_seed: int,
        output_dir: str,
        train_data_path: str,
        test_data_path: str,
    ):
        workflows_module = self._load_workflows_module()
        selected_factors = self._normalize_feature_values(oneway_factors)
        resolved_feature_list = ",".join(selected_factors) if selected_factors else feature_list
        yield from self._run_workflow_with_preview(
            "Pre-Oneway Plot",
            workflows_module.run_pre_oneway,
            lambda started_at: self._collect_pre_oneway_images(
                data_path=data_path,
                train_data_path=train_data_path,
                test_data_path=test_data_path,
                model_name=model_name,
                output_dir=output_dir,
                started_at=started_at,
            ),
            data_path=data_path,
            model_name=model_name,
            target_col=target_col,
            weight_col=weight_col,
            feature_list=resolved_feature_list,
            categorical_features=categorical_features,
            n_bins=n_bins,
            holdout_ratio=holdout_ratio,
            rand_seed=rand_seed,
            output_dir=output_dir or None,
            train_data_path=train_data_path or None,
            test_data_path=test_data_path or None,
        )

    def run_plot_direct_ui(
        self,
        cfg_path: str,
        xgb_cfg_path: str,
        resn_cfg_path: str,
        oneway_factors: Optional[Sequence[str]],
        train_data_path: str,
        test_data_path: str,
        xgb_model_file: Optional[Any] = None,
        resn_model_file: Optional[Any] = None,
        xgb_model_path: Optional[str] = None,
        resn_model_path: Optional[str] = None,
    ):
        workflows_module = self._load_workflows_module()
        selected_factors = self._normalize_feature_values(oneway_factors)
        resolved_paths = self._resolve_model_override_paths(
            xgb=(xgb_model_path, xgb_model_file),
            resn=(resn_model_path, resn_model_file),
        )
        yield from self._run_workflow_with_preview(
            "Direct Plot",
            workflows_module.run_plot_direct,
            lambda started_at: self._collect_prediction_plot_images(
                cfg_path=cfg_path,
                xgb_cfg_path=xgb_cfg_path,
                resn_cfg_path=resn_cfg_path,
                started_at=started_at,
            ),
            cfg_path=cfg_path,
            xgb_cfg_path=xgb_cfg_path,
            resn_cfg_path=resn_cfg_path,
            oneway_features=selected_factors or None,
            train_data_path=train_data_path or None,
            test_data_path=test_data_path or None,
            xgb_model_path=resolved_paths["xgb"],
            resn_model_path=resolved_paths["resn"],
            model_search_dir=str(self.working_dir),
        )

    def run_plot_embed_ui(
        self,
        cfg_path: str,
        xgb_cfg_path: str,
        resn_cfg_path: str,
        ft_cfg_path: str,
        use_runtime_ft_embedding: bool,
        oneway_factors: Optional[Sequence[str]],
        train_data_path: str,
        test_data_path: str,
        xgb_model_file: Optional[Any] = None,
        resn_model_file: Optional[Any] = None,
        ft_model_file: Optional[Any] = None,
        xgb_model_path: Optional[str] = None,
        resn_model_path: Optional[str] = None,
        ft_model_path: Optional[str] = None,
    ):
        workflows_module = self._load_workflows_module()
        selected_factors = self._normalize_feature_values(oneway_factors)
        resolved_paths = self._resolve_model_override_paths(
            xgb=(xgb_model_path, xgb_model_file),
            resn=(resn_model_path, resn_model_file),
            ft=(ft_model_path, ft_model_file),
        )
        yield from self._run_workflow_with_preview(
            "Embed Plot",
            workflows_module.run_plot_embed,
            lambda started_at: self._collect_prediction_plot_images(
                cfg_path=cfg_path,
                xgb_cfg_path=xgb_cfg_path,
                resn_cfg_path=resn_cfg_path,
                started_at=started_at,
            ),
            cfg_path=cfg_path,
            xgb_cfg_path=xgb_cfg_path,
            resn_cfg_path=resn_cfg_path,
            ft_cfg_path=ft_cfg_path,
            use_runtime_ft_embedding=use_runtime_ft_embedding,
            oneway_features=selected_factors or None,
            train_data_path=train_data_path or None,
            test_data_path=test_data_path or None,
            xgb_model_path=resolved_paths["xgb"],
            resn_model_path=resolved_paths["resn"],
            ft_model_path=resolved_paths["ft"],
            model_search_dir=str(self.working_dir),
        )

    def run_predict_ui(
        self,
        ft_cfg_path: str,
        xgb_cfg_path: str,
        resn_cfg_path: str,
        input_path: str,
        output_path: str,
        model_name: str,
        model_keys: str,
        ft_model_file: Optional[Any] = None,
        xgb_model_file: Optional[Any] = None,
        resn_model_file: Optional[Any] = None,
        ft_model_path: Optional[str] = None,
        xgb_model_path: Optional[str] = None,
        resn_model_path: Optional[str] = None,
    ):
        workflows_module = self._load_workflows_module()
        resolved_paths = self._resolve_model_override_paths(
            ft=(ft_model_path, ft_model_file),
            xgb=(xgb_model_path, xgb_model_file),
            resn=(resn_model_path, resn_model_file),
        )
        yield from self._run_workflow(
            "Prediction",
            workflows_module.run_predict_ft_embed,
            ft_cfg_path=ft_cfg_path,
            xgb_cfg_path=xgb_cfg_path or None,
            resn_cfg_path=resn_cfg_path or None,
            input_path=input_path,
            output_path=output_path,
            model_name=model_name or None,
            model_keys=model_keys,
            ft_model_path=resolved_paths["ft"],
            xgb_model_path=resolved_paths["xgb"],
            resn_model_path=resolved_paths["resn"],
            model_search_dir=str(self.working_dir),
        )

    def run_compare_ui(
        self,
        model_key: str,
        direct_cfg_path: str,
        ft_cfg_path: str,
        ft_embed_cfg_path: str,
        label_direct: str,
        label_ft: str,
        use_runtime_ft_embedding: bool,
        n_bins_override: int,
        train_data_path: str,
        test_data_path: str,
        direct_model_file: Optional[Any] = None,
        ft_embed_model_file: Optional[Any] = None,
        ft_model_file: Optional[Any] = None,
        direct_model_path: Optional[str] = None,
        ft_embed_model_path: Optional[str] = None,
        ft_model_path: Optional[str] = None,
    ):
        model_key_norm = str(model_key or "").strip().lower()
        if model_key_norm not in {"xgb", "resn"}:
            raise ValueError("model_key must be one of: xgb, resn.")
        label = "Compare XGB" if model_key_norm == "xgb" else "Compare ResNet"
        resolved_paths = self._resolve_model_override_paths(
            direct=(direct_model_path, direct_model_file),
            ft_embed=(ft_embed_model_path, ft_embed_model_file),
            ft=(ft_model_path, ft_model_file),
        )
        yield from self._run_compare_ui(
            model_key=model_key_norm,
            label=label,
            direct_cfg_path=direct_cfg_path,
            ft_cfg_path=ft_cfg_path,
            ft_embed_cfg_path=ft_embed_cfg_path,
            label_direct=label_direct,
            label_ft=label_ft,
            use_runtime_ft_embedding=use_runtime_ft_embedding,
            n_bins_override=n_bins_override,
            train_data_path=train_data_path,
            test_data_path=test_data_path,
            direct_model_path=resolved_paths["direct"] or "",
            ft_embed_model_path=resolved_paths["ft_embed"] or "",
            ft_model_path=resolved_paths["ft"] or "",
        )

    def _run_compare_ui(
        self,
        *,
        model_key: str,
        label: str,
        direct_cfg_path: str,
        ft_cfg_path: str,
        ft_embed_cfg_path: str,
        label_direct: str,
        label_ft: str,
        use_runtime_ft_embedding: bool,
        n_bins_override: int,
        train_data_path: str,
        test_data_path: str,
        direct_model_path: str,
        ft_embed_model_path: str,
        ft_model_path: str,
    ):
        workflows_module = self._load_workflows_module()
        yield from self._run_workflow_with_preview(
            label,
            workflows_module.run_compare_ft_embed,
            lambda started_at: self._collect_compare_images(
                direct_cfg_path=direct_cfg_path,
                started_at=started_at,
            ),
            direct_cfg_path=direct_cfg_path,
            ft_cfg_path=ft_cfg_path,
            ft_embed_cfg_path=ft_embed_cfg_path,
            model_key=model_key,
            label_direct=label_direct,
            label_ft=label_ft,
            use_runtime_ft_embedding=use_runtime_ft_embedding,
            n_bins_override=n_bins_override,
            train_data_path=train_data_path or None,
            test_data_path=test_data_path or None,
            direct_model_path=direct_model_path or None,
            ft_embed_model_path=ft_embed_model_path or None,
            ft_model_path=ft_model_path or None,
            model_search_dir=str(self.working_dir),
        )

    def run_double_lift_ui(
        self,
        data_path: str,
        pred_col_1: str,
        pred_col_2: str,
        target_col: str,
        weight_col: str,
        n_bins: int,
        label1: str,
        label2: str,
        pred1_weighted: bool,
        pred2_weighted: bool,
        actual_weighted: bool,
        holdout_ratio: float,
        split_strategy: str,
        split_group_col: str,
        split_time_col: str,
        split_time_ascending: bool,
        rand_seed: int,
        output_path: str,
    ):
        workflows_module = self._load_workflows_module()
        yield from self._run_workflow_with_preview(
            "Double Lift",
            workflows_module.run_double_lift_from_file,
            lambda started_at: self._collect_double_lift_images(
                data_path=data_path,
                output_path=output_path,
                started_at=started_at,
            ),
            data_path=data_path,
            pred_col_1=pred_col_1,
            pred_col_2=pred_col_2,
            target_col=target_col,
            weight_col=weight_col,
            n_bins=n_bins,
            label1=label1 or None,
            label2=label2 or None,
            pred1_weighted=pred1_weighted,
            pred2_weighted=pred2_weighted,
            actual_weighted=actual_weighted,
            holdout_ratio=holdout_ratio,
            split_strategy=split_strategy,
            split_group_col=split_group_col or None,
            split_time_col=split_time_col or None,
            split_time_ascending=split_time_ascending,
            rand_seed=rand_seed,
            output_path=output_path or None,
        )
