"""Configuration and path-resolution methods for PricingApp."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

class AppControllerConfigMixin:
    def load_json_config(self, file_path) -> tuple[str, Dict[str, Any], str]:
        """Load configuration from uploaded JSON file."""
        if not file_path:
            return "No file uploaded", {}, ""

        try:
            path = self._resolve_user_path(str(file_path), base_dir=self.working_dir)
            with open(path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.current_config = config
            self.current_config_path = path
            self.current_config_dir = path.parent
            config_json = json.dumps(config, indent=2, ensure_ascii=False)
            return f"Configuration loaded successfully from {path.name}", config, config_json
        except Exception as e:
            return f"Error loading config: {str(e)}", {}, ""

    def set_working_dir(self, working_dir: str) -> tuple[str, str]:
        """Set working directory used for relative paths and generated configs."""
        try:
            scope_root = getattr(self, "user_workspace_root", None)
            raw = str(working_dir or "").strip()
            if not raw:
                resolved = Path(scope_root).resolve() if scope_root is not None else Path.cwd().resolve()
            else:
                candidate = Path(raw).expanduser()
                if not candidate.is_absolute():
                    candidate = (self.working_dir / candidate).resolve()
                else:
                    candidate = candidate.resolve()
                if not candidate.exists():
                    return f"Working directory does not exist: {candidate}", str(self.working_dir)
                if not candidate.is_dir():
                    return f"Working directory is not a folder: {candidate}", str(self.working_dir)
                if scope_root is not None and not self._is_relative_to(candidate, Path(scope_root).resolve()):
                    return (
                        f"Working directory must stay inside your workspace: {scope_root}",
                        str(self.working_dir),
                    )
                resolved = candidate

            self.working_dir = resolved
            return f"Working directory set to: {resolved}", str(resolved)
        except Exception as e:
            return f"Error setting working directory: {str(e)}", str(self.working_dir)

    def list_directory_candidates(
        self,
        root_dir: str,
        *,
        max_depth: int = 2,
        max_items: int = 300,
    ) -> tuple[str, list[str], str]:
        """List directories for manual working-directory selection."""
        try:
            raw_root = str(root_dir or "").strip()
            candidate = Path(raw_root).expanduser() if raw_root else self.working_dir
            if not candidate.is_absolute():
                candidate = (self.working_dir / candidate).resolve()
            else:
                candidate = candidate.resolve()
            scope_root = getattr(self, "user_workspace_root", None)
            if scope_root is not None:
                scope_root_obj = Path(scope_root).resolve()
                if not self._is_relative_to(candidate, scope_root_obj):
                    candidate = scope_root_obj

            if not candidate.exists():
                fallback = str(self.working_dir)
                return f"Browse root does not exist: {candidate}", [fallback], fallback
            if not candidate.is_dir():
                fallback = str(self.working_dir)
                return f"Browse root is not a folder: {candidate}", [fallback], fallback

            depth_limit = max(0, int(max_depth))
            item_limit = max(1, int(max_items))
            root = candidate
            root_parts = len(root.parts)
            dirs: list[str] = []

            for current, child_dirs, _ in os.walk(root):
                current_path = Path(current).resolve()
                rel_depth = len(current_path.parts) - root_parts
                dirs.append(str(current_path))

                if rel_depth >= depth_limit:
                    child_dirs[:] = []

                if len(dirs) >= item_limit:
                    break

            dirs = sorted(dict.fromkeys(dirs))
            selected = str(self.working_dir) if str(self.working_dir) in dirs else str(root)
            truncated = len(dirs) >= item_limit
            status = f"Found {len(dirs)} folders under: {root}"
            if truncated:
                status += f" (limited to first {item_limit})"
            return status, dirs, selected
        except Exception as e:
            fallback = str(self.working_dir)
            return f"Error listing folders: {str(e)}", [fallback], fallback

    def _resolve_workdir_path(self, relative_path: Optional[str] = None) -> Path:
        root = Path(self.working_dir).resolve()
        raw = str(relative_path or "").strip()
        if not raw:
            return root
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = (root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if not self._is_relative_to(candidate, root):
            raise ValueError(f"Path is outside current work_dir: {candidate}")
        return candidate

    def list_workdir_entries(
        self,
        subdir: str = "",
        *,
        include_hidden: bool = False,
        max_items: int = 500,
    ) -> tuple[str, list[Dict[str, Any]]]:
        """List files/folders under working directory (optionally under a subdir)."""
        try:
            target = self._resolve_workdir_path(subdir)
            if not target.exists():
                return f"Path does not exist: {target}", []
            if not target.is_dir():
                return f"Path is not a folder: {target}", []

            limit = max(1, int(max_items))
            rows: list[Dict[str, Any]] = []
            root = Path(self.working_dir).resolve()
            children = sorted(
                target.iterdir(),
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
            for item in children:
                name = item.name
                if not include_hidden and name.startswith("."):
                    continue
                try:
                    stat = item.stat()
                    size = int(stat.st_size) if item.is_file() else None
                    modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                except OSError:
                    size = None
                    modified = ""
                rel = str(item.resolve().relative_to(root))
                rows.append(
                    {
                        "name": name,
                        "type": "dir" if item.is_dir() else "file",
                        "size": size,
                        "modified": modified,
                        "path": rel.replace("\\", "/"),
                    }
                )
                if len(rows) >= limit:
                    break

            status = f"Listed {len(rows)} entries under: {target}"
            if len(rows) >= limit:
                status += f" (limited to {limit})"
            return status, rows
        except Exception as e:
            return f"Error listing work_dir files: {str(e)}", []

    def save_workdir_upload(
        self,
        *,
        file_name: str,
        content_bytes: bytes,
        subdir: str = "",
        overwrite: bool = True,
    ) -> str:
        """Save uploaded file into work_dir/subdir."""
        if not file_name:
            return "Upload file name is empty."
        try:
            target_dir = self._resolve_workdir_path(subdir)
            target_dir.mkdir(parents=True, exist_ok=True)
            if not target_dir.is_dir():
                return f"Upload target is not a folder: {target_dir}"
            safe_name = Path(str(file_name)).name
            target_path = (target_dir / safe_name).resolve()
            root = Path(self.working_dir).resolve()
            if not self._is_relative_to(target_path, root):
                return f"Upload path is outside work_dir: {target_path}"
            if target_path.exists() and not overwrite:
                return f"File already exists (overwrite disabled): {target_path}"
            target_path.write_bytes(content_bytes)
            return f"Uploaded to: {target_path}"
        except Exception as e:
            return f"Upload failed: {str(e)}"

    def delete_workdir_entry(self, relative_path: str, *, recursive: bool = True) -> str:
        """Delete file/folder under current working directory."""
        try:
            target = self._resolve_workdir_path(relative_path)
            root = Path(self.working_dir).resolve()
            if target == root:
                return "Refuse to delete current work_dir root."
            if not target.exists():
                return f"Path not found: {target}"
            if target.is_file():
                target.unlink()
                return f"Deleted file: {target}"
            if target.is_dir():
                if recursive:
                    shutil.rmtree(target)
                    return f"Deleted folder recursively: {target}"
                target.rmdir()
                return f"Deleted empty folder: {target}"
            return f"Unsupported file type: {target}"
        except Exception as e:
            return f"Delete failed: {str(e)}"

    def create_workdir_folder(self, relative_path: str) -> str:
        """Create folder (and parents) under current working directory."""
        raw = str(relative_path or "").strip()
        if not raw:
            return "Folder path is empty."
        try:
            target = self._resolve_workdir_path(raw)
            root = Path(self.working_dir).resolve()
            if not self._is_relative_to(target, root):
                return f"Folder path is outside work_dir: {target}"
            target.mkdir(parents=True, exist_ok=True)
            return f"Folder ready: {target}"
        except Exception as e:
            return f"Create folder failed: {str(e)}"

    def _resolve_user_path(self, value: str, *, base_dir: Optional[Path] = None) -> Path:
        raw = str(value or "").strip()
        if not raw:
            raise ValueError("Path value is empty.")
        path = Path(raw).expanduser()
        root = (base_dir or self.working_dir or Path.cwd()).resolve()
        if not path.is_absolute():
            path = (root / path).resolve()
        else:
            path = path.resolve()
        return self._validate_allowed_user_path(path)

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _allowed_user_roots(self) -> list[Path]:
        scope_root = getattr(self, "user_workspace_root", None)
        if scope_root is not None:
            return [Path(scope_root).resolve()]
        candidates: list[Optional[Path]] = [
            self.working_dir,
            self.current_config_dir,
            self.current_workflow_config_dir,
            Path.cwd(),
        ]
        roots: list[Path] = []
        seen: set[str] = set()
        for item in candidates:
            if item is None:
                continue
            resolved = Path(item).resolve()
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            roots.append(resolved)
        return roots

    def _validate_allowed_user_path(self, path: Path) -> Path:
        roots = self._allowed_user_roots()
        if any(self._is_relative_to(path, root) for root in roots):
            return path
        root_preview = ", ".join(str(root) for root in roots[:4])
        raise ValueError(
            f"Path is outside allowed roots: {path}. Allowed roots include: {root_preview}"
        )

    def _default_base_dir(self, preferred: Optional[Path] = None) -> Path:
        return (preferred or self.working_dir or Path.cwd()).resolve()

    def _resolve_override_path(
        self,
        manual_path: Optional[str],
        uploaded_file: Optional[Any] = None,
    ) -> Optional[str]:
        """Resolve override path from manual textbox or uploaded file path."""
        manual = str(manual_path or "").strip()
        resolved_raw: Optional[str] = manual or None
        if resolved_raw is None:
            if uploaded_file is None:
                return None

            if isinstance(uploaded_file, str):
                path_val = uploaded_file.strip()
                resolved_raw = path_val or None
            elif isinstance(uploaded_file, dict):
                for key in ("path", "name"):
                    value = uploaded_file.get(key)
                    if isinstance(value, str) and value.strip():
                        resolved_raw = value.strip()
                        break
            else:
                # Gradio may return file-like objects in some versions.
                name_attr = getattr(uploaded_file, "name", None)
                if isinstance(name_attr, str) and name_attr.strip():
                    resolved_raw = name_attr.strip()

        if not resolved_raw:
            return None
        resolved = self._resolve_user_path(resolved_raw, base_dir=self.working_dir)
        return str(resolved)

    @staticmethod
    def _parse_json_dict(raw_json: str, field_name: str) -> Dict[str, Any]:
        text = str(raw_json or "").strip()
        if not text:
            return {}
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError(f"{field_name} must be a JSON object.")
        return obj

    @staticmethod
    def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge override into base (override takes precedence)."""
        for key, value in override.items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)
            ):
                AppControllerConfigMixin._deep_merge_dict(base[key], value)
            else:
                base[key] = value
        return base

    @staticmethod
    def _normalize_feature_values(raw: Any) -> list[str]:
        if raw is None:
            return []
        values: list[str] = []
        if isinstance(raw, str):
            values = [x.strip() for x in raw.split(",") if x.strip()]
        elif isinstance(raw, Sequence):
            values = [str(x).strip() for x in raw if str(x).strip()]
        else:
            values = [str(raw).strip()] if str(raw).strip() else []
        seen: set[str] = set()
        deduped: list[str] = []
        for item in values:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    def suggest_oneway_factors(self, cfg_path: str) -> tuple[str, list[str], list[str]]:
        """Load oneway factor candidates from config.feature_list."""
        raw = str(cfg_path or "").strip()
        if not raw:
            return "Plot config path is empty.", [], []

        try:
            path_obj = self._resolve_user_path(raw, base_dir=self.working_dir)
            if not path_obj.exists():
                raw_path = Path(raw).expanduser()
                if not raw_path.is_absolute():
                    examples_candidate = (self.working_dir / "examples" / raw_path).resolve()
                    if examples_candidate.exists():
                        path_obj = examples_candidate
                    else:
                        return f"Config not found: {path_obj}", [], []
                else:
                    return f"Config not found: {path_obj}", [], []

            payload = json.loads(path_obj.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return f"Invalid config format: {path_obj}", [], []

            features = self._normalize_feature_values(payload.get("feature_list"))
            if not features:
                return f"No feature_list found in: {path_obj.name}", [], []

            default_selected = features[:1]
            return (
                f"Loaded {len(features)} factors from {path_obj.name} (default selects first factor).",
                features,
                default_selected,
            )
        except Exception as exc:
            return f"Failed to load oneway factors: {exc}", [], []

    @staticmethod
    def _int_or_none(value, default=0) -> Optional[int]:
        """Parse a numeric UI value: returns None if <= default, else int."""
        try:
            v = int(value)
        except (TypeError, ValueError):
            return None
        return v if v > default else None

    @staticmethod
    def _str_or_none(value) -> Optional[str]:
        """Return stripped string or None if empty."""
        s = str(value or "").strip()
        return s if s else None

    @staticmethod
    def _parse_csv_values(raw: Any) -> list[str]:
        return [x.strip() for x in str(raw or "").split(",") if x.strip()]

    def _prepare_build_inputs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "model_list": self._parse_csv_values(params.get("model_list")),
            "model_categories": self._parse_csv_values(params.get("model_categories")),
            "feature_list": self._parse_csv_values(params.get("feature_list")),
            "categorical_features": self._parse_csv_values(params.get("categorical_features")),
            "model_keys": self._parse_csv_values(params.get("model_keys")),
            "xgb_chunk_size": self._int_or_none(params.get("xgb_chunk_size")),
            "resn_predict_batch_size": self._int_or_none(params.get("resn_predict_batch_size")),
            "ft_predict_batch_size": self._int_or_none(params.get("ft_predict_batch_size")),
            "xgb_search_space": self._parse_json_dict(
                params.get("xgb_search_space_json", ""), "xgb_search_space_json"
            ),
            "resn_search_space": self._parse_json_dict(
                params.get("resn_search_space_json", ""), "resn_search_space_json"
            ),
            "ft_search_space": self._parse_json_dict(
                params.get("ft_search_space_json", ""), "ft_search_space_json"
            ),
            "ft_unsupervised_search_space": self._parse_json_dict(
                params.get("ft_unsupervised_search_space_json", ""),
                "ft_unsupervised_search_space_json",
            ),
            "config_overrides": self._parse_json_dict(
                params.get("config_overrides_json", ""), "config_overrides_json"
            ),
        }

    def _build_extra_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        extra: Dict[str, Any] = {}

        if self._str_or_none(params.get("split_group_col")):
            extra["split_group_col"] = str(params["split_group_col"]).strip()
        if self._str_or_none(params.get("split_time_col")):
            extra["split_time_col"] = str(params["split_time_col"]).strip()
        extra["split_time_ascending"] = bool(params.get("split_time_ascending"))

        cv_strategy_val = self._str_or_none(params.get("cv_strategy"))
        if cv_strategy_val:
            extra["cv_strategy"] = cv_strategy_val
        cv_splits_val = self._int_or_none(params.get("cv_splits"))
        if cv_splits_val:
            extra["cv_splits"] = cv_splits_val
        if self._str_or_none(params.get("cv_group_col")):
            extra["cv_group_col"] = str(params["cv_group_col"]).strip()
        if self._str_or_none(params.get("cv_time_col")):
            extra["cv_time_col"] = str(params["cv_time_col"]).strip()
        extra["cv_time_ascending"] = bool(params.get("cv_time_ascending"))

        ft_num_val = self._int_or_none(params.get("ft_num_numeric_tokens"))
        if ft_num_val:
            extra["ft_num_numeric_tokens"] = ft_num_val
        ft_oof_val = self._int_or_none(params.get("ft_oof_folds"))
        if ft_oof_val:
            extra["ft_oof_folds"] = ft_oof_val
        ft_oof_strategy = self._str_or_none(params.get("ft_oof_strategy"))
        if ft_oof_strategy:
            extra["ft_oof_strategy"] = ft_oof_strategy
        extra["ft_oof_shuffle"] = bool(params.get("ft_oof_shuffle"))

        try:
            extra["resn_weight_decay"] = float(params.get("resn_weight_decay"))
        except (TypeError, ValueError):
            pass

        extra["gnn_use_approx_knn"] = bool(params.get("gnn_use_approx_knn"))
        extra["gnn_approx_knn_threshold"] = int(params.get("gnn_approx_knn_threshold"))
        extra["gnn_max_gpu_knn_nodes"] = int(params.get("gnn_max_gpu_knn_nodes"))
        extra["gnn_knn_gpu_mem_ratio"] = float(params.get("gnn_knn_gpu_mem_ratio"))
        extra["gnn_knn_gpu_mem_overhead"] = float(params.get("gnn_knn_gpu_mem_overhead"))
        gnn_cache = self._str_or_none(params.get("gnn_graph_cache"))
        if gnn_cache:
            extra["gnn_graph_cache"] = gnn_cache

        geo_nmes = self._parse_csv_values(params.get("geo_feature_nmes"))
        if geo_nmes:
            extra["geo_feature_nmes"] = geo_nmes
        if self._str_or_none(params.get("region_province_col")):
            extra["region_province_col"] = str(params["region_province_col"]).strip()
        if self._str_or_none(params.get("region_city_col")):
            extra["region_city_col"] = str(params["region_city_col"]).strip()
        extra["region_effect_alpha"] = float(params.get("region_effect_alpha"))
        extra["geo_token_hidden_dim"] = int(params.get("geo_token_hidden_dim"))
        extra["geo_token_layers"] = int(params.get("geo_token_layers"))
        extra["geo_token_dropout"] = float(params.get("geo_token_dropout"))
        extra["geo_token_k_neighbors"] = int(params.get("geo_token_k_neighbors"))
        extra["geo_token_learning_rate"] = float(params.get("geo_token_learning_rate"))
        extra["geo_token_epochs"] = int(params.get("geo_token_epochs"))

        extra["final_ensemble"] = bool(params.get("final_ensemble"))
        extra["final_ensemble_k"] = int(params.get("final_ensemble_k"))
        extra["final_refit"] = bool(params.get("final_refit"))
        extra["reuse_best_params"] = bool(params.get("reuse_best_params"))

        bo_limit = self._int_or_none(params.get("bo_sample_limit"))
        if bo_limit:
            extra["bo_sample_limit"] = bo_limit

        extra["plot"] = {
            "enable": bool(params.get("plot_enable")),
            "n_bins": int(params.get("plot_n_bins")),
            "oneway": bool(params.get("plot_oneway")),
            "oneway_pred": bool(params.get("plot_oneway_pred")),
            "pre_oneway": bool(params.get("plot_pre_oneway")),
            "double_lift": bool(params.get("plot_double_lift")),
        }

        cal_max = self._int_or_none(params.get("calibration_max_rows"))
        extra["calibration"] = {
            "enable": bool(params.get("calibration_enable")),
            "method": str(params.get("calibration_method") or "sigmoid"),
            "max_rows": cal_max,
            "seed": int(params.get("calibration_seed")),
        }

        thr_max = self._int_or_none(params.get("threshold_max_rows"))
        extra["threshold"] = {
            "enable": bool(params.get("threshold_enable")),
            "value": None,
            "metric": str(params.get("threshold_metric") or "f1"),
            "min_positive_rate": None,
            "grid": int(params.get("threshold_grid")),
            "max_rows": thr_max,
            "seed": int(params.get("threshold_seed")),
        }

        extra["bootstrap"] = {
            "enable": bool(params.get("bootstrap_enable")),
            "metrics": [],
            "n_samples": int(params.get("bootstrap_n_samples")),
            "ci": float(params.get("bootstrap_ci")),
            "seed": int(params.get("bootstrap_seed")),
        }
        return extra

    def build_config_from_ui(
        self,
        # ── Core Data & Task ──
        data_dir: str,
        model_list: str,
        model_categories: str,
        target: str,
        weight: str,
        feature_list: str,
        categorical_features: str,
        task_type: str,
        distribution: str,
        binary_resp_nme: str,
        prop_test: float,
        holdout_ratio: float,
        val_ratio: float,
        split_strategy: str,
        rand_seed: int,
        epochs: int,
        output_dir: str,
        use_gpu: bool,
        model_keys: str,
        max_evals: int,
        plot_curves: bool,
        # ── Split & Pre-split ──
        split_group_col: str,
        split_time_col: str,
        split_time_ascending: bool,
        train_data_path: str,
        test_data_path: str,
        split_cache_path: str,
        split_cache_force_rebuild: bool,
        # ── Cross-Validation ──
        cv_strategy: str,
        cv_splits: int,
        cv_group_col: str,
        cv_time_col: str,
        cv_time_ascending: bool,
        # ── XGBoost ──
        xgb_max_depth_max: int,
        xgb_n_estimators_max: int,
        xgb_gpu_id: int,
        xgb_use_dmatrix: bool,
        xgb_chunk_size: int,
        xgb_cleanup_per_fold: bool,
        xgb_cleanup_synchronize: bool,
        xgb_search_space_json: str,
        # ── ResNet ──
        resn_use_lazy_dataset: bool,
        resn_predict_batch_size: int,
        resn_weight_decay: float,
        resn_cleanup_per_fold: bool,
        resn_cleanup_synchronize: bool,
        resn_search_space_json: str,
        # ── FT-Transformer ──
        ft_role: str,
        ft_feature_prefix: str,
        ft_num_numeric_tokens: int,
        ft_use_lazy_dataset: bool,
        ft_predict_batch_size: int,
        ft_oof_folds: int,
        ft_oof_strategy: str,
        ft_oof_shuffle: bool,
        ft_cleanup_per_fold: bool,
        ft_cleanup_synchronize: bool,
        ft_search_space_json: str,
        ft_unsupervised_search_space_json: str,
        # ── GNN ──
        gnn_use_approx_knn: bool,
        gnn_approx_knn_threshold: int,
        gnn_max_gpu_knn_nodes: int,
        gnn_knn_gpu_mem_ratio: float,
        gnn_knn_gpu_mem_overhead: float,
        gnn_max_fit_rows: int,
        gnn_max_predict_rows: int,
        gnn_predict_chunk_rows: int,
        gnn_graph_cache: str,
        gnn_cleanup_per_fold: bool,
        gnn_cleanup_synchronize: bool,
        # ── Distributed ──
        nproc_per_node: int,
        ddp_min_rows: int,
        use_resn_ddp: bool,
        use_ft_ddp: bool,
        use_resn_data_parallel: bool,
        use_ft_data_parallel: bool,
        use_gnn_data_parallel: bool,
        # ── Preprocessing ──
        build_oht: bool,
        oht_sparse_csr: bool,
        keep_unscaled_oht: bool,
        infer_categorical_max_unique: int,
        infer_categorical_max_ratio: float,
        # ── Geographic ──
        geo_feature_nmes: str,
        region_province_col: str,
        region_city_col: str,
        region_effect_alpha: float,
        geo_token_hidden_dim: int,
        geo_token_layers: int,
        geo_token_dropout: float,
        geo_token_k_neighbors: int,
        geo_token_learning_rate: float,
        geo_token_epochs: int,
        # ── Ensemble & Refit ──
        final_ensemble: bool,
        final_ensemble_k: int,
        final_refit: bool,
        reuse_best_params: bool,
        # ── Output & Caching ──
        optuna_study_prefix: str,
        optuna_cleanup_synchronize: bool,
        cache_predictions: bool,
        prediction_cache_format: str,
        dataloader_workers: int,
        bo_sample_limit: int,
        # ── Plot Settings ──
        plot_enable: bool,
        plot_n_bins: int,
        plot_oneway: bool,
        plot_oneway_pred: bool,
        plot_pre_oneway: bool,
        plot_double_lift: bool,
        # ── Calibration ──
        calibration_enable: bool,
        calibration_method: str,
        calibration_max_rows: int,
        calibration_seed: int,
        # ── Threshold ──
        threshold_enable: bool,
        threshold_metric: str,
        threshold_grid: int,
        threshold_max_rows: int,
        threshold_seed: int,
        # ── Bootstrap ──
        bootstrap_enable: bool,
        bootstrap_n_samples: int,
        bootstrap_ci: float,
        bootstrap_seed: int,
        # ── Advanced ──
        config_overrides_json: str,
    ) -> tuple[str, str]:
        """Build configuration from UI parameters."""
        try:
            params = dict(locals())
            params.pop("self", None)
            parsed = self._prepare_build_inputs(params)

            config = self.config_builder.build_config(
                data_dir=data_dir,
                model_list=parsed["model_list"],
                model_categories=parsed["model_categories"],
                target=target,
                weight=weight,
                feature_list=parsed["feature_list"],
                categorical_features=parsed["categorical_features"],
                task_type=task_type,
                distribution=self._str_or_none(distribution),
                binary_resp_nme=self._str_or_none(binary_resp_nme),
                prop_test=prop_test,
                holdout_ratio=holdout_ratio,
                val_ratio=val_ratio,
                split_strategy=split_strategy,
                train_data_path=self._str_or_none(train_data_path),
                test_data_path=self._str_or_none(test_data_path),
                split_cache_path=self._str_or_none(split_cache_path),
                split_cache_force_rebuild=split_cache_force_rebuild,
                rand_seed=rand_seed,
                epochs=epochs,
                output_dir=output_dir,
                use_gpu=use_gpu,
                model_keys=parsed["model_keys"],
                max_evals=max_evals,
                build_oht=build_oht,
                oht_sparse_csr=oht_sparse_csr,
                keep_unscaled_oht=keep_unscaled_oht,
                plot_curves=plot_curves,
                infer_categorical_max_unique=int(infer_categorical_max_unique),
                infer_categorical_max_ratio=float(infer_categorical_max_ratio),
                optuna_study_prefix=str(optuna_study_prefix or "pricing"),
                xgb_max_depth_max=xgb_max_depth_max,
                xgb_n_estimators_max=xgb_n_estimators_max,
                xgb_gpu_id=xgb_gpu_id,
                xgb_cleanup_per_fold=xgb_cleanup_per_fold,
                xgb_cleanup_synchronize=xgb_cleanup_synchronize,
                xgb_use_dmatrix=xgb_use_dmatrix,
                xgb_chunk_size=parsed["xgb_chunk_size"],
                xgb_search_space=parsed["xgb_search_space"],
                cache_predictions=cache_predictions,
                prediction_cache_format=prediction_cache_format,
                dataloader_workers=int(dataloader_workers),
                use_resn_data_parallel=use_resn_data_parallel,
                use_ft_data_parallel=use_ft_data_parallel,
                use_gnn_data_parallel=use_gnn_data_parallel,
                use_resn_ddp=use_resn_ddp,
                use_ft_ddp=use_ft_ddp,
                ddp_min_rows=int(ddp_min_rows),
                ft_role=ft_role,
                ft_feature_prefix=ft_feature_prefix,
                ft_cleanup_per_fold=ft_cleanup_per_fold,
                ft_cleanup_synchronize=ft_cleanup_synchronize,
                ft_use_lazy_dataset=ft_use_lazy_dataset,
                ft_predict_batch_size=parsed["ft_predict_batch_size"],
                ft_search_space=parsed["ft_search_space"],
                ft_unsupervised_search_space=parsed["ft_unsupervised_search_space"],
                resn_cleanup_per_fold=resn_cleanup_per_fold,
                resn_cleanup_synchronize=resn_cleanup_synchronize,
                resn_use_lazy_dataset=resn_use_lazy_dataset,
                resn_predict_batch_size=parsed["resn_predict_batch_size"],
                resn_search_space=parsed["resn_search_space"],
                gnn_cleanup_per_fold=gnn_cleanup_per_fold,
                gnn_cleanup_synchronize=gnn_cleanup_synchronize,
                gnn_max_fit_rows=self._int_or_none(gnn_max_fit_rows),
                gnn_max_predict_rows=self._int_or_none(gnn_max_predict_rows),
                gnn_predict_chunk_rows=self._int_or_none(gnn_predict_chunk_rows),
                optuna_cleanup_synchronize=optuna_cleanup_synchronize,
                nproc_per_node=int(nproc_per_node),
            )

            extra = self._build_extra_config(params)

            # Deep-merge extra params into config
            config = self._deep_merge_dict(config, extra)

            # Deep-merge user-provided JSON overrides last (highest priority)
            config_overrides = parsed["config_overrides"]
            if config_overrides:
                config = self._deep_merge_dict(config, config_overrides)

            is_valid, msg = self.config_builder.validate_config(config)
            if not is_valid:
                return f"Validation failed: {msg}", ""

            self.current_config = config
            self.current_config_path = None
            self.current_config_dir = self.working_dir
            config_json = json.dumps(config, indent=2, ensure_ascii=False)
            return "Configuration built successfully", config_json

        except Exception as e:
            return f"Error building config: {str(e)}", ""

    def save_config(self, config_json: str, filename: str) -> str:
        """Save current configuration to file."""
        if not config_json:
            return "No configuration to save"

        try:
            config_path = self._resolve_user_path(filename, base_dir=self._default_base_dir(self.current_config_dir))
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(json.loads(config_json), f,
                          indent=2, ensure_ascii=False)
            return f"Configuration saved to {config_path}"
        except Exception as e:
            return f"Error saving config: {str(e)}"

    def load_workflow_config(self, file_path) -> tuple[str, str]:
        """Load workflow configuration (plot/predict/compare/pre-oneway)."""
        if not file_path:
            return "No file uploaded", ""

        try:
            path = self._resolve_user_path(str(file_path), base_dir=self.working_dir)
            with open(path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.current_workflow_config = config
            self.current_workflow_config_path = path
            self.current_workflow_config_dir = path.parent
            config_json = json.dumps(config, indent=2, ensure_ascii=False)
            return f"Workflow config loaded from {path.name}", config_json
        except Exception as e:
            return f"Error loading workflow config: {str(e)}", ""

    @staticmethod
    def _to_csv(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(x).strip() for x in value if str(x).strip())
        return str(value)

    def _resolve_path_value(
        self,
        base_dir: Path,
        value: Any,
        field_name: str,
        *,
        required: bool = True,
    ) -> Optional[str]:
        raw = str(value or "").strip()
        if not raw:
            if required:
                raise ValueError(f"{field_name} is required.")
            return None
        path = self._resolve_user_path(raw, base_dir=base_dir)
        return str(path)

    def _run_workflow_from_config(self, config: Dict[str, Any], base_dir: Path) -> str:
        workflow_cfg = config.get("workflow", config)
        if not isinstance(workflow_cfg, dict):
            raise ValueError("workflow must be a JSON object.")
        workflows_module = self._load_workflows_module()

        mode = str(workflow_cfg.get("mode", "")).strip().lower()
        if not mode:
            raise ValueError(
                "workflow.mode is required. Supported modes: "
                "pre_oneway, plot_direct, plot_embed, predict_ft_embed, compare_xgb, compare_resn, compare, double_lift."
            )

        if mode in {"pre_oneway", "pre-oneway", "oneway_pre"}:
            holdout_ratio_raw = workflow_cfg.get("holdout_ratio", 0.25)
            holdout_ratio = None if holdout_ratio_raw is None else float(holdout_ratio_raw)
            output_dir = self._resolve_path_value(
                base_dir, workflow_cfg.get("output_dir"), "output_dir", required=False
            )
            return workflows_module.run_pre_oneway(
                data_path=self._resolve_path_value(base_dir, workflow_cfg.get("data_path"), "data_path"),
                model_name=str(workflow_cfg.get("model_name", "")).strip(),
                target_col=str(workflow_cfg.get("target_col", "")).strip(),
                weight_col=str(workflow_cfg.get("weight_col", "")).strip(),
                feature_list=self._to_csv(workflow_cfg.get("feature_list", "")),
                categorical_features=self._to_csv(workflow_cfg.get("categorical_features", "")),
                n_bins=int(workflow_cfg.get("n_bins", 10)),
                holdout_ratio=holdout_ratio,
                rand_seed=int(workflow_cfg.get("rand_seed", 13)),
                output_dir=output_dir,
            )

        if mode == "plot_direct":
            cfg_path = workflow_cfg.get("cfg_path", workflow_cfg.get("plot_cfg_path"))
            return workflows_module.run_plot_direct(
                cfg_path=self._resolve_path_value(base_dir, cfg_path, "cfg_path"),
                xgb_cfg_path=self._resolve_path_value(base_dir, workflow_cfg.get("xgb_cfg_path"), "xgb_cfg_path"),
                resn_cfg_path=self._resolve_path_value(base_dir, workflow_cfg.get("resn_cfg_path"), "resn_cfg_path"),
                model_search_dir=str(self.working_dir),
            )

        if mode == "plot_embed":
            cfg_path = workflow_cfg.get("cfg_path", workflow_cfg.get("plot_cfg_path"))
            return workflows_module.run_plot_embed(
                cfg_path=self._resolve_path_value(base_dir, cfg_path, "cfg_path"),
                xgb_cfg_path=self._resolve_path_value(base_dir, workflow_cfg.get("xgb_cfg_path"), "xgb_cfg_path"),
                resn_cfg_path=self._resolve_path_value(base_dir, workflow_cfg.get("resn_cfg_path"), "resn_cfg_path"),
                ft_cfg_path=self._resolve_path_value(base_dir, workflow_cfg.get("ft_cfg_path"), "ft_cfg_path"),
                use_runtime_ft_embedding=bool(workflow_cfg.get("use_runtime_ft_embedding", False)),
                model_search_dir=str(self.working_dir),
            )

        if mode in {"predict_ft_embed", "predict"}:
            xgb_cfg_path = self._resolve_path_value(
                base_dir, workflow_cfg.get("xgb_cfg_path"), "xgb_cfg_path", required=False
            )
            resn_cfg_path = self._resolve_path_value(
                base_dir, workflow_cfg.get("resn_cfg_path"), "resn_cfg_path", required=False
            )
            return workflows_module.run_predict_ft_embed(
                ft_cfg_path=self._resolve_path_value(base_dir, workflow_cfg.get("ft_cfg_path"), "ft_cfg_path"),
                xgb_cfg_path=xgb_cfg_path,
                resn_cfg_path=resn_cfg_path,
                input_path=self._resolve_path_value(base_dir, workflow_cfg.get("input_path"), "input_path"),
                output_path=self._resolve_path_value(base_dir, workflow_cfg.get("output_path"), "output_path"),
                model_name=(str(workflow_cfg.get("model_name", "")).strip() or None),
                model_keys=self._to_csv(workflow_cfg.get("model_keys", "")),
                model_search_dir=str(self.working_dir),
            )

        if mode in {"compare_xgb", "compare_resn", "compare"}:
            if mode == "compare_xgb":
                model_key = "xgb"
            elif mode == "compare_resn":
                model_key = "resn"
            else:
                model_key = str(workflow_cfg.get("model_key", "xgb")).strip().lower()
            if model_key not in {"xgb", "resn"}:
                raise ValueError("compare mode only supports model_key in {'xgb', 'resn'}.")

            n_bins_override_raw = workflow_cfg.get("n_bins_override", 10)
            n_bins_override = None if n_bins_override_raw is None else int(n_bins_override_raw)
            return workflows_module.run_compare_ft_embed(
                direct_cfg_path=self._resolve_path_value(base_dir, workflow_cfg.get("direct_cfg_path"), "direct_cfg_path"),
                ft_cfg_path=self._resolve_path_value(base_dir, workflow_cfg.get("ft_cfg_path"), "ft_cfg_path"),
                ft_embed_cfg_path=self._resolve_path_value(base_dir, workflow_cfg.get("ft_embed_cfg_path"), "ft_embed_cfg_path"),
                model_key=model_key,
                label_direct=str(workflow_cfg.get("label_direct", "Direct")).strip(),
                label_ft=str(workflow_cfg.get("label_ft", "FT")).strip(),
                use_runtime_ft_embedding=bool(workflow_cfg.get("use_runtime_ft_embedding", False)),
                n_bins_override=n_bins_override,
                model_search_dir=str(self.working_dir),
            )

        if mode in {"double_lift", "double-lift"}:
            holdout_ratio_raw = workflow_cfg.get("holdout_ratio", 0.0)
            holdout_ratio = None if holdout_ratio_raw is None else float(holdout_ratio_raw)
            output_path = self._resolve_path_value(
                base_dir, workflow_cfg.get("output_path"), "output_path", required=False
            )
            train_data_path = self._resolve_path_value(
                base_dir, workflow_cfg.get("train_data_path"), "train_data_path", required=False
            )
            test_data_path = self._resolve_path_value(
                base_dir, workflow_cfg.get("test_data_path"), "test_data_path", required=False
            )
            split_cache_path = self._resolve_path_value(
                base_dir, workflow_cfg.get("split_cache_path"), "split_cache_path", required=False
            )
            return workflows_module.run_double_lift_from_file(
                data_path=self._resolve_path_value(base_dir, workflow_cfg.get("data_path"), "data_path"),
                train_data_path=train_data_path,
                test_data_path=test_data_path,
                pred_col_1=str(workflow_cfg.get("pred_col_1", workflow_cfg.get("pred_col1", ""))).strip(),
                pred_col_2=str(workflow_cfg.get("pred_col_2", workflow_cfg.get("pred_col2", ""))).strip(),
                target_col=str(workflow_cfg.get("target_col", workflow_cfg.get("target", ""))).strip(),
                weight_col=str(workflow_cfg.get("weight_col", workflow_cfg.get("weight", "weights"))).strip(),
                n_bins=int(workflow_cfg.get("n_bins", 10)),
                label1=str(workflow_cfg.get("label1", "")).strip() or None,
                label2=str(workflow_cfg.get("label2", "")).strip() or None,
                pred1_weighted=bool(workflow_cfg.get("pred1_weighted", False)),
                pred2_weighted=bool(workflow_cfg.get("pred2_weighted", False)),
                actual_weighted=bool(workflow_cfg.get("actual_weighted", False)),
                holdout_ratio=holdout_ratio,
                split_strategy=str(workflow_cfg.get("split_strategy", "random")).strip(),
                split_group_col=str(workflow_cfg.get("split_group_col", "")).strip() or None,
                split_time_col=str(workflow_cfg.get("split_time_col", "")).strip() or None,
                split_time_ascending=bool(workflow_cfg.get("split_time_ascending", True)),
                rand_seed=int(workflow_cfg.get("rand_seed", 13)),
                split_cache_path=split_cache_path,
                split_cache_force_rebuild=bool(workflow_cfg.get("split_cache_force_rebuild", False)),
                output_path=output_path,
            )

        raise ValueError(
            f"Unsupported workflow mode: {mode}. "
            "Supported: pre_oneway, plot_direct, plot_embed, predict_ft_embed, compare_xgb, compare_resn, compare, double_lift."
        )

