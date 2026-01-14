from __future__ import annotations

from datetime import timedelta
import gc
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
try:  # pragma: no cover
    import torch.distributed as dist  # type: ignore
except Exception:  # pragma: no cover
    dist = None  # type: ignore
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    ShuffleSplit,
    TimeSeriesSplit,
)
from sklearn.preprocessing import StandardScaler

from ..config_preprocess import BayesOptConfig, OutputManager
from ..utils import DistributedUtils, EPS, ensure_parent_dir
from ins_pricing.utils import get_logger, GPUMemoryManager, DeviceManager

# Module-level logger
_logger = get_logger("ins_pricing.trainer")

class _OrderSplitter:
    def __init__(self, splitter, order: np.ndarray) -> None:
        self._splitter = splitter
        self._order = np.asarray(order)

    def split(self, X, y=None, groups=None):
        order = self._order
        X_ord = X.iloc[order] if hasattr(X, "iloc") else X[order]
        for tr_idx, val_idx in self._splitter.split(X_ord, y=y, groups=groups):
            yield order[tr_idx], order[val_idx]

# =============================================================================
# Trainer system
# =============================================================================


class TrainerBase:
    def __init__(self, context: "BayesOptModel", label: str, model_name_prefix: str) -> None:
        self.ctx = context
        self.label = label
        self.model_name_prefix = model_name_prefix
        self.model = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_trial = None
        self.study_name: Optional[str] = None
        self.enable_distributed_optuna: bool = False
        self._distributed_forced_params: Optional[Dict[str, Any]] = None

    def _export_preprocess_artifacts(self) -> Dict[str, Any]:
        dummy_columns: List[str] = []
        if getattr(self.ctx, "train_oht_data", None) is not None:
            dummy_columns = list(self.ctx.train_oht_data.columns)
        return {
            "factor_nmes": list(getattr(self.ctx, "factor_nmes", []) or []),
            "cate_list": list(getattr(self.ctx, "cate_list", []) or []),
            "num_features": list(getattr(self.ctx, "num_features", []) or []),
            "var_nmes": list(getattr(self.ctx, "var_nmes", []) or []),
            "cat_categories": dict(getattr(self.ctx, "cat_categories_for_shap", {}) or {}),
            "dummy_columns": dummy_columns,
            "numeric_scalers": dict(getattr(self.ctx, "numeric_scalers", {}) or {}),
            "weight_nme": str(getattr(self.ctx, "weight_nme", "")),
            "resp_nme": str(getattr(self.ctx, "resp_nme", "")),
            "binary_resp_nme": getattr(self.ctx, "binary_resp_nme", None),
            "drop_first": True,
        }

    def _dist_barrier(self, reason: str) -> None:
        """DDP barrier wrapper used by distributed Optuna.

        To debug "trial finished but next trial never starts" hangs, set these
        environment variables (either in shell or config.json `env`):
        - `BAYESOPT_DDP_BARRIER_DEBUG=1` to print barrier enter/exit per-rank
        - `BAYESOPT_DDP_BARRIER_TIMEOUT=300` to fail fast instead of waiting forever
        - `TORCH_DISTRIBUTED_DEBUG=DETAIL` and `NCCL_DEBUG=INFO` for PyTorch/NCCL logs
        """
        if dist is None:
            return
        try:
            if not getattr(dist, "is_available", lambda: False)():
                return
            if not dist.is_initialized():
                return
        except Exception:
            return

        timeout_seconds = int(os.environ.get("BAYESOPT_DDP_BARRIER_TIMEOUT", "1800"))
        debug_barrier = os.environ.get("BAYESOPT_DDP_BARRIER_DEBUG", "").strip() in {"1", "true", "TRUE", "yes", "YES"}
        rank = None
        world = None
        if debug_barrier:
            try:
                rank = dist.get_rank()
                world = dist.get_world_size()
                print(f"[DDP][{self.label}] entering barrier({reason}) rank={rank}/{world}", flush=True)
            except Exception:
                debug_barrier = False
        try:
            timeout = timedelta(seconds=timeout_seconds)
            backend = None
            try:
                backend = dist.get_backend()
            except Exception:
                backend = None

            # `monitored_barrier` is only implemented for GLOO; using it under NCCL
            # will raise and can itself trigger a secondary hang. Prefer an async
            # barrier with timeout for NCCL.
            monitored = getattr(dist, "monitored_barrier", None)
            if backend == "gloo" and callable(monitored):
                monitored(timeout=timeout)
            else:
                work = None
                try:
                    work = dist.barrier(async_op=True)
                except TypeError:
                    work = None
                if work is not None:
                    wait = getattr(work, "wait", None)
                    if callable(wait):
                        try:
                            wait(timeout=timeout)
                        except TypeError:
                            wait()
                    else:
                        dist.barrier()
                else:
                    dist.barrier()
            if debug_barrier:
                print(f"[DDP][{self.label}] exit barrier({reason}) rank={rank}/{world}", flush=True)
        except Exception as exc:
            print(
                f"[DDP][{self.label}] barrier failed during {reason}: {exc}",
                flush=True,
            )
            raise

    @property
    def config(self) -> BayesOptConfig:
        return self.ctx.config

    @property
    def output(self) -> OutputManager:
        return self.ctx.output_manager

    def _get_model_filename(self) -> str:
        ext = 'pkl' if self.label in ['Xgboost', 'GLM'] else 'pth'
        return f'01_{self.ctx.model_nme}_{self.model_name_prefix}.{ext}'

    def _resolve_optuna_storage_url(self) -> Optional[str]:
        storage = getattr(self.config, "optuna_storage", None)
        if not storage:
            return None
        storage_str = str(storage).strip()
        if not storage_str:
            return None
        if "://" in storage_str or storage_str == ":memory:":
            return storage_str
        path = Path(storage_str)
        path = path.resolve()
        ensure_parent_dir(str(path))
        return f"sqlite:///{path.as_posix()}"

    def _resolve_optuna_study_name(self) -> str:
        prefix = getattr(self.config, "optuna_study_prefix",
                         None) or "bayesopt"
        raw = f"{prefix}_{self.ctx.model_nme}_{self.model_name_prefix}"
        safe = "".join([c if c.isalnum() or c in "._-" else "_" for c in raw])
        return safe.lower()

    def tune(self, max_evals: int, objective_fn=None) -> None:
        # Generic Optuna tuning loop.
        if objective_fn is None:
            # If subclass doesn't provide objective_fn, default to cross_val.
            objective_fn = self.cross_val

        if self._should_use_distributed_optuna():
            self._distributed_tune(max_evals, objective_fn)
            return

        total_trials = max(1, int(max_evals))
        progress_counter = {"count": 0}

        def objective_wrapper(trial: optuna.trial.Trial) -> float:
            should_log = DistributedUtils.is_main_process()
            if should_log:
                current_idx = progress_counter["count"] + 1
                print(
                    f"[Optuna][{self.label}] Trial {current_idx}/{total_trials} started "
                    f"(trial_id={trial.number})."
                )
            try:
                result = objective_fn(trial)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(
                        f"[Optuna][{self.label}] OOM detected. Pruning trial and clearing CUDA cache."
                    )
                    self._clean_gpu()
                    raise optuna.TrialPruned() from exc
                raise
            finally:
                self._clean_gpu()
                if should_log:
                    progress_counter["count"] = progress_counter["count"] + 1
                    trial_state = getattr(trial, "state", None)
                    state_repr = getattr(trial_state, "name", "OK")
                    print(
                        f"[Optuna][{self.label}] Trial {progress_counter['count']}/{total_trials} finished "
                        f"(status={state_repr})."
                    )
            return result

        storage_url = self._resolve_optuna_storage_url()
        study_name = self._resolve_optuna_study_name()
        study_kwargs: Dict[str, Any] = {
            "direction": "minimize",
            "sampler": optuna.samplers.TPESampler(seed=self.ctx.rand_seed),
        }
        if storage_url:
            study_kwargs.update(
                storage=storage_url,
                study_name=study_name,
                load_if_exists=True,
            )

        study = optuna.create_study(**study_kwargs)
        self.study_name = getattr(study, "study_name", None)

        def checkpoint_callback(check_study: optuna.study.Study, _trial) -> None:
            # Persist best_params after each trial to allow safe resume.
            try:
                best = getattr(check_study, "best_trial", None)
                if best is None:
                    return
                best_params = getattr(best, "params", None)
                if not best_params:
                    return
                params_path = self.output.result_path(
                    f'{self.ctx.model_nme}_bestparams_{self.label.lower()}.csv'
                )
                pd.DataFrame(best_params, index=[0]).to_csv(
                    params_path, index=False)
            except Exception:
                return

        completed_states = (
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL,
        )
        completed = len(study.get_trials(states=completed_states))
        progress_counter["count"] = completed
        remaining = max(0, total_trials - completed)
        if remaining > 0:
            study.optimize(
                objective_wrapper,
                n_trials=remaining,
                callbacks=[checkpoint_callback],
            )
        self.best_params = study.best_params
        self.best_trial = study.best_trial

        # Save best params to CSV for reproducibility.
        params_path = self.output.result_path(
            f'{self.ctx.model_nme}_bestparams_{self.label.lower()}.csv'
        )
        pd.DataFrame(self.best_params, index=[0]).to_csv(
            params_path, index=False)

    def train(self) -> None:
        raise NotImplementedError

    def _unwrap_module(self, module: torch.nn.Module) -> torch.nn.Module:
        """Unwrap DDP or DataParallel wrapper to get the base module."""
        from torch.nn.parallel import DistributedDataParallel as DDP
        if isinstance(module, (DDP, torch.nn.DataParallel)):
            return module.module
        return module

    def save(self) -> None:
        if self.model is None:
            print(f"[save] Warning: No model to save for {self.label}")
            return

        path = self.output.model_path(self._get_model_filename())
        if self.label in ['Xgboost', 'GLM']:
            payload = {
                "model": self.model,
                "preprocess_artifacts": self._export_preprocess_artifacts(),
            }
            joblib.dump(payload, path)
        else:
            # PyTorch models: save state_dict without DDP/DataParallel wrappers
            # to ensure cross-platform compatibility.
            payload = {
                "preprocess_artifacts": self._export_preprocess_artifacts(),
            }
            if hasattr(self.model, 'resnet'):  # ResNetSklearn model
                # Unwrap DDP/DataParallel and move to CPU
                resnet = self._unwrap_module(self.model.resnet)
                resnet_cpu = resnet.to("cpu")
                payload["state_dict"] = resnet_cpu.state_dict()
                payload["best_params"] = dict(self.best_params or {})
            elif hasattr(self.model, 'ft'):  # FTTransformerSklearn model
                # Unwrap DDP/DataParallel and save state_dict (not full model object)
                # to avoid serialization issues with DDP wrappers
                ft = self._unwrap_module(self.model.ft)
                ft_cpu = ft.to("cpu")
                payload["state_dict"] = ft_cpu.state_dict()
                payload["best_params"] = dict(self.best_params or {})
                # Save model configuration for reconstruction
                payload["model_config"] = {
                    "model_nme": getattr(self.model, "model_nme", ""),
                    "num_cols": list(getattr(self.model, "num_cols", [])),
                    "cat_cols": list(getattr(self.model, "cat_cols", [])),
                    "d_model": getattr(self.model, "d_model", 64),
                    "n_heads": getattr(self.model, "n_heads", 8),
                    "n_layers": getattr(self.model, "n_layers", 4),
                    "dropout": getattr(self.model, "dropout", 0.1),
                    "task_type": getattr(self.model, "task_type", "regression"),
                    "tw_power": getattr(self.model, "tw_power", 1.5),
                    "num_geo": getattr(self.model, "num_geo", 0),
                    "num_numeric_tokens": getattr(self.model, "num_numeric_tokens", None),
                    "cat_cardinalities": getattr(self.model, "cat_cardinalities", None),
                    "cat_categories": {k: list(v) for k, v in getattr(self.model, "cat_categories", {}).items()},
                    "_num_mean": getattr(self.model, "_num_mean", None),
                    "_num_std": getattr(self.model, "_num_std", None),
                }
                # Convert numpy arrays to lists for JSON serialization
                if payload["model_config"]["_num_mean"] is not None:
                    payload["model_config"]["_num_mean"] = payload["model_config"]["_num_mean"].tolist() if hasattr(payload["model_config"]["_num_mean"], "tolist") else payload["model_config"]["_num_mean"]
                if payload["model_config"]["_num_std"] is not None:
                    payload["model_config"]["_num_std"] = payload["model_config"]["_num_std"].tolist() if hasattr(payload["model_config"]["_num_std"], "tolist") else payload["model_config"]["_num_std"]
            else:
                # Generic PyTorch model fallback
                if hasattr(self.model, 'to'):
                    self.model.to("cpu")
                payload["model"] = self.model
            torch.save(payload, path)

    def load(self) -> None:
        path = self.output.model_path(self._get_model_filename())
        if not os.path.exists(path):
            print(f"[load] Warning: Model file not found: {path}")
            return

        if self.label in ['Xgboost', 'GLM']:
            loaded = joblib.load(path)
            if isinstance(loaded, dict) and "model" in loaded:
                self.model = loaded.get("model")
            else:
                self.model = loaded
        else:
            # PyTorch loading depends on the model structure.
            if self.label == 'ResNet' or self.label == 'ResNetClassifier':
                # ResNet requires reconstructing the skeleton; handled by subclass.
                pass
            else:
                # FT-Transformer: load state_dict and reconstruct model
                loaded = torch.load(path, map_location='cpu', weights_only=False)
                if isinstance(loaded, dict):
                    if "state_dict" in loaded and "model_config" in loaded:
                        # New format: state_dict + model_config
                        state_dict = loaded.get("state_dict")
                        model_config = loaded.get("model_config", {})
                        self.best_params = loaded.get("best_params", {})

                        # Import FTTransformerSklearn for reconstruction
                        from ..models import FTTransformerSklearn

                        # Reconstruct model from config
                        model = FTTransformerSklearn(
                            model_nme=model_config.get("model_nme", ""),
                            num_cols=model_config.get("num_cols", []),
                            cat_cols=model_config.get("cat_cols", []),
                            d_model=model_config.get("d_model", 64),
                            n_heads=model_config.get("n_heads", 8),
                            n_layers=model_config.get("n_layers", 4),
                            dropout=model_config.get("dropout", 0.1),
                            task_type=model_config.get("task_type", "regression"),
                            tweedie_power=model_config.get("tw_power", 1.5),
                            num_numeric_tokens=model_config.get("num_numeric_tokens"),
                            use_data_parallel=False,
                            use_ddp=False,
                        )
                        # Restore internal state
                        model.num_geo = model_config.get("num_geo", 0)
                        model.cat_cardinalities = model_config.get("cat_cardinalities")
                        model.cat_categories = {k: pd.Index(v) for k, v in model_config.get("cat_categories", {}).items()}
                        if model_config.get("_num_mean") is not None:
                            model._num_mean = np.array(model_config["_num_mean"], dtype=np.float32)
                        if model_config.get("_num_std") is not None:
                            model._num_std = np.array(model_config["_num_std"], dtype=np.float32)

                        # Build the model architecture and load weights
                        # We need a dummy dataframe to initialize the model
                        if model.cat_cardinalities is not None:
                            from ..models.model_ft_components import FTTransformerCore
                            core = FTTransformerCore(
                                num_numeric=len(model.num_cols),
                                cat_cardinalities=model.cat_cardinalities,
                                d_model=model.d_model,
                                n_heads=model.n_heads,
                                n_layers=model.n_layers,
                                dropout=model.dropout,
                                task_type=model.task_type,
                                num_geo=model.num_geo,
                                num_numeric_tokens=model.num_numeric_tokens,
                            )
                            model.ft = core
                            model.ft.load_state_dict(state_dict)

                        self._move_to_device(model)
                        self.model = model
                    elif "model" in loaded:
                        # Legacy format: full model object
                        loaded_model = loaded.get("model")
                        if loaded_model is not None:
                            self._move_to_device(loaded_model)
                        self.model = loaded_model
                    else:
                        # Unknown format
                        print(f"[load] Warning: Unknown model format in {path}")
                else:
                    # Very old format: direct model object
                    if loaded is not None:
                        self._move_to_device(loaded)
                    self.model = loaded

    def _move_to_device(self, model_obj):
        """Move model to the best available device using shared DeviceManager."""
        DeviceManager.move_to_device(model_obj)

    def _should_use_distributed_optuna(self) -> bool:
        if not self.enable_distributed_optuna:
            return False
        rank_env = os.environ.get("RANK")
        world_env = os.environ.get("WORLD_SIZE")
        local_env = os.environ.get("LOCAL_RANK")
        if rank_env is None or world_env is None or local_env is None:
            return False
        try:
            world_size = int(world_env)
        except Exception:
            return False
        return world_size > 1

    def _distributed_is_main(self) -> bool:
        return DistributedUtils.is_main_process()

    def _distributed_send_command(self, payload: Dict[str, Any]) -> None:
        if not self._should_use_distributed_optuna() or not self._distributed_is_main():
            return
        if dist is None:
            return
        DistributedUtils.setup_ddp()
        if not dist.is_initialized():
            return
        message = [payload]
        dist.broadcast_object_list(message, src=0)

    def _distributed_prepare_trial(self, params: Dict[str, Any]) -> None:
        if not self._should_use_distributed_optuna():
            return
        if not self._distributed_is_main():
            return
        if dist is None:
            return
        self._distributed_send_command({"type": "RUN", "params": params})
        if not dist.is_initialized():
            return
        # STEP 2 (DDP/Optuna): make sure all ranks start the trial together.
        self._dist_barrier("prepare_trial")

    def _distributed_worker_loop(self, objective_fn: Callable[[Optional[optuna.trial.Trial]], float]) -> None:
        if dist is None:
            print(
                f"[Optuna][Worker][{self.label}] torch.distributed unavailable. Worker exit.",
                flush=True,
            )
            return
        DistributedUtils.setup_ddp()
        if not dist.is_initialized():
            print(
                f"[Optuna][Worker][{self.label}] DDP init failed. Worker exit.",
                flush=True,
            )
            return
        while True:
            message = [None]
            dist.broadcast_object_list(message, src=0)
            payload = message[0]
            if not isinstance(payload, dict):
                continue
            cmd = payload.get("type")
            if cmd == "STOP":
                best_params = payload.get("best_params")
                if best_params is not None:
                    self.best_params = best_params
                break
            if cmd == "RUN":
                params = payload.get("params") or {}
                self._distributed_forced_params = params
                # STEP 2 (DDP/Optuna): align worker with rank0 before running objective_fn.
                self._dist_barrier("worker_start")
                try:
                    objective_fn(None)
                except optuna.TrialPruned:
                    pass
                except Exception as exc:
                    print(
                        f"[Optuna][Worker][{self.label}] Exception: {exc}", flush=True)
                finally:
                    self._clean_gpu()
                    # STEP 2 (DDP/Optuna): align worker with rank0 after objective_fn returns/raises.
                    self._dist_barrier("worker_end")

    def _distributed_tune(self, max_evals: int, objective_fn: Callable[[optuna.trial.Trial], float]) -> None:
        if dist is None:
            print(
                f"[Optuna][{self.label}] torch.distributed unavailable. Fallback to single-process.",
                flush=True,
            )
            prev = self.enable_distributed_optuna
            self.enable_distributed_optuna = False
            try:
                self.tune(max_evals, objective_fn)
            finally:
                self.enable_distributed_optuna = prev
            return
        DistributedUtils.setup_ddp()
        if not dist.is_initialized():
            rank_env = os.environ.get("RANK", "0")
            if str(rank_env) != "0":
                print(
                    f"[Optuna][{self.label}] DDP init failed on worker. Skip.",
                    flush=True,
                )
                return
            print(
                f"[Optuna][{self.label}] DDP init failed. Fallback to single-process.",
                flush=True,
            )
            prev = self.enable_distributed_optuna
            self.enable_distributed_optuna = False
            try:
                self.tune(max_evals, objective_fn)
            finally:
                self.enable_distributed_optuna = prev
            return
        if not self._distributed_is_main():
            self._distributed_worker_loop(objective_fn)
            return

        total_trials = max(1, int(max_evals))
        progress_counter = {"count": 0}

        def objective_wrapper(trial: optuna.trial.Trial) -> float:
            should_log = True
            if should_log:
                current_idx = progress_counter["count"] + 1
                print(
                    f"[Optuna][{self.label}] Trial {current_idx}/{total_trials} started "
                    f"(trial_id={trial.number})."
                )
            try:
                result = objective_fn(trial)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(
                        f"[Optuna][{self.label}] OOM detected. Pruning trial and clearing CUDA cache."
                    )
                    self._clean_gpu()
                    raise optuna.TrialPruned() from exc
                raise
            finally:
                self._clean_gpu()
                if should_log:
                    progress_counter["count"] = progress_counter["count"] + 1
                    trial_state = getattr(trial, "state", None)
                    state_repr = getattr(trial_state, "name", "OK")
                    print(
                        f"[Optuna][{self.label}] Trial {progress_counter['count']}/{total_trials} finished "
                        f"(status={state_repr})."
                    )
                # STEP 2 (DDP/Optuna): a trial-end sync point; debug with BAYESOPT_DDP_BARRIER_DEBUG=1.
                self._dist_barrier("trial_end")
            return result

        storage_url = self._resolve_optuna_storage_url()
        study_name = self._resolve_optuna_study_name()
        study_kwargs: Dict[str, Any] = {
            "direction": "minimize",
            "sampler": optuna.samplers.TPESampler(seed=self.ctx.rand_seed),
        }
        if storage_url:
            study_kwargs.update(
                storage=storage_url,
                study_name=study_name,
                load_if_exists=True,
            )
        study = optuna.create_study(**study_kwargs)
        self.study_name = getattr(study, "study_name", None)

        def checkpoint_callback(check_study: optuna.study.Study, _trial) -> None:
            try:
                best = getattr(check_study, "best_trial", None)
                if best is None:
                    return
                best_params = getattr(best, "params", None)
                if not best_params:
                    return
                params_path = self.output.result_path(
                    f'{self.ctx.model_nme}_bestparams_{self.label.lower()}.csv'
                )
                pd.DataFrame(best_params, index=[0]).to_csv(
                    params_path, index=False)
            except Exception:
                return

        completed_states = (
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL,
        )
        completed = len(study.get_trials(states=completed_states))
        progress_counter["count"] = completed
        remaining = max(0, total_trials - completed)
        try:
            if remaining > 0:
                study.optimize(
                    objective_wrapper,
                    n_trials=remaining,
                    callbacks=[checkpoint_callback],
                )
            self.best_params = study.best_params
            self.best_trial = study.best_trial
            params_path = self.output.result_path(
                f'{self.ctx.model_nme}_bestparams_{self.label.lower()}.csv'
            )
            pd.DataFrame(self.best_params, index=[0]).to_csv(
                params_path, index=False)
        finally:
            self._distributed_send_command(
                {"type": "STOP", "best_params": self.best_params})

    def _clean_gpu(self):
        """Clean up GPU memory using shared GPUMemoryManager."""
        GPUMemoryManager.clean()

    def _standardize_fold(self,
                          X_train: pd.DataFrame,
                          X_val: pd.DataFrame,
                          columns: Optional[List[str]] = None
                          ) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
        """Fit StandardScaler on the training fold and transform train/val features.

        Args:
            X_train: training features.
            X_val: validation features.
            columns: columns to scale (default: all).

        Returns:
            Scaled train/val features and the fitted scaler.
        """
        scaler = StandardScaler()
        cols = list(columns) if columns else list(X_train.columns)
        X_train_scaled = X_train.copy(deep=True)
        X_val_scaled = X_val.copy(deep=True)
        if cols:
            scaler.fit(X_train_scaled[cols])
            X_train_scaled[cols] = scaler.transform(X_train_scaled[cols])
            X_val_scaled[cols] = scaler.transform(X_val_scaled[cols])
        return X_train_scaled, X_val_scaled, scaler

    def _resolve_train_val_indices(
        self,
        X_all: pd.DataFrame,
        *,
        allow_default: bool = False,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        val_ratio = float(self.ctx.prop_test) if self.ctx.prop_test is not None else 0.25
        if not (0.0 < val_ratio < 1.0):
            if not allow_default:
                return None
            val_ratio = 0.25
        if len(X_all) < 10:
            return None

        strategy = str(getattr(self.ctx.config, "cv_strategy", "random") or "random").strip().lower()
        if strategy in {"time", "timeseries", "temporal"}:
            time_col = getattr(self.ctx.config, "cv_time_col", None)
            if not time_col:
                raise ValueError("cv_time_col is required for time cv_strategy.")
            if time_col not in self.ctx.train_data.columns:
                raise KeyError(f"cv_time_col '{time_col}' not in train_data.")
            ascending = bool(getattr(self.ctx.config, "cv_time_ascending", True))
            order_index = self.ctx.train_data[time_col].sort_values(ascending=ascending).index
            index_set = set(X_all.index)
            order_index = [idx for idx in order_index if idx in index_set]
            order = X_all.index.get_indexer(order_index)
            order = order[order >= 0]
            cutoff = int(len(order) * (1.0 - val_ratio))
            if cutoff <= 0 or cutoff >= len(order):
                raise ValueError(
                    f"prop_test={val_ratio} leaves no data for train/val split.")
            return order[:cutoff], order[cutoff:]

        if strategy in {"group", "grouped"}:
            group_col = getattr(self.ctx.config, "cv_group_col", None)
            if not group_col:
                raise ValueError("cv_group_col is required for group cv_strategy.")
            if group_col not in self.ctx.train_data.columns:
                raise KeyError(f"cv_group_col '{group_col}' not in train_data.")
            groups = self.ctx.train_data.reindex(X_all.index)[group_col]
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=val_ratio,
                random_state=self.ctx.rand_seed,
            )
            train_idx, val_idx = next(splitter.split(X_all, groups=groups))
            return train_idx, val_idx

        splitter = ShuffleSplit(
            n_splits=1,
            test_size=val_ratio,
            random_state=self.ctx.rand_seed,
        )
        train_idx, val_idx = next(splitter.split(X_all))
        return train_idx, val_idx

    def _resolve_time_sample_indices(
        self,
        X_all: pd.DataFrame,
        sample_limit: int,
    ) -> Optional[pd.Index]:
        if sample_limit <= 0:
            return None
        strategy = str(getattr(self.ctx.config, "cv_strategy", "random") or "random").strip().lower()
        if strategy not in {"time", "timeseries", "temporal"}:
            return None
        time_col = getattr(self.ctx.config, "cv_time_col", None)
        if not time_col:
            raise ValueError("cv_time_col is required for time cv_strategy.")
        if time_col not in self.ctx.train_data.columns:
            raise KeyError(f"cv_time_col '{time_col}' not in train_data.")
        ascending = bool(getattr(self.ctx.config, "cv_time_ascending", True))
        order_index = self.ctx.train_data[time_col].sort_values(ascending=ascending).index
        index_set = set(X_all.index)
        order_index = [idx for idx in order_index if idx in index_set]
        if not order_index:
            return None
        if len(order_index) > sample_limit:
            order_index = order_index[-sample_limit:]
        return pd.Index(order_index)

    def _resolve_ensemble_splits(
        self,
        X_all: pd.DataFrame,
        *,
        k: int,
    ) -> Tuple[Optional[Iterable[Tuple[np.ndarray, np.ndarray]]], int]:
        k = max(2, int(k))
        n_samples = len(X_all)
        if n_samples < 2:
            return None, 0

        strategy = str(getattr(self.ctx.config, "cv_strategy", "random") or "random").strip().lower()
        if strategy in {"group", "grouped"}:
            group_col = getattr(self.ctx.config, "cv_group_col", None)
            if not group_col:
                raise ValueError("cv_group_col is required for group cv_strategy.")
            if group_col not in self.ctx.train_data.columns:
                raise KeyError(f"cv_group_col '{group_col}' not in train_data.")
            groups = self.ctx.train_data.reindex(X_all.index)[group_col]
            n_groups = int(groups.nunique(dropna=False))
            if n_groups < 2:
                return None, 0
            if k > n_groups:
                k = n_groups
            if k < 2:
                return None, 0
            splitter = GroupKFold(n_splits=k)
            return splitter.split(X_all, y=None, groups=groups), k

        if strategy in {"time", "timeseries", "temporal"}:
            time_col = getattr(self.ctx.config, "cv_time_col", None)
            if not time_col:
                raise ValueError("cv_time_col is required for time cv_strategy.")
            if time_col not in self.ctx.train_data.columns:
                raise KeyError(f"cv_time_col '{time_col}' not in train_data.")
            ascending = bool(getattr(self.ctx.config, "cv_time_ascending", True))
            order_index = self.ctx.train_data[time_col].sort_values(ascending=ascending).index
            index_set = set(X_all.index)
            order_index = [idx for idx in order_index if idx in index_set]
            order = X_all.index.get_indexer(order_index)
            order = order[order >= 0]
            if len(order) < 2:
                return None, 0
            if len(order) <= k:
                k = max(2, len(order) - 1)
            if k < 2:
                return None, 0
            splitter = TimeSeriesSplit(n_splits=k)
            return _OrderSplitter(splitter, order).split(X_all), k

        if n_samples < k:
            k = n_samples
        if k < 2:
            return None, 0
        splitter = KFold(
            n_splits=k,
            shuffle=True,
            random_state=self.ctx.rand_seed,
        )
        return splitter.split(X_all), k

    def cross_val_generic(
            self,
            trial: optuna.trial.Trial,
            hyperparameter_space: Dict[str, Callable[[optuna.trial.Trial], Any]],
            data_provider: Callable[[], Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]],
            model_builder: Callable[[Dict[str, Any]], Any],
            metric_fn: Callable[[pd.Series, np.ndarray, Optional[pd.Series]], float],
            sample_limit: Optional[int] = None,
            preprocess_fn: Optional[Callable[[
                pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]] = None,
            fit_predict_fn: Optional[
                Callable[[Any, pd.DataFrame, pd.Series, Optional[pd.Series],
                          pd.DataFrame, pd.Series, Optional[pd.Series],
                          optuna.trial.Trial], np.ndarray]
            ] = None,
            cleanup_fn: Optional[Callable[[Any], None]] = None,
            splitter: Optional[Iterable[Tuple[np.ndarray, np.ndarray]]] = None) -> float:
        """Generic holdout/CV helper to reuse tuning workflows.

        Args:
            trial: current Optuna trial.
            hyperparameter_space: sampler dict keyed by parameter name.
            data_provider: callback returning (X, y, sample_weight).
            model_builder: callback to build a model per fold.
            metric_fn: loss/score function taking y_true, y_pred, weight.
            sample_limit: optional sample cap; random sample if exceeded.
            preprocess_fn: optional per-fold preprocessing (X_train, X_val).
            fit_predict_fn: optional custom fit/predict logic for validation.
            cleanup_fn: optional cleanup callback per fold.
            splitter: optional (train_idx, val_idx) iterator; defaults to cv_strategy config.

        Returns:
            Mean validation metric across folds.
        """
        params: Optional[Dict[str, Any]] = None
        if self._distributed_forced_params is not None:
            params = self._distributed_forced_params
            self._distributed_forced_params = None
        else:
            if trial is None:
                raise RuntimeError(
                    "Missing Optuna trial for parameter sampling.")
            params = {name: sampler(trial)
                      for name, sampler in hyperparameter_space.items()}
            if self._should_use_distributed_optuna():
                self._distributed_prepare_trial(params)
        X_all, y_all, w_all = data_provider()
        cfg_limit = getattr(self.ctx.config, "bo_sample_limit", None)
        if cfg_limit is not None:
            cfg_limit = int(cfg_limit)
            if cfg_limit > 0:
                sample_limit = cfg_limit if sample_limit is None else min(sample_limit, cfg_limit)
        if sample_limit is not None and len(X_all) > sample_limit:
            sampled_idx = self._resolve_time_sample_indices(X_all, int(sample_limit))
            if sampled_idx is None:
                sampled_idx = X_all.sample(
                    n=sample_limit,
                    random_state=self.ctx.rand_seed
                ).index
            X_all = X_all.loc[sampled_idx]
            y_all = y_all.loc[sampled_idx]
            w_all = w_all.loc[sampled_idx] if w_all is not None else None

        if splitter is None:
            strategy = str(getattr(self.ctx.config, "cv_strategy", "random") or "random").strip().lower()
            val_ratio = float(self.ctx.prop_test) if self.ctx.prop_test is not None else 0.25
            if not (0.0 < val_ratio < 1.0):
                val_ratio = 0.25
            cv_splits = getattr(self.ctx.config, "cv_splits", None)
            if cv_splits is None:
                cv_splits = max(2, int(round(1 / val_ratio)))
            cv_splits = max(2, int(cv_splits))

            if strategy in {"group", "grouped"}:
                group_col = getattr(self.ctx.config, "cv_group_col", None)
                if not group_col:
                    raise ValueError("cv_group_col is required for group cv_strategy.")
                if group_col not in self.ctx.train_data.columns:
                    raise KeyError(f"cv_group_col '{group_col}' not in train_data.")
                groups = self.ctx.train_data.reindex(X_all.index)[group_col]
                split_iter = GroupKFold(n_splits=cv_splits).split(X_all, y_all, groups=groups)
            elif strategy in {"time", "timeseries", "temporal"}:
                time_col = getattr(self.ctx.config, "cv_time_col", None)
                if not time_col:
                    raise ValueError("cv_time_col is required for time cv_strategy.")
                if time_col not in self.ctx.train_data.columns:
                    raise KeyError(f"cv_time_col '{time_col}' not in train_data.")
                ascending = bool(getattr(self.ctx.config, "cv_time_ascending", True))
                order_index = self.ctx.train_data[time_col].sort_values(ascending=ascending).index
                index_set = set(X_all.index)
                order_index = [idx for idx in order_index if idx in index_set]
                order = X_all.index.get_indexer(order_index)
                order = order[order >= 0]
                if len(order) <= cv_splits:
                    cv_splits = max(2, len(order) - 1)
                if cv_splits < 2:
                    raise ValueError("Not enough samples for time-series CV.")
                split_iter = _OrderSplitter(TimeSeriesSplit(n_splits=cv_splits), order).split(X_all)
            else:
                split_iter = ShuffleSplit(
                    n_splits=cv_splits,
                    test_size=val_ratio,
                    random_state=self.ctx.rand_seed
                ).split(X_all)
        else:
            if hasattr(splitter, "split"):
                split_iter = splitter.split(X_all, y_all, groups=None)
            else:
                split_iter = splitter

        losses: List[float] = []
        for train_idx, val_idx in split_iter:
            X_train = X_all.iloc[train_idx]
            y_train = y_all.iloc[train_idx]
            X_val = X_all.iloc[val_idx]
            y_val = y_all.iloc[val_idx]
            w_train = w_all.iloc[train_idx] if w_all is not None else None
            w_val = w_all.iloc[val_idx] if w_all is not None else None

            if preprocess_fn:
                X_train, X_val = preprocess_fn(X_train, X_val)

            model = model_builder(params)
            try:
                if fit_predict_fn:
                    y_pred = fit_predict_fn(
                        model, X_train, y_train, w_train,
                        X_val, y_val, w_val, trial
                    )
                else:
                    fit_kwargs = {}
                    if w_train is not None:
                        fit_kwargs["sample_weight"] = w_train
                    model.fit(X_train, y_train, **fit_kwargs)
                    y_pred = model.predict(X_val)
                losses.append(metric_fn(y_val, y_pred, w_val))
            finally:
                if cleanup_fn:
                    cleanup_fn(model)
                self._clean_gpu()

        return float(np.mean(losses))

    # Prediction + caching logic.
    def _predict_and_cache(self,
                           model,
                           pred_prefix: str,
                           use_oht: bool = False,
                           design_fn=None,
                           predict_kwargs_train: Optional[Dict[str, Any]] = None,
                           predict_kwargs_test: Optional[Dict[str, Any]] = None,
                           predict_fn: Optional[Callable[..., Any]] = None) -> None:
        if design_fn:
            X_train = design_fn(train=True)
            X_test = design_fn(train=False)
        elif use_oht:
            X_train = self.ctx.train_oht_scl_data[self.ctx.var_nmes]
            X_test = self.ctx.test_oht_scl_data[self.ctx.var_nmes]
        else:
            X_train = self.ctx.train_data[self.ctx.factor_nmes]
            X_test = self.ctx.test_data[self.ctx.factor_nmes]

        predictor = predict_fn or model.predict
        preds_train = predictor(X_train, **(predict_kwargs_train or {}))
        preds_test = predictor(X_test, **(predict_kwargs_test or {}))
        preds_train = np.asarray(preds_train)
        preds_test = np.asarray(preds_test)

        if preds_train.ndim <= 1 or (preds_train.ndim == 2 and preds_train.shape[1] == 1):
            col_name = f'pred_{pred_prefix}'
            self.ctx.train_data[col_name] = preds_train.reshape(-1)
            self.ctx.test_data[col_name] = preds_test.reshape(-1)
            self.ctx.train_data[f'w_{col_name}'] = (
                self.ctx.train_data[col_name] *
                self.ctx.train_data[self.ctx.weight_nme]
            )
            self.ctx.test_data[f'w_{col_name}'] = (
                self.ctx.test_data[col_name] *
                self.ctx.test_data[self.ctx.weight_nme]
            )
            self._maybe_cache_predictions(pred_prefix, preds_train, preds_test)
            return

        # Vector outputs (e.g., embeddings) are expanded into pred_<prefix>_0.. columns.
        if preds_train.ndim != 2:
            raise ValueError(
                f"Unexpected prediction shape for '{pred_prefix}': {preds_train.shape}")
        if preds_test.ndim != 2 or preds_test.shape[1] != preds_train.shape[1]:
            raise ValueError(
                f"Train/test prediction dims mismatch for '{pred_prefix}': "
                f"{preds_train.shape} vs {preds_test.shape}")
        for j in range(preds_train.shape[1]):
            col_name = f'pred_{pred_prefix}_{j}'
            self.ctx.train_data[col_name] = preds_train[:, j]
            self.ctx.test_data[col_name] = preds_test[:, j]
        self._maybe_cache_predictions(pred_prefix, preds_train, preds_test)

    def _cache_predictions(self,
                           pred_prefix: str,
                           preds_train,
                           preds_test) -> None:
        preds_train = np.asarray(preds_train)
        preds_test = np.asarray(preds_test)
        if preds_train.ndim <= 1 or (preds_train.ndim == 2 and preds_train.shape[1] == 1):
            if preds_test.ndim > 1:
                preds_test = preds_test.reshape(-1)
            col_name = f'pred_{pred_prefix}'
            self.ctx.train_data[col_name] = preds_train.reshape(-1)
            self.ctx.test_data[col_name] = preds_test.reshape(-1)
            self.ctx.train_data[f'w_{col_name}'] = (
                self.ctx.train_data[col_name] *
                self.ctx.train_data[self.ctx.weight_nme]
            )
            self.ctx.test_data[f'w_{col_name}'] = (
                self.ctx.test_data[col_name] *
                self.ctx.test_data[self.ctx.weight_nme]
            )
            self._maybe_cache_predictions(pred_prefix, preds_train, preds_test)
            return

        if preds_train.ndim != 2:
            raise ValueError(
                f"Unexpected prediction shape for '{pred_prefix}': {preds_train.shape}")
        if preds_test.ndim != 2 or preds_test.shape[1] != preds_train.shape[1]:
            raise ValueError(
                f"Train/test prediction dims mismatch for '{pred_prefix}': "
                f"{preds_train.shape} vs {preds_test.shape}")
        for j in range(preds_train.shape[1]):
            col_name = f'pred_{pred_prefix}_{j}'
            self.ctx.train_data[col_name] = preds_train[:, j]
            self.ctx.test_data[col_name] = preds_test[:, j]
        self._maybe_cache_predictions(pred_prefix, preds_train, preds_test)

    def _maybe_cache_predictions(self, pred_prefix: str, preds_train, preds_test) -> None:
        cfg = getattr(self.ctx, "config", None)
        if cfg is None or not bool(getattr(cfg, "cache_predictions", False)):
            return
        fmt = str(getattr(cfg, "prediction_cache_format", "parquet") or "parquet").lower()
        cache_dir = getattr(cfg, "prediction_cache_dir", None)
        if cache_dir:
            target_dir = Path(str(cache_dir))
            if not target_dir.is_absolute():
                target_dir = Path(self.output.result_dir) / target_dir
        else:
            target_dir = Path(self.output.result_dir) / "predictions"
        target_dir.mkdir(parents=True, exist_ok=True)

        def _build_frame(preds, split_label: str) -> pd.DataFrame:
            arr = np.asarray(preds)
            if arr.ndim <= 1:
                return pd.DataFrame({f"pred_{pred_prefix}": arr.reshape(-1)})
            cols = [f"pred_{pred_prefix}_{i}" for i in range(arr.shape[1])]
            return pd.DataFrame(arr, columns=cols)

        for split_label, preds in [("train", preds_train), ("test", preds_test)]:
            frame = _build_frame(preds, split_label)
            filename = f"{self.ctx.model_nme}_{pred_prefix}_{split_label}.{ 'csv' if fmt == 'csv' else 'parquet' }"
            path = target_dir / filename
            try:
                if fmt == "csv":
                    frame.to_csv(path, index=False)
                else:
                    frame.to_parquet(path, index=False)
            except Exception:
                pass

    def _resolve_best_epoch(self,
                            history: Optional[Dict[str, List[float]]],
                            default_epochs: int) -> int:
        if not history:
            return max(1, int(default_epochs))
        vals = history.get("val") or []
        if not vals:
            return max(1, int(default_epochs))
        best_idx = int(np.nanargmin(vals))
        return max(1, best_idx + 1)

    def _fit_predict_cache(self,
                           model,
                           X_train,
                           y_train,
                           sample_weight,
                           pred_prefix: str,
                           use_oht: bool = False,
                           design_fn=None,
                           fit_kwargs: Optional[Dict[str, Any]] = None,
                           sample_weight_arg: Optional[str] = 'sample_weight',
                           predict_kwargs_train: Optional[Dict[str, Any]] = None,
                           predict_kwargs_test: Optional[Dict[str, Any]] = None,
                           predict_fn: Optional[Callable[..., Any]] = None,
                           record_label: bool = True) -> None:
        fit_kwargs = fit_kwargs.copy() if fit_kwargs else {}
        if sample_weight is not None and sample_weight_arg:
            fit_kwargs.setdefault(sample_weight_arg, sample_weight)
        model.fit(X_train, y_train, **fit_kwargs)
        if record_label:
            self.ctx.model_label.append(self.label)
        self._predict_and_cache(
            model,
            pred_prefix,
            use_oht=use_oht,
            design_fn=design_fn,
            predict_kwargs_train=predict_kwargs_train,
            predict_kwargs_test=predict_kwargs_test,
            predict_fn=predict_fn)

