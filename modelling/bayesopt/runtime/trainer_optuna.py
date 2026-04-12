from __future__ import annotations

from datetime import timedelta
import os
from pathlib import Path
import threading
from typing import Any, Callable, Dict, Optional

import optuna
import pandas as pd

try:  # pragma: no cover
    import torch.distributed as dist  # type: ignore
except Exception:  # pragma: no cover
    dist = None  # type: ignore

from ins_pricing.modelling.bayesopt.artifacts import best_params_csv_path
from ins_pricing.modelling.bayesopt.utils.distributed_utils import DistributedUtils
from ins_pricing.utils import ensure_parent_dir, get_logger, log_print

_logger = get_logger("ins_pricing.trainer")


def _log(*args, **kwargs) -> None:
    log_print(_logger, *args, **kwargs)


class TrainerOptunaMixin:
    @staticmethod
    def _default_optuna_storage_from_output_dir(output_dir: Any) -> Optional[str]:
        raw = str(output_dir or "").strip()
        if not raw:
            return None
        return str(Path(raw) / "optuna" / "bayesopt.sqlite3")

    def _resolve_model_key_for_namespace(self) -> str:
        label = str(getattr(self, "label", "") or "").strip().lower()
        prefix = str(getattr(self, "model_name_prefix", "") or "").strip().lower()
        if "xgboost" in label or "xgboost" in prefix:
            return "xgb"
        if label == "glm" or prefix == "glm":
            return "glm"
        if "resnet" in label or "resnet" in prefix:
            return "resn"
        if "fttransformer" in label or "fttransformer" in prefix:
            return "ft"
        if "gnn" in label or "gnn" in prefix:
            return "gnn"
        return prefix or label or "unknown"

    def _resolve_best_params_label(self) -> str:
        return str(self.label)

    def _sanitize_best_params_payload(
        self,
        params: Optional[Dict[str, Any]],
        *,
        context: str,
    ) -> Dict[str, Any]:
        sanitized = dict(params or {})
        sanitize_fn = getattr(self, "_sanitize_best_params", None)
        if callable(sanitize_fn):
            sanitized = dict(sanitize_fn(sanitized, context=context) or {})
        return sanitized

    @staticmethod
    def _get_trials(
        study: optuna.study.Study,
        *,
        states: tuple[optuna.trial.TrialState, ...],
    ) -> list[optuna.trial.FrozenTrial]:
        """Read trials with minimal copying when supported by Optuna."""
        try:
            return study.get_trials(states=states, deepcopy=False)
        except TypeError:
            return study.get_trials(states=states)

    @staticmethod
    def _trial_sort_key(trial: optuna.trial.FrozenTrial) -> tuple[float, int]:
        raw_value = getattr(trial, "value", None)
        try:
            value = float(raw_value) if raw_value is not None else float("inf")
        except Exception:
            value = float("inf")
        try:
            number = int(getattr(trial, "number", -1))
        except Exception:
            number = -1
        return value, number

    def _resolve_best_trial_with_params(
        self,
        study: optuna.study.Study,
        *,
        context: str,
    ) -> tuple[Optional[optuna.trial.FrozenTrial], Dict[str, Any]]:
        try:
            study_best_trial = study.best_trial
        except Exception:
            study_best_trial = None

        selected_trial = study_best_trial
        selected_params: Dict[str, Any] = {}
        if selected_trial is not None:
            selected_params = self._sanitize_best_params_payload(
                getattr(selected_trial, "params", None),
                context=f"{context}_best_trial",
            )
        if selected_params:
            return selected_trial, selected_params

        complete_trials = self._get_trials(
            study,
            states=(optuna.trial.TrialState.COMPLETE,),
        )
        nonempty_trials: list[tuple[optuna.trial.FrozenTrial, Dict[str, Any]]] = []
        for trial in complete_trials:
            params = self._sanitize_best_params_payload(
                getattr(trial, "params", None),
                context=f"{context}_complete_trial_{getattr(trial, 'number', 'unknown')}",
            )
            if params:
                nonempty_trials.append((trial, params))

        if not nonempty_trials:
            return selected_trial, {}

        nonempty_trials.sort(key=lambda item: self._trial_sort_key(item[0]))
        fallback_trial, fallback_params = nonempty_trials[0]
        if study_best_trial is not None:
            _log(
                f"[Optuna][{self.label}] Best trial "
                f"(trial_id={getattr(study_best_trial, 'number', 'unknown')}, "
                f"value={getattr(study_best_trial, 'value', None)}) has empty params; "
                f"fallback to non-empty trial "
                f"(trial_id={getattr(fallback_trial, 'number', 'unknown')}, "
                f"value={getattr(fallback_trial, 'value', None)}).",
                flush=True,
            )
        return fallback_trial, fallback_params

    def _wait_with_deadline_fallback(
        self,
        wait_fn: Callable[[], Any],
        *,
        timeout_seconds: int,
        reason: str,
    ) -> None:
        done = threading.Event()
        holder: Dict[str, Any] = {"exc": None}

        def _target() -> None:
            try:
                wait_fn()
            except BaseException as exc:  # pragma: no cover - passthrough guard
                holder["exc"] = exc
            finally:
                done.set()

        threading.Thread(target=_target, daemon=True).start()
        if not done.wait(timeout=max(1, int(timeout_seconds))):
            raise TimeoutError(
                f"[DDP][{self.label}] barrier timed out after {timeout_seconds}s "
                f"during {reason} (wait() without timeout support)."
            )
        if holder["exc"] is not None:
            raise holder["exc"]

    def _dist_barrier(self, reason: str) -> None:
        """DDP barrier wrapper used by distributed Optuna."""
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
                _log(f"[DDP][{self.label}] entering barrier({reason}) rank={rank}/{world}", flush=True)
            except Exception:
                debug_barrier = False
        try:
            timeout = timedelta(seconds=timeout_seconds)
            backend = None
            try:
                backend = dist.get_backend()
            except Exception:
                backend = None

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
                            self._wait_with_deadline_fallback(
                                wait,
                                timeout_seconds=timeout_seconds,
                                reason=reason,
                            )
                    else:
                        dist.barrier()
                else:
                    dist.barrier()
            if debug_barrier:
                _log(f"[DDP][{self.label}] exit barrier({reason}) rank={rank}/{world}", flush=True)
        except Exception as exc:
            _log(
                f"[DDP][{self.label}] barrier failed during {reason}: {exc}",
                flush=True,
            )
            raise

    def _resolve_optuna_storage_url(self) -> Optional[str]:
        storage = getattr(self.config, "optuna_storage", None)
        if not storage:
            storage = self._default_optuna_storage_from_output_dir(
                getattr(self.config, "output_dir", None)
            )
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

    def _optuna_cleanup_sync(self) -> bool:
        return bool(getattr(self.config, "optuna_cleanup_synchronize", False))

    def _best_params_csv_path(self) -> str:
        path = best_params_csv_path(
            self.output.result_dir,
            self.ctx.model_nme,
            self._resolve_best_params_label(),
        )
        ensure_parent_dir(str(path))
        return str(path)

    def _persist_best_params_csv(self, params: Dict[str, Any]) -> None:
        path = self._best_params_csv_path()
        pd.DataFrame(params, index=[0]).to_csv(path, index=False)
        _log(
            f"[Optuna][{self.label}] best_params saved to {path} "
            f"(keys={sorted(list(params.keys()))})",
            flush=True,
        )

    def _create_study(self) -> optuna.study.Study:
        """Create or load an Optuna study and cache study name."""
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
        return study

    def _make_objective_wrapper(
        self,
        objective_fn: Callable[[optuna.trial.Trial], float],
        total_trials: int,
        progress_counter: Dict[str, int],
        *,
        barrier_on_end: bool = False,
    ) -> Callable[[optuna.trial.Trial], float]:
        """Create objective wrapper with OOM handling and cleanup/logging."""

        def objective_wrapper(trial: optuna.trial.Trial) -> float:
            should_log = DistributedUtils.is_main_process()
            if should_log:
                current_idx = progress_counter["count"] + 1
                _log(
                    f"[Optuna][{self.label}] Trial {current_idx}/{total_trials} started "
                    f"(trial_id={trial.number})."
                )
            status_repr = "OK"
            try:
                result = objective_fn(trial)
            except optuna.TrialPruned:
                status_repr = "PRUNED"
                raise
            except RuntimeError as exc:
                exc_text = str(exc)
                exc_lower = exc_text.lower()
                if "out of memory" in exc_lower:
                    status_repr = "PRUNED"
                    _log(
                        f"[Optuna][{self.label}] OOM detected. Pruning trial and clearing CUDA cache."
                    )
                    self._clean_gpu(synchronize=True)
                    raise optuna.TrialPruned() from exc
                dataloader_markers = (
                    "dataloader worker",
                    "exited unexpectedly",
                    "connection reset by peer",
                    "killed by signal: aborted",
                    "resource_sharer",
                )
                if any(marker in exc_lower for marker in dataloader_markers):
                    status_repr = "PRUNED"
                    _log(
                        f"[Optuna][{self.label}] DataLoader worker failure detected; pruning trial. "
                        f"detail={exc_text}",
                        flush=True,
                    )
                    self._clean_gpu(synchronize=True)
                    raise optuna.TrialPruned(
                        f"DataLoader worker failure: {exc_text}"
                    ) from exc
                status_repr = "FAIL"
                raise
            except Exception:
                status_repr = "FAIL"
                raise
            finally:
                self._clean_gpu(synchronize=self._optuna_cleanup_sync())
                if should_log:
                    progress_counter["count"] = progress_counter["count"] + 1
                    _log(
                        f"[Optuna][{self.label}] Trial {progress_counter['count']}/{total_trials} finished "
                        f"(status={status_repr})."
                    )
                if barrier_on_end:
                    self._dist_barrier("trial_end")
            return result

        return objective_wrapper

    def _make_checkpoint_callback(self) -> Callable[[optuna.study.Study, Any], None]:
        """Create callback that persists current best params."""

        def checkpoint_callback(check_study: optuna.study.Study, _trial) -> None:
            try:
                _best_trial, best_params = self._resolve_best_trial_with_params(
                    check_study,
                    context="optuna_checkpoint",
                )
                if not best_params:
                    return
                self._persist_best_params_csv(best_params)
            except Exception as exc:
                _log(
                    f"[Optuna][{self.label}] checkpoint callback skipped due to error: {exc}",
                    flush=True,
                )
                return

        return checkpoint_callback

    def _run_study_and_extract(
        self,
        study: optuna.study.Study,
        objective_wrapper: Callable[[optuna.trial.Trial], float],
        checkpoint_callback: Callable[[optuna.study.Study, Any], None],
        total_trials: int,
        progress_counter: Dict[str, int],
    ) -> None:
        """Run study optimization and populate best params/trial artifacts."""
        completed_states = (
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL,
        )
        completed = len(self._get_trials(study, states=completed_states))
        progress_counter["count"] = completed
        remaining = max(0, total_trials - completed)
        if remaining < 1:
            has_complete = bool(self._get_trials(
                study,
                states=(optuna.trial.TrialState.COMPLETE,),
            ))
            if not has_complete:
                _log(
                    f"[Optuna][{self.label}] Study has no completed trial yet; "
                    "running one recovery trial.",
                    flush=True,
                )
                remaining = 1
        if remaining > 0:
            study.optimize(
                objective_wrapper,
                n_trials=remaining,
                callbacks=[checkpoint_callback],
            )

        complete_count = len(self._get_trials(
            study,
            states=(optuna.trial.TrialState.COMPLETE,),
        ))
        if complete_count < 1:
            pruned_count = len(self._get_trials(
                study,
                states=(optuna.trial.TrialState.PRUNED,),
            ))
            fail_count = len(self._get_trials(
                study,
                states=(optuna.trial.TrialState.FAIL,),
            ))
            study_name = getattr(study, "study_name", None) or "<unnamed>"
            raise RuntimeError(
                f"[Optuna][{self.label}] No completed trials in study '{study_name}' "
                f"(complete=0, pruned={pruned_count}, failed={fail_count}, "
                f"max_evals={total_trials}). Increase max_evals or relax pruning/"
                "search constraints (for XGB, try lowering xgb_max_depth_max and "
                "xgb_n_estimators_max), then rerun."
            )

        selected_trial, selected_params = self._resolve_best_trial_with_params(
            study,
            context="optuna_extract",
        )

        self.best_trial = selected_trial
        self.best_params = selected_params
        if not self.best_params:
            raise RuntimeError(
                f"[Optuna][{self.label}] Best trial has empty params and no "
                "completed trial provides usable params; cannot continue. "
                "This can happen when reusing an Optuna study created with an "
                "empty search space. Consider clearing/rotating optuna_storage "
                "or updating optuna_study_prefix."
            )
        self._persist_best_params_csv(self.best_params)

    def tune(self, max_evals: int, objective_fn=None) -> None:
        if objective_fn is None:
            objective_fn = self.cross_val

        if self._should_use_distributed_optuna():
            self._distributed_tune(max_evals, objective_fn)
            return

        total_trials = max(1, int(max_evals))
        progress_counter = {"count": 0}
        study = self._create_study()
        self._optuna_total_trials = total_trials
        objective_wrapper = self._make_objective_wrapper(
            objective_fn, total_trials, progress_counter, barrier_on_end=False)
        checkpoint_callback = self._make_checkpoint_callback()
        try:
            self._run_study_and_extract(
                study,
                objective_wrapper,
                checkpoint_callback,
                total_trials,
                progress_counter,
            )
        finally:
            self._optuna_total_trials = None

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
        self._dist_barrier("prepare_trial")

    def _distributed_worker_loop(self, objective_fn: Callable[[Optional[optuna.trial.Trial]], float]) -> None:
        if dist is None:
            _log(
                f"[Optuna][Worker][{self.label}] torch.distributed unavailable. Worker exit.",
                flush=True,
            )
            return
        DistributedUtils.setup_ddp()
        if not dist.is_initialized():
            _log(
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
                    self.best_params = self._sanitize_best_params_payload(
                        best_params,
                        context="ddp_stop_payload",
                    )
                break
            if cmd == "RUN":
                params = payload.get("params") or {}
                self._distributed_forced_params = params
                self._dist_barrier("worker_start")
                try:
                    objective_fn(None)
                except optuna.TrialPruned:
                    pass
                except Exception as exc:
                    _log(
                        f"[Optuna][Worker][{self.label}] Exception: {exc}", flush=True)
                finally:
                    self._clean_gpu(synchronize=self._optuna_cleanup_sync())
                    self._dist_barrier("worker_end")

    def _distributed_tune(self, max_evals: int, objective_fn: Callable[[optuna.trial.Trial], float]) -> None:
        if dist is None:
            raise RuntimeError(
                f"[Optuna][{self.label}] distributed mode requires torch.distributed."
            )
        DistributedUtils.setup_ddp()
        if not dist.is_initialized():
            rank_env = os.environ.get("RANK", "0")
            if str(rank_env) != "0":
                _log(
                    f"[Optuna][{self.label}] DDP init failed on worker. Skip.",
                    flush=True,
                )
                return
            raise RuntimeError(
                f"[Optuna][{self.label}] distributed mode requested but DDP init failed."
            )
        if not self._distributed_is_main():
            self._distributed_worker_loop(objective_fn)
            return

        total_trials = max(1, int(max_evals))
        progress_counter = {"count": 0}
        study = self._create_study()
        self._optuna_total_trials = total_trials
        objective_wrapper = self._make_objective_wrapper(
            objective_fn, total_trials, progress_counter, barrier_on_end=True)
        checkpoint_callback = self._make_checkpoint_callback()
        try:
            self._run_study_and_extract(
                study,
                objective_wrapper,
                checkpoint_callback,
                total_trials,
                progress_counter,
            )
        finally:
            self._optuna_total_trials = None
            self._distributed_send_command(
                {"type": "STOP", "best_params": self.best_params})


__all__ = ["TrainerOptunaMixin"]
