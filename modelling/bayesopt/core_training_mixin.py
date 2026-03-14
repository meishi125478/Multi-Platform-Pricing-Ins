from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

from ins_pricing.modelling.bayesopt.artifacts import (
    best_params_csv_path,
    load_best_params_csv,
)
from ins_pricing.utils import get_logger, log_print
from ins_pricing.utils.io import IOUtils

_logger = get_logger("ins_pricing.modelling.bayesopt.core")


def _log(*args, **kwargs) -> None:
    log_print(_logger, *args, **kwargs)


class BayesOptTrainingMixin:
    def _require_trainer(self, model_key: str) -> "TrainerBase":
        trainer = self.trainers.get(model_key)
        if trainer is None:
            raise KeyError(f"Unknown model key: {model_key}")
        return trainer

    def _pred_vector_columns(self, pred_prefix: str) -> List[str]:
        """Return vector feature columns like pred_<prefix>_0.. sorted by suffix."""
        col_prefix = f"pred_{pred_prefix}_"
        cols = [c for c in self.train_data.columns if c.startswith(col_prefix)]

        def sort_key(name: str):
            tail = name.rsplit("_", 1)[-1]
            try:
                return (0, int(tail))
            except Exception:
                return (1, tail)

        cols.sort(key=sort_key)
        return cols

    def _inject_pred_features(self, pred_prefix: str) -> List[str]:
        """Inject pred_<prefix> or pred_<prefix>_i columns into features and return names."""
        cols = self._pred_vector_columns(pred_prefix)
        if cols:
            self.add_numeric_features_from_columns(cols)
            return cols
        scalar_col = f"pred_{pred_prefix}"
        if scalar_col in self.train_data.columns:
            self.add_numeric_feature_from_column(scalar_col)
            return [scalar_col]
        return []

    def _maybe_load_best_params(self, model_key: str, trainer: "TrainerBase") -> None:
        def _sanitize_loaded(params: Optional[Dict[str, Any]], source: str) -> Optional[Dict[str, Any]]:
            if not isinstance(params, dict):
                return None
            sanitize_fn = getattr(trainer, "_sanitize_best_params", None)
            if callable(sanitize_fn):
                return dict(sanitize_fn(dict(params), context=source) or {})
            return dict(params)

        def _apply_loaded(
            params: Optional[Dict[str, Any]],
            *,
            source: str,
            message: str,
            study_name: Optional[str] = None,
        ) -> None:
            trainer.best_params = _sanitize_loaded(params, source)
            trainer.best_trial = None
            if study_name is not None:
                trainer.study_name = study_name
            _log(message)

        # 1) If best_params_files is specified, load and skip tuning.
        best_params_files = getattr(self.config, "best_params_files", None) or {}
        best_params_file = best_params_files.get(model_key)
        if best_params_file and not trainer.best_params:
            _apply_loaded(
                IOUtils.load_params_file(best_params_file),
                source="best_params_file",
                message=(
                    f"[Optuna][{trainer.label}] Loaded best_params from "
                    f"{best_params_file}; skip tuning."
                ),
            )

        # 2) If reuse_best_params is enabled, prefer version snapshots; else load legacy CSV.
        reuse_params = bool(getattr(self.config, "reuse_best_params", False))
        if reuse_params and not trainer.best_params:
            payload = self.version_manager.load_latest(f"{model_key}_best")
            best_params = None if payload is None else payload.get("best_params")
            if best_params:
                _apply_loaded(
                    best_params,
                    source="version_snapshot",
                    message=(
                        f"[Optuna][{trainer.label}] Reusing best_params "
                        "from versions snapshot."
                    ),
                    study_name=(
                        payload.get("study_name")
                        if isinstance(payload, dict)
                        else None
                    ),
                )
                return

            params_path = best_params_csv_path(
                self.output_manager.result_dir,
                self.model_nme,
                trainer.label,
            )
            params = load_best_params_csv(
                self.output_manager.result_dir,
                self.model_nme,
                trainer.label,
            )
            if params is not None:
                _apply_loaded(
                    params,
                    source="best_params_csv",
                    message=(
                        f"[Optuna][{trainer.label}] Reusing best_params from "
                        f"{params_path}."
                    ),
                )

    # Generic optimization entry point.
    def optimize_model(self, model_key: str, max_evals: int = 100):
        if model_key not in self.trainers:
            _log(f"Warning: Unknown model key: {model_key}")
            return

        trainer = self._require_trainer(model_key)
        self._maybe_load_best_params(model_key, trainer)

        should_tune = not trainer.best_params
        if should_tune:
            if model_key == "ft" and str(self.config.ft_role) == "unsupervised_embedding":
                if hasattr(trainer, "cross_val_unsupervised"):
                    trainer.tune(
                        max_evals,
                        objective_fn=getattr(trainer, "cross_val_unsupervised")
                    )
                else:
                    raise RuntimeError(
                        "FT trainer does not support unsupervised Optuna objective.")
            else:
                trainer.tune(max_evals)

        if model_key == "ft" and str(self.config.ft_role) != "model":
            prefix = str(self.config.ft_feature_prefix or "ft_emb")
            role = str(self.config.ft_role)
            if role == "embedding":
                trainer.train_as_feature(
                    pred_prefix=prefix, feature_mode="embedding")
            elif role == "unsupervised_embedding":
                trainer.pretrain_unsupervised_as_feature(
                    pred_prefix=prefix,
                    params=trainer.best_params
                )
            else:
                raise ValueError(
                    f"Unsupported ft_role='{role}', expected 'model'/'embedding'/'unsupervised_embedding'.")

            # Inject generated prediction/embedding columns as features (scalar or vector).
            self._inject_pred_features(prefix)
            # Do not add FT as a standalone model label; downstream models handle evaluation.
        else:
            trainer.train()

        if bool(getattr(self.config, "final_ensemble", False)):
            k = int(getattr(self.config, "final_ensemble_k", 3) or 3)
            if (
                k > 1
                and not (model_key == "ft" and str(self.config.ft_role) != "model")
            ):
                if hasattr(trainer, "ensemble_predict"):
                    trainer.ensemble_predict(k)
                else:
                    _log(
                        f"[Ensemble] Trainer '{model_key}' does not support ensemble prediction.",
                        flush=True,
                    )

        setattr(self, f"{model_key}_best", trainer.model)
        # Save a snapshot for traceability.
        study_name = getattr(trainer, "study_name", None)
        if study_name is None and trainer.best_trial is not None:
            study_obj = getattr(trainer.best_trial, "study", None)
            study_name = getattr(study_obj, "study_name", None)
        try:
            config_snapshot = asdict(self.config)
        except Exception as exc:
            _log(
                f"[VersionManager] Failed to serialize config with asdict: {exc}. "
                "Saving fallback config snapshot.",
                flush=True,
            )
            config_snapshot = {
                "model_nme": self.model_nme,
                "task_type": self.task_type,
                "error": f"asdict_failed: {type(exc).__name__}",
            }
        snapshot = {
            "model_key": model_key,
            "timestamp": datetime.now().isoformat(),
            "best_params": trainer.best_params,
            "study_name": study_name,
            "config": config_snapshot,
        }
        self.version_manager.save(f"{model_key}_best", snapshot)

    def add_numeric_feature_from_column(self, col_name: str) -> None:
        """Add an existing column as a feature and sync one-hot/scaled tables."""
        if col_name not in self.train_data.columns or col_name not in self.test_data.columns:
            raise KeyError(
                f"Column '{col_name}' must exist in both train_data and test_data.")

        if col_name not in self.factor_nmes:
            self.factor_nmes.append(col_name)
        if col_name not in self.config.factor_nmes:
            self.config.factor_nmes.append(col_name)

        if col_name not in self.cate_list and col_name not in self.num_features:
            self.num_features.append(col_name)

        if self.train_oht_data is not None and self.test_oht_data is not None:
            self.train_oht_data[col_name] = self.train_data[col_name].values
            self.test_oht_data[col_name] = self.test_data[col_name].values
        if self.train_oht_scl_data is not None and self.test_oht_scl_data is not None:
            scaler = StandardScaler()
            tr = self.train_data[col_name].to_numpy(
                dtype=np.float32, copy=False).reshape(-1, 1)
            te = self.test_data[col_name].to_numpy(
                dtype=np.float32, copy=False).reshape(-1, 1)
            self.train_oht_scl_data[col_name] = scaler.fit_transform(
                tr).reshape(-1)
            self.test_oht_scl_data[col_name] = scaler.transform(te).reshape(-1)

        if col_name not in self.var_nmes:
            self.var_nmes.append(col_name)

    def add_numeric_features_from_columns(self, col_names: List[str]) -> None:
        if not col_names:
            return

        missing = [
            col for col in col_names
            if col not in self.train_data.columns or col not in self.test_data.columns
        ]
        if missing:
            raise KeyError(
                f"Column(s) {missing} must exist in both train_data and test_data."
            )

        for col_name in col_names:
            if col_name not in self.factor_nmes:
                self.factor_nmes.append(col_name)
            if col_name not in self.config.factor_nmes:
                self.config.factor_nmes.append(col_name)
            if col_name not in self.cate_list and col_name not in self.num_features:
                self.num_features.append(col_name)
            if col_name not in self.var_nmes:
                self.var_nmes.append(col_name)

        if self.train_oht_data is not None and self.test_oht_data is not None:
            self.train_oht_data[col_names] = self.train_data[col_names].to_numpy(copy=False)
            self.test_oht_data[col_names] = self.test_data[col_names].to_numpy(copy=False)

        if self.train_oht_scl_data is not None and self.test_oht_scl_data is not None:
            scaler = StandardScaler()
            tr = self.train_data[col_names].to_numpy(dtype=np.float32, copy=False)
            te = self.test_data[col_names].to_numpy(dtype=np.float32, copy=False)
            self.train_oht_scl_data[col_names] = scaler.fit_transform(tr)
            self.test_oht_scl_data[col_names] = scaler.transform(te)

    def prepare_ft_as_feature(self, max_evals: int = 50, pred_prefix: str = "ft_feat") -> str:
        """Train FT as a feature generator and return the downstream column name."""
        ft_trainer = self._require_trainer("ft")
        ft_trainer.tune(max_evals=max_evals)
        if hasattr(ft_trainer, "train_as_feature"):
            ft_trainer.train_as_feature(pred_prefix=pred_prefix)
        else:
            ft_trainer.train()
        feature_col = f"pred_{pred_prefix}"
        self.add_numeric_feature_from_column(feature_col)
        return feature_col

    def prepare_ft_embedding_as_features(self, max_evals: int = 50, pred_prefix: str = "ft_emb") -> List[str]:
        """Train FT and inject pooled embeddings as vector features pred_<prefix>_0.. ."""
        ft_trainer = self._require_trainer("ft")
        ft_trainer.tune(max_evals=max_evals)
        if hasattr(ft_trainer, "train_as_feature"):
            ft_trainer.train_as_feature(
                pred_prefix=pred_prefix, feature_mode="embedding")
        else:
            raise RuntimeError(
                "FT trainer does not support embedding feature mode.")
        cols = self._pred_vector_columns(pred_prefix)
        if not cols:
            raise RuntimeError(
                f"No embedding columns were generated for prefix '{pred_prefix}'.")
        self.add_numeric_features_from_columns(cols)
        return cols

    def prepare_ft_unsupervised_embedding_as_features(self,
                                                      pred_prefix: str = "ft_uemb",
                                                      params: Optional[Dict[str,
                                                                            Any]] = None,
                                                      mask_prob_num: float = 0.15,
                                                      mask_prob_cat: float = 0.15,
                                                      num_loss_weight: float = 1.0,
                                                      cat_loss_weight: float = 1.0) -> List[str]:
        """Export embeddings after FT self-supervised masked reconstruction pretraining."""
        ft_trainer = self._require_trainer("ft")
        if not hasattr(ft_trainer, "pretrain_unsupervised_as_feature"):
            raise RuntimeError(
                "FT trainer does not support unsupervised pretraining.")
        ft_trainer.pretrain_unsupervised_as_feature(
            pred_prefix=pred_prefix,
            params=params,
            mask_prob_num=mask_prob_num,
            mask_prob_cat=mask_prob_cat,
            num_loss_weight=num_loss_weight,
            cat_loss_weight=cat_loss_weight
        )
        cols = self._pred_vector_columns(pred_prefix)
        if not cols:
            raise RuntimeError(
                f"No embedding columns were generated for prefix '{pred_prefix}'.")
        self.add_numeric_features_from_columns(cols)
        return cols

    # GLM Bayesian optimization wrapper.
    def bayesopt_glm(self, max_evals=50):
        self.optimize_model("glm", max_evals)

    # XGBoost Bayesian optimization wrapper.
    def bayesopt_xgb(self, max_evals=100):
        self.optimize_model("xgb", max_evals)

    # ResNet Bayesian optimization wrapper.
    def bayesopt_resnet(self, max_evals=100):
        self.optimize_model("resn", max_evals)

    # GNN Bayesian optimization wrapper.
    def bayesopt_gnn(self, max_evals=50):
        self.optimize_model("gnn", max_evals)

    # FT-Transformer Bayesian optimization wrapper.
    def bayesopt_ft(self, max_evals=50):
        self.optimize_model("ft", max_evals)

    def save_model(self, model_name=None):
        keys = [model_name] if model_name else self.trainers.keys()
        for key in keys:
            if key in self.trainers:
                self.trainers[key].save()
            else:
                if model_name:  # Only warn when the user specifies a model name.
                    _log(f"[save_model] Warning: Unknown model key {key}")

    def load_model(self, model_name=None):
        keys = [model_name] if model_name else self.trainers.keys()
        for key in keys:
            if key in self.trainers:
                self.trainers[key].load()
                # Sync context fields.
                trainer = self.trainers[key]
                if trainer.model is not None:
                    setattr(self, f"{key}_best", trainer.model)
            else:
                if model_name:
                    _log(f"[load_model] Warning: Unknown model key {key}")


__all__ = ["BayesOptTrainingMixin"]
