from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch

from ins_pricing.modelling.bayesopt.config_runtime import OutputManager, VersionManager
from ins_pricing.modelling.bayesopt.config_schema import BayesOptConfig
from ins_pricing.modelling.bayesopt.dataset_preprocessor import DatasetPreprocessor
from ins_pricing.modelling.bayesopt.core_geo_preprocess_mixin import BayesOptGeoPreprocessMixin
from ins_pricing.modelling.bayesopt.core_training_mixin import BayesOptTrainingMixin
from ins_pricing.modelling.bayesopt.model_explain_mixin import BayesOptExplainMixin
from ins_pricing.modelling.bayesopt.model_plotting_mixin import BayesOptPlottingMixin
from ins_pricing.modelling.bayesopt.models import GraphNeuralNetSklearn
from ins_pricing.modelling.bayesopt.trainers import FTTrainer, GLMTrainer, GNNTrainer, ResNetTrainer, XGBTrainer
from ins_pricing.modelling.bayesopt.trainers.cv_utils import CVStrategyResolver
from ins_pricing.utils import DeviceManager, set_global_seed, get_logger, log_print
from ins_pricing.utils.losses import (
    normalize_distribution_name,
    resolve_effective_loss_name,
    resolve_xgb_objective,
)

_logger = get_logger("ins_pricing.modelling.bayesopt.core")


def _log(*args, **kwargs) -> None:
    log_print(_logger, *args, **kwargs)


class _ResolvedCVSplitter:
    """Adapter exposing a sklearn-like split() API via CVStrategyResolver."""

    def __init__(
        self,
        resolver: CVStrategyResolver,
        *,
        n_splits: int,
        val_ratio: float,
    ) -> None:
        self._resolver = resolver
        self._n_splits = max(2, int(n_splits))
        self._val_ratio = float(val_ratio)

    def split(self, X, y=None, groups=None):  # pylint: disable=unused-argument
        split_iter, _ = self._resolver.create_cv_splitter(
            X_all=X,
            y_all=y,
            n_splits=self._n_splits,
            val_ratio=self._val_ratio,
        )
        for tr_idx, val_idx in split_iter:
            yield tr_idx, val_idx

# BayesOpt orchestration and SHAP utilities
# =============================================================================
class BayesOptModel(
    BayesOptTrainingMixin,
    BayesOptGeoPreprocessMixin,
    BayesOptPlottingMixin,
    BayesOptExplainMixin,
):
    def __init__(self, train_data, test_data, config: BayesOptConfig):
        """Orchestrate BayesOpt training across multiple trainers.

        Args:
            train_data: Training DataFrame.
            test_data: Test DataFrame.
            config: BayesOptConfig instance with all configuration.

        Examples:
            config = BayesOptConfig(
                model_nme="my_model",
                resp_nme="target",
                weight_nme="weight",
                factor_nmes=["feat1", "feat2"]
            )
            model = BayesOptModel(train_df, test_df, config=config)
        """
        if not isinstance(config, BayesOptConfig):
            raise TypeError(
                f"config must be a BayesOptConfig instance, got {type(config).__name__}"
            )
        # Work on an internal copy so caller-provided config object is never mutated.
        cfg = deepcopy(config)
        self.config = cfg
        self.model_nme = cfg.model_nme
        self.task_type = cfg.task_type
        configured_distribution = getattr(cfg, "distribution", None)
        configured_loss_name = getattr(cfg, "loss_name", None)
        normalized_distribution = normalize_distribution_name(
            getattr(cfg, "distribution", None),
            self.task_type,
        )
        self.distribution = None if normalized_distribution == "auto" else normalized_distribution
        self.loss_name = resolve_effective_loss_name(
            getattr(cfg, "loss_name", None),
            task_type=self.task_type,
            model_name=self.model_nme,
            distribution=self.distribution,
        )
        if hasattr(self.config, "distribution"):
            self.config.distribution = self.distribution
        self.resp_nme = cfg.resp_nme
        self.weight_nme = cfg.weight_nme
        self.factor_nmes = cfg.factor_nmes
        self.binary_resp_nme = cfg.binary_resp_nme
        self.cate_list = list(cfg.cate_list or [])
        self.prop_test = cfg.prop_test
        self.epochs = cfg.epochs
        self.rand_seed = cfg.rand_seed if cfg.rand_seed is not None else np.random.randint(
            1, 10000)
        set_global_seed(int(self.rand_seed))
        self.use_gpu = bool(
            cfg.use_gpu
            and (torch.cuda.is_available() or DeviceManager.is_mps_available())
        )
        self.output_manager = OutputManager(
            cfg.output_dir or os.getcwd(), self.model_nme)

        bundle_path = self._resolve_preprocess_bundle_path()
        load_preprocess_bundle = bool(
            getattr(self.config, "load_preprocess_bundle", False)
        )
        save_preprocess_bundle = bool(
            getattr(self.config, "save_preprocess_bundle", False)
        )

        if load_preprocess_bundle:
            if bundle_path is None:
                raise ValueError(
                    "load_preprocess_bundle=True requires preprocess_bundle_path."
                )
            self._load_preprocess_bundle(bundle_path)
        else:
            if train_data is None or test_data is None:
                raise ValueError(
                    "train_data/test_data must be provided unless "
                    "load_preprocess_bundle=True."
                )
            preprocessor = DatasetPreprocessor(train_data, test_data, cfg).run()
            self.train_data = preprocessor.train_data
            self.test_data = preprocessor.test_data
            self.train_oht_data = preprocessor.train_oht_data
            self.test_oht_data = preprocessor.test_oht_data
            self.train_oht_scl_data = preprocessor.train_oht_scl_data
            self.test_oht_scl_data = preprocessor.test_oht_scl_data
            self.var_nmes = preprocessor.var_nmes
            self.num_features = preprocessor.num_features
            self.cat_categories_for_shap = preprocessor.cat_categories_for_shap
            self.numeric_scalers = preprocessor.numeric_scalers
            self.ohe_feature_names = list(
                getattr(preprocessor, "ohe_feature_names", []) or []
            )
            self.oht_sparse_csr = bool(
                getattr(preprocessor, "oht_sparse_csr", False)
            )
            if getattr(self.config, "save_preprocess", False):
                artifact_path = getattr(self.config, "preprocess_artifact_path", None)
                if artifact_path:
                    target = Path(str(artifact_path))
                    if not target.is_absolute():
                        target = Path(self.output_manager.result_dir) / target
                else:
                    target = Path(self.output_manager.result_path(
                        f"{self.model_nme}_preprocess.json"
                    ))
                preprocessor.save_artifacts(target)
            if save_preprocess_bundle and bundle_path is not None:
                self._save_preprocess_bundle(bundle_path)
        self.geo_token_cols: List[str] = []
        self.train_geo_tokens: Optional[pd.DataFrame] = None
        self.test_geo_tokens: Optional[pd.DataFrame] = None
        self.geo_gnn_model: Optional[GraphNeuralNetSklearn] = None
        self._add_region_effect()

        self.cv = self._build_cv_splitter()
        if self.task_type == 'classification':
            self.obj = 'binary:logistic'
        else:  # regression task
            self.obj = resolve_xgb_objective(self.loss_name)
        _log(
            "[RunConfig] "
            f"model={self.model_nme} "
            f"task_type={self.task_type} "
            f"distribution_cfg={configured_distribution!r} "
            f"distribution_resolved={(self.distribution if self.distribution is not None else 'auto')!r} "
            f"loss_name_cfg={configured_loss_name!r} "
            f"loss_name_resolved={self.loss_name!r} "
            f"xgb_objective={self.obj!r} "
            f"ft_role={getattr(cfg, 'ft_role', None)!r}",
            flush=True,
        )
        self.fit_params = {
            'sample_weight': self.train_data[self.weight_nme].values
        }
        self.model_label: List[str] = []
        self.optuna_storage = cfg.optuna_storage
        self.optuna_study_prefix = cfg.optuna_study_prefix or "bayesopt"
        self.glm_best = None
        self.xgb_best = None
        self.resn_best = None
        self.ft_best = None
        self.gnn_best = None

        # Keep trainers in a dict for unified access and easy extension.
        self.trainers: Dict[str, TrainerBase] = {
            'glm': GLMTrainer(self),
            'xgb': XGBTrainer(self),
            'resn': ResNetTrainer(self),
            'ft': FTTrainer(self),
            'gnn': GNNTrainer(self),
        }
        self._prepare_geo_tokens()
        self.version_manager = VersionManager(self.output_manager)
        self._objective_service = None
        self._data_registry = None

    def _build_cv_splitter(self) -> _ResolvedCVSplitter:
        val_ratio = float(self.prop_test) if self.prop_test is not None else 0.25
        if not (0.0 < val_ratio < 1.0):
            val_ratio = 0.25
        cv_splits = getattr(self.config, "cv_splits", None)
        if cv_splits is None:
            cv_splits = max(2, int(round(1 / val_ratio)))
        resolver = CVStrategyResolver(
            self.config,
            self.train_data,
            self.rand_seed,
        )
        return _ResolvedCVSplitter(
            resolver,
            n_splits=int(cv_splits),
            val_ratio=val_ratio,
        )
