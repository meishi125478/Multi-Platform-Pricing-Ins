from __future__ import annotations

from .gradients import (
    ft_integrated_gradients,
    gradient_x_input_torch,
    integrated_gradients_multi_input_torch,
    integrated_gradients_torch,
    resnet_integrated_gradients,
    summarize_attributions,
)
from .metrics import (
    auc_score,
    logloss,
    mae,
    mape,
    gamma_deviance,
    poisson_deviance,
    r2_score,
    rmse,
    tweedie_deviance,
    resolve_metric,
)
from .permutation import permutation_importance
from .shap_utils import (
    compute_shap_core,
    compute_shap_ft,
    compute_shap_glm,
    compute_shap_resn,
    compute_shap_xgb,
)

__all__ = [
    "auc_score",
    "logloss",
    "mae",
    "mape",
    "gamma_deviance",
    "poisson_deviance",
    "tweedie_deviance",
    "r2_score",
    "rmse",
    "resolve_metric",
    "permutation_importance",
    "gradient_x_input_torch",
    "integrated_gradients_torch",
    "integrated_gradients_multi_input_torch",
    "summarize_attributions",
    "resnet_integrated_gradients",
    "ft_integrated_gradients",
    "compute_shap_core",
    "compute_shap_glm",
    "compute_shap_xgb",
    "compute_shap_resn",
    "compute_shap_ft",
]
