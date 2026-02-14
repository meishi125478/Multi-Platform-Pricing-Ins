from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from ins_pricing.modelling.bayesopt.config_preprocess import BayesOptConfig, OutputManager


@runtime_checkable
class TrainerContext(Protocol):
    """Typed context interface consumed by trainer implementations."""

    config: BayesOptConfig
    model_nme: str
    task_type: str
    loss_name: str
    resp_nme: str
    weight_nme: str
    factor_nmes: list[str]
    cate_list: list[str]
    var_nmes: list[str]
    num_features: list[str]
    ohe_feature_names: list[str]
    oht_sparse_csr: bool
    use_gpu: bool
    prop_test: float
    epochs: int
    rand_seed: int
    obj: str
    fit_params: dict[str, Any]
    cv: Any
    train_data: Any
    test_data: Any
    train_oht_data: Any
    train_oht_scl_data: Any
    test_oht_scl_data: Any
    train_geo_tokens: Any
    test_geo_tokens: Any
    geo_token_cols: list[str]
    geo_gnn_model: Any
    output_manager: OutputManager
    xgb_best: Any
    resn_best: Any
    gnn_best: Any
    glm_best: Any
    ft_best: Any
    model_label: list[str]

    def default_tweedie_power(self, obj: Optional[str] = None) -> Optional[float]: ...
    def _build_geo_tokens(self, params_override: Optional[dict[str, Any]] = None): ...
