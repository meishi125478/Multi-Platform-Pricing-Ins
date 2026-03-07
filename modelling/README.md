# Modelling

ML training, evaluation, explainability, and visualization.

## Submodules

| Path | Description |
|------|-------------|
| `bayesopt/` | BayesOpt training core (GLM / XGB / ResNet / FT-Transformer / GNN) |
| `plotting/` | Model-agnostic curves, diagnostics, importance, and geo visualization |
| `explain/` | Permutation importance, SHAP, integrated gradients |
| `evaluation.py` | Calibration, threshold selection, bootstrap CI, metrics report |

## Common Usage

### Python API (config-based)

```python
from ins_pricing.modelling.bayesopt import BayesOptConfig
from ins_pricing.modelling import BayesOptModel

config = BayesOptConfig.from_file("config.json")
model = BayesOptModel(train_data, test_data, config=config)
model.optimize_model("xgb", max_evals=50)
model.optimize_model("resn", max_evals=30)
```

### CLI

```bash
python ins_pricing/cli/BayesOpt_entry.py --config-json config.json
```

### Explainability

```bash
python ins_pricing/cli/Explain_entry.py --config-json config_explain.json
```

```python
# Permutation importance
from ins_pricing.modelling.explain import permutation_importance
results = permutation_importance(predict_fn, X, y, metric="auto", n_repeats=5)

# Integrated gradients (ResNet / FT)
from ins_pricing.modelling.explain import resnet_integrated_gradients, ft_integrated_gradients
attr = resnet_integrated_gradients(model, X, steps=50)

# SHAP
from ins_pricing.modelling.explain import compute_shap_xgb
shap_result = compute_shap_xgb(ctx, on_train=False, n_background=500)
```

### Plotting

```python
from ins_pricing.modelling.plotting import curves, diagnostics, importance, geo

# Lift / double-lift
fig = curves.plot_lift_curve(pred, actual, weight=weight)
fig = curves.plot_double_lift_curve(pred1, actual1, pred2, actual2)

# Calibration, ROC, KS
fig = curves.plot_calibration_curve(y_true, y_pred)
fig = curves.plot_roc_curves(y_true, {"xgb": y_pred_xgb, "resn": y_pred_resn})

# Oneway diagnostics
fig = diagnostics.plot_oneway(df, "age_band", weight_col="exposure", target_col="claim_amt")

# Feature importance / SHAP
fig = importance.plot_feature_importance(imp_series, top_n=30)
fig = importance.plot_shap_importance(shap_values, feature_names)

# Geo
fig = geo.plot_geo_heatmap_on_map(df, "longitude", "latitude", "residual")
```

### Evaluation

```python
from ins_pricing.modelling.evaluation import (
    calibrate_predictions,
    select_threshold,
    bootstrap_ci,
    metrics_report,
)

cal = calibrate_predictions(y_true, y_pred, method="sigmoid")
adjusted = cal.predict(y_pred_test)

threshold_info = select_threshold(y_true, y_pred, metric="f1")
ci = bootstrap_ci(rmse_fn, y_true, y_pred, n_samples=200, ci=0.95)
report = metrics_report(y_true, y_pred, task_type="regression", weight=w)
```

## Loss Functions

Loss resolution priority:
1. `distribution` (if set) overrides `loss_name`
2. `loss_name` (if set and not `auto`)
3. Legacy auto inference (from `model_nme`) when both are `auto`/unset

| distribution | resolved loss_name | notes |
| --- | --- | --- |
| `tweedie` | `tweedie` | Tweedie power can be tuned |
| `poisson` | `poisson` | Tweedie power fixed at `1.0` |
| `gamma` | `gamma` | Tweedie power fixed at `2.0` |
| `gaussian`, `normal`, `mse` | `mse` | L2 regression |
| `laplace`, `laplacian`, `mae` | `mae` | L1 regression |
| `bernoulli`, `binomial`, `logistic`, `binary` | `logloss` | Classification |

Supported explicit `loss_name` values:
- Regression: `auto`, `tweedie`, `poisson`, `gamma`, `mse`, `mae`
- Classification: `auto`, `logloss`, `bce`

Detailed BayesOpt-level behavior is documented in `ins_pricing/modelling/bayesopt/README.md`.

## Output Layout

Training writes to `output_dir/` with subdirectories:
- `plot/` - plots and diagnostics
- `Results/` - metrics, params, snapshots
- `model/` - saved model artifacts

Models load from `output_dir/model` by default (override with `explain.model_dir`).
