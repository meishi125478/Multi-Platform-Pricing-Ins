# modelling

## Purpose

`modelling` owns model training, tuning orchestration, explainability, plotting, and evaluation.
It is the package entry for BayesOpt workflows and related analysis utilities.

## Use When / Not For

- Use when training/tuning GLM, XGB, ResNet, FT, or GNN models with config-driven workflows.
- Use when generating model explanations (permutation, SHAP, integrated gradients) and model plots.
- Not for premium table computation and calibration loops (handled by `pricing`).
- Not for runtime model registry/deployment operations (handled by `governance` and `production`).

## Public Entrypoints

- Core facade: `BayesOptConfig`, `BayesOptModel`
- Subpackages: `bayesopt`, `plotting`, `explain`, `evaluation`
- BayesOpt-level trainer/model exports are also available from `ins_pricing.modelling`

Loss/distribution details are owned by the BayesOpt deep-dive doc and are not duplicated here.

## Minimal Flow

```python
from ins_pricing.modelling import BayesOptConfig, BayesOptModel

config = BayesOptConfig.from_file("config.json")
model = BayesOptModel(train_data, test_data, config=config)
model.optimize_model("xgb", max_evals=50)
```

## Further Reading

- BayesOpt internals and loss/distribution mapping: [bayesopt/README.md](bayesopt/README.md)
- Public export index: [../docs/api_reference.md](../docs/api_reference.md)
- Package navigation: [../README.md](../README.md)
