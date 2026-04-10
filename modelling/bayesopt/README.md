# BayesOpt

BayesOpt is the training and tuning core used by `ins_pricing.modelling`.
This is the owner document for BayesOpt-specific behavior, including detailed
loss/distribution resolution.

## Purpose

- Provide a config-driven orchestration facade (`BayesOptModel`) over trainer backends.
- Keep trainer/model implementations pluggable by model family (`glm`, `xgb`, `resn`, `ft`, `gnn`).
- Centralize persistence/checkpoints/output layout for training workflows.

## Minimal Run

```python
from ins_pricing.modelling import BayesOptConfig, BayesOptModel

config = BayesOptConfig.from_file("config.json")
model = BayesOptModel(train_data, test_data, config=config)
model.optimize_model("xgb", max_evals=50)
```

## Key Components

- `config_schema.py`: `BayesOptConfig` schema and config loading.
- `core.py` and mixins: orchestration facade for training, explainability, plotting.
- `trainers/`: backend implementations (`GLMTrainer`, `XGBTrainer`, `ResNetTrainer`,
  `FTTrainer`, `GNNTrainer`).
- `models/`: model classes/wrappers used by torch/GNN/transformer trainers.
- `runtime/`: Optuna orchestration, persistence, and CV prediction helpers.

## Loss And Distribution Mapping (Owner)

Resolution order:

1. `distribution` wins when set.
2. Otherwise explicit `loss_name` is used.
3. If both are unset or `auto`, standard auto behavior applies.

| task_type | distribution values | resolved loss_name | xgboost objective | tweedie power |
| --- | --- | --- | --- | --- |
| regression | `tweedie` | `tweedie` | `reg:tweedie` | tuned in `[1.0, 2.0]` when enabled |
| regression | `poisson` | `poisson` | `count:poisson` | `1.0` |
| regression | `gamma` | `gamma` | `reg:gamma` | `2.0` |
| regression | `gaussian`, `normal`, `mse` | `mse` | `reg:squarederror` | not used |
| regression | `laplace`, `laplacian`, `mae` | `mae` | `reg:absoluteerror` | not used |
| classification | `bernoulli`, `binomial`, `logistic`, `binary` | `logloss` | `binary:logistic` | not used |

## FT Roles

- `model`: FT runs as predictor and writes `pred_ft`.
- `embedding`: FT trains with labels and exports embedding features.
- `unsupervised_embedding`: FT trains without labels and exports embeddings.

## Output Layout

`output_dir/` contains:

- `plot/`: plots and diagnostics
- `Results/`: metrics, params, snapshots
- `model/`: persisted model artifacts

## Runtime Notes

- Relative config paths are resolved from the config file directory.
- Multi-GPU distributed runs use `torchrun` with `runner.nproc_per_node`.
- Optional cleanup flags (`*_cleanup_per_fold`, `*_cleanup_synchronize`) trade speed for memory.

## Further Reading

- Modelling package boundary: [../README.md](../README.md)
- Public export index: [../../docs/api_reference.md](../../docs/api_reference.md)
