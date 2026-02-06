# BayesOpt

BayesOpt is the training/tuning core for GLM, XGBoost, ResNet, FT-Transformer, and GNN workflows.
It supports JSON-driven CLI runs and a Python API for notebooks/scripts.

## Recommended API (config-based)

```python
from ins_pricing.modelling.bayesopt import BayesOptConfig
from ins_pricing.modelling import BayesOptModel

config = BayesOptConfig(
    model_nme="my_model",
    resp_nme="target",
    weight_nme="weight",
    factor_nmes=["f1", "f2"],
    cate_list=["f2"],
    task_type="regression",
    epochs=50,
    output_dir="./Results",
)

model = BayesOptModel(train_data, test_data, config=config)
model.optimize_model("xgb", max_evals=50)
```

## Load config from file

```python
from ins_pricing.modelling.bayesopt import BayesOptConfig
from ins_pricing.modelling import BayesOptModel

config = BayesOptConfig.from_file("config.json")
model = BayesOptModel(train_data, test_data, config=config)
```

## CLI entry

```bash
python ins_pricing/cli/BayesOpt_entry.py --config-json config_template.json
```

## Loss and Distribution Mapping

You can configure either `loss_name`, `distribution`, or both.

Resolution order:
1. If `distribution` is set, it takes precedence and is mapped to the corresponding `loss_name`.
2. If `distribution` is not set, use `loss_name` directly.
3. If both are unset/`auto`, keep legacy auto behavior (infer from `model_nme`).

| task_type | distribution (accepted values) | resolved loss_name | xgboost objective | tweedie power |
| --- | --- | --- | --- | --- |
| regression | `tweedie` | `tweedie` | `reg:tweedie` | tuned in `[1.0, 2.0]` when enabled |
| regression | `poisson` | `poisson` | `count:poisson` | `1.0` |
| regression | `gamma` | `gamma` | `reg:gamma` | `2.0` |
| regression | `gaussian`, `normal`, `mse` | `mse` | `reg:squarederror` | not used |
| regression | `laplace`, `laplacian`, `mae` | `mae` | `reg:absoluteerror` | not used |
| classification | `bernoulli`, `binomial`, `logistic`, `binary` | `logloss` | `binary:logistic` | not used |

Example:

```json
{
  "task_type": "regression",
  "distribution": "poisson",
  "loss_name": "auto"
}
```

## FT roles

- `model`: FT is a prediction model (writes `pred_ft`).
- `embedding`: FT trains with labels but exports embeddings (`pred_<prefix>_*`).
- `unsupervised_embedding`: FT trains without labels and exports embeddings.

## Output layout

`output_dir/` contains:
- `plot/` plots and diagnostics
- `Results/` metrics, params, and snapshots
- `model/` saved models

## XGBoost GPU tips

- Use `xgb_gpu_id` to select a specific GPU on multi-GPU Linux systems.
- Per-fold GPU cleanup is disabled by default to avoid long idle gaps caused by CUDA sync.
  - If you need to reclaim memory between folds, set `xgb_cleanup_per_fold=true`.
  - If you still need a full device sync, set `xgb_cleanup_synchronize=true` (slower).
- `xgb_use_dmatrix=true` switches XGBoost to `xgb.train` + DMatrix/QuantileDMatrix for better throughput.
- External-memory DMatrix (file-backed) is disabled; pass in-memory arrays/dataframes.

## Torch model cleanup

To reduce CPU↔GPU thrash, fold-level cleanup for FT/ResNet/GNN is off by default.
Enable if you see memory pressure:
- `ft_cleanup_per_fold`, `ft_cleanup_synchronize`
- `resn_cleanup_per_fold`, `resn_cleanup_synchronize`
- `gnn_cleanup_per_fold`, `gnn_cleanup_synchronize`
- `optuna_cleanup_synchronize` controls whether trial-level cleanup syncs CUDA (default false)

## Notes

- Relative paths in config are resolved from the config file directory.
- For multi-GPU, use `torchrun` and set `runner.nproc_per_node` in config.
