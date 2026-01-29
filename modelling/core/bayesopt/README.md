# BayesOpt

BayesOpt is the training/tuning core for GLM, XGBoost, ResNet, FT-Transformer, and GNN workflows.
It supports JSON-driven CLI runs and a Python API for notebooks/scripts.

## Recommended API (config-based)

```python
from ins_pricing.modelling.core.bayesopt import BayesOptConfig
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
from ins_pricing.modelling.core.bayesopt import BayesOptConfig
from ins_pricing.modelling import BayesOptModel

config = BayesOptConfig.from_file("config.json")
model = BayesOptModel(train_data, test_data, config=config)
```

## CLI entry

```bash
python ins_pricing/cli/BayesOpt_entry.py --config-json config_template.json
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

## Notes

- Relative paths in config are resolved from the config file directory.
- For multi-GPU, use `torchrun` and set `runner.nproc_per_node` in config.
