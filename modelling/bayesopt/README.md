# BayesOpt

BayesOpt is the training/tuning core for GLM, XGBoost, ResNet, FT-Transformer, and GNN workflows.
It supports JSON-driven CLI runs and a Python API for notebooks/scripts.

## Python API (config-based)

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

### Load config from file

```python
config = BayesOptConfig.from_file("config.json")
model = BayesOptModel(train_data, test_data, config=config)
```

## CLI

```bash
python ins_pricing/cli/BayesOpt_entry.py --config-json config_template.json
```

## Architecture

```
bayesopt/
  config_schema.py        BayesOptConfig dataclass and JSON loading
  core.py                 BayesOptModel orchestrator (composes mixins)
  core_training_mixin.py  optimize_model(), save/load, feature addition
  model_explain_mixin.py  permutation, SHAP, integrated gradients methods
  model_plotting_mixin.py plot_oneway, plot_calibration, plot_lift, etc.
  artifacts.py            OutputManager, VersionManager
  checkpoints.py          Checkpoint save/load utilities
  trainers/
    __init__.py            TrainerBase, GLMTrainer, XGBTrainer, ResNetTrainer, FTTrainer, GNNTrainer
    trainer_glm.py         GLM (statsmodels) with Optuna tuning
    trainer_xgb.py         XGBoost with GPU/DMatrix/chunked training
    trainer_resn.py        ResNet with DDP support
    trainer_ft.py          FT-Transformer (model / embedding / unsupervised_embedding)
    trainer_gnn.py         GNN with graph construction and subsampling
    cv_utils.py            Cross-validation fold generation
  models/
    __init__.py            Model class exports
    model_ft_components.py FeatureTokenizer, ScaledTransformerEncoderLayer, FTTransformerCore
    model_ft_trainer.py    FTTransformerSklearn wrapper
    model_resn.py          ResBlock, ResNetSequential, ResNetSklearn
    model_gnn.py           SimpleGraphLayer, SimpleGNN, GraphNeuralNetSklearn
  runtime/
    trainer_optuna.py      Optuna study orchestration
    trainer_persistence.py Model persistence utilities
    trainer_cv_prediction.py  Cached CV prediction generation
  utils/
    torch_runtime.py       Torch training loop helpers
    torch_trainer_mixin.py Shared torch trainer utilities
```

## Extensibility and Refactoring Direction

- Keep `BayesOptModel` as the orchestration facade and treat trainers as pluggable strategy units.
  - New model families should implement `TrainerBase` and be registered through trainer modules.
- Keep data/side-effect boundaries explicit.
  - `runtime/` handles execution flow and persistence orchestration.
  - `trainers/` and `models/` focus on model behavior and optimization logic.
- Prefer backward-compatible config evolution.
  - Additive config fields should be the default.
  - Behavior changes should remain explicit in config flags or versioned runtime logic.
- Future roadmap
  - Introduce clearer trainer registry wiring to reduce `if/else` growth in orchestration.
  - Isolate reusable preprocessing and CV components for cross-model reuse.
  - Expand contract tests for config parsing, checkpoint compatibility, and artifact schema stability.

## BayesOptModel Key Methods

### Training
- `optimize_model(model_key, max_evals=100)` - train: `"glm"`, `"xgb"`, `"resn"`, `"ft"`, `"gnn"`
- `add_numeric_feature_from_column(col_name)` / `add_numeric_features_from_columns(col_names)`
- `save_model()` / `load_model()`

### Explainability (via mixin)
- `compute_permutation_importance(model_key, ...)`
- `compute_shap_xgb(...)`, `compute_shap_glm(...)`, `compute_shap_resn(...)`, `compute_shap_ft(...)`
- `compute_integrated_gradients_resn(...)`, `compute_integrated_gradients_ft(...)`

### Plotting (via mixin)
- `plot_oneway(...)`, `plot_calibration(...)`, `plot_roc(...)`, `plot_lift(...)`, `plot_ks(...)`

## BayesOptConfig Key Fields

### Required
- `model_nme`, `resp_nme`, `weight_nme`, `factor_nmes`

### Task
- `task_type` (`"regression"` / `"classification"`), `distribution`, `loss_name`
- `cate_list`, `binary_resp_nme`

### Training
- `prop_test`, `rand_seed`, `epochs`, `use_gpu`
- `cv_strategy` (`random` / `group` / `time` / `stratified`), `cv_splits`

### Model-specific search spaces
- `xgb_search_space`, `resn_search_space`, `ft_search_space`, `ft_unsupervised_search_space`

## Loss and Distribution Mapping

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

## FT Roles

- `model`: FT is a prediction model (writes `pred_ft`).
- `embedding`: FT trains with labels but exports embeddings (`pred_<prefix>_*`).
- `unsupervised_embedding`: FT trains without labels and exports embeddings.

## Output Layout

`output_dir/` contains:
- `plot/` - plots and diagnostics
- `Results/` - metrics, params, and snapshots
- `model/` - saved models

## XGBoost GPU Tips

- Use `xgb_gpu_id` to select a specific GPU on multi-GPU Linux systems.
- Per-fold GPU cleanup is disabled by default to avoid long idle gaps caused by CUDA sync.
  - `xgb_cleanup_per_fold=true` to reclaim memory between folds.
  - `xgb_cleanup_synchronize=true` for a full device sync (slower).
- `xgb_use_dmatrix=true` switches to `xgb.train` + DMatrix/QuantileDMatrix for better throughput.
- `xgb_chunk_size` enables chunked incremental boosting to reduce peak memory.
  - Example: `xgb_chunk_size=200000` trains on 200k-row chunks.
  - In chunk mode `early_stopping_rounds` is ignored.
- `stream_split_csv=true` enables CSV streaming random split at entry runtime.

## Torch Model Cleanup

Fold-level cleanup for FT/ResNet/GNN is off by default.
Enable if you see memory pressure:
- `ft_cleanup_per_fold`, `ft_cleanup_synchronize`
- `resn_cleanup_per_fold`, `resn_cleanup_synchronize`
- `gnn_cleanup_per_fold`, `gnn_cleanup_synchronize`
- `optuna_cleanup_synchronize` controls trial-level cleanup sync (default false)
- `resn_use_lazy_dataset=true` (default) avoids full in-memory tensor copy for ResNet.
- `resn_predict_batch_size` controls batched ResNet inference to avoid OOM.
- `ft_use_lazy_dataset=true` (default) avoids full tensor materialization during FT training.
- `ft_predict_batch_size` controls FT batched inference for cached predictions/embeddings.
- `gnn_max_fit_rows` caps GNN train/val rows to avoid graph OOM.
- `gnn_max_predict_rows` and `gnn_predict_chunk_rows` gate large GNN prediction requests.

## Notes

- Relative paths in config are resolved from the config file directory.
- For multi-GPU, use `torchrun` and set `runner.nproc_per_node` in config.
