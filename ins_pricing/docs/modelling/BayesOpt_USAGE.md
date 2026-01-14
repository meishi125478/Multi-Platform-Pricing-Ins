# BayesOpt Usage Guide (Framework + How-To)

This document explains the overall framework, config fields, and recommended usage for the training/tuning/stacking pipeline under `ins_pricing/modelling/`. It is mainly for:

- Batch training via JSON config using `ins_pricing/cli/BayesOpt_entry.py` (can be combined with `torchrun`)
- Calling the Python API directly in notebooks/scripts via `ins_pricing.BayesOpt` or `ins_pricing.bayesopt`

---

## 1. Which file should you run?

Files related to this workflow in `ins_pricing/modelling/`:

- `ins_pricing/modelling/core/bayesopt/`: Core subpackage (data preprocessing, Trainer, Optuna tuning, FT embedding/self-supervised pretraining, plotting, SHAP, etc)
- `ins_pricing/modelling/core/BayesOpt.py`: Compatibility entry that re-exports the new subpackage for older import paths
- `ins_pricing/cli/BayesOpt_entry.py`: CLI batch entry (reads multiple CSVs from config, trains/tunes/saves/plots; supports DDP)
- `ins_pricing/cli/BayesOpt_incremental.py`: Incremental training entry (append data and reuse params/models; for production incremental scenarios)
- `ins_pricing/cli/utils/cli_common.py`: Shared CLI helpers (path resolution, model name generation, plotting selection)
- `ins_pricing/__init__.py`: Makes `ins_pricing/` importable (e.g. `from ins_pricing import BayesOptModel` or `from ins_pricing import bayesopt`)
- `ins_pricing/cli/utils/notebook_utils.py`: Notebook helpers (build and run BayesOpt_entry and watchdog commands)
- `ins_pricing/cli/Pricing_Run.py`: Unified runner (notebook/script only needs a config; `runner` decides entry/incremental/DDP/watchdog)
- `ins_pricing/examples/modelling/config_template.json`: Common config template (recommended to copy and edit)
- `ins_pricing/examples/modelling/config_incremental_template.json`: Sample incremental training config (used by `Pricing_incremental.ipynb`)
- `ins_pricing/examples/modelling/config_explain_template.json`: Explain workflow config template
- `user_packages legacy/Try/config_Pricing_FT_Stack.json`: Historical "FT stacking" config example
- Notebooks (demo): `ins_pricing/examples/modelling/Pricing_Run.ipynb`, `ins_pricing/examples/modelling/PricingSingle.ipynb`, `ins_pricing/examples/modelling/Explain_Run.ipynb`
- Deprecated examples: see `user_packages legacy/Try/*_deprecate.ipynb`

Note: `ins_pricing/examples/modelling/` is kept in the repo only; the PyPI package does not include this directory.

---

## 2. Overall framework (from data to model pipeline)

### 2.1 Typical flow for a single training job (BayesOpt_entry)

Core logic in `BayesOpt_entry.py` (each dataset `model_name.csv` runs once):

1. Read `config.json`, build dataset names from `model_list x model_categories` (e.g. `od_bc`)
2. Load data from `data_dir/<model_name>.csv`
3. Split train/test with `split_strategy` (`random` / `time` / `group`)
4. Construct `BayesOptModel(train_df, test_df, ...)`
5. Run by FT role and model selection:
   - If `ft_role != "model"`: run FT first (tune/train/export embedding columns), then run base models (XGB/ResNet/GLM, etc)
   - If `ft_role == "model"`: FT itself is a prediction model and can be tuned/trained in parallel with others
6. Save models and parameter snapshots, optionally plot

Extra: `BayesOpt_entry.py` / `BayesOpt_incremental.py` resolve relative paths in config as "relative to the config.json directory" (for example, if config is in `ins_pricing/examples/modelling/`, then `./Data` means `ins_pricing/examples/modelling/Data`). Currently supported path fields: `data_dir` / `output_dir` / `optuna_storage` / `gnn_graph_cache` / `best_params_files`.

If you want notebook runs to only change config (no code changes), use `ins_pricing/examples/modelling/Pricing_Run.ipynb` (it calls `ins_pricing/cli/Pricing_Run.py`). Add a `runner` field in config to control entry/incremental/DDP/watchdog.

### 2.2 Core components in the BayesOpt subpackage

Under `ins_pricing/modelling/core/bayesopt/`:

- `BayesOptConfig`: unified config (epochs, feature lists, FT role, DDP/DP, etc)
- `DatasetPreprocessor`: preprocessing once in `BayesOptModel` init:
  - create `w_act` (weighted actual), optional `w_binary_act`
  - cast categorical columns to `category`
  - create `train_oht_data/test_oht_data` (one-hot)
  - create `train_oht_scl_data/test_oht_scl_data` (one-hot with standardized numeric columns)
- `TrainerBase`: base trainer with `tune()` (Optuna), `train()`, `save()/load()`, and distributed Optuna sync for DDP
- Trainers (`BayesOptModel.trainers`):
  - `GLMTrainer`: statsmodels GLM
  - `XGBTrainer`: xgboost
  - `ResNetTrainer`: PyTorch MLP/ResNet style
  - `FTTrainer`: FT-Transformer (supports 3 roles)
  - `GNNTrainer`: GNN (standalone model `gnn`, or used to generate geo tokens for FT)
- `OutputManager`: unified output paths (`plot/`, `Results/`, `model/`)
- `VersionManager`: save/load snapshots (`Results/versions/*_ft_best.json`, etc)

### 2.3 BayesOpt subpackage structure (read in code order)

`BayesOpt` is now a subpackage (`ins_pricing/modelling/core/bayesopt/`). Recommended order:

1) **Tools and utilities**

- `IOUtils / TrainingUtils / PlotUtils`: I/O, training utilities (batch size, tweedie loss, free_cuda), plotting helpers
- `DistributedUtils`: DDP init, rank/world_size helpers

2) **TorchTrainerMixin (common components for torch tabular training)**

- DataLoader: `_build_dataloader()` / `_build_val_dataloader()` (prints batch/accum/workers)
- Loss: `_compute_losses()` / `_compute_weighted_loss()` (regression uses tweedie; classification uses BCEWithLogits)
- Early stop: `_early_stop_update()`

3) **Sklearn-style model classes (core training objects)**

- `ResNetSklearn`: `fit/predict/set_params`, holds `ResNetSequential`, supports DP/DDP
- `FTTransformerSklearn`: `fit/predict/fit_unsupervised`, supports embedding output, DP/DDP
- `GraphNeuralNetSklearn`: `fit/predict/set_params`, used for geo tokens (CPU/GPU graph build, adjacency cache)

4) **Config and preprocessing/output management**

- `BayesOptConfig`: aggregated config for task, training, parallelism, FT role (built in `BayesOptModel`)
- `OutputManager`: manage `plot/Results/model` under output root
- `VersionManager`: write snapshots to `Results/versions/` and read latest (for best_params reuse)
- `DatasetPreprocessor`: runs in `BayesOptModel.__init__`, generates data views and derived columns

5) **Trainer system (Optuna + training + cached predictions)**

- `TrainerBase`: `tune()` (Optuna), `save()/load()`, distributed Optuna sync for DDP
- `cross_val_generic()`: generic CV/holdout evaluation logic (trainer supplies model_builder/metric_fn/fit_predict_fn)
- `_fit_predict_cache()` / `_predict_and_cache()`: after training, write predictions back to `BayesOptModel.train_data/test_data`

6) **Orchestrator BayesOptModel**

- `BayesOptModel.optimize_model(model_key, max_evals)`: unified entry, responsible for:
  - selecting objective (e.g. self-supervised objective when `ft_role=unsupervised_embedding`)
  - "FT as feature" mode: export `pred_<prefix>_*` and inject into downstream features
  - saving snapshots (for reuse/backtracking)
- `save_model/load_model`, `plot_*`, `compute_shap_*`, etc

### 2.4 Key call chain (from entry to disk)

Using `BayesOpt_entry.py` as an example:

1. `BayesOpt_entry.train_from_config()` reads CSV and builds `BayesOptModel(...)`
2. `BayesOptModel.optimize_model(model_key)`
3. `TrainerBase.tune()` (if `reuse_best_params` is false or no historical params found)
   - calls `Trainer.cross_val()` or FT self-supervised `Trainer.cross_val_unsupervised()`
   - inside `cross_val_generic()`:
     - sample Optuna params
     - build model `model_builder(params)`
     - train and evaluate on validation via `metric_fn(...)`
4. `Trainer.train()` trains the final model with `best_params` and caches prediction columns
5. `Trainer.save()` saves model files; `BayesOptModel.optimize_model()` saves parameter snapshots

**Optuna under DDP (distributed coordination)**:

- Only rank0 drives Optuna sampling; trial params are broadcast to other ranks
- Non-rank0 processes do not sample; they receive params and run the same objective (multi-GPU sync)

### 2.5 Data views and cached columns (used by training/plotting)

`DatasetPreprocessor` creates common columns in `train_data/test_data`:

- `w_act`: `target * weight`
- (if `binary_resp_nme` provided) `w_binary_act`: `binary_target * weight`

After training, `TrainerBase._predict_and_cache()` writes predictions back:

- **Scalar prediction models**:
  - `pred_<prefix>` (e.g. `pred_xgb/pred_resn/pred_ft`)
  - `w_pred_<prefix>` (column name `w_pred_xgb`; computed as `pred_<prefix> * weight`)
- **Multi-dim output (embedding)**:
  - `pred_<prefix>_0 .. pred_<prefix>_{k-1}` (e.g. `pred_ft_emb_0..`)
  - these multi-dim columns do not have `w_` weighted columns

These prediction columns are used by lift/dlift/oneway plotting and downstream stacking.

### 2.6 Sklearn-style model classes: details and usage

Below are the three sklearn-style model classes in `bayesopt` (usually created by trainers, but can be used directly).

#### 2.6.1 ResNetSklearn (`class ResNetSklearn`)

Purpose: train a residual MLP on one-hot/standardized tabular features (regression uses Softplus, classification outputs logits).

Key parameters (common):

- `input_dim`: input dimension (typically number of one-hot columns)
- `hidden_dim`, `block_num`: width and number of residual blocks
- `learning_rate`, `epochs`, `patience`
- `use_data_parallel` / `use_ddp`

Key methods:

- `fit(X_train, y_train, w_train, X_val, y_val, w_val, trial=...)`
- `predict(X_test)`: classification uses sigmoid; regression clips to positive
- `set_params(params: dict)`: trainer writes `best_params` back to model

Minimal manual example:

```python
from ins_pricing.BayesOpt import ResNetSklearn

# Use the one-hot standardized view from DatasetPreprocessor for X_train/X_val.
resn = ResNetSklearn(model_nme="od_bc", input_dim=X_train.shape[1], task_type="regression", epochs=50)
resn.set_params({"hidden_dim": 32, "block_num": 4, "learning_rate": 1e-3})
resn.fit(X_train, y_train, w_train, X_val, y_val, w_val)
y_pred = resn.predict(X_val)
```

#### 2.6.2 FTTransformerSklearn (`class FTTransformerSklearn`)

Purpose: learn Transformer representations on numeric/categorical features; supports three output modes:

- supervised prediction: `predict()` returns scalar predictions
- embedding output: `predict(return_embedding=True)` returns `(N, d_model)` embeddings
- self-supervised masked reconstruction: `fit_unsupervised()` (used by `ft_role=unsupervised_embedding`)

Key details:

- Numeric columns are `nan_to_num` and standardized by train mean/std in `_tensorize_split()` (reduces AMP overflow risk)
- Categorical columns record train `categories` on first build; inference uses the same categories; unknown/missing maps to "unknown index" (`len(categories)`)
- DDP uses `DistributedSampler`; the self-supervised head is computed inside forward to avoid DDP "ready twice" errors

Key methods:

- `fit(X_train, y_train, w_train, X_val, y_val, w_val, trial=..., geo_train=..., geo_val=...)`
- `predict(X_test, geo_tokens=None, return_embedding=False)`
- `fit_unsupervised(X_train, X_val=None, mask_prob_num=..., mask_prob_cat=..., ...) -> float`

Minimal manual example (self-supervised pretrain + embeddings):

```python
from ins_pricing.BayesOpt import FTTransformerSklearn

ft = FTTransformerSklearn(
    model_nme="od_bc",
    num_cols=num_cols,
    cat_cols=cat_cols,
    d_model=64,
    n_heads=4,
    n_layers=4,
    dropout=0.1,
    epochs=30,
    use_ddp=False,
)

val_loss = ft.fit_unsupervised(train_df, X_val=test_df, mask_prob_num=0.2, mask_prob_cat=0.2)
emb = ft.predict(test_df, return_embedding=True)   # shape: (N, d_model)
```

#### 2.6.3 GraphNeuralNetSklearn (`class GraphNeuralNetSklearn`)

Purpose: build a graph from `geo_feature_nmes` and train a small GNN to generate geo tokens for FT.

Key details:

- Graph building: kNN (approx via pynndescent if available; GPU graph build with PyG when memory allows)
- Adjacency cache: `graph_cache_path`
- Training: full-graph training (one forward per epoch), good for moderate-size geo features

Key methods:

- `fit(X_train, y_train, w_train, X_val, y_val, w_val, trial=...)`
- `predict(X)`: regression clips positive; classification uses sigmoid
- `set_params(params: dict)`: rebuilds the backbone after structural changes

> In most stacking workflows you do not need to call it manually: when `geo_feature_nmes` is provided in config, `BayesOptModel` builds and caches geo tokens during init.

### 2.7 Mapping between Trainer and Sklearn models (who calls what)

To unify tuning and final training/saving, `bayesopt` uses two layers:

- **Trainer (tuning/scheduling layer)**: Optuna, CV/holdout, feature view selection, save/load, prediction caching
- **Sklearn-style model (execution layer)**: only fit/predict (plus minimal helpers), no Optuna or output paths

Mapping overview:

- `GLMTrainer` -> statsmodels GLM (not a `*Sklearn` class; trainer builds design matrix and caches `pred_glm/w_pred_glm`)
- `XGBTrainer` -> `xgb.XGBRegressor` (`enable_categorical=True`, choose `gpu_hist/hist` based on `use_gpu`)
- `ResNetTrainer` -> `ResNetSklearn`
  - Feature view: usually `train_oht_scl_data/test_oht_scl_data` with `var_nmes` (one-hot + standardize)
  - Cached columns: `pred_resn/w_pred_resn`
- `FTTrainer` -> `FTTransformerSklearn`
  - Feature view: raw `train_data/test_data` with `factor_nmes` (numeric + category columns; category columns must be declared in `cate_list`)
  - `ft_role=model`: cache `pred_ft/w_pred_ft`
  - `ft_role=embedding/unsupervised_embedding`: cache `pred_<prefix>_0..` and inject into downstream `factor_nmes`
- `GraphNeuralNetSklearn`: primarily used by `BayesOptModel` to generate geo tokens (when `geo_feature_nmes` is set)

---

## 3. Three FT roles (decide whether to stack)

FT role is controlled by `ft_role` (from config or CLI `--ft-role`):

### 3.1 `ft_role="model"` (FT as a prediction model)

- Goal: train FT directly from `X -> y`, generate `pred_ft` / `w_pred_ft`
- FT participates in lift/dlift/SHAP evaluation

### 3.2 `ft_role="embedding"` (supervised training, export embeddings only)

- Goal: still train with `X -> y` (embedding quality influenced by supervised signal)
- Export pooled embedding feature columns: `pred_<ft_feature_prefix>_0..`
- These columns are injected into `factor_nmes` for downstream base models (stacking)
- FT itself is not evaluated as a standalone model in lift/SHAP

### 3.3 `ft_role="unsupervised_embedding"` (masked pretrain + embeddings)

- Goal: do not use `y`; run masked reconstruction on inputs `X` (numeric + categorical)
- Export `pred_<ft_feature_prefix>_0..` and inject to downstream features
- Suitable for "representation first, base model decision" two-stage stacking

---

## 4. What does Optuna optimize?

### 4.1 Supervised models (GLM/XGB/ResNet/FT-as-model)

- `TrainerBase.tune()` calls each trainer's `cross_val()` and minimizes validation metric (default direction `minimize`)
- Regression typically uses Tweedie deviance or related loss; classification uses logloss

### 4.2 FT self-supervised (`unsupervised_embedding`)

When `ft_role="unsupervised_embedding"`, `BayesOptModel.optimize_model("ft")` calls:

- `FTTrainer.cross_val_unsupervised()` (Optuna objective)
- Objective: validation loss of masked reconstruction (smaller is better)
  - Numeric: MSE only on masked positions (multiplied by `num_loss_weight`)
  - Categorical: cross-entropy only on masked positions (multiplied by `cat_loss_weight`)

Note:
- `n_heads` is not searched by default; it is derived from `d_model` with divisibility guarantees (see `FTTrainer._resolve_adaptive_heads()`).

---

## 5. Output directories and files (convention)

Output root comes from `output_dir` (config) or CLI `--output-dir`. Under it:

- `plot/`: plots (loss curves, lift/dlift/oneway, etc)
- `Results/`: params, metrics, version snapshots
  - `Results/<model>_bestparams_<trainer>.csv`: best params per trainer after tuning
  - `Results/versions/<timestamp>_<model_key>_best.json`: snapshots (best_params and config)
- `model/`: model files
  - GLM/XGB: `pkl`
  - PyTorch: `pth` (ResNet usually saves state_dict; FT usually saves full object)

---

## 6. Config fields (JSON) - common

Start by copying `ins_pricing/examples/modelling/config_template.json`. Examples: `ins_pricing/examples/modelling/config_template.json`, `ins_pricing/examples/modelling/config_incremental_template.json`, `user_packages legacy/Try/config_Pricing_FT_Stack.json`.

### 6.1 Path resolution rules (important)

- `BayesOpt_entry.py` / `BayesOpt_incremental.py` resolve relative paths in config as "relative to the config.json directory".
  - Example: config in `ins_pricing/examples/modelling/` and `data_dir: "./Data"` means `ins_pricing/examples/modelling/Data`.
  - Fields resolved: `data_dir` / `output_dir` / `optuna_storage` / `gnn_graph_cache` / `best_params_files`.
- If `optuna_storage` looks like a URL (contains `://`), it is passed to Optuna as-is; otherwise it is resolved as a file path and converted to absolute.

**Data and task**

- `data_dir` (str): directory of CSV files (`<model_name>.csv` per dataset)
- `model_list` (list[str]) / `model_categories` (list[str]): build dataset names (cartesian product)
- `target` (str): target column name
- `weight` (str): weight column name
- `feature_list` (list[str]): feature column names (recommended to provide explicitly; otherwise inferred in `BayesOptModel`)
- `categorical_features` (list[str]): categorical column names (if empty, inferred in `BayesOptModel`)
- `binary_resp_nme` (str|null, optional): binary target column (for conversion curves, etc)
- `task_type` (str, optional): `"regression"` / `"classification"`, default `"regression"`

**Training and split**

- `prop_test` (float): train/test split ratio (entry splits train/test; trainers also do CV/holdout), typical `(0, 0.5]`, default `0.25`
- `split_strategy` (str): `"random"` / `"time"` / `"group"` (applies in `BayesOpt_entry.py` and `Explain_entry.py`)
- `split_time_col` (str|null): required when `split_strategy="time"` (time order for holdout)
- `split_time_ascending` (bool): time sort direction, default `true`
- `split_group_col` (str|null): required when `split_strategy="group"` (group holdout)
- `cv_strategy` (str|null): CV strategy for Optuna folds (`"random"` / `"time"` / `"group"`); if null, defaults to `split_strategy`
- `cv_time_col` (str|null): required when `cv_strategy="time"` (time order for CV)
- `cv_time_ascending` (bool): time sort direction for CV, default `true`
- `cv_group_col` (str|null): required when `cv_strategy="group"` (group CV)
- `cv_splits` (int|null): explicit CV fold count (otherwise derived from `prop_test`)
- `rand_seed` (int): random seed, default `13`
- `epochs` (int): NN epochs (ResNet/FT/GNN), default `50`
- `use_gpu` (bool, optional): prefer GPU (actual usage depends on `torch.cuda.is_available()`)
- `resn_weight_decay` (float, optional): ResNet weight decay (L2), default `1e-4`
- `final_ensemble` (bool, optional): enable k-fold model averaging during final training, default `false`
- `final_ensemble_k` (int, optional): number of folds for averaging, default `3`
- `final_refit` (bool, optional): enable refit after early stop with full data, default `true`

Note: when `cv_strategy="time"` and a sampling cap is applied (e.g. `bo_sample_limit` or FT unsupervised `max_rows_for_ft_bo`), the subset is chosen in time order (no random sampling).

**FT stacking**

- `ft_role` (str): `"model"` / `"embedding"` / `"unsupervised_embedding"`
  - `"model"`: FT acts as prediction model and outputs `pred_ft`
  - `"embedding"`: FT is supervised but only exports embedding feature columns `pred_<prefix>_*`, not evaluated as final model
  - `"unsupervised_embedding"`: FT uses masked reconstruction pretraining, exports `pred_<prefix>_*`
- `ft_feature_prefix` (str): prefix for exported features (creates `pred_<prefix>_0..`)
- `ft_num_numeric_tokens` (int|null): number of numeric tokens for FT; default equals number of numeric features
- `stack_model_keys` (list[str]): when `ft_role != "model"` and you want base models after FT, specify trainers to run, e.g. `["xgb","resn"]` or `["all"]`

**Parallelism and DDP**

- `use_resn_ddp` / `use_ft_ddp` / `use_gnn_ddp` (bool): use DDP (requires `torchrun`/`nproc_per_node>1`)
- `use_resn_data_parallel` / `use_ft_data_parallel` / `use_gnn_data_parallel` (bool): allow DataParallel as fallback

**Reuse historical best params (skip Optuna)**

- `reuse_best_params` (bool): `true/false`
  - `true`: try `Results/versions/*_<model_key>_best.json` first, else fall back to `Results/<model>_bestparams_*.csv`
  - if not found, runs Optuna normally
- `best_params_files` (dict, optional): explicit best param files, format `{"xgb":"./Results/xxx.csv","ft":"./Results/xxx.json"}`
  - supports `.csv/.tsv` (read first row) and `.json` (`{"best_params": {...}}` or direct dict)
  - if provided, reads directly and skips Optuna

**Optuna resume (recommended)**

- `optuna_storage` (str|null): Optuna storage (sqlite recommended)
  - example: `"./Results/optuna/bayesopt.sqlite3"` (resolved to absolute path)
  - or: `"sqlite:///E:/path/to/bayesopt.sqlite3"` (URL passed as-is)
- `optuna_study_prefix` (str): study name prefix; keep fixed for resuming

**XGBoost search caps (avoid very slow trials)**

- `xgb_max_depth_max` (int): max depth cap, default `25`
- `xgb_n_estimators_max` (int): tree count cap, default `500`

**GNN and geo tokens (optional)**

- `gnn_use_approx_knn` (bool): prefer approximate kNN for large samples
- `gnn_approx_knn_threshold` (int): row threshold to switch to approximate kNN
- `gnn_graph_cache` (str|null): adjacency/graph cache path
- `gnn_max_gpu_knn_nodes` (int): force CPU kNN above this node count (avoid GPU OOM)
- `gnn_knn_gpu_mem_ratio` (float): fraction of free GPU memory allowed for kNN
- `gnn_knn_gpu_mem_overhead` (float): memory overhead multiplier for kNN
- `geo_feature_nmes` (list[str]): raw columns for geo tokens (empty means no geo tokens)
- `region_province_col` / `region_city_col` (str|null): province/city columns (for region_effect features)
- `region_effect_alpha` (float): partial pooling strength (>=0)

**Plotting (optional)**

- `plot_curves` (bool): plot at end of run
- `plot` (dict): recommended unified plot settings
  - `plot.enable` (bool)
  - `plot.n_bins` (int): bin count
  - `plot.oneway` (bool)
  - `plot.lift_models` (list[str]): model keys for lift plots (e.g. `["xgb","resn"]`), empty means all trained models
  - `plot.double_lift` (bool)
  - `plot.double_lift_pairs` (list): supports `["xgb,resn"]` or `[["xgb","resn"]]`

**Standalone plotting (recommended)**

`ins_pricing.plotting` provides plotting utilities decoupled from training. You can use DataFrames or arrays to compare models:

- `plotting.curves`: lift/double lift/ROC/PR/KS/calibration/conversion lift
- `plotting.diagnostics`: loss curve, one-way plots
- `plotting.importance`: feature importance (supports SHAP summary)
- `plotting.geo`: geo heatmaps/contours (with map tiles for heatmap/contour)

Example (standalone):

```python
from ins_pricing.plotting import curves, importance, geo

# Lift / Double Lift
curves.plot_lift_curve(pred, w_act, weight, n_bins=10, save_path="plot/lift.png")
curves.plot_double_lift_curve(pred1, pred2, w_act, weight, n_bins=10, save_path="plot/dlift.png")

# ROC / PR (multi-model comparison)
curves.plot_roc_curves(y_true, {"xgb": pred_xgb, "resn": pred_resn}, save_path="plot/roc.png")
curves.plot_pr_curves(y_true, {"xgb": pred_xgb, "resn": pred_resn}, save_path="plot/pr.png")

# Feature importance
importance.plot_feature_importance({"x1": 0.32, "x2": 0.18}, save_path="plot/importance.png")

# Geo heat/contour
geo.plot_geo_heatmap(df, x_col="lon", y_col="lat", value_col="loss", bins=50, save_path="plot/geo_heat.png")
geo.plot_geo_contour(df, x_col="lon", y_col="lat", value_col="loss", levels=12, save_path="plot/geo_contour.png")

# Map heatmap (requires contextily)
geo.plot_geo_heatmap_on_map(df, lon_col="lon", lat_col="lat", value_col="loss", bins=80, save_path="plot/map_heat.png")
```

Map functions use lat/lon (EPSG:4326) by default and auto-scale view to data bounds.

The training flow also uses this plotting package (`plot_oneway`/`plot_lift`/`plot_dlift`/`plot_conversion_lift`/loss curves) for consistent maintenance.

**Model explanation (standalone module, light + deep)**

`ins_pricing.explain` provides model explanation methods decoupled from training:

- Light: permutation importance (for XGB/ResNet/FT, global)
- Deep: integrated gradients (for ResNet/FT, mainly numeric features)
- Classic: SHAP (KernelExplainer, for GLM/XGB/ResNet/FT, requires `shap`)

SHAP is optional; a prompt appears if not installed.

Example:

```python
from ins_pricing.explain import (
    permutation_importance,
    resnet_integrated_gradients,
    ft_integrated_gradients,
    compute_shap_xgb,
)

# permutation importance
imp = permutation_importance(
    predict_fn=model.predict,
    X=X_valid,
    y=y_valid,
    sample_weight=w_valid,
    metric="rmse",
    n_repeats=5,
)

# ResNet integrated gradients
ig_resn = resnet_integrated_gradients(resn_model, X_valid_scl, steps=50)

# FT integrated gradients (categorical fixed; numeric/geo participate)
ig_ft = ft_integrated_gradients(ft_model, X_valid, geo_tokens=geo_tokens, steps=50)

# SHAP for XGB (BayesOptModel as context)
shap_xgb = compute_shap_xgb(model, n_background=500, n_samples=200, on_train=False)
```

BayesOptModel also provides convenience wrappers:

```python
model.compute_permutation_importance("resn", on_train=False, metric="rmse")
model.compute_integrated_gradients_resn(on_train=False, steps=50)
model.compute_integrated_gradients_ft(on_train=False, steps=50)
model.compute_shap_xgb(on_train=False)
model.compute_shap_glm(on_train=False)
```

**Explain batch via config**

Use `Explain_entry.py` with config to load trained models under `output_dir/model` and run explanations on the validation set:

```bash
python ins_pricing/cli/Explain_entry.py --config-json ins_pricing/examples/modelling/config_explain_template.json
```

Notebook option: `ins_pricing/examples/modelling/Explain_Run.ipynb`.

**Environment variable injection (optional)**

- `env`: values are set via `os.environ.setdefault()` (e.g. thread limits, CUDA debug)

### 6.2 Notebook unified run: runner field (recommended)

All `Pricing_*.ipynb` are thin wrappers: they only call `Pricing_Run.run("<config.json>")`, and the run mode is controlled by config `runner`.

Notebook usage (recommended):

```python
from ins_pricing.cli.Pricing_Run import run
run("examples/modelling/config_template.json")
```

CLI usage (optional):

```bash
python ins_pricing/cli/Pricing_Run.py --config-json ins_pricing/examples/modelling/config_template.json
```

`runner` supports three modes:

- `runner.mode="entry"`: run `BayesOpt_entry.py`
  - `runner.model_keys` (list[str]): `["glm","xgb","resn","ft","gnn"]` or includes `"all"`
  - `runner.nproc_per_node` (int): `1` (single process) or `>=2` (torchrun/DDP)
  - `runner.max_evals` (int): Optuna trials per model (default `50`)
  - `runner.plot_curves` (bool): add `--plot-curves`
  - `runner.ft_role` (str|null): if set, overrides config `ft_role`

- `runner.mode="incremental"`: run `BayesOpt_incremental.py`
  - `runner.incremental_args` (list[str]): equivalent to CLI args for the incremental script
    - common: `--incremental-dir/--incremental-file`, `--merge-keys`, `--timestamp-col`, `--model-keys`, `--max-evals`, `--update-base-data`, `--summary-json`, etc

- `runner.mode="explain"`: run `Explain_entry.py`
  - `runner.explain_args` (list[str]): equivalent to CLI args for the explain script

watchdog (available in both modes):

- `runner.use_watchdog` (bool): enable watchdog
- `runner.idle_seconds` (int): seconds without output to treat as stuck
- `runner.max_restarts` (int): max restarts
- `runner.restart_delay_seconds` (int): delay between restarts

---

## 7. CLI: BayesOpt_entry.py examples

### 7.0 Quick args reference (BayesOpt_entry.py)

Common CLI args for `BayesOpt_entry.py` (`--config-json` is required):

- `--config-json` (required, str): config path (recommend `ins_pricing/examples/modelling/xxx.json` or absolute path)
- `--model-keys` (list[str]): `glm` / `xgb` / `resn` / `ft` / `gnn` / `all`
- `--stack-model-keys` (list[str]): only when `ft_role != model`; same values as `--model-keys`
- `--max-evals` (int): Optuna trials per dataset per model
- `--plot-curves` (flag): enable plotting (also controlled by `plot_curves`/`plot.enable` in config)
- `--output-dir` (str): override config `output_dir`
- `--reuse-best-params` (flag): override config and reuse historical params to skip Optuna

DDP/DP (override config):

- `--use-resn-ddp` / `--use-ft-ddp` / `--use-gnn-ddp` (flag): force DDP for trainer
- `--use-resn-dp` / `--use-ft-dp` / `--use-gnn-dp` (flag): enable DataParallel fallback

GNN graph build (override config):

- `--gnn-no-ann` (flag): disable approximate kNN
- `--gnn-ann-threshold` (int): override `gnn_approx_knn_threshold`
- `--gnn-graph-cache` (str): override `gnn_graph_cache`
- `--gnn-max-gpu-nodes` (int): override `gnn_max_gpu_knn_nodes`
- `--gnn-gpu-mem-ratio` (float): override `gnn_knn_gpu_mem_ratio`
- `--gnn-gpu-mem-overhead` (float): override `gnn_knn_gpu_mem_overhead`

FT feature mode:

- `--ft-role` (str): `model` / `embedding` / `unsupervised_embedding`
- `--ft-feature-prefix` (str): feature prefix (e.g. `ft_emb`)
- `--ft-as-feature` (flag): compatibility alias (if config ft_role is default, set to `embedding`)

### 7.1 Direct train/tune (single machine)

```bash
python ins_pricing/cli/BayesOpt_entry.py ^
  --config-json ins_pricing/examples/modelling/config_template.json ^
  --model-keys xgb resn ^
  --max-evals 50
```

### 7.2 FT stacking: self-supervised FT then base models (single machine or torchrun)

If config already has `ft_role=unsupervised_embedding`, you can omit `--ft-role`.

```bash
python ins_pricing/cli/BayesOpt_entry.py ^
  --config-json "user_packages legacy/Try/config_Pricing_FT_Stack.json" ^
  --model-keys xgb resn ^
  --max-evals 50
```

DDP (multi-GPU) example:

```bash
torchrun --standalone --nproc_per_node=2 ^
  ins_pricing/cli/BayesOpt_entry.py ^
  --config-json "user_packages legacy/Try/config_Pricing_FT_Stack.json" ^
  --model-keys xgb resn ^
  --use-ft-ddp ^
  --max-evals 50
```

### 7.3 Reuse historical best params (skip tuning)

```bash
python ins_pricing/cli/BayesOpt_entry.py ^
  --config-json "user_packages legacy/Try/config_Pricing_FT_Stack.json" ^
  --model-keys xgb resn ^
  --reuse-best-params
```

### 7.4 Quick args reference (BayesOpt_incremental.py)

`BayesOpt_incremental.py` has many args; the common combo is incremental data source + merge/dedupe + models to retrain.

Common args:

- `--config-json` (required, str): reuse the same config (must include `data_dir/model_list/model_categories/target/weight/feature_list/categorical_features`)
- `--model-names` (list[str], optional): update only certain datasets (default uses `model_list x model_categories`)
- `--model-keys` (list[str]): `glm` / `xgb` / `resn` / `ft` / `gnn` / `all`
- `--incremental-dir` (Path) or `--incremental-file` (Path): incremental CSV source (choose one)
- `--incremental-template` (str): filename template for `--incremental-dir` (default `{model_name}_incremental.csv`)
- `--merge-keys` (list[str]): primary keys for dedupe after merge
- `--dedupe-keep` (str): `first` / `last`
- `--timestamp-col` (str|null): timestamp column for ordering before dedupe
- `--timestamp-descending` (flag): descending timestamp (default ascending)
- `--max-evals` (int): trial count when re-tuning is needed
- `--force-retune` (flag): force retune even if historical params exist
- `--skip-retune-missing` (flag): skip if params missing (default re-tunes)
- `--update-base-data` (flag): overwrite base CSV with merged data after success
- `--persist-merged-dir` (Path|null): optionally save merged snapshot to a separate dir
- `--summary-json` (Path|null): output summary
- `--plot-curves` (flag): plot
- `--dry-run` (flag): only merge and stats, no training

---

## 8. Python API: minimal runnable example (recommended to get working first)

This example shows "self-supervised FT embeddings, then XGB" (only key calls shown):

```python
import pandas as pd
from sklearn.model_selection import train_test_split

import ins_pricing.BayesOpt as ropt

df = pd.read_csv("./Data/od_bc.csv")
train_df, test_df = train_test_split(df, test_size=0.25, random_state=13)

model = ropt.BayesOptModel(
    train_df=train_df,
    test_df=test_df,
    model_nme="od_bc",
    resp_nme="reponse",
    weight_nme="weights",
    factor_nmes=[...],          # same as config feature_list
    cate_list=[...],            # same as config categorical_features
    epochs=50,
    use_ft_ddp=False,
    ft_role="unsupervised_embedding",
    ft_feature_prefix="ft_emb",
    output_dir="./Results",
)

# 1) FT masked self-supervised pretrain + export embeddings + inject to factor_nmes
model.optimize_model("ft", max_evals=30)

# 2) Base model tune/train (uses injected pred_ft_emb_* features)
model.optimize_model("xgb", max_evals=50)

# 3) Save (or save one model only)
model.save_model()
```

For time-based splits in Python, keep chronological order and slice:

```python
df = df.sort_values("as_of_date")
cutoff = int(len(df) * 0.75)
train_df = df.iloc[:cutoff]
test_df = df.iloc[cutoff:]
```

### 8.x Tuning stuck / resume (recommended)

If a trial hangs for a long time (e.g. the 17th trial runs for hours), stop the run and add Optuna persistent storage in `config.json`. The next run will resume from completed trials and keep total trials equal to `max_evals`.

Some XGBoost parameter combos can be extremely slow; use the cap fields to narrow the search space.

**config.json example:**
```json
{
  "optuna_storage": "./Results/optuna/pricing.sqlite3",
  "optuna_study_prefix": "pricing",
  "xgb_max_depth_max": 12,
  "xgb_n_estimators_max": 300
}
```

**Continue training with current best params (no tuning)**
- Set `"reuse_best_params": true` in `config.json`: it prefers `Results/versions/*_xgb_best.json` or `Results/<model>_bestparams_xgboost.csv` and trains directly.
- Or specify `"best_params_files"` (by `model_key`) to read from files and skip Optuna:

```json
{
  "best_params_files": {
    "xgb": "./Results/od_bc_bestparams_xgboost.csv",
    "ft": "./Results/od_bc_bestparams_fttransformer.csv"
  }
}
```

**Auto-detect hangs and restart (Watchdog)**
If a trial hangs with no output for hours, use `ins_pricing/cli/watchdog_run.py` to monitor output: when stdout/stderr is idle for `idle_seconds`, it kills the `torchrun` process tree and restarts. With `optuna_storage`, restarts resume remaining trials.

```bash
python ins_pricing/cli/watchdog_run.py --idle-seconds 7200 --max-restarts 50 -- ^
  python -m torch.distributed.run --standalone --nproc_per_node=2 ^
  ins_pricing/cli/BayesOpt_entry.py --config-json config.json --model-keys xgb resn --max-evals 50
```

---

## 9. Model usage examples (CLI and Python)

Examples by model/trainer. All examples follow the same data contract: CSV must include `target/weight/feature_list` columns; categorical columns listed in `categorical_features`.

> Note: `model_key` follows `BayesOpt_entry.py`: `glm` / `xgb` / `resn` / `ft` / `gnn`.

### 9.1 GLM (`model_key="glm"`)

**CLI**

```bash
python ins_pricing/cli/BayesOpt_entry.py ^
  --config-json ins_pricing/examples/modelling/config_template.json ^
  --model-keys glm ^
  --max-evals 50
```

**Python**

```python
model.optimize_model("glm", max_evals=50)
model.trainers["glm"].save()
```

Use case: fast, interpretable baseline and sanity check.

### 9.2 XGBoost (`model_key="xgb"`)

**CLI**

```bash
python ins_pricing/cli/BayesOpt_entry.py ^
  --config-json ins_pricing/examples/modelling/config_template.json ^
  --model-keys xgb ^
  --max-evals 100
```

**Python**

```python
model.optimize_model("xgb", max_evals=100)
model.trainers["xgb"].save()
```

Use case: strong baseline, friendly to feature engineering/stacked features (including FT embeddings).

### 9.3 ResNet (`model_key="resn"`)

ResNetTrainer uses PyTorch, and uses one-hot/standardized views for training and CV (good for high-dimensional one-hot inputs).

**CLI (single machine)**

```bash
python ins_pricing/cli/BayesOpt_entry.py ^
  --config-json ins_pricing/examples/modelling/config_template.json ^
  --model-keys resn ^
  --max-evals 50
```

**CLI (DDP, multi-GPU)**

```bash
torchrun --standalone --nproc_per_node=2 ^
  ins_pricing/cli/BayesOpt_entry.py ^
  --config-json ins_pricing/examples/modelling/config_template.json ^
  --model-keys resn ^
  --use-resn-ddp ^
  --max-evals 50
```

**Python**

```python
model.optimize_model("resn", max_evals=50)
model.trainers["resn"].save()
```

### 9.4 FT-Transformer: as prediction model (`ft_role="model"`)

FT outputs `pred_ft` and participates in lift/SHAP (if enabled).

**CLI**

```bash
python ins_pricing/cli/BayesOpt_entry.py ^
  --config-json ins_pricing/examples/modelling/config_template.json ^
  --model-keys ft ^
  --ft-role model ^
  --max-evals 50
```

**Python**

```python
model.config.ft_role = "model"
model.optimize_model("ft", max_evals=50)
```

### 9.5 FT-Transformer: supervised but export embeddings only (`ft_role="embedding"`)

FT is not evaluated as a standalone model; it writes embedding features (`pred_<prefix>_0..`) and injects them into downstream features.

**CLI (generate features with FT, then train base models)**

```bash
python ins_pricing/cli/BayesOpt_entry.py ^
  --config-json "user_packages legacy/Try/config_Pricing_FT_Stack.json" ^
  --model-keys xgb resn ^
  --ft-role embedding ^
  --max-evals 50
```

**Python**

```python
model.config.ft_role = "embedding"
model.config.ft_feature_prefix = "ft_emb"
model.optimize_model("ft", max_evals=50)      # generate pred_ft_emb_* and inject to factor_nmes
model.optimize_model("xgb", max_evals=100)    # train/tune with injected features
```

### 9.6 FT-Transformer: masked self-supervised pretrain + embeddings (`ft_role="unsupervised_embedding"`)

This is a two-stage stacking mode: representation learning first, base model decision later. Optuna objective is validation loss of masked reconstruction (not `tw_power`).

**CLI (recommended: use sample config)**

```bash
python ins_pricing/cli/BayesOpt_entry.py ^
  --config-json "user_packages legacy/Try/config_Pricing_FT_Stack.json" ^
  --model-keys xgb resn ^
  --max-evals 50
```

**CLI (DDP, multi-GPU)**

```bash
torchrun --standalone --nproc_per_node=2 ^
  ins_pricing/cli/BayesOpt_entry.py ^
  --config-json "user_packages legacy/Try/config_Pricing_FT_Stack.json" ^
  --model-keys xgb resn ^
  --use-ft-ddp ^
  --max-evals 50
```

**Python**

```python
model.config.ft_role = "unsupervised_embedding"
model.config.ft_feature_prefix = "ft_emb"
model.optimize_model("ft", max_evals=50)      # self-supervised pretrain + export pred_ft_emb_*
model.optimize_model("xgb", max_evals=100)
model.optimize_model("resn", max_evals=50)
```

### 9.7 GNN (`model_key="gnn"`) and geo tokens

GNN can run as a standalone model with Optuna tuning/training: it trains on one-hot/standardized features and writes `pred_gnn` / `w_pred_gnn` to `train_data/test_data`.

**CLI**

```bash
python ins_pricing/cli/BayesOpt_entry.py ^
  --config-json ins_pricing/examples/modelling/config_template.json ^
  --model-keys gnn ^
  --max-evals 50
```

GNN can also generate geo tokens: when config includes `geo_feature_nmes`, it trains a geo encoder to produce `geo_token_*` and injects those tokens into FT.

Implementation: geo token generation is handled by `GNNTrainer.prepare_geo_tokens()`. Tokens are stored in `BayesOptModel.train_geo_tokens/test_geo_tokens` and used as FT inputs during training/prediction.

---

## 9. FAQ (quick checks)

### 9.1 torchrun OMP_NUM_THREADS warning

This is a common torchrun message: it sets per-process threads to 1 to avoid CPU overload. You can override it via config `env`.

### 9.2 Optuna loss shows inf

This usually means NaN/inf during training or validation (numeric overflow, data issues, etc). Check:

- data ranges and NaNs (use `nan_to_num`, scaling)
- learning rate and AMP (reduce LR or disable AMP)
- gradient clipping (already enabled for torch models)
- unstable configs (cap XGBoost depth/estimators)
