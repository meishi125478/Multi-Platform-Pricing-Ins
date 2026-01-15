# New Config-Based API Examples (v0.2.10+)

**Updated**: 2026-01-15
**New in**: v0.2.10

This guide shows how to use the new simplified config-based API for `BayesOptModel`.

---

## Why the New API?

**Before (Old API - 56 parameters)**:
```python
model = BayesOptModel(
    train_data, test_data,
    model_nme="my_model",
    resp_nme="response",
    weight_nme="weights",
    factor_nmes=feature_list,
    cate_list=categorical_features,
    prop_test=0.25,
    rand_seed=13,
    epochs=50,
    use_gpu=True,
    use_resn_ddp=False,
    use_ft_ddp=False,
    use_gnn_ddp=False,
    output_dir="./Results",
    # ... 40+ more parameters!
)
```

**After (New API - Clean and Simple)**:
```python
config = BayesOptConfig(
    model_nme="my_model",
    resp_nme="response",
    weight_nme="weights",
    factor_nmes=feature_list,
    cate_list=categorical_features,
    prop_test=0.25,
    rand_seed=13,
    epochs=50,
    use_gpu=True,
    output_dir="./Results"
)

model = BayesOptModel(train_data, test_data, config=config)
```

**Benefits**:
- ✅ Cleaner code - configuration separate from instantiation
- ✅ Reusable configurations - save/load/modify configs easily
- ✅ Better IDE support - auto-completion for all parameters
- ✅ Type safety - validation at config creation time

---

## Example 1: Basic Usage

```python
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from ins_pricing.modelling.core.bayesopt import BayesOptModel, BayesOptConfig

# Load data
work_dir = Path.cwd()
data_path = work_dir / 'Data/od_bc.csv'
raw = pd.read_csv(data_path)

# Split data
train_data, test_data = train_test_split(
    raw, test_size=0.25, random_state=13
)

# Define features
feature_list = [
    'age_owner', 'gender_owner', 'plt_zone',
    'cheling_year', 'carbrand', 'price', ...
]

categorical_features = [
    'gender_owner', 'plt_zone', 'carbrand', ...
]

# Create configuration
config = BayesOptConfig(
    model_nme="od_bc",
    resp_nme="response",
    weight_nme="weights",
    factor_nmes=feature_list,
    cate_list=categorical_features,
    task_type="regression",

    # Training settings
    prop_test=0.25,
    rand_seed=13,
    epochs=50,
    use_gpu=True,

    # Output
    output_dir=str(work_dir / "Results")
)

# Create model (much cleaner!)
model = BayesOptModel(train_data, test_data, config=config)

# Train models
model.bayesopt_xgb(max_evals=100)
model.trainers['xgb'].save()

model.bayesopt_resnet(max_evals=50)
model.trainers['resn'].save()

model.bayesopt_ft(max_evals=50)
model.trainers['ft'].save()
```

---

## Example 2: Reusable Configurations

```python
from ins_pricing.modelling.core.bayesopt import BayesOptConfig

# Create base configuration
base_config = BayesOptConfig(
    model_nme="base_model",
    resp_nme="response",
    weight_nme="weights",
    factor_nmes=feature_list,
    cate_list=categorical_features,
    epochs=50,
    use_gpu=True,
    output_dir="./Results"
)

# Experiment 1: Default settings
model1 = BayesOptModel(train_data, test_data, config=base_config)

# Experiment 2: Enable DDP for faster training
from dataclasses import replace

config_ddp = replace(
    base_config,
    model_nme="model_with_ddp",
    use_resn_ddp=True,
    use_ft_ddp=True
)
model2 = BayesOptModel(train_data, test_data, config=config_ddp)

# Experiment 3: More epochs, ensemble
config_ensemble = replace(
    base_config,
    model_nme="model_ensemble",
    epochs=100,
    final_ensemble=True,
    final_ensemble_k=5
)
model3 = BayesOptModel(train_data, test_data, config=config_ensemble)
```

---

## Example 3: Loading Configuration from JSON

**Good News**: The existing JSON config files in `ins_pricing/examples/` already match `BayesOptConfig` structure perfectly! No changes needed.

```python
import json
from ins_pricing.modelling.core.bayesopt import BayesOptConfig, BayesOptModel

# Load from existing config files (config_quicktest.json, config_template.json, etc.)
with open('config_quicktest.json', 'r') as f:
    config_dict = json.load(f)

# Create config from dictionary (existing configs work as-is!)
config = BayesOptConfig(
    model_nme=config_dict['model_list'][0],  # Extract from config
    resp_nme=config_dict['target'],
    weight_nme=config_dict['weight'],
    factor_nmes=config_dict['feature_list'],
    cate_list=config_dict['categorical_features'],
    epochs=config_dict['epochs'],
    use_gpu=config_dict['use_gpu'],
    output_dir=config_dict['output_dir'],
    # All other fields can be passed from config_dict
    **{k: v for k, v in config_dict.items() if k not in [
        'data_dir', 'model_list', 'model_categories', 'target', 'weight',
        'feature_list', 'plot', 'env', 'runner'  # Filter out non-BayesOptConfig fields
    ]}
)

model = BayesOptModel(train_data, test_data, config=config)
```

**Tip**: Your existing JSON config files (like `config_quicktest.json`, `config_template.json`) are already compatible! The field names in these files match `BayesOptConfig` parameters.

---

## Example 4: Updated PricingSingle.ipynb Pattern

Here's how to update your existing notebooks:

**Old pattern**:
```python
model_basic = BayesOptModel(
    train_data, test_data,
    model, tgt, wght, feature_list,
    cate_list=categorical_features,
    prop_test=0.25,
    rand_seed=rand_seed,
    epochs=50,
    use_resn_data_parallel=False,
    use_ft_data_parallel=False,
    use_gnn_data_parallel=False,
    use_resn_ddp=False,
    use_ft_ddp=False,
    use_gnn_ddp=False,
    output_dir=str(output_dir),
)
```

**New pattern**:
```python
config = BayesOptConfig(
    model_nme=model,
    resp_nme=tgt,
    weight_nme=wght,
    factor_nmes=feature_list,
    cate_list=categorical_features,
    prop_test=0.25,
    rand_seed=rand_seed,
    epochs=50,
    use_resn_data_parallel=False,
    use_ft_data_parallel=False,
    use_gnn_data_parallel=False,
    use_resn_ddp=False,
    use_ft_ddp=False,
    use_gnn_ddp=False,
    output_dir=str(output_dir),
)

model_basic = BayesOptModel(train_data, test_data, config=config)
```

---

## Example 5: Cross-Validation Strategies

```python
# Strategy 1: Random split (default)
config_random = BayesOptConfig(
    model_nme="cv_random",
    resp_nme="response",
    weight_nme="weights",
    factor_nmes=feature_list,
    cv_strategy="random",
    cv_splits=5
)

# Strategy 2: Stratified (for classification)
config_stratified = BayesOptConfig(
    model_nme="cv_stratified",
    resp_nme="response",
    weight_nme="weights",
    factor_nmes=feature_list,
    task_type="classification",
    cv_strategy="stratified",
    cv_splits=5
)

# Strategy 3: Grouped (for hierarchical data)
config_grouped = BayesOptConfig(
    model_nme="cv_grouped",
    resp_nme="response",
    weight_nme="weights",
    factor_nmes=feature_list,
    cv_strategy="grouped",
    cv_splits=5,
    cv_group_col="policy_id"  # Group by policy
)

# Strategy 4: Time series
config_timeseries = BayesOptConfig(
    model_nme="cv_timeseries",
    resp_nme="response",
    weight_nme="weights",
    factor_nmes=feature_list,
    cv_strategy="timeseries",
    cv_splits=5,
    cv_time_col="policy_date",
    cv_time_ascending=True
)
```

---

## Example 6: Advanced Settings

```python
config = BayesOptConfig(
    # Basic settings
    model_nme="advanced_model",
    resp_nme="response",
    weight_nme="weights",
    factor_nmes=feature_list,
    cate_list=categorical_features,

    # Training
    epochs=100,
    use_gpu=True,

    # Distributed training (requires torchrun)
    use_resn_ddp=True,
    use_ft_ddp=True,

    # Optuna settings (for checkpoint resume)
    optuna_storage="sqlite:///optuna_study.db",
    optuna_study_prefix="my_experiment",

    # Reuse previous best parameters
    reuse_best_params=True,
    best_params_files={
        'xgb': './Results/xgb_params.json',
        'resn': './Results/resn_params.json'
    },

    # Final model settings
    final_ensemble=True,  # Average predictions from K models
    final_ensemble_k=5,   # Number of models to average
    final_refit=True,     # Refit on full dataset

    # XGBoost constraints
    xgb_max_depth_max=25,
    xgb_n_estimators_max=500,

    # GNN settings
    gnn_use_approx_knn=True,
    gnn_approx_knn_threshold=50000,
    gnn_graph_cache="./cache/gnn_graph.pkl",

    # FT Transformer settings
    ft_num_numeric_tokens=None,  # Auto-detect
    ft_role="model",  # or "embedding"

    # Output
    output_dir="./Results",
    save_preprocess=True,
    preprocess_artifact_path="./preprocessor.pkl"
)

model = BayesOptModel(train_data, test_data, config=config)
```

---

## Example 7: Config Modification

```python
# Start with a base configuration
config = BayesOptConfig(
    model_nme="base",
    resp_nme="response",
    weight_nme="weights",
    factor_nmes=feature_list,
    epochs=50
)

# Modify specific fields for experiments
config.epochs = 100  # Increase epochs
config.final_ensemble = True  # Enable ensemble
config.use_resn_ddp = True  # Enable DDP

# Create model with modified config
model = BayesOptModel(train_data, test_data, config=config)
```

---

## Complete Working Example

```python
#!/usr/bin/env python3
"""Complete example using new config-based API."""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from ins_pricing.modelling.core.bayesopt import BayesOptModel, BayesOptConfig

# Setup paths
work_dir = Path.cwd()
data_dir = work_dir / 'Data'
output_dir = work_dir / 'Results'
output_dir.mkdir(exist_ok=True)

# Load data
print("Loading data...")
raw = pd.read_csv(data_dir / 'od_bc.csv')
raw.fillna(0, inplace=True)

# Split data
train_data, test_data = train_test_split(
    raw, test_size=0.25, random_state=13
)

# Define features
feature_list = [
    'age_owner', 'gender_owner', 'plt_zone',
    'cheling_year', 'carbrand', 'carkind',
    'cartype', 'price', 'seat_num'
]

categorical_features = [
    'gender_owner', 'plt_zone', 'carbrand',
    'carkind', 'cartype'
]

# Create configuration
print("Creating configuration...")
config = BayesOptConfig(
    model_nme="od_bc",
    resp_nme="response",
    weight_nme="weights",
    factor_nmes=feature_list,
    cate_list=categorical_features,
    task_type="regression",
    prop_test=0.25,
    rand_seed=13,
    epochs=50,
    use_gpu=True,
    output_dir=str(output_dir)
)

# Create model
print("Creating BayesOptModel...")
model = BayesOptModel(train_data, test_data, config=config)

# Train XGBoost
print("Training XGBoost...")
model.bayesopt_xgb(max_evals=100)
model.trainers['xgb'].save()
print("✓ XGBoost model saved")

# Train ResNet
print("Training ResNet...")
model.bayesopt_resnet(max_evals=50)
model.trainers['resn'].save()
print("✓ ResNet model saved")

# Train FT Transformer
print("Training FT Transformer...")
model.bayesopt_ft(max_evals=50)
model.trainers['ft'].save()
print("✓ FT Transformer model saved")

print(f"\n✓ All models saved to: {output_dir}")
```

---

## Migration Checklist

When updating your notebooks to the new API:

- [ ] Import `BayesOptConfig` from `ins_pricing.modelling.core.bayesopt`
- [ ] Create a config object with all your settings
- [ ] Replace multi-parameter `BayesOptModel(...)` with `BayesOptModel(train, test, config=config)`
- [ ] Test that your notebook still works
- [ ] Enjoy cleaner, more maintainable code!

---

## Important Note About JSON Config Files

**The existing JSON config files do NOT need to be changed!**

Files like `config_quicktest.json`, `config_template.json`, `config_ft_unsupervised_*.json` already have the correct structure that maps to `BayesOptConfig` fields. They were designed this way from the start and continue to work perfectly with the new API.

**What this means**:
- ✅ Existing JSON configs work as-is
- ✅ No need to update config files
- ✅ Just update how you load them in your code (see Example 3)

## FAQ

### Q: Do I have to update my old notebooks?

**A**: No! The old API still works (with a deprecation warning). Update at your convenience.

### Q: How do I suppress the deprecation warning?

**A**: Just update to the new API! The warning disappears when you use the config parameter.

### Q: Can I save my configuration for later?

**A**: Yes! Convert to dict and save as JSON:
```python
import json
from dataclasses import asdict

config_dict = asdict(config)
with open('config.json', 'w') as f:
    json.dump(config_dict, f, indent=2)
```

### Q: Do I need to update my existing JSON config files?

**A**: No! The existing config files (`config_quicktest.json`, `config_template.json`, etc.) already match the `BayesOptConfig` structure. They work perfectly as-is.

### Q: Where can I find all available configuration parameters?

**A**: Check [config_preprocess.py](../modelling/core/bayesopt/config_preprocess.py) for the full `BayesOptConfig` dataclass definition.

---

## See Also

- [PHASE2_REFACTORING_SUMMARY.md](../modelling/core/bayesopt/PHASE2_REFACTORING_SUMMARY.md) - Technical details of the API change
- [TEMPLATE_GUIDE.md](TEMPLATE_GUIDE.md) - General template usage guide
- [README.md](README.md) - Examples directory overview

---

**End of New API Examples**
