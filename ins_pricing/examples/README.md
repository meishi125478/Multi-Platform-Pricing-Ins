# Insurance Pricing Modelling Examples

This directory contains **ready-to-use template notebooks** for training and evaluating insurance pricing models.

**Quick Start**: 📖 [Template Usage Guide](TEMPLATE_GUIDE.md) - Copy, modify parameters, run!

---

## Recent Updates (2026-01)

### New Config-Based API (v0.2.10+)

**🎉 Simplified API**: `BayesOptModel` now accepts a single configuration object instead of 56 individual parameters!

**Before** (Old API):
```python
model = BayesOptModel(
    train_data, test_data,
    model_nme="my_model", resp_nme="response", weight_nme="weights",
    factor_nmes=features, cate_list=categorical_features,
    prop_test=0.25, rand_seed=13, epochs=50, use_gpu=True,
    # ... 40+ more parameters!
)
```

**After** (New API):
```python
config = BayesOptConfig(
    model_nme="my_model", resp_nme="response", weight_nme="weights",
    factor_nmes=features, cate_list=categorical_features,
    prop_test=0.25, rand_seed=13, epochs=50, use_gpu=True
)
model = BayesOptModel(train_data, test_data, config=config)
```

**Benefits**: Cleaner code, reusable configurations, better IDE support, type safety

📖 **Full Guide**: [NEW_API_EXAMPLES.md](NEW_API_EXAMPLES.md) - Complete examples and migration guide

### Path Compatibility Changes

All example notebooks have been updated to work correctly when the package is installed via pip:

- **Before**: Used `repo_root` and hardcoded paths like `repo_root / 'ins_pricing/examples/...'`
- **After**: Use `work_dir = Path.cwd()` and relative paths from the current working directory

This means you can now:
1. Install the package via pip: `pip install ins_pricing`
2. Copy the example notebooks and config files to any directory
3. Run them from that directory without path conflicts

### Plot Organization

All plots are automatically organized into subdirectories by type:

```
Results/
└── plot/
    └── {model_name}/
        ├── loss/             # Training loss curves (always generated)
        ├── oneway/
        │   ├── pre/          # Pre-model oneway plots
        │   └── post/         # Post-model oneway plots with predictions
        ├── lift/             # Lift curve plots
        └── double_lift/      # Double lift comparison plots
```

Plot directories are automatically created when plots are saved.

#### Training vs. Evaluation Plots

The plotting system has two types of plots with different control mechanisms:

1. **Training Loss Curves** (`plot/{model_name}/loss/`)
   - **Always generated** during training for ResNet and FT Transformer models
   - Shows training and validation loss over epochs
   - Not controlled by `plot_curves` or `plot.enable` settings
   - Useful for monitoring training convergence

2. **Evaluation Plots** (`plot/{model_name}/oneway/`, `lift/`, etc.)
   - Generated only when `plot_curves: true` or `plot.enable: true`
   - Includes oneway analysis, lift curves, double-lift comparisons
   - Can be time-consuming for large datasets
   - **Recommended approach**:
     - Set `"plot_curves": false` during training
     - Use dedicated plotting notebooks ([Plot_LoadModel.ipynb](Plot_LoadModel.ipynb), [Plot_Oneway_Pre.ipynb](Plot_Oneway_Pre.ipynb)) after training

This separation allows you to:
- Monitor training progress via loss curves
- Skip expensive evaluation plots during training
- Generate comprehensive visualizations later with full control

## Quick Start (No Config Files)

The simplest way to get started - no configuration files needed!

### 1. Pre-model Analysis

Open [Plot_Oneway_Pre.ipynb](Plot_Oneway_Pre.ipynb):
```python
# Just set these variables in the notebook
use_config_file = False
data_path = work_dir / 'Data/od_bc.csv'
model_name = 'od_bc'
target_col = 'response'
weight_col = 'weights'
feature_list = ['age_owner', 'gender_owner', ...]  # Your features
```

Run the notebook - plots will be generated without any config file!

### 2. Train Models

Use [PricingSingle.ipynb](PricingSingle.ipynb) with minimal setup:
```python
data_dir = work_dir / 'Data'
feature_list = [...]  # Your features
categorical_features = [...]  # Your categorical features
```

### 3. Post-model Visualization

Coming soon: simplified mode for [Plot_LoadModel.ipynb](Plot_LoadModel.ipynb) without config files.

---

## Full Usage Guide

For more advanced workflows with configuration files:

### 1. Prepare Your Environment

If installed via pip:
```bash
# Create a working directory
mkdir my_pricing_project
cd my_pricing_project

# Copy example files (config and notebooks) here
# Or create your own config files
```

If running from source:
```bash
cd ins_pricing/examples
```

### 2. Prepare Your Data

Place your data files in a `Data` subdirectory:
```
my_pricing_project/
├── Data/
│   └── od_bc.csv
├── config_template.json
└── PricingSingle.ipynb
```

### 3. Run the Examples

All notebooks now work relative to your current directory:
- Configuration files: `work_dir / 'config_name.json'`
- Data files: `work_dir / 'Data/model_name.csv'`
- Output: `work_dir / 'Results/'`

## Available Notebooks

### Training Examples

- **[PricingSingle.ipynb](PricingSingle.ipynb)**: Complete single-model training pipeline
  - Trains XGBoost, ResNet, and FT Transformer models
  - Generates lift curves and other evaluation plots

- **[Train_FT_Embed_XGBResN.ipynb](Train_FT_Embed_XGBResN.ipynb)**: Two-step training with embeddings
  - Step 1: Train FT Transformer to generate embeddings
  - Step 2: Train XGBoost/ResNet on augmented data with embeddings

### Prediction Examples

- **[Predict_FT_Embed_XGB.ipynb](Predict_FT_Embed_XGB.ipynb)**: Load saved models and predict on new data
  - Loads FT model to generate embeddings
  - Loads XGB/ResNet model for final predictions

### Analysis Examples

- **[Plot_Oneway_Pre.ipynb](Plot_Oneway_Pre.ipynb)**: Pre-model oneway analysis
  - **No config file required** - specify data path and features directly
  - Optional config mode for consistency with training
  - Analyze feature distributions before modeling
  - Plots saved to `Results/plot/{model_name}/oneway/pre/`

- **[Plot_LoadModel.ipynb](Plot_LoadModel.ipynb)**: Post-model visualization
  - **Two modes**: Simple (direct paths) or Config (FT->XGB/ResNet workflows)
  - Load saved models and generate comprehensive plots
  - Includes oneway, lift, and double-lift curves
  - Plots organized by type in subdirectories

- **[Explain_Run.ipynb](Explain_Run.ipynb)**: Model explanation and interpretation
  - Supports permutation importance, SHAP, integrated gradients
  - Works with all model types (XGBoost, ResNet, FT)

## Configuration Files

All notebooks use JSON configuration files. Key settings:

```json
{
  "data_dir": "./Data",
  "output_dir": "./Results",
  "plot_path_style": "nested",  // Options: "nested" (default), "flat"

  // Training: disable evaluation plots, keep loss curves
  "plot_curves": false,         // Set to false during training
  "plot": {
    "enable": false,            // Disable expensive evaluation plots
    "n_bins": 10,
    "oneway": false,
    "oneway_pred": false,
    "pre_oneway": false,
    "lift_models": [],
    "double_lift": false
  }
}
```

**Note**: Loss curves for neural network models (ResNet, FT Transformer) are always generated regardless of these settings.

### Plot Path Styles

- **nested** (default): Organizes plots by type: `plot/{model}/oneway/`, `plot/{model}/lift/`
- **flat**: All plots in one directory: `plot/`

## Tips

1. **Working Directory**: All paths are relative to where you run the notebook
2. **Data Location**: Always use `Data/` subdirectory for data files
3. **Output Location**: Results are saved to `Results/` by default
4. **Plot Organization**: Plots are automatically organized by type when using `plot_path_style: "nested"`
5. **Config Files**: Keep config files in the same directory as notebooks for simplest setup

## Example Directory Structure

After running the examples:

```
my_pricing_project/
├── Data/
│   └── od_bc.csv
├── config_template.json
├── config_xgb_from_ft_unsupervised.json (auto-generated)
├── PricingSingle.ipynb
├── Plot_LoadModel.ipynb
└── Results/
    ├── model/                    # Saved models
    ├── predictions/              # Prediction cache
    ├── optuna/                   # Hyperparameter tuning results
    ├── reports/                  # Analysis reports
    └── plot/
        └── od_bc/
            ├── loss/            # Training loss curves (auto-generated)
            ├── oneway/
            │   ├── pre/         # Feature analysis before modeling
            │   └── post/        # Feature impact after modeling
            ├── lift/            # Model performance curves
            └── double_lift/     # Model comparison curves
```

## Troubleshooting

**Q: Config file not found**
- Make sure config files are in the same directory where you run the notebook
- Or adjust the path: `cfg_path = work_dir / 'configs' / 'config.json'`

**Q: Data file not found**
- Data should be in a `Data/` subdirectory of your working directory
- Or adjust `data_dir` in the config file

**Q: Plots not organized by type**
- Set `"plot_path_style": "nested"` in your config file
- Flat style puts all plots in one directory

**Q: Want to speed up training**
- Set `"plot_curves": false` in config to skip evaluation plots during training
- Loss curves will still be generated for neural networks
- Generate evaluation plots later using [Plot_LoadModel.ipynb](Plot_LoadModel.ipynb)

**Q: Permission errors on Windows**
- Run notebooks from a directory you own (not in Program Files)
- Avoid spaces in directory paths
