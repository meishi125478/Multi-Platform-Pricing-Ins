# Changes to Example Notebooks and Configurations

## Summary

Updated all example notebooks and training configurations to:
1. **Simplified directory structure**: Moved all notebooks and configs from `examples/modelling/` to `examples/` root
2. Use relative paths compatible with pip-installed packages
3. Separate training and plotting workflows for better performance
4. Organize plots by category automatically

## Date: 2026-01-15

---

## 0. Directory Structure Simplification

### Change
Moved all example files from `examples/modelling/` to `examples/` root directory.

### New Structure
```
ins_pricing/examples/
├── *.ipynb                      # All notebook templates
├── config_*.json                # All configuration files
├── README.md                    # Main documentation
├── TEMPLATE_GUIDE.md           # Usage guide
├── CHANGES.md                  # This file
└── CHECKLIST.md                # Verification checklist
```

### Benefits
- **Simpler path**: Copy from `examples/` instead of `examples/modelling/`
- **Cleaner structure**: All templates in one place
- **Easier discovery**: No nested directories to navigate

---

## 1. Path Compatibility Updates

### Problem
- Examples used hardcoded paths like `repo_root / 'ins_pricing/examples/...'`
- Failed when package installed via pip and run from different directories

### Solution
Changed all notebooks to use relative paths from current working directory:

```python
# Before
repo_root = Path.cwd()
# Find ins_pricing directory...
cfg_path = repo_root / 'ins_pricing/examples/config.json'

# After
work_dir = Path.cwd()
cfg_path = work_dir / 'config.json'
```

### Modified Files
- [Plot_LoadModel.ipynb](Plot_LoadModel.ipynb)
- [Plot_Oneway_Pre.ipynb](Plot_Oneway_Pre.ipynb)
- [Explain_Run.ipynb](Explain_Run.ipynb)
- [Train_FT_Embed_XGBResN.ipynb](Train_FT_Embed_XGBResN.ipynb)
- [Predict_FT_Embed_XGB.ipynb](Predict_FT_Embed_XGB.ipynb)
- [PricingSingle.ipynb](PricingSingle.ipynb)

---

## 2. Training Plot Optimization

### Problem
- Training generated expensive evaluation plots (oneway, lift, double-lift)
- Slowed down training significantly
- Plots often needed regeneration with different settings anyway

### Solution
Disabled evaluation plots in training configurations while keeping loss curves:

```json
{
  "plot_curves": false,
  "plot": {
    "enable": false,
    "oneway": false,
    "oneway_pred": false,
    "pre_oneway": false,
    "lift_models": [],
    "double_lift": false
  }
}
```

### Plot Types

**1. Training Loss Curves** (Always Generated)
- Location: `Results/plot/{model_name}/loss/`
- Generated for: ResNet, FT Transformer models
- Purpose: Monitor training convergence
- Not affected by `plot_curves` setting

**2. Evaluation Plots** (Optional, Disabled by Default)
- Location: `Results/plot/{model_name}/oneway/`, `lift/`, `double_lift/`
- Generated for: All models when `plot_curves: true`
- Purpose: Comprehensive model evaluation
- **Recommended**: Generate separately using plotting notebooks

### Modified Configuration Files
- [config_template.json](config_template.json)
- [config_ft_unsupervised_xgb.json](config_ft_unsupervised_xgb.json)
- [config_ft_unsupervised_resn.json](config_ft_unsupervised_resn.json)
- [config_ft_unsupervised_ddp_embed.json](config_ft_unsupervised_ddp_embed.json)

### Workflow Changes

**Old Workflow**:
```
Train Model (with plots) → Takes 2x time → Often regenerate plots anyway
```

**New Workflow**:
```
Train Model (fast) → Generate Loss Curves → Use plotting notebooks for detailed analysis
```

---

## 3. Plot Organization

### Structure
Plots are automatically organized by type when using `plot_path_style: "nested"`:

```
Results/plot/{model_name}/
├── loss/              # Training loss curves (auto)
├── oneway/
│   ├── pre/          # Pre-model feature analysis
│   └── post/         # Post-model with predictions
├── lift/             # Lift curves
└── double_lift/      # Model comparisons
```

### Plotting Notebooks

Use these dedicated notebooks for comprehensive visualization:

1. **[Plot_Oneway_Pre.ipynb](Plot_Oneway_Pre.ipynb)**
   - Pre-model feature analysis
   - Fast, useful before training

2. **[Plot_LoadModel.ipynb](Plot_LoadModel.ipynb)**
   - Load saved models
   - Generate all plot types
   - Full control over visualization

3. **[Explain_Run.ipynb](Explain_Run.ipynb)**
   - Model interpretation
   - Feature importance analysis

---

## 4. Benefits

### Performance
- **Faster training**: No expensive plot generation during training
- **Parallel development**: Train models while preparing visualizations
- **Flexible plotting**: Regenerate plots with different settings without retraining

### Usability
- **Pip compatible**: Works after `pip install ins_pricing`
- **Location independent**: Run from any directory
- **Better organization**: Plots categorized automatically

### Workflow
- **Separation of concerns**: Training focuses on training, plotting on visualization
- **Iterative analysis**: Easy to regenerate plots with different parameters
- **Loss monitoring**: Always get training curves for convergence checking

---

## 5. Migration Guide

### For Existing Users

**If you have custom configs**:
1. Set `"plot_curves": false` to speed up training
2. Loss curves still generated automatically
3. Use plotting notebooks for evaluation plots

**If you have custom notebooks**:
1. Replace `repo_root` logic with `work_dir = Path.cwd()`
2. Update paths to be relative: `work_dir / 'config.json'`
3. Update paths to be relative: `work_dir / 'Data/file.csv'`

**If you need plots during training**:
- Set `"plot_curves": true` in your config
- All plot types will be generated (slower)
- Loss curves generated regardless

### Example Migration

**Before**:
```python
repo_root = Path.cwd()
if not (repo_root / 'ins_pricing').exists():
    for parent in repo_root.parents:
        if (parent / 'ins_pricing').exists():
            repo_root = parent
            sys.path.insert(0, str(repo_root))
            break

cfg_path = repo_root / 'ins_pricing/examples/config.json'
```

**After**:
```python
work_dir = Path.cwd()
cfg_path = work_dir / 'config.json'
```

---

## 6. Documentation

Added comprehensive [README.md](README.md) covering:
- Setup instructions for pip-installed package
- Directory structure recommendations
- Plot organization details
- Training vs. evaluation plot distinction
- Troubleshooting common issues

---

## Backward Compatibility

✅ **Fully backward compatible**:
- Old configs with `plot_curves: true` still work
- Loss curves always generated (no change)
- All existing functionality preserved
- Only defaults changed for better performance

❌ **Path changes require updates**:
- Notebooks using old `repo_root` pattern need updating
- Recommended to update to relative paths for pip compatibility
