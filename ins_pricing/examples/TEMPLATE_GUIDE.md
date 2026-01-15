# Template Usage Guide

All notebooks in this folder are **ready-to-use templates**. Just copy, modify parameters, and run!

## 📋 Template Philosophy

These examples are designed as **copy-paste templates**:
1. ✅ Copy notebook to your working directory
2. ✅ Modify parameters in the "CONFIGURATION" section
3. ✅ Run all cells
4. ✅ Done!

**No complex setup required** - just modify the clearly marked configuration sections.

---

## 📁 Available Templates

### 1. Pre-model Analysis

**File**: [Plot_Oneway_Pre.ipynb](Plot_Oneway_Pre.ipynb)

**What it does**: Analyze feature distributions before modeling

**How to use**:
```python
# Just modify these in the CONFIGURATION section:
data_path = work_dir / 'Data/your_data.csv'
model_name = 'your_model'
target_col = 'your_target'
weight_col = 'your_weight'
feature_list = ['feature1', 'feature2', ...]
categorical_features = ['cat1', 'cat2', ...]
```

**Output**: Oneway plots in `ResultsOnewayPre/plot/{model_name}/oneway/pre/`

---

### 2. Model Training (Simple)

**File**: [PricingSingle.ipynb](PricingSingle.ipynb)

**What it does**: Train XGBoost, ResNet, and FT Transformer models

**How to use**:
```python
# Modify these parameters:
data_dir = work_dir / 'Data'
model_list = ['od']
model_cate = ['bc']
feature_list = [...]
categorical_features = [...]
```

**Output**:
- Models in `Results/model/`
- Loss curves in `Results/plot/{model}/loss/`

---

### 3. Two-step Training (FT → XGB/ResNet)

**File**: [Train_FT_Embed_XGBResN.ipynb](Train_FT_Embed_XGBResN.ipynb)

**What it does**:
1. Train FT to generate embeddings
2. Train XGB/ResNet on augmented data with embeddings

**How to use**:
```python
# Uses config files - modify config_ft_unsupervised_ddp_embed.json
# Then run the notebook - it will:
# 1. Run FT training
# 2. Generate configs for XGB/ResNet
# 3. Run XGB/ResNet training
```

**Output**:
- FT embeddings in `DataFTUnsupervised/`
- Models in respective output directories
- Auto-generated configs for step 2

---

### 4. Prediction

**File**: [Predict_FT_Embed_XGB.ipynb](Predict_FT_Embed_XGB.ipynb)

**What it does**: Use trained FT+XGB/ResNet models to predict new data

**How to use**:
```python
# Modify these in the CONFIGURATION section:
ft_cfg_path = work_dir / 'config_ft_unsupervised_ddp_embed.json'
xgb_cfg_path = work_dir / 'config_xgb_from_ft_unsupervised.json'
input_path = work_dir / 'Data/new_data.csv'
output_path = work_dir / 'Results/predictions.csv'
model_keys = ["xgb", "resn"]  # Which models to use
```

**Output**: Predictions CSV with `pred_xgb` and/or `pred_resn` columns

---

### 5. Post-model Visualization

**File**: [Plot_LoadModel.ipynb](Plot_LoadModel.ipynb)

**What it does**: Load saved models and generate comprehensive plots

**How to use**:
```python
# Modify config paths in CONFIGURATION section:
cfg_path = work_dir / 'config_xgb_from_ft_unsupervised.json'
xgb_cfg_path = work_dir / 'config_xgb_from_ft_unsupervised.json'
resn_cfg_path = work_dir / 'config_resn_from_ft_unsupervised_ddp.json'
ft_cfg_path = work_dir / 'config_ft_unsupervised_ddp_embed.json'
```

**Output**:
- Oneway plots in `plot/{model}/oneway/post/`
- Lift curves in `plot/{model}/lift/`
- Double-lift in `plot/{model}/double_lift/`

---

### 6. Model Explanation

**File**: [Explain_Run.ipynb](Explain_Run.ipynb)

**What it does**: Generate model explanations (permutation, SHAP, etc.)

**How to use**:
```python
# Modify these:
config_path = Path("config_explain_template.json")
model_keys = ["xgb"]  # Models to explain
methods = ["permutation"]  # Explanation methods
```

**Output**: Explanation reports and plots in output directory

---

## 🎯 Common Workflow

### Scenario 1: Quick Data Exploration
```
1. Copy Plot_Oneway_Pre.ipynb
2. Set data_path and features
3. Run → Get oneway plots
```

### Scenario 2: Simple Model Training
```
1. Copy PricingSingle.ipynb
2. Set data_dir and features
3. Run → Get trained models + loss curves
4. Copy Plot_LoadModel.ipynb for detailed plots
```

### Scenario 3: Advanced Two-step Training
```
1. Copy config_ft_unsupervised_ddp_embed.json
2. Modify data_dir, features, output_dir
3. Copy Train_FT_Embed_XGBResN.ipynb
4. Run → Get FT embeddings + XGB/ResNet models
5. Copy Plot_LoadModel.ipynb for visualization
6. Copy Predict_FT_Embed_XGB.ipynb for predictions
```

---

## 📝 Configuration Sections

All notebooks have clearly marked configuration sections:

```python
# ============================================================
# CONFIGURATION - Modify these parameters for your data
# ============================================================

# Your parameters here
data_path = work_dir / 'Data/your_data.csv'
model_name = 'your_model'
# ... more parameters ...

# ============================================================
# PROCESSING (No changes needed below)
# ============================================================

# Template code - don't modify unless needed
```

**Rule of thumb**: Only modify what's in the `CONFIGURATION` section!

---

## 🔧 Customization Tips

### Minimal Changes (Most Common)
Just modify:
- `data_path` / `data_dir`
- `feature_list`
- `categorical_features`
- `target_col` / `weight_col`
- `model_name`

### Advanced Changes
You can also modify:
- `n_bins` - Number of bins for plots
- `holdout_ratio` - Train/test split ratio
- `output_dir` - Where to save results
- `rand_seed` - Random seed for reproducibility

### Expert Changes
For special cases, modify the processing sections:
- Custom data preprocessing
- Special feature engineering
- Custom plot styling
- Different model architectures

---

## 💡 Tips for Template Usage

### 1. **Directory Structure**
Always work in a clean directory:
```
my_project/
├── Data/                    # Your data files
├── Plot_Oneway_Pre.ipynb   # Copied template
├── PricingSingle.ipynb     # Copied template
└── Results/                 # Auto-generated outputs
```

### 2. **Start Simple**
- Begin with `Plot_Oneway_Pre.ipynb` to understand your data
- Then use `PricingSingle.ipynb` for quick modeling
- Graduate to advanced templates as needed

### 3. **Keep Original Templates**
- Don't modify templates in the `examples/` folder
- Copy them to your project directory first
- This way you always have clean templates

### 4. **Iterate Quickly**
- Templates are designed for fast iteration
- Modify parameters and re-run
- No need to restart from scratch

### 5. **Parameter Comments**
All parameters have inline comments explaining what they do:
```python
n_bins = 10  # Number of bins for continuous features
holdout_ratio = 0.25  # Test set ratio (0.25 = 25% test)
rand_seed = 13  # Random seed for reproducibility
```

---

## ❓ Troubleshooting

**Q: "Data file not found"**
- Make sure `data_path` points to the correct location
- Use relative paths: `work_dir / 'Data/file.csv'`
- Check that file exists before running

**Q: "Feature not in dataframe"**
- Check feature names in `feature_list` match your data
- Check for typos in feature names
- Use `df.columns.tolist()` to see available columns

**Q: "Model file not found"**
- Make sure you trained models before trying to load them
- Check `output_dir` matches between training and loading
- Verify model files exist in `{output_dir}/model/`

**Q: "Config file not found"**
- For advanced templates, make sure you run training first
- Training creates config files needed for prediction/plotting
- Or create config manually following template examples

---

## 🎓 Learning Path

1. **Beginner**:
   - Start with `Plot_Oneway_Pre.ipynb`
   - Try `PricingSingle.ipynb` with small data

2. **Intermediate**:
   - Use config files for consistency
   - Try `Train_FT_Embed_XGBResN.ipynb`
   - Generate plots with `Plot_LoadModel.ipynb`

3. **Advanced**:
   - Customize templates for your needs
   - Modify processing sections
   - Create your own template variants

---

## 📚 Additional Resources

- [README.md](README.md) - Full feature documentation
- [CHANGES.md](CHANGES.md) - Recent updates and changes
- [config_template.json](config_template.json) - Configuration reference

---

## ✨ Template Design Principles

These templates follow these principles:

1. **Self-contained**: Everything in one file
2. **Clear structure**: Configuration separated from processing
3. **Well-documented**: Comments explain every parameter
4. **Robust**: Error messages guide you to fixes
5. **Informative**: Progress messages show what's happening
6. **Reusable**: Copy once, use many times

**Goal**: You should be able to use these templates without reading any documentation!
