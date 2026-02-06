# Insurance Pricing Model Frontend

A Gradio-based web interface for configuring and running all insurance pricing model tasks from the examples folder.

## Features

- **Multiple Task Modes**: Supports all task types, automatically detected from config
  - **Training** (entry mode): Train XGB, ResNet, FT-Transformer, and GNN models
  - **Explanation** (explain mode): Generate permutation importance, SHAP values, integrated gradients
  - **Incremental** (incremental mode): Incremental batch training
  - **Watchdog** (watchdog mode): Automated monitoring and retraining
- **Dual Configuration Modes**: Manual parameter configuration or JSON file upload
- **Real-time Logging**: Live task logs displayed in the UI
- **Parameter Validation**: Automatic validation of configuration parameters
- **Config Export**: Save current configuration as JSON file for reuse
- **User-friendly Interface**: Intuitive web UI without writing code
- **Auto-Detection**: Automatically detects task mode from `config.runner.mode`
- **Plotting & Prediction Tools**: Run the plotting, prediction, and compare steps from the example notebooks

## Supported Examples

This frontend provides dedicated tabs or workflows that match the notebooks in `examples/`:

| Example Notebook | Task Mode | Description |
|-----------------|-----------|-------------|
| `01 Plot_Oneway_Pre.ipynb` | Manual plotting | Pre-model oneway analysis (can run manually, see examples) |
| `02 PricingSingle.ipynb` | `entry` | Legacy training; use config-based training tab |
| `02 Train_XGBResN.ipynb` | `entry` | Direct training of XGB/ResN models |
| `02 Train_FT_Embed_XGBResN.ipynb` | `entry` | FT-Transformer embedding + XGB/ResN training |
| `03 Plot_Embed_Model.ipynb` | Manual plotting | Post-model plotting (oneway, lift, double-lift) |
| `04 Explain_Run.ipynb` | `explain` | Model explanation and interpretability |
| `05 Predict_FT_Embed_XGB.ipynb` | Prediction | Model prediction (load config + run) |
| `06 Compare_*.ipynb` | Manual plotting | Model comparison plots |

## Installation

```bash
pip install gradio>=4.0.0
```

Or install from requirements file:

```bash
pip install -r ins_pricing/frontend/requirements.txt
```

### Recommended (Cross-Platform) Install

To avoid dependency mismatches on Linux/macOS, install the pinned frontend extras:

```bash
pip install "ins_pricing[frontend]"
```

If installing from source:

```bash
pip install -e ".[frontend]"
```

### Linux Note (gradio + huggingface_hub)

If you see `ImportError: cannot import name 'HfFolder'`, your `huggingface_hub` is too new.
Fix it with:

```bash
pip install "gradio>=4,<5" "huggingface_hub<0.24"
```

### Apple Silicon (MPS) Note

For MPS usage, install a PyTorch build with MPS support, and optionally enable fallback:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Launch Methods

### Method 1: Direct Run

```bash
python -m ins_pricing.frontend.app
```

### Method 2: Launch in Python Script

```python
from ins_pricing.frontend.app import create_ui

demo = create_ui()
demo.launch()
```

### Method 3: Custom Host and Port

```python
from ins_pricing.frontend.app import create_ui

demo = create_ui()
demo.launch(
    server_name="localhost",  # or "0.0.0.0" for external access
    server_port=8080,         # custom port
    share=False               # set True to generate public link
)
```

## Usage Guide

### 1. Configure Model Parameters

#### Option A: Upload JSON Config File (Recommended)

1. Click the **"Configuration"** tab
2. In the **"Load Configuration"** section, click **"Upload JSON Config File"**
3. Select a config file from `examples/`:
   - `config_template.json` - Full template
   - `config_xgb_direct.json` - XGBoost training
   - `config_resn_direct.json` - ResNet training
   - `config_explain_template.json` - Model explanation
   - `config_ft_unsupervised_*.json` - FT-Transformer configs
4. Click **"Load Config"** button
5. Configuration will display in the **"Current Configuration"** panel

**Important**: The `runner.mode` field in the config determines which task runs:
- `"mode": "entry"` -> Training
- `"mode": "explain"` -> Model explanation
- `"mode": "incremental"` -> Incremental training
- `"mode": "watchdog"` -> Watchdog monitoring

#### Option B: Manual Parameter Entry

Fill in parameters in the **"Manual Configuration"** section:

**Data Settings**
- **Data Directory**: Directory containing data files (e.g., `./Data`)
- **Model List**: Comma-separated model names (e.g., `od`)
- **Model Categories**: Comma-separated model categories (e.g., `bc`)
- **Target Column**: Target column name (e.g., `response`)
- **Weight Column**: Weight column name (e.g., `weights`)

**Features**
- **Feature List**: Comma-separated feature names
- **Categorical Features**: Comma-separated categorical feature names

**Model Settings**
- **Task Type**: Task type (`regression`/`binary`/`multiclass`)
- **Test Proportion**: Test set ratio (0.1-0.5)
- **Holdout Ratio**: Holdout validation ratio (0.1-0.5)
- **Validation Ratio**: Validation ratio (0.1-0.5)
- **Split Strategy**: Data split strategy (`random`/`stratified`/`time`/`group`)
- **Random Seed**: Random seed for reproducibility
- **Epochs**: Number of training epochs

**Training Settings**
- **Output Directory**: Output directory (e.g., `./Results`)
- **Use GPU**: Whether to use GPU
- **Model Keys**: Comma-separated model types (e.g., `xgb, resn`)
- **Max Evaluations**: Maximum number of evaluations

**XGBoost Settings**
- **XGB Max Depth**: XGBoost maximum depth
- **XGB Max Estimators**: XGBoost maximum number of estimators

### 2. Build Configuration

1. After filling parameters, click **"Build Configuration"** button
2. Generated JSON config will display in the **"Generated Config (JSON)"** textbox
3. You can review and edit the generated configuration
4. **Note**: Manual configuration defaults to `runner.mode = "entry"` (training)

### 3. Save Configuration (Optional)

1. Enter filename in **"Save Filename"** textbox (e.g., `my_config.json`)
2. Click **"Save Configuration"** button
3. Configuration will be saved to the specified file

### 4. Run Task

1. Switch to the **"Run Task"** tab
2. Click **"Run Task"** button to execute
3. Task status will display in the **"Task Status"** section
4. Real-time logs will appear in the **"Task Logs"** textbox below

**The system automatically detects the task mode from your config and runs the appropriate task!**

### 5. Plotting / Prediction / Compare

Use the **Plotting**, **Prediction**, and **Compare** tabs to run:
- Pre-model oneway plots
- Post-model plots (direct or FT-embed workflows)
- FT-embed predictions
- Direct vs FT-embed model comparisons

## Task Modes Explained

### Entry Mode (Training)

Standard model training mode. Trains one or more models specified in `runner.model_keys`.

**Example config snippet**:
```json
{
  "runner": {
    "mode": "entry",
    "model_keys": ["xgb", "resn"],
    "max_evals": 50
  }
}
```

**Equivalent to**: `examples/02 Train_XGBResN.ipynb`

### Explain Mode

Generates model explanations using various methods.

**Example config snippet**:
```json
{
  "runner": {
    "mode": "explain"
  },
  "explain": {
    "model_keys": ["xgb"],
    "methods": ["permutation", "shap"],
    "on_train": false,
    "permutation": {
      "n_repeats": 5,
      "max_rows": 5000
    },
    "shap": {
      "n_background": 500,
      "n_samples": 200
    }
  }
}
```

**Equivalent to**: `examples/04 Explain_Run.ipynb`

**Supported methods**:
- `permutation`: Permutation feature importance
- `shap`: SHAP values
- `integrated_gradients`: Integrated gradients (for neural models)

### Incremental Mode

Incremental batch training for continuous model updates.

**Example config snippet**:
```json
{
  "runner": {
    "mode": "incremental",
    "incremental_args": [
      "--incremental-dir", "./IncrementalBatches",
      "--incremental-template", "{model_name}_2025Q1.csv",
      "--merge-keys", "policy_id", "vehicle_id",
      "--model-keys", "xgb",
      "--update-base-data"
    ]
  }
}
```

### Watchdog Mode

Automated monitoring and retraining when new data arrives.

**Example config snippet**:
```json
{
  "runner": {
    "mode": "watchdog",
    "use_watchdog": true,
    "idle_seconds": 7200,
    "max_restarts": 50
  }
}
```

## Configuration Examples

### Minimal Training Config

```json
{
  "data_dir": "./Data",
  "model_list": ["od"],
  "model_categories": ["bc"],
  "target": "response",
  "weight": "weights",
  "feature_list": ["age", "gender", "region"],
  "categorical_features": ["gender", "region"],
  "runner": {
    "mode": "entry",
    "model_keys": ["xgb"],
    "max_evals": 50
  }
}
```

### Minimal Explain Config

```json
{
  "data_dir": "./Data",
  "model_list": ["od"],
  "model_categories": ["bc"],
  "target": "response",
  "weight": "weights",
  "output_dir": "./Results",
  "runner": {
    "mode": "explain"
  },
  "explain": {
    "model_keys": ["xgb"],
    "methods": ["permutation"]
  }
}
```

### Full Configuration Examples

Refer to configuration files in the `examples/` directory:
- `config_template.json` - Complete training template
- `config_xgb_direct.json` - XGBoost training
- `config_resn_direct.json` - ResNet training
- `config_explain_template.json` - Model explanation template
- `config_ft_unsupervised_*.json` - FT-Transformer configs
- `config_incremental_template.json` - Incremental training template

## FAQ

### Q: How do I access the frontend interface->

A: After launching, the browser will open automatically, or manually navigate to `http://localhost:7860`

### Q: Which task mode will run->

A: The task mode is determined by `config.runner.mode` in your configuration file:
- `"entry"` = Training
- `"explain"` = Explanation
- `"incremental"` = Incremental training
- `"watchdog"` = Watchdog mode

### Q: Can I interrupt the task->

A: The current version does not support interruption. Tasks must complete once started.

### Q: How do I run explanation after training->

A: First, run training with a config file. Then, load an explain config that points to the same output directory, and set `runner.mode` to `"explain"`.

### Q: What if logs don't display->

A: Check that the configuration is correct and data paths exist. Check the console for error messages.

### Q: Can I run multiple tasks simultaneously->

A: Not recommended. Wait for the current task to complete before starting a new one.

### Q: How do I run on a remote server->

A: Set `server_name="0.0.0.0"` when launching, then access via server IP and port.

```python
demo.launch(server_name="0.0.0.0", server_port=7860)
```

### Q: Where are configuration files saved->

A: By default, saved in the current working directory. You can specify a full path in "Save Filename".

### Q: How do I run plotting tasks->

A: Plotting tasks (oneway, lift, double-lift) can be run by using config files with plotting enabled. See `config_plot.json` example or manually run the plotting notebooks in `examples/`.

## Technical Architecture

- **Frontend Framework**: Gradio 4.x
- **Configuration Management**: ConfigBuilder class
- **Task Execution**: TaskRunner class (with real-time log capture and auto-detection)
- **Backend Interface**: `ins_pricing.cli.utils.notebook_utils.run_from_config` (unified entry point)

## Development Guide

### File Structure

```
ins_pricing/frontend/
- __init__.py           # Package initialization
- app.py                # Main application entry
- config_builder.py     # Configuration builder
- runner.py             # Unified task runner
- requirements.txt      # Dependencies
- README.md             # This document
- Quick Start Guide     # Quick start guide
- example_config.json   # Example configuration
- start_app.bat         # Windows launcher
- start_app.sh          # Linux/Mac launcher
```

### Extending Functionality

To add new features:

1. **Add new config parameters**: Modify the `ConfigBuilder` class in `config_builder.py`
2. **Modify UI layout**: Edit the `create_ui()` function in `app.py`
3. **Customize task handling**: Modify the `TaskRunner` class in `runner.py`

### How Task Detection Works

The `TaskRunner` reads `config.runner.mode` from your JSON file and automatically calls the appropriate backend function via `run_from_config()`. No manual routing needed!

## License

This project follows the same license as the `ins_pricing` package.

---
## Quick Start Guide

Get started with the Insurance Pricing Model Training Frontend in 3 easy steps.

### Prerequisites

1. Install the `ins_pricing` package
2. Install Gradio:
   ```bash
   pip install gradio>=4.0.0
   ```

### Step 1: Launch the Application

#### On Windows:
Double-click `start_app.bat` or run:
```bash
python -m ins_pricing.frontend.app
```

#### On Linux/Mac:
Run the shell script:
```bash
./start_app.sh
```

Or use Python directly:
```bash
python -m ins_pricing.frontend.app
```

The web interface will automatically open at `http://localhost:7860`

### Step 2: Configure Your Model

#### Option A: Upload Existing Config (Recommended)
1. Go to the **Configuration** tab
2. Click **"Upload JSON Config File"**
3. Select a config file (e.g., `config_xgb_direct.json` from `examples/`)
4. Click **"Load Config"**

#### Option B: Manual Configuration
1. Go to the **Configuration** tab
2. Scroll to **"Manual Configuration"**
3. Fill in the required fields:
   - **Data Directory**: Path to your data folder
   - **Model List**: Model name(s)
   - **Target Column**: Your target variable
   - **Weight Column**: Your weight variable
   - **Feature List**: Comma-separated features
   - **Categorical Features**: Comma-separated categorical features
4. Adjust other settings as needed
5. Click **"Build Configuration"**

### Step 3: Run Training

1. Switch to the **Run Task** tab
2. Click **"Run Task"**
3. Watch real-time logs appear below

Training will start automatically and logs will update in real-time!

### New Features

#### FT Two-Step Workflow

For advanced FT-Transformer -> XGB/ResN training:

1. **Prepare Base Config**: Create or load a base configuration
2. **Go to FT Two-Step Workflow tab**
3. **Step 1 - FT Embedding Generation**:
   - Configure DDP settings
   - Click "Prepare Step 1 Config"
   - Copy the config to Configuration tab
   - Run it in "Run Task" tab
4. **Step 2 - Train XGB/ResN**:
   - After Step 1 completes, click "Prepare Step 2 Configs"
   - Choose which models to train (XGB, ResN, or both)
   - Copy the generated configs and run them

#### Open Results Folder

- In the **Run Task** tab, click **"Open Results Folder"**
- Automatically opens the output directory in your file explorer
- Works on Windows, macOS, and Linux

### Example Configuration

Here's a minimal example to get started:

```json
{
  "data_dir": "./Data",
  "model_list": ["od"],
  "model_categories": ["bc"],
  "target": "response",
  "weight": "weights",
  "feature_list": ["age", "gender", "region"],
  "categorical_features": ["gender", "region"],
  "runner": {
    "mode": "entry",
    "model_keys": ["xgb"],
    "max_evals": 50
  }
}
```

Save this as `my_first_config.json` and upload it!

### Tips

- **Save Your Config**: After building a configuration, save it using the "Save Configuration" button for reuse
- **Check Logs**: Training logs update in real-time - watch for errors or progress indicators
- **GPU Usage**: Toggle "Use GPU" checkbox in Training Settings to enable/disable GPU acceleration
- **Model Selection**: Specify which models to train in "Model Keys" (xgb, resn, ft, gnn)
- **Open Results**: Use the "Open Results Folder" button to quickly access output files
- **FT Workflow**: Use the dedicated FT tab for automated two-step FT -> XGB/ResN training

### Troubleshooting

**Problem**: Interface doesn't load
- **Solution**: Check that port 7860 is not in use, or specify a different port

**Problem**: Configuration validation fails
- **Solution**: Ensure all required fields are filled and feature lists are properly formatted

**Problem**: Training doesn't start
- **Solution**: Verify data paths exist and configuration is valid

**Problem**: Results folder won't open
- **Solution**: Make sure the task has run at least once to create the output directory

**Problem**: Step 2 configs fail to generate
- **Solution**: Ensure Step 1 completed successfully and embedding files exist

### Next Steps

- Explore advanced options in the Configuration tab
- Try the FT Two-Step Workflow for better model performance
- Experiment with different model combinations (xgb, resn, ft)
- Try different split strategies
- Use the Explain mode for model interpretability
- Check the full [README.md](README.md) for detailed documentation

### Support

For issues or questions, refer to:
- Full documentation: [README.md](README.md)
- Example configs: `examples/`
- Package documentation: `ins_pricing/docs/`

Happy modeling!
