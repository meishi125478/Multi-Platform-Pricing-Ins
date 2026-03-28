# Insurance Pricing Model Frontend

A NiceGUI-based web interface for configuring and running all insurance pricing model tasks.

## Installation

```bash
pip install "ins_pricing[frontend]"
```

Or from source:

```bash
pip install -e ".[frontend]"
```

### Apple Silicon (MPS) Note

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Launch

```bash
python -m ins_pricing.frontend.app
```

Override host/port at runtime (PowerShell):

```powershell
$env:NICEGUI_HOST="0.0.0.0"
$env:NICEGUI_PORT="7860"
python -m ins_pricing.frontend.app
```

The web interface binds to `0.0.0.0:7860` by default.

## Public API

| Class | Module | Description |
|-------|--------|-------------|
| `ConfigBuilder` | `config_builder.py` | Build config dicts from UI parameters |
| `TaskRunner` | `runner.py` | Execute tasks with real-time log capture |
| `FTWorkflowHelper` | `ft_workflow.py` | Automate FT two-step embedding + XGB/ResN training |

All three are importable via `from ins_pricing.frontend import ConfigBuilder, TaskRunner, FTWorkflowHelper`.

## Features

- **Multiple Task Modes**: Training (`entry`), explanation (`explain`), incremental, watchdog — auto-detected from `config.runner.mode`
- **Dual Config Input**: Manual parameter entry or JSON file upload
- **Real-time Logging**: Live task logs in the UI
- **Plotting / Prediction / Compare / Double-Lift**: Dedicated tabs matching `examples/` notebooks
- **Workflow Config Tab**: Config-driven plotting/prediction/compare via `workflow.mode`
- **Working Directory Control**: Set a frontend working folder for relative paths
- **FT Two-Step Workflow**: Automated Step 1 (FT embedding) -> Step 2 (XGB/ResN on augmented features)

## Task Modes

| `runner.mode` | Description | Example Notebook |
|---------------|-------------|-----------------|
| `entry` | Model training (XGB, ResNet, FT, GNN) | `02 Train_XGBResN.ipynb` |
| `explain` | Permutation, SHAP, integrated gradients | `04 Explain_Run.ipynb` |
| `incremental` | Incremental batch training | — |
| `watchdog` | Auto-restart monitoring | — |

## Workflow Config Modes

Use the **Workflow Config** tab with `workflow.mode` for config-driven frontend workflows:

- `pre_oneway` - Pre-model oneway analysis
- `plot_direct` / `plot_embed` - Post-model plots (direct or FT-embed)
- `predict_ft_embed` - FT embedding + prediction
- `compare_xgb` / `compare_resn` / `compare` - Model comparison plots
- `double_lift` - Double-lift curves from CSV

Example:

```json
{
  "workflow": {
    "mode": "plot_direct",
    "cfg_path": "config_plot.json",
    "xgb_cfg_path": "config_xgb_direct.json",
    "resn_cfg_path": "config_resn_direct.json"
  }
}
```

## File Structure

```
ins_pricing/frontend/
  app.py                NiceGUI application entrypoint
  ui_frontend.py        UI layout, tabs, and event handlers
  app_controller.py     Runtime/config orchestration layer
  config_builder.py     ConfigBuilder class
  runner.py             TaskRunner class
  ft_workflow.py        FTWorkflowHelper for two-step workflows
  workflows_common.py   Shared utilities (path resolution, model discovery)
  workflows_predict.py  Prediction workflow (FT embed + XGB/ResN stacking)
```

## Usage Guide

### 1. Set Working Directory (Optional)

At the top of the UI, set the **Working Directory** to your project folder. All relative paths resolve from here.

### 2. Configure

**Option A — Upload JSON**: Go to **Configuration** tab, upload a config file from `examples/`, click **Load Config**.

**Option B — Manual**: Fill in parameters under **Manual Configuration**, click **Build Configuration**.

### 3. Run

Switch to **Run Task** tab, click **Run Task**. Logs appear in real-time.

### 4. FT Two-Step Workflow

1. Load a base config
2. Go to **FT Two-Step Workflow** tab
3. **Step 1**: Configure DDP, click "Prepare Step 1 Config", run in Run Task tab
4. **Step 2**: Click "Prepare Step 2 Configs" to auto-generate XGB/ResN configs with augmented features

### 5. Plotting / Prediction / Compare

Use dedicated tabs or the **Workflow Config** tab for config-driven execution matching the `examples/` notebooks.

## Technical Notes

- **Backend entry point**: `ins_pricing.cli.utils.notebook_utils.run_from_config`
- **Task detection**: `TaskRunner` reads `config.runner.mode` and routes automatically
- **Config builder defaults**: Includes data format, splitting, CV strategy, preprocessing, and Optuna search space templates
- Tasks cannot be interrupted once started; wait for completion before starting a new one
