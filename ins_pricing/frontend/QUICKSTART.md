# Quick Start Guide

Get started with the Insurance Pricing Model Training Frontend in 3 easy steps.

## Prerequisites

1. Install the `ins_pricing` package
2. Install Gradio:
   ```bash
   pip install gradio>=4.0.0
   ```

## Step 1: Launch the Application

### On Windows:
Double-click `start_app.bat` or run:
```bash
python -m ins_pricing.frontend.app
```

### On Linux/Mac:
Run the shell script:
```bash
./start_app.sh
```

Or use Python directly:
```bash
python -m ins_pricing.frontend.app
```

The web interface will automatically open at `http://localhost:7860`

## Step 2: Configure Your Model

### Option A: Upload Existing Config (Recommended)
1. Go to the **Configuration** tab
2. Click **"Upload JSON Config File"**
3. Select a config file (e.g., `config_xgb_direct.json` from `examples/`)
4. Click **"Load Config"**

### Option B: Manual Configuration
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

## Step 3: Run Training

1. Switch to the **Run Task** tab
2. Click **"Run Task"**
3. Watch real-time logs appear below

Training will start automatically and logs will update in real-time!

## New Features

### FT Two-Step Workflow

For advanced FT-Transformer → XGB/ResN training:

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

### Open Results Folder

- In the **Run Task** tab, click **"📁 Open Results Folder"**
- Automatically opens the output directory in your file explorer
- Works on Windows, macOS, and Linux

## Example Configuration

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

## Tips

- **Save Your Config**: After building a configuration, save it using the "Save Configuration" button for reuse
- **Check Logs**: Training logs update in real-time - watch for errors or progress indicators
- **GPU Usage**: Toggle "Use GPU" checkbox in Training Settings to enable/disable GPU acceleration
- **Model Selection**: Specify which models to train in "Model Keys" (xgb, resn, ft, gnn)
- **Open Results**: Use the "📁 Open Results Folder" button to quickly access output files
- **FT Workflow**: Use the dedicated FT tab for automated two-step FT → XGB/ResN training

## Troubleshooting

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

## Next Steps

- Explore advanced options in the Configuration tab
- Try the FT Two-Step Workflow for better model performance
- Experiment with different model combinations (xgb, resn, ft)
- Try different split strategies
- Use the Explain mode for model interpretability
- Check the full [README.md](README.md) for detailed documentation

## Support

For issues or questions, refer to:
- Full documentation: [README.md](README.md)
- Example configs: `ins_pricing/examples/`
- Package documentation: `ins_pricing/docs/`

Happy modeling!
