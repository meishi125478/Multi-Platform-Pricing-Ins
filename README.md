# ins_pricing

Distribution name: `ins_pricing` (import package is `ins_pricing`).

Reusable modelling and pricing utilities organized as a small toolbox with clear boundaries
between modelling, production, governance, and reporting.

## Architecture

- `cli/` — CLI entry points and shared utilities.
  - `BayesOpt_entry.py` — main training entry.
  - `Explain_entry.py` — explanation entry.
  - `watchdog_run.py` — process watchdog (auto-restart on idle).
  - `utils/` — config loading (`cli_config.py`), notebook helpers (`notebook_utils.py`), reporting (`bayesopt_runner_reporting.py`), argument parsing (`bayesopt_runner_ui.py`).
- `modelling/`
  - `bayesopt/` — BayesOpt training core (GLM / XGB / ResNet / FT / GNN). Config via `BayesOptConfig`, orchestrator via `BayesOptModel`.
  - `plotting/` — model-agnostic curves, diagnostics, importance, geo visualizations.
  - `explain/` — permutation importance, SHAP, integrated gradients.
  - `evaluation.py` — calibration, threshold selection, bootstrap CI, metrics report.
- `frontend/` — Gradio web UI (`ConfigBuilder`, `TaskRunner`, `FTWorkflowHelper`).
- `examples/` — demo configs and notebooks (repo only; not packaged).
- `pricing/` — factor tables, calibration, exposure, data quality, PSI monitoring.
- `production/` — scoring, preprocessing, inference registry, drift detection, metrics.
- `governance/` — model registry, approval workflows, audit logging, release management.
- `reporting/` — Markdown report builder and daily scheduler.
- `utils/` — validation, features, IO, losses, device management, paths, numerics, metrics, profiling, logging, safe pickle, torch compat.

## Call Flow (typical)

1. **Model training**
   - Python API: `from ins_pricing.modelling import BayesOptModel`
   - CLI: `python ins_pricing/cli/BayesOpt_entry.py --config-json ...`
   - Frontend: `python -m ins_pricing.frontend.app`
2. **Evaluation and visualization**
   - Curves: `from ins_pricing.modelling.plotting import curves`
   - Diagnostics: `from ins_pricing.modelling.plotting import diagnostics`
   - Importance: `from ins_pricing.modelling.plotting import importance`
   - Geo: `from ins_pricing.modelling.plotting import geo`
3. **Explainability**
   - `from ins_pricing.modelling.explain import permutation_importance, integrated_gradients_torch`
   - SHAP: `from ins_pricing.modelling.explain import compute_shap_xgb, compute_shap_resn, compute_shap_ft`
4. **Pricing loop**
   - `from ins_pricing.pricing import build_factor_table, rate_premium, fit_calibration_factor`
5. **Production and governance**
   - Scoring: `from ins_pricing.production import batch_score, metrics_report`
   - Inference: `from ins_pricing.production import load_predictor_from_config, predict_from_config`
   - Preprocessing: `from ins_pricing.production import load_preprocess_artifacts, apply_preprocess_artifacts`
   - Drift: `from ins_pricing.production import psi_report`
   - Registry: `from ins_pricing.governance import ModelRegistry, ReleaseManager`
6. **Reporting**
   - `from ins_pricing.reporting import ReportPayload, write_report, schedule_daily`

## Loss and Distribution Docs

- Model-level mapping and priority rules: `ins_pricing/modelling/README.md`
- BayesOpt detailed mapping table and examples: `ins_pricing/modelling/bayesopt/README.md`

## Import Policy

Use canonical import paths only:

- `ins_pricing.modelling.*`
- `ins_pricing.pricing.*`
- `ins_pricing.production.*`
- `ins_pricing.governance.*`
- `ins_pricing.reporting.*`
- `ins_pricing.frontend.*`

The root `ins_pricing` and `ins_pricing.modelling` packages expose lightweight lazy imports
so that pricing/production/governance can be used without installing heavy ML dependencies.
Heavy dependencies are only required when you import or use the related modules:
- BayesOpt: `torch`, `optuna`, `xgboost`, etc.
- Explain: `torch` (gradients), `shap` (SHAP).
- Geo plotting on basemap: `contextily`.
- Plotting: `matplotlib`.
- Inference: `torch` only when loading FT/ResNet/GNN models.
- Frontend: `gradio`.

## Inference Interface

`production.inference` provides a registry-based interface:
- `ModelSpec` describes the saved model location and config.
- `PredictorRegistry` lets you plug in custom model loaders.
- `load_predictor_from_config` loads a predictor from a training config file.
- `predict_from_config` runs batch prediction with optional chunking and parallel loading.

```python
from ins_pricing.production import load_predictor_from_config

predictor = load_predictor_from_config("config.json", "resn", device="cuda")
preds = predictor.predict(df)
```

## Multi-Platform and GPU Notes

- Install the correct PyTorch build for your platform/GPU before installing extras.
- Torch Geometric requires platform-specific wheels; follow the official PyG install guide.
- Multi-GPU uses DDP or DataParallel where supported; Windows disables CUDA DDP.
- CLI usage prefers `python -m ins_pricing.cli.BayesOpt_entry ...` but the direct script path still works.
