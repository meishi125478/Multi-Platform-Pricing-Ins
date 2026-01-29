# ins_pricing

Distribution name: ins_pricing (import package is `ins_pricing`; legacy alias `user_packages` still works).

Reusable modelling and pricing utilities organized as a small toolbox with clear boundaries
between modelling, production, governance, and reporting.

## Architecture

- `cli/`: CLI entry points and shared utilities.
- `modelling/`
  - `core/`: BayesOpt training core (GLM / XGB / ResNet / FT / GNN).
  - `plotting/`: model-agnostic curves and geo visualizations.
  - `explain/`: permutation, gradients, and SHAP helpers.
- `examples/`: demo configs and notebooks (repo only; not packaged).
- `pricing/`: factor tables, calibration, exposure, monitoring.
- `production/`: scoring, metrics, drift/PSI.
- `governance/`: registry, approval, audit workflows.
- `reporting/`: report builder and scheduler.

## Call flow (typical)

1. Model training
   - Python API: `from ins_pricing.modelling import BayesOptModel`
   - CLI: `python ins_pricing/cli/BayesOpt_entry.py --config-json ...`
2. Evaluation and visualization
   - Curves: `from ins_pricing.plotting import curves`
   - Importance: `from ins_pricing.plotting import importance`
   - Geo: `from ins_pricing.plotting import geo`
3. Explainability
   - `from ins_pricing.explain import permutation_importance, integrated_gradients_torch`
4. Pricing loop
   - `from ins_pricing.pricing import build_factor_table, rate_premium`
5. Production and governance
   - `from ins_pricing.production import batch_score, psi_report`
   - `from ins_pricing.governance import ModelRegistry, ReleaseManager`
6. Reporting
   - `from ins_pricing.reporting import build_report, write_report, schedule_daily`

## Import notes

- `ins_pricing` exposes lightweight lazy imports so that pricing/production/governance
  can be used without installing heavy ML dependencies.
- Demo notebooks/configs live in the repo under `examples/` and are not shipped
  in the PyPI package.
- Heavy dependencies are only required when you import or use the related modules:
  - BayesOpt: `torch`, `optuna`, `xgboost`, etc.
  - Explain: `torch` (gradients), `shap` (SHAP).
  - Geo plotting on basemap: `contextily`.
  - Plotting: `matplotlib`.

## Multi-platform and GPU notes

- Install the correct PyTorch build for your platform/GPU before installing extras.
- Torch Geometric requires platform-specific wheels; follow the official PyG install guide.
- Multi-GPU uses DDP or DataParallel where supported; Windows disables CUDA DDP.

## Backward-compatible imports

Legacy import paths continue to work:

- `import user_packages`
- `import user_packages.bayesopt`
- `import user_packages.plotting`
- `import user_packages.explain`
- `import user_packages.BayesOpt`
