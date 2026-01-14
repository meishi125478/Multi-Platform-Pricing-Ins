# ins_pricing

This directory contains reusable production-grade tooling and training frameworks, with a focus on the BayesOpt series.

Key contents:
- `core/bayesopt/`: core subpackage (data preprocessing, trainers, models, plotting, explainability)
- `plotting/`: standalone plotting helpers (lift/roc/importance/geo)
- `explain/`: explainability helpers (Permutation/Integrated Gradients/SHAP)
- `core/BayesOpt.py`: compatibility entry point for legacy imports
- `cli/BayesOpt_entry.py`: batch training CLI
- `cli/BayesOpt_incremental.py`: incremental training CLI
- `cli/utils/cli_common.py` / `cli/utils/notebook_utils.py`: shared CLI and notebook utilities
- `examples/modelling/config_template.json` / `examples/modelling/config_incremental_template.json`: config templates
- `cli/Explain_entry.py` / `cli/Explain_Run.py`: explainability entry points (load trained models)
- `examples/modelling/config_explain_template.json` / `examples/modelling/Explain_Run.ipynb`: explainability demo

Note: `examples/modelling/` is kept in the repo only and is not shipped in the PyPI package.
Migration note: CLI entry points now live under `cli/` and demo assets are under `examples/modelling/`.

Common usage:
- CLI: `python ins_pricing/cli/BayesOpt_entry.py --config-json ...`
- Notebook: `from ins_pricing.bayesopt import BayesOptModel`

Explainability (load trained models under `Results/model` and explain a validation set):
- CLI: `python ins_pricing/cli/Explain_entry.py --config-json ins_pricing/examples/modelling/config_explain_template.json`
- Notebook: open `ins_pricing/examples/modelling/Explain_Run.ipynb` and run it

Notes:
- Models load from `output_dir/model` by default (override with `explain.model_dir`).
- Validation data can be specified via `explain.validation_path`.

Operational notes:
- Training outputs are written to `plot/`, `Results/`, and `model/` by default.
- Keep large data and secrets outside the repo and use environment variables or `.env`.
