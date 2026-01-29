# Modelling

This directory contains reusable training tooling and frameworks centered on BayesOpt.

## Key locations

- `core/bayesopt/` - core training/tuning package
- `explain/` - explainability helpers
- `plotting/` - plotting utilities
- `ins_pricing/cli/` - CLI entry points
- `examples/` - example configs and notebooks (repo only)

## Common usage

- CLI training: `python ins_pricing/cli/BayesOpt_entry.py --config-json config_template.json`
- Notebook API: `from ins_pricing.modelling import BayesOptModel`

## Explainability

- CLI: `python ins_pricing/cli/Explain_entry.py --config-json config_explain_template.json`
- Notebook: `examples/04 Explain_Run.ipynb`

## Loss functions

Configure the regression/classification loss with `loss_name` in the BayesOpt config.

Supported `loss_name` values:
- `auto` (default): legacy behavior based on model name
- `tweedie`: Tweedie deviance
- `poisson`: Poisson deviance
- `gamma`: Gamma deviance
- `mse`: mean squared error
- `mae`: mean absolute error

Mapping summary:
- Tweedie deviance -> `tweedie`
- Poisson deviance -> `poisson`
- Gamma deviance -> `gamma`
- Mean squared error -> `mse`
- Mean absolute error -> `mae`
- Classification log loss -> `logloss` (classification only)
- Classification BCE -> `bce` (classification only)

Classification tasks:
- `loss_name` can be `auto`, `logloss`, or `bce`.
- Training uses `BCEWithLogits` for torch models; evaluation uses log loss.

Where to set `loss_name`:

```json
{
  "task_type": "regression",
  "loss_name": "mse"
}
```

Behavior notes:
- When `loss_name` is `mse` or `mae`, tuning does not sample Tweedie power.
- When `loss_name` is `poisson` or `gamma`, power is fixed (1.0 / 2.0).
- When `loss_name` is `tweedie`, power is sampled as usual.
- XGBoost objective is selected from the loss name.

## Notes

- Models load from `output_dir/model` by default (override with `explain.model_dir`).
- Training outputs are written to `plot/`, `Results/`, and `model/` under `output_dir`.
- Keep large data and secrets outside the repo; use environment variables or `.env` files.
