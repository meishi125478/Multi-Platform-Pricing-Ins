# Modelling

This directory contains reusable training tooling and frameworks centered on BayesOpt.

## Key locations

- `bayesopt/` - core training/tuning package
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

Loss resolution priority:
1. `distribution` (if set) overrides `loss_name`
2. `loss_name` (if set and not `auto`)
3. legacy auto inference (from `model_nme`) when both are `auto`/unset

| distribution | resolved loss_name | notes |
| --- | --- | --- |
| `tweedie` | `tweedie` | Tweedie power can be tuned |
| `poisson` | `poisson` | Tweedie power fixed at `1.0` |
| `gamma` | `gamma` | Tweedie power fixed at `2.0` |
| `gaussian`, `normal`, `mse` | `mse` | L2 regression |
| `laplace`, `laplacian`, `mae` | `mae` | L1 regression |
| `bernoulli`, `binomial`, `logistic`, `binary` | `logloss` | classification only |

Supported explicit `loss_name` values:
- Regression: `auto`, `tweedie`, `poisson`, `gamma`, `mse`, `mae`
- Classification: `auto`, `logloss`, `bce`

Example:

```json
{
  "task_type": "regression",
  "distribution": "poisson",
  "loss_name": "auto"
}
```

Detailed BayesOpt-level behavior is documented in `ins_pricing/modelling/bayesopt/README.md`.

## Notes

- Models load from `output_dir/model` by default (override with `explain.model_dir`).
- Training outputs are written to `plot/`, `Results/`, and `model/` under `output_dir`.
- Keep large data and secrets outside the repo; use environment variables or `.env` files.
