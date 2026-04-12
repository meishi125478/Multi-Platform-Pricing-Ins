# ins_pricing

Distribution name: `ins_pricing` (import package is also `ins_pricing`).

This repository is organized as layered tooling for modelling, pricing, production inference,
governance, reporting, and frontend workflows.

## Documentation Layers

- Layer 1 (global navigation): this file.
- Layer 2 (module boundaries and entrypoints):
  - [modelling/README.md](modelling/README.md)
  - [pricing/README.md](pricing/README.md)
  - [production/README.md](production/README.md)
  - [governance/README.md](governance/README.md)
  - [reporting/README.md](reporting/README.md)
  - [frontend/README.md](frontend/README.md)
- Layer 3 (deep-dive and reference):
  - [modelling/bayesopt/README.md](modelling/bayesopt/README.md) for BayesOpt internals and
    loss/distribution mapping.
  - [docs/api_reference.md](docs/api_reference.md) for package-level public exports.

## Architecture At A Glance

- `cli/`: training/explain/watchdog entry scripts and shared CLI helpers.
- `modelling/`: training, explainability, plotting, and evaluation.
- `pricing/`: exposure, factor tables, premium rating, calibration.
- `production/`: preprocessing, predictor loading, batch inference, monitoring metrics.
- `governance/`: registry, approvals, audit trail, release/rollback.
- `reporting/`: Markdown report generation and daily scheduling.
- `frontend/`: NiceGUI UI for config-driven workflows.
- `utils/`: shared utilities (metrics, paths, IO, numerics, logging, device helpers).

## Quick Start

```python
from ins_pricing.modelling import BayesOptConfig, BayesOptModel
from ins_pricing.pricing import rate_premium
from ins_pricing.production import load_predictor_from_config
```

## Typical Cross-Module Flow

1. Train with `modelling` (CLI, Python API, or frontend launcher).
2. Explain and evaluate with `modelling.explain`, `modelling.plotting`, and `modelling.evaluation`.
3. Build premiums with `pricing`.
4. Serve or batch-run predictions with `production`.
5. Track approvals/releases with `governance`.
6. Publish periodic summaries with `reporting`.

## Import Policy

Use canonical paths:

- `ins_pricing.modelling.*`
- `ins_pricing.pricing.*`
- `ins_pricing.production.*`
- `ins_pricing.governance.*`
- `ins_pricing.reporting.*`
- `ins_pricing.frontend.*`

`ins_pricing` and `ins_pricing.modelling` keep imports lazy so pricing/production/governance
can be imported without pulling heavy modelling dependencies.

## Torch 1.x Compatibility Check

Torch checkpoints are loaded with secure `weights_only` mode when available. On torch 1.x,
the package can fall back to trusted legacy loading through
`INS_PRICING_ALLOW_LEGACY_TORCH_LOAD=1` (default).

To validate torch 1.x runtime behavior in a torch 1.x environment:

```bash
./scripts/run_torch1_compat_smoke.sh
# or run the pytest commands directly:
python -m pytest tests/utils/test_model_loading.py -q
python -m pytest tests/production/test_inference.py tests/modelling/test_ft_checkpoints.py -q
```
