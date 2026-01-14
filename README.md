# Insurance-Pricing

A reusable toolkit for insurance modeling, pricing, governance, and reporting.

## Overview

Insurance-Pricing (ins_pricing) is an enterprise-grade Python library designed for machine learning model training, pricing calculations, and model governance workflows in the insurance industry.

### Core Modules

| Module | Description |
|--------|-------------|
| **modelling** | ML model training (GLM, XGBoost, ResNet, FT-Transformer, GNN) and model interpretability (SHAP, permutation importance) |
| **pricing** | Factor table construction, numeric binning, premium calibration, exposure calculation, PSI monitoring |
| **production** | Model prediction, batch scoring, data drift detection, production metrics monitoring |
| **governance** | Model registry, version management, approval workflows, audit logging |
| **reporting** | Report generation (Markdown format), report scheduling |
| **utils** | Data validation, performance profiling, device management, logging configuration |

### Quick Start

```python
# Model training
from ins_pricing.modelling import BayesOptModel
model = BayesOptModel(config)
model.fit(X_train, y_train)

# Pricing and calibration
from ins_pricing.pricing import build_factor_table, fit_calibration_factor
factors = build_factor_table(df, 'age', 'claim_amount')

# Production scoring
from ins_pricing.production import batch_score
scores = batch_score(df, model)

# Model governance
from ins_pricing.governance import ModelRegistry
registry = ModelRegistry('models.json')
registry.register(model_name, version, metrics=metrics)
```

### Project Structure

```
ins_pricing/
├── cli/                    # Command-line entry points
├── modelling/
│   ├── core/bayesopt/     # ML model training core
│   ├── explain/           # Model interpretability
│   └── plotting/          # Model visualization
├── pricing/               # Insurance pricing module
├── production/            # Production deployment module
├── governance/            # Model governance
├── reporting/             # Report generation
├── utils/                 # Utilities
└── tests/                 # Test suite
```

### Installation

```bash
# Basic installation
pip install ins_pricing

# Full installation (all optional dependencies)
pip install ins_pricing[all]

# Install specific extras
pip install ins_pricing[bayesopt]    # Model training
pip install ins_pricing[explain]     # Model explanation
pip install ins_pricing[plotting]    # Visualization
pip install ins_pricing[gnn]         # Graph neural networks
```

### Requirements

- Python >= 3.9
- Core dependencies: numpy >= 1.20, pandas >= 1.4

### License

Proprietary
