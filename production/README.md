# production

## Purpose

`production` owns runtime inference and monitoring helpers: preprocessing artifacts, predictor
loading, config-driven prediction execution, and production-friendly metric reporting.

## Use When / Not For

- Use when loading trained artifacts and running batch prediction in non-training contexts.
- Use when applying saved preprocessing and producing monitoring metrics/PSI outputs.
- Not for model training or hyperparameter tuning (handled by `modelling`).
- Not for model approval/deployment governance records (handled by `governance`).

## Public Entrypoints

- Inference and registry: `Predictor`, `SavedModelPredictor`, `ModelSpec`, `PredictorRegistry`,
  `register_model_loader`, `load_predictor`, `load_saved_model`, `load_predictor_from_config`,
  `predict_from_config`, `load_best_params`
- Preprocessing: `load_preprocess_artifacts`, `prepare_raw_features`, `apply_preprocess_artifacts`
- Scoring and monitoring: `batch_score`, `metrics_report`, `regression_metrics`,
  `classification_metrics`, `group_metrics`, `loss_ratio`, `psi_report`

## Minimal Flow

```python
from ins_pricing.production import load_predictor_from_config, predict_from_config

predictor = load_predictor_from_config("config.json", "xgb")
pred = predictor.predict(df)
pred2 = predict_from_config("config.json", "xgb", data=df)
```

## Further Reading

- Public export index: [../docs/api_reference.md](../docs/api_reference.md)
- Package navigation: [../README.md](../README.md)
