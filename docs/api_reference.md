# API Reference (Public Exports)

This page is the single index for package-level public exports.
It lists public symbols grouped by package, with one-line responsibilities.

## `ins_pricing`

- `modelling`: lazy-loaded modelling package namespace.
- `pricing`: lazy-loaded pricing package namespace.
- `production`: lazy-loaded production package namespace.
- `governance`: lazy-loaded governance package namespace.
- `reporting`: lazy-loaded reporting package namespace.

## `ins_pricing.modelling`

### Core Facade

- `BayesOptConfig`: config schema for BayesOpt workflows.
- `BayesOptModel`: orchestration facade for training, plotting, and explainability mixins.

### Namespaces

- `bayesopt`: BayesOpt internals package.
- `plotting`: plotting helpers for lift, calibration, ROC/PR, diagnostics, and geo.
- `explain`: permutation, SHAP, integrated-gradient explainability helpers.
- `evaluation`: calibration, thresholding, bootstrap CI, and summary metrics utilities.
- `cli`: CLI package alias exposed through modelling namespace.

### BayesOpt Exports Re-exported By `ins_pricing.modelling`

- `DatasetPreprocessor`: dataset preprocessing for BayesOpt training pipelines.
- `OutputManager`: output directory and artifact writing helper.
- `VersionManager`: model/version metadata helper.
- `FeatureTokenizer`: FT tokenizer component.
- `FTTransformerCore`: FT-Transformer core network.
- `FTTransformerSklearn`: scikit-learn style FT wrapper.
- `GraphNeuralNetSklearn`: scikit-learn style GNN wrapper.
- `MaskedTabularDataset`: masked tabular dataset wrapper for torch training.
- `ResBlock`: residual block used in ResNet models.
- `ResNetSequential`: sequential ResNet architecture container.
- `ResNetSklearn`: scikit-learn style ResNet wrapper.
- `ScaledTransformerEncoderLayer`: scaled transformer encoder layer implementation.
- `SimpleGraphLayer`: basic graph layer building block.
- `SimpleGNN`: lightweight GNN model.
- `TabularDataset`: generic tabular dataset wrapper.
- `TrainerBase`: common trainer interface contract.
- `GLMTrainer`: GLM trainer implementation.
- `XGBTrainer`: XGBoost trainer implementation.
- `ResNetTrainer`: ResNet trainer implementation.
- `FTTrainer`: FT-Transformer trainer implementation.
- `GNNTrainer`: GNN trainer implementation.
- `_xgb_cuda_available`: helper flag/function for XGBoost CUDA availability checks.

## `ins_pricing.pricing`

- `detect_leakage`: detect target leakage by direct or near-direct feature leakage.
- `profile_columns`: compute missing/unique and summary stats by column.
- `validate_schema`: validate required columns and optional dtypes.
- `compute_exposure`: calculate exposure from start/end dates.
- `aggregate_policy_level`: aggregate event-level rows into policy-level rows.
- `build_frequency_severity`: derive frequency/severity/pure-premium style columns.
- `bin_numeric`: bin numeric variables for factor building.
- `build_factor_table`: build factor/relativity tables from loss and exposure.
- `compute_base_rate`: compute portfolio base rate.
- `apply_factor_tables`: apply multiplicative factor tables to records.
- `rate_premium`: compute premium from exposure, base rate, and factors.
- `RateTable`: packaged rate table with scoring helper methods.
- `fit_calibration_factor`: fit scalar calibration factor to targets.
- `apply_calibration`: apply scalar factor to predicted premium values.
- `population_stability_index`: compute PSI for numeric or bucketed distributions.
- `psi_report`: generate feature-level PSI summary report.

## `ins_pricing.production`

- `ModelSpec`: model specification payload for registry-based loading.
- `Predictor`: predictor protocol/base class.
- `SavedModelPredictor`: predictor backed by persisted model artifacts.
- `PredictorRegistry`: registry that maps model keys to loader functions.
- `register_model_loader`: register custom loader in a registry.
- `load_predictor`: load predictor from a `ModelSpec` and registry.
- `load_saved_model`: load persisted model object and artifacts.
- `load_best_params`: load best-params file from training outputs.
- `load_predictor_from_config`: build predictor from training config and model key.
- `predict_from_config`: run prediction from config, model key, and tabular data.
- `load_preprocess_artifacts`: load serialized preprocessing artifacts.
- `prepare_raw_features`: normalize/coerce raw frame into model-ready feature set.
- `apply_preprocess_artifacts`: apply fitted preprocessing artifacts to a dataframe.
- `batch_score`: compute batch-level scoring outputs and summary report.
- `regression_metrics`: compute regression metrics dictionary.
- `classification_metrics`: compute classification metrics dictionary.
- `group_metrics`: compute grouped metrics by segmentation column.
- `metrics_report`: produce unified metrics report payload.
- `loss_ratio`: compute claims-to-premium ratio metric.
- `psi_report`: produce PSI drift report (re-export from utils metrics).

## `ins_pricing.governance`

- `ModelArtifact`: model artifact metadata record.
- `ModelVersion`: model version record with tags/metrics/status.
- `ModelRegistry`: version registry with register/promote/query operations.
- `ApprovalAction`: approval decision record.
- `ApprovalRequest`: approval request record.
- `ApprovalStore`: approval workflow storage and action manager.
- `AuditEvent`: audit log event record.
- `AuditLogger`: append-only audit logger with filtering helpers.
- `ModelRef`: deployment model pointer record.
- `DeploymentState`: environment deployment state snapshot.
- `ReleaseManager`: deploy/rollback/release lifecycle manager.

## `ins_pricing.reporting`

- `ReportPayload`: structured report input payload.
- `build_report`: build Markdown report string from payload.
- `write_report`: write Markdown report to file.
- `schedule_daily`: run callable once daily in a background thread.

## `ins_pricing.frontend`

- `ConfigBuilder`: build config dictionaries from UI inputs.
- `TaskRunner`: execute frontend tasks and stream logs.
- `FTWorkflowHelper`: helper for FT embedding two-step workflows.

## Related Docs

- Project navigation: [../README.md](../README.md)
- BayesOpt deep-dive and loss/distribution owner: [../modelling/bayesopt/README.md](../modelling/bayesopt/README.md)
