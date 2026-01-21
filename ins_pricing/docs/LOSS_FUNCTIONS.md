LOSS FUNCTIONS

Overview
This document describes the loss-function changes in ins_pricing. The training
stack now supports multiple regression losses (not just Tweedie deviance) and
propagates the selected loss into tuning, training, and inference.

Supported loss_name values
- auto (default): keep legacy behavior based on model name
- tweedie: Tweedie deviance (uses tw_power / tweedie_variance_power when tuning)
- poisson: Poisson deviance (power=1)
- gamma: Gamma deviance (power=2)
- mse: mean squared error
- mae: mean absolute error

Loss name mapping (all options)
- Tweedie deviance -> tweedie
- Poisson deviance -> poisson
- Gamma deviance -> gamma
- Mean squared error -> mse
- Mean absolute error -> mae
- Classification log loss -> logloss (classification only)
- Classification BCE -> bce (classification only)

Classification tasks
- loss_name can be auto, logloss, or bce
- training continues to use BCEWithLogits for torch models; evaluation uses logloss

Where to set loss_name
Add to any BayesOpt config JSON:

{
  "task_type": "regression",
  "loss_name": "mse"
}

Behavior changes
1) Tuning and metrics
   - When loss_name is mse/mae, tuning does not sample Tweedie power.
   - When loss_name is poisson/gamma, power is fixed (1.0/2.0).
   - When loss_name is tweedie, power is sampled as before.

2) Torch training (ResNet/FT/GNN)
   - Loss computation is routed by loss_name.
   - For tweedie/poisson/gamma, predictions are clamped positive.
   - For mse/mae, no Tweedie power is used.

3) XGBoost objective
   - loss_name controls XGB objective:
     - tweedie -> reg:tweedie
     - poisson -> count:poisson
     - gamma -> reg:gamma
     - mse -> reg:squarederror
     - mae -> reg:absoluteerror

4) Inference
   - ResNet/GNN constructors now receive loss_name.
   - When loss_name is not tweedie, tw_power is not applied at inference.

Legacy defaults (auto)
- If loss_name is omitted, behavior is unchanged:
  - model name contains "f" -> poisson
  - model name contains "s" -> gamma
  - otherwise -> tweedie

Examples
- ResNet direct training (MSE):
  "loss_name": "mse"

- FT embed -> ResNet (MSE):
  "loss_name": "mse"

- XGB direct training (unchanged):
  omit loss_name or set "loss_name": "auto"

Notes
- loss_name is global per config. If you need different losses for different
  models, split into separate configs and run them independently.
