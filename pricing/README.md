# pricing

## Purpose

`pricing` owns lightweight insurance pricing loop utilities: exposure construction, frequency and
severity preparation, factor tables, premium rating, calibration, and PSI monitoring hooks.

## Use When / Not For

- Use when you need deterministic, table-driven premium computation or post-model calibration.
- Use when raw policy/event data must be converted into exposure-aware pricing features.
- Not for model training/tuning (handled by `modelling`).
- Not for runtime model loading/serving pipelines (handled by `production`).

## Public Entrypoints

- `compute_exposure`, `aggregate_policy_level`, `build_frequency_severity`
- `bin_numeric`, `build_factor_table`
- `compute_base_rate`, `apply_factor_tables`, `rate_premium`, `RateTable`
- `fit_calibration_factor`, `apply_calibration`
- `population_stability_index`, `psi_report`

## Minimal Flow

```python
from ins_pricing.pricing import (
    compute_exposure,
    build_factor_table,
    compute_base_rate,
    rate_premium,
    fit_calibration_factor,
)

df["exposure"] = compute_exposure(df, "start_date", "end_date")
base_rate = compute_base_rate(df, loss_col="claim_amt", exposure_col="exposure")
vehicle = build_factor_table(df, factor_col="vehicle_type", loss_col="claim_amt", exposure_col="exposure", base_rate=base_rate)
premium = rate_premium(df, exposure_col="exposure", base_rate=base_rate, factor_tables={"vehicle_type": vehicle})
premium = premium * fit_calibration_factor(premium, df["claim_amt"].to_numpy())
```

## Further Reading

- Public export index: [../docs/api_reference.md](../docs/api_reference.md)
- Package navigation: [../README.md](../README.md)
