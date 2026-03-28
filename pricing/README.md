# pricing

Lightweight pricing loop utilities: data quality checks, exposure/targets,
factor tables, rate tables, calibration, and monitoring (PSI).

## Modules

| File | Description |
|------|-------------|
| `data_quality.py` | Leakage detection, column profiling, schema validation |
| `exposure.py` | Exposure calculation, policy-level aggregation, frequency/severity |
| `factors.py` | Numeric binning, factor table construction with smoothing |
| `rate_table.py` | Base rate, factor application, premium rating, `RateTable` dataclass |
| `calibration.py` | Scalar calibration factor fitting and application |

## Quick Start

```python
from ins_pricing.pricing import (
    compute_exposure,
    build_frequency_severity,
    build_factor_table,
    compute_base_rate,
    rate_premium,
    fit_calibration_factor,
)

# 1. Exposure
df["exposure"] = compute_exposure(df, "start_date", "end_date")

# 2. Frequency / severity
df = build_frequency_severity(
    df,
    exposure_col="exposure",
    claim_count_col="claim_cnt",
    claim_amount_col="claim_amt",
)

# 3. Factor table
base_rate = compute_base_rate(df, loss_col="claim_amt", exposure_col="exposure")
vehicle_table = build_factor_table(
    df,
    factor_col="vehicle_type",
    loss_col="claim_amt",
    exposure_col="exposure",
    base_rate=base_rate,
)

# 4. Premium rating
premium = rate_premium(
    df,
    exposure_col="exposure",
    base_rate=base_rate,
    factor_tables={"vehicle_type": vehicle_table},
)

# 5. Calibration
factor = fit_calibration_factor(premium, df["claim_amt"].to_numpy(), target_lr=0.65)
premium_calibrated = premium * factor
```

## API Reference

### Data Quality (`data_quality.py`)

- `detect_leakage(df, target_col, *, exclude_cols=None, corr_threshold=0.995)` - detect identical or near-identical columns to target
- `profile_columns(df, cols=None)` - missing ratios, unique counts, numeric stats
- `validate_schema(df, required_cols, dtypes=None, *, raise_on_error=True)` - check required columns and types

### Exposure (`exposure.py`)

- `compute_exposure(df, start_col, end_col, *, unit="year", ...)` - date-based exposure (day/month/year)
- `aggregate_policy_level(df, policy_keys, *, exposure_col, ...)` - event-level to policy-level
- `build_frequency_severity(df, *, exposure_col, claim_count_col, claim_amount_col, ...)` - frequency, severity, pure premium

### Factors (`factors.py`)

- `bin_numeric(series, *, bins=10, method="quantile", ...)` - quantile or uniform binning (cached)
- `build_factor_table(df, *, factor_col, loss_col, exposure_col, ...)` - rate and relativity table with optional smoothing

### Rate Table (`rate_table.py`)

- `compute_base_rate(df, *, loss_col, exposure_col, ...)` - portfolio-level base rate
- `apply_factor_tables(df, factor_tables, ...)` - multiplicative factor array
- `rate_premium(df, *, exposure_col, base_rate, factor_tables, ...)` - exposure x base_rate x factors
- `RateTable` dataclass with `score(df, *, exposure_col)` method

### Calibration (`calibration.py`)

- `fit_calibration_factor(pred, actual, *, weight=None, target_lr=None)` - scalar calibration
- `apply_calibration(pred, factor)` - apply factor to predictions

### PSI (re-exported from `ins_pricing.utils.metrics`)

- `population_stability_index(expected, actual, *, bins=10, strategy="quantile")`
- `psi_report(expected_df, actual_df, *, features=None, bins=10, strategy="quantile")`
