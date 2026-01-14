# pricing

Lightweight pricing loop utilities: data quality checks, exposure/targets,
factor tables, rate tables, calibration, and monitoring (PSI).

Quick start:

```python
from ins_pricing.pricing import (
    compute_exposure,
    build_frequency_severity,
    build_factor_table,
    compute_base_rate,
    rate_premium,
    fit_calibration_factor,
)

df["exposure"] = compute_exposure(df, "start_date", "end_date")
df = build_frequency_severity(
    df,
    exposure_col="exposure",
    claim_count_col="claim_cnt",
    claim_amount_col="claim_amt",
)

base_rate = compute_base_rate(df, loss_col="claim_amt", exposure_col="exposure")
vehicle_table = build_factor_table(
    df,
    factor_col="vehicle_type",
    loss_col="claim_amt",
    exposure_col="exposure",
    base_rate=base_rate,
)

premium = rate_premium(
    df,
    exposure_col="exposure",
    base_rate=base_rate,
    factor_tables={"vehicle_type": vehicle_table},
)

factor = fit_calibration_factor(premium, df["claim_amt"].to_numpy(), target_lr=0.65)
premium_calibrated = premium * factor
```
