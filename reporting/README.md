# reporting

Report builder and scheduler for model monitoring.

Example:

```python
from ins_pricing.reporting import ReportPayload, write_report, schedule_daily

payload = ReportPayload(
    model_name="pricing_ft",
    model_version="v1",
    metrics={"rmse": 0.12, "loss_ratio": 0.63},
    risk_trend=risk_df,
    drift_report=psi_df,
)
write_report(payload, "Reports/model_report.md")

schedule_daily(lambda: write_report(payload, "Reports/model_report.md"), run_time="02:00")
```
