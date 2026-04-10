# reporting

## Purpose

`reporting` owns Markdown report assembly and lightweight daily scheduling utilities for recurring
model monitoring outputs.

## Use When / Not For

- Use when model/monitoring data must be turned into a readable Markdown report.
- Use when a callable needs to run once per day on local process time.
- Not for model monitoring metric computation itself (handled by `production`).
- Not for enterprise job orchestration or external schedulers.

## Public Entrypoints

- `ReportPayload`
- `build_report`
- `write_report`
- `schedule_daily`

## Minimal Flow

```python
from ins_pricing.reporting import ReportPayload, write_report, schedule_daily

payload = ReportPayload(model_name="pricing_ft", model_version="v1", metrics={"rmse": 0.12})
write_report(payload, "Reports/model_report.md")
schedule_daily(lambda: write_report(payload, "Reports/model_report.md"), run_time="02:00")
```

## Further Reading

- Public export index: [../docs/api_reference.md](../docs/api_reference.md)
- Package navigation: [../README.md](../README.md)
