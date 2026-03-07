# reporting

Markdown report builder and daily scheduler for model monitoring.

## API

```python
from ins_pricing.reporting import ReportPayload, build_report, write_report, schedule_daily
```

### ReportPayload

```python
@dataclass
class ReportPayload:
    model_name: str
    model_version: str
    metrics: Dict[str, float]
    risk_trend: Optional[pd.DataFrame] = None
    drift_report: Optional[pd.DataFrame] = None
    validation_table: Optional[pd.DataFrame] = None
    extra_notes: Optional[str] = None
```

### Functions

- `build_report(payload: ReportPayload) -> str` - generate Markdown report string
- `write_report(payload: ReportPayload, output_path) -> Path` - write report to file
- `schedule_daily(job_fn, *, run_time="01:00", stop_event=None) -> threading.Thread` - run a callable daily at the specified local time in a background thread

## Example

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

# Schedule daily at 2 AM
schedule_daily(lambda: write_report(payload, "Reports/model_report.md"), run_time="02:00")
```
