from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def _df_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "_(no data)_"
    data = df.copy()
    if len(data) > max_rows:
        data = data.head(max_rows)
    headers = list(data.columns)
    rows = data.astype(str).values.tolist()
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


@dataclass
class ReportPayload:
    model_name: str
    model_version: str
    metrics: Dict[str, float]
    risk_trend: Optional[pd.DataFrame] = None
    drift_report: Optional[pd.DataFrame] = None
    validation_table: Optional[pd.DataFrame] = None
    extra_notes: Optional[str] = None


def build_report(payload: ReportPayload) -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    metrics_lines = [f"- {k}: {v:.6f}" for k, v in payload.metrics.items()]
    metrics_block = "\n".join(metrics_lines) if metrics_lines else "_(no metrics)_"

    report = [
        f"# Model Report: {payload.model_name} ({payload.model_version})",
        "",
        f"Generated at: {now}",
        "",
        "## Model Validation",
        metrics_block,
    ]

    if payload.validation_table is not None:
        report.extend(["", "### Validation Details", _df_to_markdown(payload.validation_table)])

    report.extend(["", "## Drift / Stability"])
    report.append(_df_to_markdown(payload.drift_report))

    report.extend(["", "## Risk Trend"])
    report.append(_df_to_markdown(payload.risk_trend))

    if payload.extra_notes:
        report.extend(["", "## Notes", payload.extra_notes])

    return "\n".join(report).strip() + "\n"


def write_report(payload: ReportPayload, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = build_report(payload)
    output_path.write_text(content, encoding="utf-8")
    return output_path
