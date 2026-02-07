from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AuditEvent:
    action: str
    actor: str
    timestamp: str
    metadata: Dict[str, Any]
    note: Optional[str] = None


class AuditLogger:
    """Append-only JSONL audit log with legacy interface compatibility."""

    def __init__(
        self,
        log_path: str | Path | None = None,
        *,
        audit_dir: str | Path | None = None,
    ):
        if audit_dir is not None:
            log_path = Path(audit_dir) / "audit_log.jsonl"
        if log_path is None:
            raise ValueError("Either log_path or audit_dir must be provided.")
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _read_all(self) -> List[Dict[str, Any]]:
        if not self.log_path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with self.log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                    if isinstance(payload, dict):
                        rows.append(payload)
                except json.JSONDecodeError:
                    continue
        return rows

    @staticmethod
    def _to_date(value: Any) -> Optional[date]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value).date()
            except ValueError:
                return None
        return None

    def log(
        self,
        action: str,
        *args: str,
        model_name: Optional[str] = None,
        user: Optional[str] = None,
        actor: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        note: Optional[str] = None,
    ) -> AuditEvent:
        if len(args) > 2:
            raise TypeError("log accepts at most 2 positional arguments after action.")

        # Legacy usage: log(action, model_name, user)
        if len(args) >= 1 and model_name is None and user is None and actor is None:
            if len(args) == 1 and metadata:
                # New usage in release manager: log(action, actor, metadata=...)
                actor = str(args[0])
            else:
                model_name = str(args[0])
        if len(args) >= 2 and user is None and actor is None:
            user = str(args[1])

        metadata_payload = dict(metadata or {})
        details_payload = dict(details or {})

        if model_name is None:
            model_name = (
                metadata_payload.get("model_name")
                or metadata_payload.get("name")
                or details_payload.get("model_name")
            )
        actor_value = actor or user or "system"
        # Use local wall-clock date to keep date-range filters consistent with callers
        # that pass datetime.now().date() (including tests and CLI users).
        timestamp = datetime.now().isoformat()

        payload: Dict[str, Any] = {
            "timestamp": timestamp,
            "action": action,
            "model_name": model_name,
            "user": actor_value,
            "details": details_payload,
            "metadata": metadata_payload,
        }
        if note is not None:
            payload["note"] = note

        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")

        return AuditEvent(
            action=action,
            actor=actor_value,
            timestamp=timestamp,
            metadata=metadata_payload or details_payload,
            note=note,
        )

    def get_logs(
        self,
        *,
        model_name: Optional[str] = None,
        start_date: Any = None,
        end_date: Any = None,
    ) -> List[Dict[str, Any]]:
        rows = self._read_all()
        start = self._to_date(start_date)
        end = self._to_date(end_date)

        out: List[Dict[str, Any]] = []
        for row in rows:
            row_model = (
                row.get("model_name")
                or (row.get("details") or {}).get("model_name")
                or (row.get("metadata") or {}).get("model_name")
                or (row.get("metadata") or {}).get("name")
            )
            if model_name is not None and str(row_model) != str(model_name):
                continue

            ts_date = self._to_date(row.get("timestamp"))
            if start is not None and (ts_date is None or ts_date < start):
                continue
            if end is not None and (ts_date is None or ts_date > end):
                continue
            out.append(row)
        return out

    def get_audit_trail(self, model_name: str) -> List[Dict[str, Any]]:
        return self.get_logs(model_name=model_name)
