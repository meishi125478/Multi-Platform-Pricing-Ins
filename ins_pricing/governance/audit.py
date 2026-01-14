from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class AuditEvent:
    action: str
    actor: str
    timestamp: str
    metadata: Dict[str, Any]
    note: Optional[str] = None


class AuditLogger:
    """Append-only JSONL audit log."""

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, action: str, actor: str, *, metadata: Optional[Dict[str, Any]] = None,
            note: Optional[str] = None) -> AuditEvent:
        event = AuditEvent(
            action=action,
            actor=actor,
            timestamp=datetime.utcnow().isoformat(),
            metadata=metadata or {},
            note=note,
        )
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(event), ensure_ascii=True) + "\n")
        return event
