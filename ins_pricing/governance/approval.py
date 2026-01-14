from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class ApprovalAction:
    actor: str
    decision: str
    timestamp: str
    comment: Optional[str] = None


@dataclass
class ApprovalRequest:
    model_name: str
    model_version: str
    requested_by: str
    requested_at: str
    status: str = "pending"
    actions: List[ApprovalAction] = field(default_factory=list)


class ApprovalStore:
    """Simple approval workflow stored as JSON."""

    def __init__(self, store_path: str | Path):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> List[dict]:
        if not self.store_path.exists():
            return []
        with self.store_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _save(self, payload: List[dict]) -> None:
        with self.store_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=True)

    def request(self, model_name: str, model_version: str, requested_by: str) -> ApprovalRequest:
        payload = self._load()
        req = ApprovalRequest(
            model_name=model_name,
            model_version=model_version,
            requested_by=requested_by,
            requested_at=datetime.utcnow().isoformat(),
        )
        payload.append(asdict(req))
        self._save(payload)
        return req

    def list_requests(self, model_name: Optional[str] = None) -> List[ApprovalRequest]:
        payload = self._load()
        requests = [ApprovalRequest(**entry) for entry in payload]
        if model_name is None:
            return requests
        return [req for req in requests if req.model_name == model_name]

    def act(
        self,
        model_name: str,
        model_version: str,
        *,
        actor: str,
        decision: str,
        comment: Optional[str] = None,
    ) -> ApprovalRequest:
        payload = self._load()
        found = None
        for entry in payload:
            if entry["model_name"] == model_name and entry["model_version"] == model_version:
                found = entry
                break
        if found is None:
            raise ValueError("Approval request not found.")
        action = ApprovalAction(
            actor=actor,
            decision=decision,
            timestamp=datetime.utcnow().isoformat(),
            comment=comment,
        )
        found["actions"].append(asdict(action))
        if decision.lower() in {"approve", "approved"}:
            found["status"] = "approved"
        elif decision.lower() in {"reject", "rejected"}:
            found["status"] = "rejected"
        self._save(payload)
        return ApprovalRequest(**found)
