from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ModelArtifact:
    path: str
    description: Optional[str] = None


@dataclass
class ModelVersion:
    name: str
    version: str
    created_at: str
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    artifacts: List[ModelArtifact] = field(default_factory=list)
    status: str = "candidate"
    notes: Optional[str] = None


class ModelRegistry:
    """Lightweight JSON-based model registry."""

    def __init__(self, registry_path: str | Path):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> Dict[str, List[dict]]:
        if not self.registry_path.exists():
            return {}
        with self.registry_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _save(self, payload: Dict[str, List[dict]]) -> None:
        with self.registry_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=True)

    def register(
        self,
        name: str,
        version: str,
        *,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
        artifacts: Optional[List[ModelArtifact]] = None,
        status: str = "candidate",
        notes: Optional[str] = None,
    ) -> ModelVersion:
        payload = self._load()
        created_at = datetime.utcnow().isoformat()
        entry = ModelVersion(
            name=name,
            version=version,
            created_at=created_at,
            metrics=metrics or {},
            tags=tags or {},
            artifacts=artifacts or [],
            status=status,
            notes=notes,
        )
        payload.setdefault(name, []).append(asdict(entry))
        self._save(payload)
        return entry

    def list_versions(self, name: str) -> List[ModelVersion]:
        payload = self._load()
        versions = payload.get(name, [])
        return [ModelVersion(**v) for v in versions]

    def get_version(self, name: str, version: str) -> Optional[ModelVersion]:
        for entry in self.list_versions(name):
            if entry.version == version:
                return entry
        return None

    def promote(
        self, name: str, version: str, *, new_status: str = "production"
    ) -> None:
        payload = self._load()
        if name not in payload:
            raise ValueError("Model not found in registry.")
        updated = False
        for entry in payload[name]:
            if entry["version"] == version:
                entry["status"] = new_status
                updated = True
            elif new_status == "production":
                if entry.get("status") == "production":
                    entry["status"] = "archived"
        if not updated:
            raise ValueError("Version not found in registry.")
        self._save(payload)
