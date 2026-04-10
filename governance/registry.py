from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ins_pricing.exceptions import GovernanceError


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
        self._io_lock = threading.RLock()

    def _load_unlocked(self) -> Dict[str, List[dict]]:
        if not self.registry_path.exists():
            return {}
        with self.registry_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            return {}
        return {str(k): list(v) for k, v in payload.items()}

    def _load(self) -> Dict[str, List[dict]]:
        with self._io_lock:
            return self._load_unlocked()

    def _save_unlocked(self, payload: Dict[str, List[dict]]) -> None:
        tmp_path = self.registry_path.with_suffix(f"{self.registry_path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=True)
            fh.flush()
        tmp_path.replace(self.registry_path)

    def _save(self, payload: Dict[str, List[dict]]) -> None:
        with self._io_lock:
            self._save_unlocked(payload)

    @staticmethod
    def _version_key(version: Any) -> Tuple[int, ...]:
        text = str(version or "")
        out: List[int] = []
        for token in text.replace("-", ".").split("."):
            if token.isdigit():
                out.append(int(token))
            else:
                digits = "".join(ch for ch in token if ch.isdigit())
                out.append(int(digits) if digits else 0)
        return tuple(out or [0])

    def _normalize_entry(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        entry = dict(metadata)
        name = entry.get("model_name") or entry.get("name")
        version = entry.get("version")
        if not name:
            raise GovernanceError("Model metadata must include 'model_name' or 'name'.")
        if not version:
            raise GovernanceError("Model metadata must include 'version'.")

        entry["model_name"] = str(name)
        entry["name"] = str(name)
        entry["version"] = str(version)
        entry.setdefault("created_at", datetime.utcnow().isoformat())
        entry.setdefault("metrics", {})
        entry.setdefault("status", "candidate")

        artifacts = entry.get("artifacts")
        if artifacts is not None:
            normalized_artifacts: List[Dict[str, Any]] = []
            for artifact in artifacts:
                if isinstance(artifact, ModelArtifact):
                    normalized_artifacts.append(asdict(artifact))
                elif isinstance(artifact, dict):
                    normalized_artifacts.append(dict(artifact))
                else:
                    normalized_artifacts.append({"path": str(artifact)})
            entry["artifacts"] = normalized_artifacts
        return entry

    @staticmethod
    def _to_model_version(entry: Dict[str, Any]) -> ModelVersion:
        artifacts_payload = entry.get("artifacts") or []
        artifacts: List[ModelArtifact] = []
        for item in artifacts_payload:
            if isinstance(item, dict):
                artifacts.append(
                    ModelArtifact(
                        path=str(item.get("path", "")),
                        description=item.get("description"),
                    )
                )
            else:
                artifacts.append(ModelArtifact(path=str(item)))
        return ModelVersion(
            name=str(entry.get("model_name") or entry.get("name")),
            version=str(entry.get("version")),
            created_at=str(entry.get("created_at") or ""),
            metrics=dict(entry.get("metrics") or {}),
            tags=dict(entry.get("tags") or {}),
            artifacts=artifacts,
            status=str(entry.get("status", "candidate")),
            notes=entry.get("notes"),
        )

    def register(
        self,
        name: str | Dict[str, Any],
        version: Optional[str] = None,
        *,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
        artifacts: Optional[List[ModelArtifact]] = None,
        status: str = "candidate",
        notes: Optional[str] = None,
    ) -> ModelVersion:
        if isinstance(name, dict):
            entry = self._normalize_entry(name)
        else:
            entry = self._normalize_entry(
                {
                    "model_name": name,
                    "version": version,
                    "metrics": metrics or {},
                    "tags": tags or {},
                    "artifacts": artifacts or [],
                    "status": status,
                    "notes": notes,
                }
            )

        with self._io_lock:
            payload = self._load_unlocked()
            model_name = entry["model_name"]
            versions = payload.setdefault(model_name, [])
            if any(str(v.get("version")) == str(entry["version"]) for v in versions):
                raise GovernanceError(
                    f"Model '{model_name}' version '{entry['version']}' already exists."
                )
            versions.append(entry)
            self._save_unlocked(payload)
        return self._to_model_version(entry)

    def exists(self, name: str) -> bool:
        payload = self._load()
        return bool(payload.get(name))

    def list_all(self) -> List[Dict[str, Any]]:
        payload = self._load()
        out: List[Dict[str, Any]] = []
        for name in sorted(payload.keys()):
            out.append(self.get_latest(name))
        return out

    def get(self, name: str, version: Optional[str] = None) -> Dict[str, Any]:
        if version is None:
            return self.get_latest(name)
        payload = self._load()
        for entry in payload.get(name, []):
            if str(entry.get("version")) == str(version):
                return dict(entry)
        raise GovernanceError(f"Model '{name}' version '{version}' not found.")

    def update(
        self,
        name: str,
        updates: Dict[str, Any],
        *,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._io_lock:
            payload = self._load_unlocked()
            if name not in payload or not payload[name]:
                raise GovernanceError(f"Model '{name}' not found.")
            entries = payload[name]

            target_idx = -1
            if version is None:
                versions = [self._version_key(v.get("version")) for v in entries]
                target_idx = int(max(range(len(entries)), key=lambda i: versions[i]))
            else:
                for idx, entry in enumerate(entries):
                    if str(entry.get("version")) == str(version):
                        target_idx = idx
                        break
            if target_idx < 0:
                raise GovernanceError(f"Model '{name}' version '{version}' not found.")

            entries[target_idx] = {**entries[target_idx], **dict(updates)}
            payload[name] = entries
            self._save_unlocked(payload)
            return dict(entries[target_idx])

    def delete(self, name: str, *, version: Optional[str] = None) -> None:
        with self._io_lock:
            payload = self._load_unlocked()
            if name not in payload:
                return
            if version is None:
                payload.pop(name, None)
            else:
                kept = [v for v in payload[name] if str(v.get("version")) != str(version)]
                if kept:
                    payload[name] = kept
                else:
                    payload.pop(name, None)
            self._save_unlocked(payload)

    def get_versions(self, name: str) -> List[str]:
        payload = self._load()
        versions = [str(v.get("version")) for v in payload.get(name, []) if v.get("version") is not None]
        return sorted(versions, key=self._version_key)

    def get_latest(self, name: str) -> Dict[str, Any]:
        payload = self._load()
        entries = payload.get(name, [])
        if not entries:
            raise GovernanceError(f"Model '{name}' not found.")
        latest = max(entries, key=lambda x: self._version_key(x.get("version")))
        return dict(latest)

    def list_versions(self, name: str) -> List[ModelVersion]:
        payload = self._load()
        versions = payload.get(name, [])
        return [self._to_model_version(v) for v in versions]

    def get_version(self, name: str, version: str) -> Optional[ModelVersion]:
        payload = self._load()
        for entry in payload.get(name, []):
            if str(entry.get("version")) == str(version):
                return self._to_model_version(entry)
        return None

    def promote(
        self, name: str, version: str, *, new_status: str = "production"
    ) -> None:
        with self._io_lock:
            payload = self._load_unlocked()
            if name not in payload:
                raise GovernanceError("Model not found in registry.")
            updated = False
            for entry in payload[name]:
                if str(entry.get("version")) == str(version):
                    entry["status"] = new_status
                    updated = True
                elif new_status == "production" and entry.get("status") == "production":
                    entry["status"] = "archived"
            if not updated:
                raise GovernanceError("Version not found in registry.")
            self._save_unlocked(payload)
