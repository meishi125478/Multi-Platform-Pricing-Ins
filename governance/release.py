from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ins_pricing.exceptions import GovernanceError
from ins_pricing.governance.audit import AuditLogger
from ins_pricing.governance.registry import ModelRegistry


@dataclass
class ModelRef:
    name: str
    version: str
    activated_at: str
    actor: Optional[str] = None
    note: Optional[str] = None


@dataclass
class DeploymentState:
    env: str
    active: Optional[ModelRef] = None
    history: List[ModelRef] = field(default_factory=list)
    updated_at: Optional[str] = None


class ReleaseManager:
    """Environment release manager with rollback support."""

    def __init__(
        self,
        state_dir: str | Path | None = None,
        *,
        release_dir: str | Path | None = None,
        registry: Optional[ModelRegistry] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        if release_dir is not None:
            state_dir = release_dir
        if state_dir is None:
            raise ValueError("Either state_dir or release_dir must be provided.")
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.registry = registry
        self.audit_logger = audit_logger
        self._release_manifest_path = self.state_dir / "releases.json"

    @staticmethod
    def _load_json(path: Path, *, default: dict) -> dict:
        if not path.exists():
            return dict(default)
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            return dict(default)
        merged = dict(default)
        merged.update(payload)
        return merged

    @staticmethod
    def _save_json(path: Path, payload: dict) -> None:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=True)

    def _load_release_manifest(self) -> dict:
        payload = self._load_json(self._release_manifest_path, default={"releases": {}, "production": {}})
        payload.setdefault("releases", {})
        payload.setdefault("production", {})
        return payload

    def _save_release_manifest(self, payload: dict) -> None:
        self._save_json(self._release_manifest_path, payload)

    @staticmethod
    def _set_production_release(payload: dict, *, release_id: str) -> dict:
        releases = payload.get("releases", {})
        if release_id not in releases:
            raise GovernanceError(f"Release '{release_id}' not found.")
        target = releases[release_id]
        model_name = str(target["model_name"])
        previous_id = payload.get("production", {}).get(model_name)
        now = datetime.utcnow().isoformat()

        if previous_id and previous_id in releases and previous_id != release_id:
            releases[previous_id]["status"] = "archived"
            releases[previous_id]["updated_at"] = now

        target["status"] = "production"
        target["updated_at"] = now
        payload.setdefault("production", {})[model_name] = release_id
        return target

    def _state_path(self, env: str) -> Path:
        return self.state_dir / f"{env}.json"

    def _load(self, env: str) -> DeploymentState:
        path = self._state_path(env)
        payload = self._load_json(path, default={"env": env, "active": None, "history": [], "updated_at": None})
        active = payload.get("active")
        history = payload.get("history", [])
        return DeploymentState(
            env=payload.get("env", env),
            active=ModelRef(**active) if active else None,
            history=[ModelRef(**item) for item in history],
            updated_at=payload.get("updated_at"),
        )

    def _save(self, state: DeploymentState) -> None:
        payload = {
            "env": state.env,
            "active": asdict(state.active) if state.active else None,
            "history": [asdict(item) for item in state.history],
            "updated_at": state.updated_at,
        }
        path = self._state_path(state.env)
        self._save_json(path, payload)

    def get_active(self, env: str) -> Optional[ModelRef]:
        state = self._load(env)
        return state.active

    def list_history(self, env: str) -> List[ModelRef]:
        return self._load(env).history

    def deploy(
        self,
        env: str,
        name: str,
        version: str,
        *,
        actor: Optional[str] = None,
        note: Optional[str] = None,
        update_registry_status: bool = True,
        registry_status: str = "production",
    ) -> DeploymentState:
        state = self._load(env)
        if state.active and state.active.name == name and state.active.version == version:
            return state

        if state.active is not None:
            state.history.append(state.active)

        now = datetime.utcnow().isoformat()
        state.active = ModelRef(
            name=name,
            version=version,
            activated_at=now,
            actor=actor,
            note=note,
        )
        state.updated_at = now
        self._save(state)

        if self.registry and update_registry_status:
            self.registry.promote(name, version, new_status=registry_status)

        if self.audit_logger:
            self.audit_logger.log(
                "deploy",
                actor or "unknown",
                metadata={"env": env, "name": name, "version": version},
                note=note,
            )

        return state

    def rollback(
        self,
        env: str,
        *,
        actor: Optional[str] = None,
        note: Optional[str] = None,
        update_registry_status: bool = False,
        registry_status: str = "production",
    ) -> DeploymentState:
        state = self._load(env)
        if not state.history:
            raise ValueError("No history available to rollback.")

        previous = state.history.pop()
        now = datetime.utcnow().isoformat()
        state.active = ModelRef(
            name=previous.name,
            version=previous.version,
            activated_at=now,
            actor=actor or previous.actor,
            note=note or previous.note,
        )
        state.updated_at = now
        self._save(state)

        if self.registry and update_registry_status:
            self.registry.promote(previous.name, previous.version, new_status=registry_status)

        if self.audit_logger:
            self.audit_logger.log(
                "rollback",
                actor or "unknown",
                metadata={"env": env, "name": previous.name, "version": previous.version},
                note=note,
            )

        return state

    # ---------------------------------------------------------------------
    # Legacy release API compatibility (tests/governance/test_release.py)
    # ---------------------------------------------------------------------
    def create_release(
        self,
        model_name: str,
        version: str,
        artifacts: Optional[List[str]] = None,
    ) -> str:
        now = datetime.utcnow().isoformat()
        release_id = f"{model_name}:{version}:{int(datetime.utcnow().timestamp() * 1_000_000)}"
        payload = self._load_release_manifest()
        payload["releases"][release_id] = {
            "release_id": release_id,
            "model_name": model_name,
            "version": version,
            "artifacts": list(artifacts or []),
            "status": "candidate",
            "created_at": now,
            "updated_at": now,
        }
        self._save_release_manifest(payload)
        return release_id

    def release_exists(self, release_id: str) -> bool:
        payload = self._load_release_manifest()
        return release_id in payload.get("releases", {})

    def get_release_info(self, release_id: str) -> dict:
        payload = self._load_release_manifest()
        info = payload.get("releases", {}).get(release_id)
        if info is None:
            raise GovernanceError(f"Release '{release_id}' not found.")
        return dict(info)

    def promote_to_production(self, release_id: str) -> None:
        payload = self._load_release_manifest()
        target = self._set_production_release(payload, release_id=release_id)
        self._save_release_manifest(payload)

        if self.registry is not None:
            try:
                self.registry.promote(str(target["model_name"]), str(target["version"]), new_status="production")
            except Exception:
                pass
        if self.audit_logger is not None:
            self.audit_logger.log(
                "release_promoted",
                model_name=str(target["model_name"]),
                user="system",
                details={"release_id": release_id, "version": str(target["version"])},
            )

    def rollback_to(self, release_id: str) -> None:
        payload = self._load_release_manifest()
        target = self._set_production_release(payload, release_id=release_id)
        self._save_release_manifest(payload)

        if self.audit_logger is not None:
            self.audit_logger.log(
                "release_rollback",
                model_name=str(target["model_name"]),
                user="system",
                details={"release_id": release_id, "version": str(target["version"])},
            )

    def get_production_release(self, model_name: str) -> dict:
        payload = self._load_release_manifest()
        release_id = payload.get("production", {}).get(model_name)
        if release_id is None:
            raise GovernanceError(f"No production release for model '{model_name}'.")
        info = payload.get("releases", {}).get(release_id)
        if info is None:
            raise GovernanceError(f"Release '{release_id}' not found.")
        return dict(info)
