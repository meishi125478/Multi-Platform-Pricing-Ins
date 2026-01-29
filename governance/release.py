from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .audit import AuditLogger
from .registry import ModelRegistry


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
        state_dir: str | Path,
        *,
        registry: Optional[ModelRegistry] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.registry = registry
        self.audit_logger = audit_logger

    def _state_path(self, env: str) -> Path:
        return self.state_dir / f"{env}.json"

    def _load(self, env: str) -> DeploymentState:
        path = self._state_path(env)
        if not path.exists():
            return DeploymentState(env=env)
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
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
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=True)

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
