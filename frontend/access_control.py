"""RBAC helpers for the frontend UI."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Dict, Iterable, List, Optional, Set


class AuthorizationError(PermissionError):
    """Raised when authentication or authorization fails."""


ALL_PERMISSIONS = (
    "system:view",
    "task:run",
    "config:edit",
    "account:manage",
)


DEFAULT_ROLES = {
    "admin": {
        "description": "Full access to frontend runtime operations and account management.",
        "permissions": list(ALL_PERMISSIONS),
    },
    "operator": {
        "description": "Can configure and run tasks, but cannot manage accounts.",
        "permissions": ["system:view", "task:run", "config:edit"],
    },
    "viewer": {
        "description": "Read-only visibility into server runtime status.",
        "permissions": ["system:view"],
    },
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_permissions(permissions: Iterable[str]) -> List[str]:
    return sorted({str(item).strip() for item in permissions if str(item).strip()})


@dataclass(frozen=True)
class AuthenticatedFrontendUser:
    username: str
    roles: List[str]
    permissions: List[str]


class FrontendAuthStore:
    """Simple JSON-backed RBAC store with salted-password authentication."""

    def __init__(
        self,
        store_path: str | Path,
        *,
        bootstrap_admin_username: str = "admin",
        bootstrap_admin_password: Optional[str] = None,
    ) -> None:
        self.store_path = Path(store_path).expanduser().resolve()
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self.bootstrap_admin_username = bootstrap_admin_username.strip() or "admin"
        env_password = os.environ.get("INS_PRICING_FRONTEND_ADMIN_PASSWORD")
        self._default_admin_password_in_use = not bootstrap_admin_password and not env_password
        self.bootstrap_admin_password = bootstrap_admin_password or env_password or "admin123!"
        self._ensure_bootstrap_state()

    @property
    def default_admin_password_in_use(self) -> bool:
        return self._default_admin_password_in_use

    def _load(self) -> Dict[str, Dict[str, dict]]:
        if not self.store_path.exists():
            return {"roles": {}, "users": {}}
        with self.store_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return {"roles": {}, "users": {}}
        roles = payload.get("roles")
        users = payload.get("users")
        return {
            "roles": dict(roles) if isinstance(roles, dict) else {},
            "users": dict(users) if isinstance(users, dict) else {},
        }

    def _save(self, payload: Dict[str, Dict[str, dict]]) -> None:
        with self.store_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)

    @staticmethod
    def _hash_password(password: str, salt: str) -> str:
        raw = f"{salt}:{password}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _make_user_record(password: str, roles: Iterable[str]) -> Dict[str, object]:
        salt = secrets.token_hex(16)
        now = _utcnow_iso()
        return {
            "salt": salt,
            "password_hash": FrontendAuthStore._hash_password(password, salt),
            "roles": sorted({str(role).strip() for role in roles if str(role).strip()}),
            "active": True,
            "created_at": now,
            "updated_at": now,
        }

    def _ensure_bootstrap_state(self) -> None:
        with self._lock:
            payload = self._load()
            changed = False
            roles = payload.setdefault("roles", {})
            users = payload.setdefault("users", {})

            for role_name, role_cfg in DEFAULT_ROLES.items():
                normalized = {
                    "description": str(role_cfg.get("description", "")),
                    "permissions": _normalize_permissions(role_cfg.get("permissions", [])),
                }
                existing = roles.get(role_name)
                if existing != normalized:
                    roles[role_name] = normalized
                    changed = True

            if self.bootstrap_admin_username not in users:
                users[self.bootstrap_admin_username] = self._make_user_record(
                    self.bootstrap_admin_password,
                    roles=["admin"],
                )
                changed = True

            if changed:
                self._save(payload)

    @staticmethod
    def _permissions_for_roles(
        role_names: Iterable[str],
        payload: Dict[str, Dict[str, dict]],
    ) -> Set[str]:
        roles = payload.get("roles", {})
        permissions: Set[str] = set()
        for role_name in role_names:
            role_cfg = roles.get(role_name)
            if not isinstance(role_cfg, dict):
                continue
            permissions.update(_normalize_permissions(role_cfg.get("permissions", [])))
        return permissions

    def _build_user(self, username: str, payload: Dict[str, Dict[str, dict]]) -> AuthenticatedFrontendUser:
        users = payload.get("users", {})
        record = users.get(username)
        if not isinstance(record, dict):
            raise AuthorizationError("User not found.")
        if not bool(record.get("active", True)):
            raise AuthorizationError("User is disabled.")
        role_names = sorted(
            {
                str(role).strip()
                for role in record.get("roles", [])
                if str(role).strip()
            }
        )
        permissions = sorted(self._permissions_for_roles(role_names, payload))
        return AuthenticatedFrontendUser(
            username=username,
            roles=role_names,
            permissions=permissions,
        )

    def authenticate(self, username: str, password: str) -> AuthenticatedFrontendUser:
        username = (username or "").strip()
        if not username or not password:
            raise AuthorizationError("Username and password are required.")

        with self._lock:
            payload = self._load()
            record = payload.get("users", {}).get(username)
            if not isinstance(record, dict):
                raise AuthorizationError("Invalid username or password.")
            if not bool(record.get("active", True)):
                raise AuthorizationError("User is disabled.")

            salt = str(record.get("salt", ""))
            expected_hash = str(record.get("password_hash", ""))
            actual_hash = self._hash_password(password, salt)
            if not hmac.compare_digest(actual_hash, expected_hash):
                raise AuthorizationError("Invalid username or password.")
            return self._build_user(username, payload)

    def get_user(self, username: str) -> AuthenticatedFrontendUser:
        with self._lock:
            payload = self._load()
            return self._build_user(username, payload)

    def has_permission(self, username: str, permission: str) -> bool:
        try:
            user = self.get_user(username)
        except AuthorizationError:
            return False
        return str(permission).strip() in user.permissions

    def require_permission(self, username: str, permission: str) -> None:
        if not self.has_permission(username, permission):
            raise AuthorizationError(f"Permission denied: {permission}")

    def list_users(self) -> List[Dict[str, object]]:
        with self._lock:
            payload = self._load()
            users = payload.get("users", {})
            result: List[Dict[str, object]] = []
            for username in sorted(users.keys()):
                record = users.get(username)
                if not isinstance(record, dict):
                    continue
                result.append(
                    {
                        "username": username,
                        "roles": sorted(
                            {
                                str(role).strip()
                                for role in record.get("roles", [])
                                if str(role).strip()
                            }
                        ),
                        "active": bool(record.get("active", True)),
                        "created_at": record.get("created_at"),
                        "updated_at": record.get("updated_at"),
                    }
                )
            return result

    def list_roles(self) -> Dict[str, Dict[str, object]]:
        with self._lock:
            payload = self._load()
            roles = payload.get("roles", {})
            result: Dict[str, Dict[str, object]] = {}
            for role_name, role_cfg in sorted(roles.items()):
                if not isinstance(role_cfg, dict):
                    continue
                result[role_name] = {
                    "description": str(role_cfg.get("description", "")),
                    "permissions": _normalize_permissions(role_cfg.get("permissions", [])),
                }
            return result

    def create_user(self, username: str, password: str, roles: Iterable[str]) -> Dict[str, object]:
        username = (username or "").strip()
        if not username:
            raise ValueError("username is required")
        if not password:
            raise ValueError("password is required")

        role_set = sorted({str(role).strip() for role in roles if str(role).strip()})
        if not role_set:
            raise ValueError("at least one role is required")

        with self._lock:
            payload = self._load()
            users = payload.setdefault("users", {})
            roles_cfg = payload.setdefault("roles", {})
            missing = [role for role in role_set if role not in roles_cfg]
            if missing:
                raise ValueError(f"unknown roles: {missing}")
            if username in users:
                raise ValueError(f"user already exists: {username}")
            users[username] = self._make_user_record(password, role_set)
            self._save(payload)
            return {"username": username, "roles": role_set, "active": True}

    def set_user_roles(self, username: str, roles: Iterable[str]) -> Dict[str, object]:
        username = (username or "").strip()
        if not username:
            raise ValueError("username is required")

        role_set = sorted({str(role).strip() for role in roles if str(role).strip()})
        if not role_set:
            raise ValueError("at least one role is required")

        with self._lock:
            payload = self._load()
            users = payload.setdefault("users", {})
            roles_cfg = payload.setdefault("roles", {})
            if username not in users:
                raise ValueError(f"user not found: {username}")
            missing = [role for role in role_set if role not in roles_cfg]
            if missing:
                raise ValueError(f"unknown roles: {missing}")
            users[username]["roles"] = role_set
            users[username]["updated_at"] = _utcnow_iso()
            self._save(payload)
            return {"username": username, "roles": role_set}

    def set_user_active(self, username: str, active: bool) -> Dict[str, object]:
        username = (username or "").strip()
        if not username:
            raise ValueError("username is required")

        with self._lock:
            payload = self._load()
            users = payload.setdefault("users", {})
            if username not in users:
                raise ValueError(f"user not found: {username}")

            new_active = bool(active)
            if not new_active:
                admin_count = 0
                for _, record in users.items():
                    if not isinstance(record, dict):
                        continue
                    if not bool(record.get("active", True)):
                        continue
                    role_names = {
                        str(role).strip()
                        for role in record.get("roles", [])
                        if str(role).strip()
                    }
                    if "admin" in role_names:
                        admin_count += 1
                record = users[username]
                role_names = {
                    str(role).strip()
                    for role in record.get("roles", [])
                    if str(role).strip()
                }
                if "admin" in role_names and admin_count <= 1:
                    raise ValueError("cannot disable the last active admin")

            users[username]["active"] = new_active
            users[username]["updated_at"] = _utcnow_iso()
            self._save(payload)
            return {"username": username, "active": new_active}

    def set_user_password(self, username: str, new_password: str) -> Dict[str, object]:
        username = (username or "").strip()
        if not username:
            raise ValueError("username is required")
        if not new_password:
            raise ValueError("new_password is required")

        with self._lock:
            payload = self._load()
            users = payload.setdefault("users", {})
            record = users.get(username)
            if not isinstance(record, dict):
                raise ValueError(f"user not found: {username}")

            salt = secrets.token_hex(16)
            record["salt"] = salt
            record["password_hash"] = self._hash_password(new_password, salt)
            record["updated_at"] = _utcnow_iso()
            self._save(payload)
            return {"username": username}
