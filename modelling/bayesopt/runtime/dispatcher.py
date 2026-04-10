from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

_VALID_MODEL_KEYS = {"glm", "xgb", "resn", "ft", "gnn"}


@dataclass(frozen=True)
class EngineDecision:
    model_key: str
    ft_role: str
    supported: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_key": self.model_key,
            "ft_role": self.ft_role,
            "supported": self.supported,
            "reason": self.reason,
        }


def _runtime_supported(model_key: str, *, ft_role: str) -> bool:
    # Current capability matrix: glm/xgb/resn/gnn and all FT roles.
    if model_key in {"glm", "xgb", "resn", "gnn"}:
        return True
    if model_key == "ft":
        return ft_role in {"model", "embedding", "unsupervised_embedding"}
    return False


def resolve_engine_decision(_config: Any, *, model_key: str, ft_role: str) -> EngineDecision:
    resolved_model_key = str(model_key).strip().lower()
    role = str(ft_role or "model").strip().lower()

    if resolved_model_key not in _VALID_MODEL_KEYS:
        raise ValueError(
            f"Unsupported model_key={resolved_model_key!r}; expected one of "
            f"{sorted(_VALID_MODEL_KEYS)}."
        )

    if _runtime_supported(resolved_model_key, ft_role=role):
        return EngineDecision(
            model_key=resolved_model_key,
            ft_role=role,
            supported=True,
            reason="enabled in runtime capability matrix",
        )

    raise ValueError(
        "Runtime dispatch is not enabled for "
        f"model_key={resolved_model_key!r}, ft_role={role!r}."
    )
