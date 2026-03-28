from __future__ import annotations

from pathlib import Path


def test_double_lift_target_default_value_is_response() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "frontend" / "ui_frontend.py").read_text(encoding="utf-8")
    assert 'dl_tgt = ui.input("Target", value="response")' in source
