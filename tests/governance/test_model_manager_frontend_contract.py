from __future__ import annotations

import re
from pathlib import Path


def test_model_manager_frontend_dom_contract() -> None:
    project_root = Path(__file__).resolve().parents[3]
    web_root = project_root / "model_manager_tool" / "web"

    html = (web_root / "index.html").read_text(encoding="utf-8")
    app_js = (web_root / "app.js").read_text(encoding="utf-8")

    html_ids = set(re.findall(r'id="([^"]+)"', html))
    js_ids = set(re.findall(r'\$\("([^"]+)"\)', app_js))

    missing_ids = sorted(js_ids - html_ids)
    assert not missing_ids, f"Missing DOM ids referenced by app.js: {missing_ids}"

    expected_tabs = {"import", "versions", "deploy", "audit", "admin"}
    button_tabs = set(re.findall(r'data-tab="([^"]+)"', html))
    panel_tabs = set(re.findall(r'id="tab-([^"]+)"', html))

    assert expected_tabs <= button_tabs
    assert expected_tabs <= panel_tabs
