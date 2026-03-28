"""
NiceGUI-based frontend entrypoint for Insurance Pricing Model Training.

Launch:
    python -m ins_pricing.frontend.app
"""

from __future__ import annotations

import os

from nicegui import ui

from ins_pricing.frontend.app_controller import PricingApp
from ins_pricing.frontend.ui_frontend import PricingFrontend

_pricing_app = PricingApp()


@ui.page("/")
def index():
    frontend = PricingFrontend(_pricing_app)
    frontend.build()


def main():
    """Launch the NiceGUI frontend server."""
    server_name = os.environ.get("NICEGUI_HOST", "0.0.0.0").strip() or "0.0.0.0"
    server_port = 7860
    port_env = os.environ.get("NICEGUI_PORT", "").strip()
    if port_env:
        server_port = int(port_env)

    ui.run(
        title="Insurance Pricing Model Training",
        host=server_name,
        port=server_port,
        reload=False,
        show=False,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
