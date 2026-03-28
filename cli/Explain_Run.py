from __future__ import annotations

from pathlib import Path
from typing import Optional
if __package__ in {None, ""}:
    from _entry_bootstrap import ensure_repo_root  # type: ignore
else:
    from ._entry_bootstrap import ensure_repo_root

ensure_repo_root(__file__, __package__)

if __package__ in {None, ""}:
    from _config_runner import run_config, run_config_main  # type: ignore
else:
    from ._config_runner import run_config, run_config_main


def run(config_json: str | Path) -> None:
    """Run explain by config.json (runner.mode=explain)."""
    run_config(config_json)


def main(argv: Optional[list[str]] = None) -> None:
    run_config_main(
        "Explain_Run: run explain by config.json (runner.mode=explain).",
        argv,
    )


if __name__ == "__main__":
    main()
