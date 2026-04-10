from __future__ import annotations

from pathlib import Path

from ins_pricing.cli.utils.cli_config import (
    normalize_config_paths,
    resolve_runtime_config,
)


def test_normalize_config_paths_derives_optuna_storage_from_output_dir(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{}", encoding="utf-8")

    normalized = normalize_config_paths(
        {"output_dir": "./ResultsRun"},
        config_path,
    )

    assert normalized["output_dir"] == str((config_path.parent / "ResultsRun").resolve())
    assert normalized["optuna_storage"] == str(
        (config_path.parent / "ResultsRun" / "optuna" / "bayesopt.sqlite3").resolve()
    )


def test_resolve_runtime_config_derives_optuna_storage_from_output_dir() -> None:
    runtime_cfg = resolve_runtime_config({"output_dir": "/tmp/results_auto"})

    assert runtime_cfg["optuna_storage"] == "/tmp/results_auto/optuna/bayesopt.sqlite3"
