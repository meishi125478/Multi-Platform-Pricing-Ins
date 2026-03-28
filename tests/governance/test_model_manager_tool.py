from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from model_manager_tool.auth import AuthorizationError
from model_manager_tool.manager import ModelManagerService


def _write_bytes(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_import_model_with_mixed_artifacts(tmp_path: Path) -> None:
    service = ModelManagerService(tmp_path / "model_manager")

    model_file = _write_bytes(tmp_path / "tmp_artifacts" / "model.pkl", b"binary-model")
    factor_excel = _write_bytes(tmp_path / "tmp_artifacts" / "factors.xlsx", b"not-real-excel")

    imported = service.import_model(
        model_name="pricing_model",
        version="1.0.0",
        actor="admin",
        artifact_paths=[model_file, factor_excel],
        model_type="xgb",
        metrics={"mae": 0.31},
        tags={"channel": "direct"},
        notes="initial import",
    )

    assert imported["model_name"] == "pricing_model"
    assert imported["version"] == "1.0.0"

    versions = service.list_model_versions("pricing_model")
    assert len(versions) == 1
    manifest = versions[0].get("artifact_manifest", [])
    assert len(manifest) == 2
    kinds = {item["kind"] for item in manifest}
    assert "model_file" in kinds
    assert "factor_table_excel" in kinds


def test_import_model_reports_progress_callbacks(tmp_path: Path) -> None:
    service = ModelManagerService(tmp_path / "model_manager")
    model_file = _write_bytes(tmp_path / "tmp_artifacts" / "model.pkl", b"binary-model")

    events: list[dict[str, Any]] = []

    imported = service.import_model(
        model_name="progress_model",
        version="1.0.0",
        actor="admin",
        artifact_paths=[model_file],
        progress_callback=lambda payload: events.append(dict(payload)),
    )

    assert imported["model_name"] == "progress_model"
    assert events
    assert events[0]["stage"] == "started"
    assert events[-1]["stage"] == "completed"
    assert float(events[-1]["value"]) == 1.0
    assert any(item["stage"] == "artifact_complete" for item in events)


def test_import_model_persists_posix_relative_paths(tmp_path: Path) -> None:
    service = ModelManagerService(tmp_path / "model_manager")
    model_file = _write_bytes(tmp_path / "tmp_artifacts" / "model.bin", b"model-v-posix")

    imported = service.import_model(
        model_name="path_model",
        version="1.0.0",
        actor="admin",
        artifact_paths=[model_file],
    )

    manifest = imported.get("artifact_manifest", [])
    assert len(manifest) == 1
    rel_path = str(manifest[0].get("relative_path", ""))
    assert rel_path
    assert "\\" not in rel_path
    assert "/" in rel_path


def test_purge_all_models_clears_registry_artifacts_and_release_state(tmp_path: Path) -> None:
    service = ModelManagerService(tmp_path / "model_manager")
    model_file = _write_bytes(tmp_path / "tmp_artifacts" / "model.bin", b"model-v1")

    service.import_model(
        model_name="purge_model",
        version="1.0.0",
        actor="admin",
        artifact_paths=[model_file],
    )
    service.deploy_model(env="staging", model_name="purge_model", version="1.0.0", actor="admin")

    summary = service.purge_all_models(actor="admin", clear_audit=False)

    assert summary["model_count"] == 1
    assert summary["version_count"] == 1
    assert summary["artifact_file_count"] >= 1
    assert service.list_models() == []
    assert list(service.artifact_dir.rglob("*")) == []
    env_state = service.get_environment_state("staging")
    assert env_state["active"] is None
    assert env_state["history"] == []
    logs = service.get_audit_logs(limit=20)
    assert logs
    assert logs[0].get("action") == "model_store_purged"


def test_purge_all_models_can_clear_audit_log(tmp_path: Path) -> None:
    service = ModelManagerService(tmp_path / "model_manager")
    model_file = _write_bytes(tmp_path / "tmp_artifacts" / "model.bin", b"model-v2")

    service.import_model(
        model_name="purge_audit_model",
        version="1.0.0",
        actor="admin",
        artifact_paths=[model_file],
    )
    assert service.get_audit_logs(limit=20)

    summary = service.purge_all_models(actor="admin", clear_audit=True)

    assert summary["clear_audit"] is True
    logs = service.get_audit_logs(limit=20)
    assert len(logs) == 1
    assert logs[0].get("action") == "model_store_purged"


def test_deploy_and_rollback(tmp_path: Path) -> None:
    service = ModelManagerService(tmp_path / "model_manager")

    v1 = _write_bytes(tmp_path / "v1" / "model.bin", b"v1")
    v2 = _write_bytes(tmp_path / "v2" / "model.bin", b"v2")

    service.import_model(
        model_name="risk_model",
        version="1.0.0",
        actor="admin",
        artifact_paths=[v1],
    )
    service.import_model(
        model_name="risk_model",
        version="2.0.0",
        actor="admin",
        artifact_paths=[v2],
    )

    service.deploy_model(env="production", model_name="risk_model", version="1.0.0", actor="admin")
    service.deploy_model(env="production", model_name="risk_model", version="2.0.0", actor="admin")

    current = service.get_environment_state("production")
    assert current["active"]["version"] == "2.0.0"
    assert len(current["history"]) == 1

    service.rollback_environment(env="production", actor="admin", note="rollback test")
    rolled_back = service.get_environment_state("production")
    assert rolled_back["active"]["version"] == "1.0.0"


def test_permission_guard_blocks_unauthorized_action(tmp_path: Path) -> None:
    service = ModelManagerService(tmp_path / "model_manager")

    service.create_user("viewer_1", "pass123", ["viewer"])
    auth_user = service.authenticate("viewer_1", "pass123")
    assert "model:view" in auth_user.permissions
    assert "model:import" not in auth_user.permissions

    with pytest.raises(AuthorizationError):
        service.require_permission("viewer_1", "model:import")


def test_import_model_persists_excel_auto_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = ModelManagerService(tmp_path / "model_manager")
    excel_file = _write_bytes(tmp_path / "tmp_artifacts" / "model_config.xlsx", b"not-real-excel")

    fake_profile = {
        "sheet_count": 4,
        "sheets": [
            {"sheet": "factor_definitions", "rows_sampled": 2, "columns": ["factor_name", "data_type"]},
        ],
        "config_auto_import": {
            "parser_version": "excel_config_v1",
            "has_structured_config": True,
            "sheet_names": ["factor_definitions", "dropdown_options", "factor_values", "calculation_logic"],
            "sections": {
                "factor_definitions": [{"factor_name": "vehicle_age", "data_type": "int"}],
                "dropdown_options": [{"factor_name": "channel", "option_value": "direct"}],
                "factor_values": [{"factor_name": "vehicle_age", "factor_value": "5", "score": 1.2}],
                "calculation_logic": [{"target": "premium", "expression": "=base*score"}],
            },
            "sheet_kind_map": {"factor_definitions": "factor_definitions"},
            "dropdown_validations": [],
            "formula_cells": [],
            "warnings": [],
        },
    }

    def _fake_inspect_excel(cls: type[ModelManagerService], path: Path) -> Dict[str, Any]:
        assert path.suffix.lower() == ".xlsx"
        return fake_profile

    monkeypatch.setattr(ModelManagerService, "_inspect_excel", classmethod(_fake_inspect_excel))

    imported = service.import_model(
        model_name="excel_config_model",
        version="1.0.0",
        actor="admin",
        artifact_paths=[excel_file],
        model_type="excel",
    )

    manifest = imported.get("artifact_manifest", [])
    assert len(manifest) == 1
    assert manifest[0].get("kind") == "factor_table_excel"
    assert manifest[0].get("excel_profile", {}).get("config_auto_import", {}).get("has_structured_config") is True

    excel_configs = imported.get("excel_auto_configs", [])
    assert len(excel_configs) == 1
    config = excel_configs[0].get("config", {})
    assert config.get("has_structured_config") is True
    assert config.get("sections", {}).get("factor_definitions", [])[0]["factor_name"] == "vehicle_age"


def test_inspect_excel_uses_detected_headers_for_unnamed_columns(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    pytest.importorskip("pandas")

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "factor_definitions"
    sheet["A2"] = "factor_name"
    sheet["B2"] = "data_type"
    sheet["C2"] = "default_value"
    sheet["A3"] = "driver_age"
    sheet["B3"] = "int"
    sheet["C3"] = 30

    excel_path = tmp_path / "pricing_config.xlsx"
    workbook.save(excel_path)

    profile = ModelManagerService._inspect_excel(excel_path)
    sheet_entry = next(item for item in profile.get("sheets", []) if item.get("sheet") == "factor_definitions")

    assert "factor_name" in sheet_entry.get("columns", [])
    assert not any(str(col).lower().startswith("unnamed:") for col in sheet_entry.get("columns", []))
    assert sheet_entry.get("columns_source") == "openpyxl_detected_header"


def test_inspect_excel_extracts_named_reference_context(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    pytest.importorskip("pandas")

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "pricing_sheet"
    sheet["A2"] = "base_rate"
    sheet["B2"] = 1.15

    defined_name = openpyxl.workbook.defined_name.DefinedName(
        name="rate_factor",
        attr_text="'pricing_sheet'!$B$2",
    )
    workbook.defined_names.add(defined_name)

    excel_path = tmp_path / "named_reference.xlsx"
    workbook.save(excel_path)

    profile = ModelManagerService._inspect_excel(excel_path)
    named_refs = profile.get("config_auto_import", {}).get("named_references", [])
    match = next((item for item in named_refs if item.get("name") == "rate_factor"), None)

    assert match is not None
    assert match.get("anchor_cell") == "B2"
    assert match.get("neighbor_labels", {}).get("left") == "base_rate"


def test_inspect_excel_extracts_input_cells_and_formula_dependencies(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    pytest.importorskip("pandas")

    workbook = openpyxl.Workbook()
    input_sheet = workbook.active
    input_sheet.title = "Input"
    input_sheet["A1"] = "Age"
    input_sheet["B1"] = 35
    input_sheet["A2"] = "Channel"
    input_sheet["B2"] = "direct"

    validation = openpyxl.worksheet.datavalidation.DataValidation(
        type="list",
        formula1='"direct,agent"',
        allow_blank=True,
    )
    validation.add("B2")
    input_sheet.add_data_validation(validation)

    workbook.defined_names.add(
        openpyxl.workbook.defined_name.DefinedName(
            name="age_input",
            attr_text="'Input'!$B$1",
        )
    )

    calc_sheet = workbook.create_sheet("Calc")
    calc_sheet["A1"] = "=age_input*2"
    calc_sheet["A2"] = "='Input'!$B$2&\"_x\""

    excel_path = tmp_path / "pricing_logic.xlsx"
    workbook.save(excel_path)

    profile = ModelManagerService._inspect_excel(excel_path)
    config = profile.get("config_auto_import", {})

    formula_cells = config.get("formula_cells", [])
    a1_formula = next(
        item for item in formula_cells if item.get("sheet") == "Calc" and item.get("cell") == "A1"
    )
    assert "age_input" in a1_formula.get("dependencies", {}).get("named_refs", [])

    input_cells = config.get("input_cells", [])
    age_cell = next(
        item for item in input_cells if item.get("sheet") == "Input" and item.get("cell") == "B1"
    )
    assert "age_input" in age_cell.get("defined_names", [])
    assert any(ref.get("sheet") == "Calc" and ref.get("cell") == "A1" for ref in age_cell.get("referenced_by", []))
    assert age_cell.get("neighbor_labels", {}).get("left") == "Age"

    channel_cell = next(
        item for item in input_cells if item.get("sheet") == "Input" and item.get("cell") == "B2"
    )
    assert "direct" in channel_cell.get("validation", {}).get("inline_options", [])

    calc_graph = config.get("calculation_graph", {})
    outputs = calc_graph.get("output_cells", [])
    assert any(item.get("sheet") == "Calc" and item.get("cell") == "A2" for item in outputs)

    edges = calc_graph.get("edges", [])
    assert any(
        edge.get("from") == "Calc::A1" and edge.get("to") == "Input::B1" and edge.get("via") == "named_ref"
        for edge in edges
    )
    assert any(
        edge.get("from") == "Calc::A2" and edge.get("to") == "Input::B2" and edge.get("via") == "cell_ref"
        for edge in edges
    )


def test_excel_calculator_schema_and_single_calculation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    service = ModelManagerService(tmp_path / "model_manager")

    workbook = openpyxl.Workbook()
    input_sheet = workbook.active
    input_sheet.title = "Input"
    input_sheet["A1"] = "Age"
    input_sheet["B1"] = 30
    input_sheet["A2"] = "Channel"
    input_sheet["B2"] = "direct"
    calc_sheet = workbook.create_sheet("Calc")
    calc_sheet["A1"] = 0
    calc_sheet["A2"] = ""
    excel_path = tmp_path / "tmp_artifacts" / "single_calc.xlsx"
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(excel_path)

    fake_profile = {
        "sheet_count": 2,
        "sheets": [
            {"sheet": "Input", "rows_sampled": 2, "columns": ["Age", "Channel"]},
            {"sheet": "Calc", "rows_sampled": 2, "columns": ["result_1", "result_2"]},
        ],
        "config_auto_import": {
            "parser_version": "excel_config_v1",
            "has_structured_config": True,
            "sheet_names": ["Input", "Calc"],
            "named_references": [
                {
                    "name": "age_input",
                    "sheet": "Input",
                    "anchor_cell": "B1",
                    "neighbor_labels": {"left": "Age"},
                }
            ],
            "input_cells": [
                {
                    "sheet": "Input",
                    "cell": "B1",
                    "value": 30,
                    "neighbor_labels": {"left": "Age"},
                    "defined_names": ["age_input"],
                },
                {
                    "sheet": "Input",
                    "cell": "B2",
                    "value": "direct",
                    "neighbor_labels": {"left": "Channel"},
                    "validation": {"inline_options": ["direct", "agent"]},
                    "defined_names": [],
                },
            ],
            "formula_cells": [
                {"sheet": "Calc", "cell": "A1", "formula": "=age_input*2"},
                {"sheet": "Calc", "cell": "A2", "formula": "='Input'!$B$2&\"_x\""},
            ],
            "calculation_graph": {
                "output_cells": [
                    {"id": "Calc::A1", "sheet": "Calc", "cell": "A1", "formula": "=age_input*2"},
                    {"id": "Calc::A2", "sheet": "Calc", "cell": "A2", "formula": "='Input'!$B$2&\"_x\""},
                ],
                "nodes": [],
                "edges": [],
            },
            "dropdown_validations": [],
            "warnings": [],
        },
    }

    def _fake_inspect_excel(cls: type[ModelManagerService], path: Path) -> Dict[str, Any]:
        assert path.suffix.lower() == ".xlsx"
        return fake_profile

    monkeypatch.setattr(ModelManagerService, "_inspect_excel", classmethod(_fake_inspect_excel))

    imported = service.import_model(
        model_name="single_calc_model",
        version="1.0.0",
        actor="admin",
        artifact_paths=[excel_path],
        model_type="excel",
    )
    assert imported["model_name"] == "single_calc_model"

    artifacts = service.list_excel_calculator_artifacts("single_calc_model", "1.0.0")
    assert len(artifacts) == 1
    artifact_path = str(artifacts[0]["artifact_path"])

    schema = service.get_excel_calculator_schema(
        model_name="single_calc_model",
        version="1.0.0",
        artifact_path=artifact_path,
    )
    assert len(schema.get("input_fields", [])) == 2
    assert len(schema.get("output_cells", [])) == 2
    channel_field = next(item for item in schema["input_fields"] if item["key"] == "Input::B2")
    assert channel_field["options"] == ["direct", "agent"]

    result = service.run_excel_single_calculation(
        model_name="single_calc_model",
        version="1.0.0",
        artifact_path=artifact_path,
        inputs={
            "age_input": 40,
            "Input::B2": "agent",
        },
    )
    assert result["errors"] == []

    output_map = {
        f"{item.get('sheet')}::{item.get('cell')}": item.get("value")
        for item in result.get("output_values", [])
    }
    assert output_map["Calc::A1"] == 80.0
    assert output_map["Calc::A2"] == "agent_x"


def test_excel_calculator_schema_from_imported_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = ModelManagerService(tmp_path / "model_manager")
    excel_file = _write_bytes(tmp_path / "tmp_artifacts" / "schema_only.xlsx", b"not-real-excel")

    fake_profile = {
        "sheet_count": 1,
        "sheets": [{"sheet": "Input", "rows_sampled": 2, "columns": ["factor", "value"]}],
        "config_auto_import": {
            "parser_version": "excel_config_v1",
            "has_structured_config": True,
            "sheet_names": ["Input"],
            "named_references": [],
            "input_cells": [
                {
                    "sheet": "Input",
                    "cell": "B2",
                    "value": 12,
                    "neighbor_labels": {"left": "base_factor"},
                    "defined_names": [],
                }
            ],
            "formula_cells": [
                {"sheet": "Input", "cell": "C2", "formula": "=B2*2"},
            ],
            "calculation_graph": {
                "output_cells": [
                    {"id": "Input::C2", "sheet": "Input", "cell": "C2", "formula": "=B2*2"},
                ],
                "nodes": [],
                "edges": [],
            },
            "dropdown_validations": [],
            "warnings": [],
        },
    }

    def _fake_inspect_excel(cls: type[ModelManagerService], path: Path) -> Dict[str, Any]:
        assert path.suffix.lower() == ".xlsx"
        return fake_profile

    monkeypatch.setattr(ModelManagerService, "_inspect_excel", classmethod(_fake_inspect_excel))

    service.import_model(
        model_name="schema_model",
        version="1.0.0",
        actor="admin",
        artifact_paths=[excel_file],
        model_type="excel",
    )

    artifacts = service.list_excel_calculator_artifacts("schema_model", "1.0.0")
    assert len(artifacts) == 1
    artifact_path = str(artifacts[0]["artifact_path"])

    schema = service.get_excel_calculator_schema(
        model_name="schema_model",
        version="1.0.0",
        artifact_path=artifact_path,
    )
    assert schema["artifact_path"] == artifact_path
    assert len(schema.get("input_fields", [])) == 1
    assert len(schema.get("output_cells", [])) == 1
    assert schema["input_fields"][0]["label"] == "base_factor"
    assert schema["input_fields"][0]["neighbor_labels"]["left"] == "base_factor"
    assert isinstance(schema.get("calculation_graph", {}), dict)
    assert "nodes" in schema["calculation_graph"]
    assert "edges" in schema["calculation_graph"]


def test_excel_calculator_schema_rebuilds_graph_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = ModelManagerService(tmp_path / "model_manager")
    excel_file = _write_bytes(tmp_path / "tmp_artifacts" / "schema_rebuild.xlsx", b"not-real-excel")

    fake_profile = {
        "sheet_count": 1,
        "sheets": [{"sheet": "Input", "rows_sampled": 2, "columns": ["factor", "value"]}],
        "config_auto_import": {
            "parser_version": "excel_config_v1",
            "has_structured_config": True,
            "sheet_names": ["Input"],
            "named_references": [],
            "input_cells": [
                {
                    "sheet": "Input",
                    "cell": "B2",
                    "value": 12,
                    "neighbor_labels": {"left": "base_factor"},
                    "defined_names": [],
                }
            ],
            "formula_cells": [
                {
                    "sheet": "Input",
                    "cell": "C2",
                    "formula": "=B2*2",
                    "dependencies": {"cell_refs": [{"sheet": "Input", "cell": "B2"}], "named_refs": []},
                }
            ],
            "dropdown_validations": [],
            "warnings": [],
        },
    }

    def _fake_inspect_excel(cls: type[ModelManagerService], path: Path) -> Dict[str, Any]:
        assert path.suffix.lower() == ".xlsx"
        return fake_profile

    monkeypatch.setattr(ModelManagerService, "_inspect_excel", classmethod(_fake_inspect_excel))

    service.import_model(
        model_name="schema_rebuild_model",
        version="1.0.0",
        actor="admin",
        artifact_paths=[excel_file],
        model_type="excel",
    )
    artifact_path = str(service.list_excel_calculator_artifacts("schema_rebuild_model", "1.0.0")[0]["artifact_path"])

    schema = service.get_excel_calculator_schema(
        model_name="schema_rebuild_model",
        version="1.0.0",
        artifact_path=artifact_path,
    )
    graph = schema.get("calculation_graph", {})
    assert len(graph.get("nodes", [])) >= 2
    assert len(graph.get("edges", [])) >= 1


def test_input_label_prefers_meaningful_neighbor_context() -> None:
    item = {
        "sheet": "Input",
        "cell": "B2",
        "neighbor_labels": {
            "left": "100",
            "up": "=A1*2",
            "right": "vehicle age",
            "down": "",
        },
        "defined_names": ["age_input"],
    }
    label = ModelManagerService._input_label_from_metadata(item)
    assert label == "vehicle age"


def test_input_label_falls_back_to_humanized_defined_name() -> None:
    item = {
        "sheet": "Input",
        "cell": "B2",
        "neighbor_labels": {
            "left": "12",
            "up": "34",
        },
        "defined_names": ["base_premium_rate"],
    }
    label = ModelManagerService._input_label_from_metadata(item)
    assert label == "base premium rate"
