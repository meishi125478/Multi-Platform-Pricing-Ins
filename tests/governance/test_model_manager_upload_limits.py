from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from model_manager_tool import server


def test_resolve_max_upload_bytes_accepts_positive_mb() -> None:
    assert server._resolve_max_upload_bytes(8) == 8 * 1024 * 1024


def test_resolve_max_upload_bytes_rejects_non_positive_value() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        server._resolve_max_upload_bytes(0)


def test_parse_bool_flag_recognizes_truthy_values() -> None:
    assert server._parse_bool_flag(True) is True
    assert server._parse_bool_flag("1") is True
    assert server._parse_bool_flag("YES") is True
    assert server._parse_bool_flag("on") is True


def test_parse_bool_flag_recognizes_falsy_values() -> None:
    assert server._parse_bool_flag(False) is False
    assert server._parse_bool_flag(None) is False
    assert server._parse_bool_flag("0") is False
    assert server._parse_bool_flag("no") is False


def test_parse_args_reads_enable_purge_all_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_MANAGER_ENABLE_PURGE_ALL", "true")
    args = server.parse_args([])
    assert args.enable_purge_all is True


def test_copy_uploaded_file_with_limit_allows_file_within_limit(tmp_path: Path) -> None:
    source = BytesIO(b"pricing-model")
    target = tmp_path / "model.bin"

    size = server._copy_uploaded_file_with_limit(
        src_file=source,
        target_path=target,
        max_bytes=64,
    )

    assert size == len(b"pricing-model")
    assert target.read_bytes() == b"pricing-model"


def test_copy_uploaded_file_with_limit_blocks_oversized_file(tmp_path: Path) -> None:
    source = BytesIO(b"oversized-content")
    target = tmp_path / "too_large.bin"

    with pytest.raises(ValueError, match="exceeds upload limit"):
        server._copy_uploaded_file_with_limit(
            src_file=source,
            target_path=target,
            max_bytes=8,
        )


def test_patch_starlette_request_form_limit_raises_default_part_size(monkeypatch: pytest.MonkeyPatch) -> None:
    StarletteRequest = pytest.importorskip("starlette.requests").Request
    MultiPartParser = pytest.importorskip("starlette.formparsers").MultiPartParser
    captured: dict[str, int] = {}
    original_form = StarletteRequest.form
    original_spool_max_size = int(getattr(MultiPartParser, "spool_max_size", 0) or 0)
    original_class_max_part_size = int(getattr(MultiPartParser, "max_part_size", 0) or 0)

    def _fake_original_form(self, *, max_files: int = 1000, max_fields: int = 1000, max_part_size: int = 1024 * 1024):
        captured["max_part_size"] = int(max_part_size)
        return object()

    monkeypatch.setattr(StarletteRequest, "form", _fake_original_form)
    monkeypatch.setattr(server, "_ORIGINAL_STARLETTE_REQUEST_FORM", None)
    monkeypatch.setattr(server, "_PATCHED_REQUEST_FORM_MIN_PART_SIZE_BYTES", server.DEFAULT_MAX_UPLOAD_BYTES)

    class _DummyRequest:
        pass

    patched_spool_max_size = 0
    patched_class_max_part_size = 0
    try:
        server._patch_starlette_request_form_limit(min_part_size_bytes=8 * 1024 * 1024)
        StarletteRequest.form(_DummyRequest())
        patched_spool_max_size = int(getattr(MultiPartParser, "spool_max_size", 0) or 0)
        patched_class_max_part_size = int(getattr(MultiPartParser, "max_part_size", 0) or 0)
    finally:
        monkeypatch.setattr(StarletteRequest, "form", original_form)
        MultiPartParser.spool_max_size = original_spool_max_size
        MultiPartParser.max_part_size = original_class_max_part_size

    assert captured["max_part_size"] == 8 * 1024 * 1024
    assert patched_spool_max_size >= 8 * 1024 * 1024
    assert patched_class_max_part_size >= 8 * 1024 * 1024


def test_compact_import_result_for_ui_summarizes_excel_payload() -> None:
    result = {
        "model_name": "pricing_model",
        "version": "1.0.0",
        "status": "candidate",
        "artifact_manifest": [
            {
                "artifact_name": "factors.xlsx",
                "kind": "factor_table_excel",
                "size_bytes": 1024,
                "relative_path": "artifacts/pricing_model/1.0.0/factors.xlsx",
                "excel_profile": {
                    "config_auto_import": {
                        "has_structured_config": True,
                        "sheet_names": ["Input", "Calc"],
                        "named_references": [{"name": "age_input"}],
                        "input_cells": [{"cell": "B1"}, {"cell": "B2"}],
                        "formula_cells": [{"cell": "A1"}],
                        "calculation_graph": {
                            "nodes": [{"id": "Calc::A1"}, {"id": "Input::B1"}],
                            "edges": [{"from": "Calc::A1", "to": "Input::B1"}],
                        },
                    }
                },
            }
        ],
        "excel_auto_configs": [{"artifact_name": "factors.xlsx"}],
    }

    compact = server._compact_import_result_for_ui(result)

    assert compact["model_name"] == "pricing_model"
    assert compact["artifact_count"] == 1
    assert compact["excel_auto_config_count"] == 1
    assert compact["artifacts_truncated"] is False
    excel_summary = compact["artifacts_preview"][0]["excel_summary"]
    assert excel_summary["sheet_count"] == 2
    assert excel_summary["named_reference_count"] == 1
    assert excel_summary["input_cell_count"] == 2
    assert excel_summary["formula_cell_count"] == 1
    assert excel_summary["calculation_graph_node_count"] == 2
    assert excel_summary["calculation_graph_edge_count"] == 1


def test_compact_import_result_for_ui_truncates_artifact_preview() -> None:
    result = {
        "artifact_manifest": [
            {
                "artifact_name": f"artifact_{idx}.bin",
                "kind": "model_file",
                "size_bytes": 10 + idx,
                "relative_path": f"artifacts/artifact_{idx}.bin",
            }
            for idx in range(13)
        ]
    }

    compact = server._compact_import_result_for_ui(result)

    assert compact["artifact_count"] == 13
    assert len(compact["artifacts_preview"]) == 12
    assert compact["artifacts_truncated"] is True
