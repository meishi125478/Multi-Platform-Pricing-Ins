from __future__ import annotations

from pathlib import Path

import ins_pricing.frontend.workflows_common as workflows_common


def test_discover_model_file_prefers_direct_paths_without_recursive_glob(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_name = "demo"
    model_key = "xgb"
    filename = workflows_common._model_artifact_filename(model_name, model_key)
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / filename
    model_path.write_bytes(b"ok")

    original_glob = Path.glob

    def _guard_glob(self: Path, pattern: str):
        if pattern.startswith("**"):
            raise AssertionError("recursive glob should not be reached for direct hit")
        return original_glob(self, pattern)

    monkeypatch.setattr(Path, "glob", _guard_glob, raising=False)
    found = workflows_common._discover_model_file(
        model_name=model_name,
        model_key=model_key,
        search_roots=[tmp_path],
        output_roots=[tmp_path],
    )
    assert found == model_path.resolve()


def test_discover_model_file_recursive_lookup_avoids_generic_fallback_when_model_dir_matches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_name = "demo"
    model_key = "xgb"
    filename = workflows_common._model_artifact_filename(model_name, model_key)
    nested_model = tmp_path / "a" / "b" / "model" / filename
    nested_model.parent.mkdir(parents=True, exist_ok=True)
    nested_model.write_bytes(b"ok")

    original_glob = Path.glob
    patterns: list[str] = []

    def _count_glob(self: Path, pattern: str):
        if pattern.startswith("**"):
            patterns.append(pattern)
        return original_glob(self, pattern)

    monkeypatch.setattr(Path, "glob", _count_glob, raising=False)

    found = workflows_common._discover_model_file(
        model_name=model_name,
        model_key=model_key,
        search_roots=[tmp_path],
        output_roots=None,
    )
    assert found == nested_model.resolve()
    assert len(patterns) > 0
    assert f"**/{filename}" not in patterns


def test_discover_model_file_refreshes_direct_hits_when_newer_file_appears(
    tmp_path: Path,
) -> None:
    model_name = "demo"
    model_key = "xgb"
    filename = workflows_common._model_artifact_filename(model_name, model_key)
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    (root_a / "model").mkdir(parents=True, exist_ok=True)
    (root_b / "model").mkdir(parents=True, exist_ok=True)
    file_a = root_a / "model" / filename
    file_b = root_b / "model" / filename
    file_a.write_bytes(b"a")

    first = workflows_common._discover_model_file(
        model_name=model_name,
        model_key=model_key,
        search_roots=[root_a, root_b],
        output_roots=[root_a, root_b],
    )
    assert first == file_a.resolve()

    file_b.write_bytes(b"b")
    second = workflows_common._discover_model_file(
        model_name=model_name,
        model_key=model_key,
        search_roots=[root_a, root_b],
        output_roots=[root_a, root_b],
    )
    assert second == file_b.resolve()


def test_discover_model_file_refreshes_recursive_hits_when_newer_file_appears(
    tmp_path: Path,
) -> None:
    model_name = "demo"
    model_key = "xgb"
    filename = workflows_common._model_artifact_filename(model_name, model_key)
    old_path = tmp_path / "old" / "run1" / "model" / filename
    new_path = tmp_path / "new" / "run2" / "model" / filename
    old_path.parent.mkdir(parents=True, exist_ok=True)
    new_path.parent.mkdir(parents=True, exist_ok=True)
    old_path.write_bytes(b"old")

    first = workflows_common._discover_model_file(
        model_name=model_name,
        model_key=model_key,
        search_roots=[tmp_path],
        output_roots=None,
    )
    assert first == old_path.resolve()

    new_path.write_bytes(b"new")
    second = workflows_common._discover_model_file(
        model_name=model_name,
        model_key=model_key,
        search_roots=[tmp_path],
        output_roots=None,
    )
    assert second == new_path.resolve()
