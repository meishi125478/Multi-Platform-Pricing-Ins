from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from ins_pricing.exceptions import ModelLoadError
from ins_pricing.utils.model_loading import load_pickle_artifact


def test_load_pickle_artifact_allows_safe_payload(tmp_path):
    path = tmp_path / "safe.pkl"
    with path.open("wb") as fh:
        pickle.dump({"values": [1, 2, 3]}, fh, protocol=pickle.HIGHEST_PROTOCOL)

    payload = load_pickle_artifact(path)
    assert payload == {"values": [1, 2, 3]}


def test_load_pickle_artifact_blocks_untrusted_global_by_default(tmp_path):
    path = tmp_path / "unsafe.pkl"
    with path.open("wb") as fh:
        pickle.dump(Path("abc"), fh, protocol=pickle.HIGHEST_PROTOCOL)

    with pytest.raises(ModelLoadError):
        load_pickle_artifact(path, allow_unsafe=False)
