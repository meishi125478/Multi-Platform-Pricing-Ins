from __future__ import annotations

from pathlib import Path

import pandas as pd

from ins_pricing.frontend.workflows_prediction_utils import (
    build_ft_embedding_frames,
    load_raw_splits,
    resolve_model_output_override,
)


def test_load_raw_splits_with_explicit_files(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(train_path, index=False)
    pd.DataFrame({"x": [3]}).to_csv(test_path, index=False)

    train_raw, test_raw, raw, use_explicit = load_raw_splits(
        split_cfg={},
        data_cfg={},
        data_cfg_path=tmp_path / "cfg.json",
        model_name="demo",
        train_data_path=str(train_path),
        test_data_path=str(test_path),
    )

    assert use_explicit is True
    assert raw is None
    assert len(train_raw) == 2
    assert len(test_raw) == 1


def test_build_ft_embedding_frames_precomputed_alignment(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text("{}", encoding="utf-8")
    embed_dir = tmp_path / "embed"
    embed_dir.mkdir(parents=True, exist_ok=True)
    embed_df = pd.DataFrame({"pred_ft_emb_0": [0.1, 0.2, 0.3, 0.4]})
    embed_df.to_csv(embed_dir / "demo.csv", index=False)

    raw = pd.DataFrame({"a": [10, 20, 30, 40]})
    train_raw = raw.iloc[[0, 2]].copy()
    test_raw = raw.iloc[[1, 3]].copy()

    train_df, test_df = build_ft_embedding_frames(
        use_runtime_ft_embedding=False,
        train_raw=train_raw,
        test_raw=test_raw,
        raw=raw,
        use_explicit_split=False,
        model_name="demo",
        ft_cfg={"output_dir": "./results", "ft_feature_prefix": "ft_emb"},
        ft_cfg_path=cfg_path,
        search_roots=[tmp_path],
        ft_model_path=None,
        embed_cfg={"data_dir": "./embed", "data_format": "csv"},
        embed_cfg_path=cfg_path,
    )

    assert train_df["pred_ft_emb_0"].tolist() == [0.1, 0.3]
    assert test_df["pred_ft_emb_0"].tolist() == [0.2, 0.4]


def test_resolve_model_output_override_with_explicit_file(tmp_path: Path) -> None:
    model_name = "demo_model"
    model_dir = tmp_path / "results" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / f"01_{model_name}_Xgboost.pkl"
    model_file.write_bytes(b"placeholder")

    resolved = resolve_model_output_override(
        model_name=model_name,
        model_key="xgb",
        model_path=str(model_file),
        search_roots=[tmp_path],
        output_root=tmp_path / "results",
        label="xgb_model_path",
    )

    assert resolved == (tmp_path / "results").resolve()
