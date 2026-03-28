"""Prediction workflows used by the frontend UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from ins_pricing.production.inference import load_predictor_from_config

from .workflows_common import (
    _build_search_roots,
    _discover_model_file,
    _load_ft_embedding_model,
    _load_pickled_model_payload,
    _resolve_model_output_dir,
    _resolve_output_dir,
)


def _resolve_model_file_for_prediction(
    *,
    model_name: str,
    model_key: str,
    model_path: Optional[str],
    label: str,
    search_roots: Sequence[Path],
    default_output_root: Optional[Path] = None,
) -> Path:
    raw_path = str(model_path or "").strip()

    if raw_path:
        candidate = Path(raw_path).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"{label} not found: {candidate}")
        if candidate.is_file():
            return candidate
        discovered = _discover_model_file(
            model_name=model_name,
            model_key=model_key,
            search_roots=[candidate],
            output_roots=[candidate],
        )
        if discovered is None:
            raise FileNotFoundError(
                f"{label} is a directory but model artifact was not found under: {candidate}"
            )
        print(f"[Info] Found {model_key} model from directory override: {discovered}")
        return discovered

    discovered = _discover_model_file(
        model_name=model_name,
        model_key=model_key,
        search_roots=search_roots,
        output_roots=[default_output_root] if default_output_root is not None else None,
    )
    if discovered is not None:
        print(f"[Info] Auto-discovered {model_key} model: {discovered}")
        return discovered

    expected = (
        default_output_root / "model"
        if default_output_root is not None
        else Path("<search_roots>")
    )
    raise FileNotFoundError(
        f"{label} not found. Expected model artifact under {expected} "
        "or provide an explicit uploaded model file."
    )


def run_predict_ft_embed(
    *,
    ft_cfg_path: str,
    xgb_cfg_path: Optional[str],
    resn_cfg_path: Optional[str],
    input_path: str,
    output_path: str,
    model_name: Optional[str],
    model_keys: str,
    ft_model_path: Optional[str] = None,
    xgb_model_path: Optional[str] = None,
    resn_model_path: Optional[str] = None,
    model_search_dir: Optional[str] = None,
) -> str:
    ft_cfg_path = Path(ft_cfg_path).resolve()
    xgb_cfg_path = Path(xgb_cfg_path).resolve() if xgb_cfg_path else None
    resn_cfg_path = Path(resn_cfg_path).resolve() if resn_cfg_path else None
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input data not found: {input_path}")

    keys = [k.strip() for k in model_keys.split(",") if k.strip()]
    if not keys:
        raise ValueError("model_keys is empty.")

    ft_cfg = json.loads(ft_cfg_path.read_text(encoding="utf-8"))
    xgb_cfg = json.loads(xgb_cfg_path.read_text(encoding="utf-8")) if xgb_cfg_path else None
    resn_cfg = json.loads(resn_cfg_path.read_text(encoding="utf-8")) if resn_cfg_path else None

    if model_name is None:
        model_list = list(ft_cfg.get("model_list") or [])
        model_categories = list(ft_cfg.get("model_categories") or [])
        if len(model_list) != 1 or len(model_categories) != 1:
            raise ValueError("Set model_name when multiple models exist.")
        model_name = f"{model_list[0]}_{model_categories[0]}"

    ft_output_dir = Path(_resolve_output_dir(ft_cfg, ft_cfg_path)).resolve()
    xgb_output_dir = (
        Path(_resolve_output_dir(xgb_cfg, xgb_cfg_path)).resolve()
        if xgb_cfg and xgb_cfg_path
        else None
    )
    resn_output_dir = (
        Path(_resolve_output_dir(resn_cfg, resn_cfg_path)).resolve()
        if resn_cfg and resn_cfg_path
        else None
    )
    ft_prefix = ft_cfg.get("ft_feature_prefix", "ft_emb")
    xgb_task_type = str(xgb_cfg.get("task_type", "regression")) if xgb_cfg else None
    search_roots = _build_search_roots(
        model_search_dir,
        ft_cfg_path.parent,
        xgb_cfg_path.parent if xgb_cfg_path else None,
        resn_cfg_path.parent if resn_cfg_path else None,
        input_path.parent,
        Path.cwd(),
    )

    if ft_cfg.get("geo_feature_nmes"):
        raise ValueError("FT with geo tokens is not supported in this workflow.")

    import torch

    print("Loading FT model...")
    ft_model_file = _resolve_model_file_for_prediction(
        model_name=model_name,
        model_key="ft",
        model_path=ft_model_path,
        label="ft_model_path",
        search_roots=search_roots,
        default_output_root=ft_output_dir,
    )
    ft_model = _load_ft_embedding_model(ft_model_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(ft_model, "device"):
        ft_model.device = device
    if hasattr(ft_model, "to"):
        ft_model.to(device)
    if hasattr(ft_model, "ft"):
        ft_model.ft.to(device)

    df_new = pd.read_csv(input_path)
    emb = ft_model.predict(df_new, return_embedding=True)
    emb_cols = [f"pred_{ft_prefix}_{i}" for i in range(emb.shape[1])]
    df_with_emb = df_new.copy()
    df_with_emb[emb_cols] = emb
    result = df_with_emb.copy()

    if "xgb" in keys:
        xgb_model_file = _resolve_model_file_for_prediction(
            model_name=model_name,
            model_key="xgb",
            model_path=xgb_model_path,
            label="xgb_model_path",
            search_roots=search_roots,
            default_output_root=xgb_output_dir,
        )
        xgb_payload = _load_pickled_model_payload(xgb_model_file)
        if isinstance(xgb_payload, dict) and "model" in xgb_payload:
            xgb_model = xgb_payload["model"]
            feature_list = xgb_payload.get("preprocess_artifacts", {}).get("factor_nmes")
        else:
            xgb_model = xgb_payload
            feature_list = None
        if not feature_list and xgb_cfg:
            feature_list = xgb_cfg.get("feature_list") or []
        if not feature_list:
            raise ValueError("Feature list missing for XGB model.")

        X = df_with_emb[feature_list]
        if (xgb_task_type or "regression") == "classification" and hasattr(xgb_model, "predict_proba"):
            pred = xgb_model.predict_proba(X)[:, 1]
        else:
            pred = xgb_model.predict(X)
        result["pred_xgb"] = pred

    if "resn" in keys:
        if not resn_cfg_path:
            raise ValueError("resn model selected but resn_cfg_path is missing.")
        resn_model_file = _resolve_model_file_for_prediction(
            model_name=model_name,
            model_key="resn",
            model_path=resn_model_path,
            label="resn_model_path",
            search_roots=search_roots,
            default_output_root=resn_output_dir,
        )
        resn_output_override = _resolve_model_output_dir(
            str(resn_model_file),
            "resolved_resn_model_path",
        )
        resn_predictor = load_predictor_from_config(
            resn_cfg_path,
            "resn",
            model_name=model_name,
            output_dir=resn_output_override,
        )
        pred_resn = resn_predictor.predict(df_with_emb)
        result["pred_resn"] = pred_resn

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")
    return str(output_path)

