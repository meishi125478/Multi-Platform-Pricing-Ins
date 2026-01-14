from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
except Exception as exc:  # pragma: no cover
    torch = None
    _torch_import_error = exc
else:
    _torch_import_error = None


def _require_torch(func_name: str) -> None:
    if torch is None:
        raise RuntimeError(f"{func_name} requires torch: {_torch_import_error}")


def _prepare_tensor(arr, device) -> "torch.Tensor":
    _require_torch("_prepare_tensor")
    if isinstance(arr, torch.Tensor):
        return arr.to(device=device, dtype=torch.float32)
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def _prepare_baseline(inputs: "torch.Tensor", baseline) -> "torch.Tensor":
    if baseline is None or baseline == "zeros":
        base = torch.zeros_like(inputs)
    elif isinstance(baseline, str):
        if baseline == "mean":
            base_vec = inputs.mean(dim=0, keepdim=True)
        elif baseline == "median":
            base_vec = inputs.median(dim=0, keepdim=True).values
        else:
            raise ValueError("baseline must be None, 'zeros', 'mean', 'median', or array-like.")
        base = base_vec.repeat(inputs.shape[0], 1)
    else:
        base = _prepare_tensor(baseline, inputs.device)
        if base.ndim == 1:
            base = base.reshape(1, -1).repeat(inputs.shape[0], 1)
        if base.shape != inputs.shape:
            raise ValueError("baseline shape must match inputs shape.")
    return base


def _select_output(output: "torch.Tensor", target: Optional[int]) -> "torch.Tensor":
    if output.ndim == 2 and output.shape[1] > 1:
        if target is None:
            raise ValueError("target must be provided for multi-class outputs.")
        output = output[:, int(target)]
    return output.reshape(-1)


def gradient_x_input_torch(
    forward_fn: Callable[["torch.Tensor"], "torch.Tensor"],
    inputs,
    *,
    target: Optional[int] = None,
    device: Optional[str] = None,
) -> np.ndarray:
    """Single-step gradient * input (fast but rough attribution)."""
    _require_torch("gradient_x_input_torch")
    device = device or "cpu"
    X = _prepare_tensor(inputs, device)
    X.requires_grad_(True)
    with torch.enable_grad():
        output = forward_fn(X)
        output = _select_output(output, target)
        grads = torch.autograd.grad(
            outputs=output,
            inputs=X,
            grad_outputs=torch.ones_like(output),
            create_graph=False,
            retain_graph=False,
        )[0]
    return (grads * X).detach().cpu().numpy()


def integrated_gradients_torch(
    forward_fn: Callable[["torch.Tensor"], "torch.Tensor"],
    inputs,
    *,
    baseline=None,
    steps: int = 50,
    batch_size: int = 256,
    target: Optional[int] = None,
    device: Optional[str] = None,
) -> np.ndarray:
    """Integrated gradients for a single tensor input."""
    _require_torch("integrated_gradients_torch")
    device = device or "cpu"
    steps = max(1, int(steps))
    batch_size = max(1, int(batch_size))

    X_full = _prepare_tensor(inputs, device)
    baseline_full = _prepare_baseline(X_full, baseline)

    n_rows = X_full.shape[0]
    out = np.zeros_like(X_full.detach().cpu().numpy(), dtype=np.float32)
    alphas = torch.linspace(0.0, 1.0, steps, device=device)

    with torch.enable_grad():
        for start in range(0, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            X = X_full[start:end]
            base = baseline_full[start:end]
            total_grad = torch.zeros_like(X)
            for alpha in alphas:
                scaled = base + alpha * (X - base)
                scaled.requires_grad_(True)
                output = forward_fn(scaled)
                output = _select_output(output, target)
                grads = torch.autograd.grad(
                    outputs=output,
                    inputs=scaled,
                    grad_outputs=torch.ones_like(output),
                    create_graph=False,
                    retain_graph=False,
                )[0]
                total_grad += grads
            avg_grad = total_grad / float(steps)
            attr = (X - base) * avg_grad
            out[start:end] = attr.detach().cpu().numpy()
    return out


def integrated_gradients_multi_input_torch(
    forward_fn: Callable[..., "torch.Tensor"],
    inputs: Sequence,
    *,
    baselines: Optional[Sequence] = None,
    steps: int = 50,
    batch_size: int = 256,
    target: Optional[int] = None,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, ...]:
    """Integrated gradients for multiple tensor inputs (e.g., numeric + geo)."""
    _require_torch("integrated_gradients_multi_input_torch")
    device = device or "cpu"
    steps = max(1, int(steps))
    batch_size = max(1, int(batch_size))

    tensors = [_prepare_tensor(inp, device) for inp in inputs]
    if baselines is None:
        baselines = [None for _ in tensors]
    base_tensors = [_prepare_baseline(t, b) for t, b in zip(tensors, baselines)]

    n_rows = tensors[0].shape[0]
    outputs = [np.zeros_like(t.detach().cpu().numpy(), dtype=np.float32) for t in tensors]
    alphas = torch.linspace(0.0, 1.0, steps, device=device)

    with torch.enable_grad():
        for start in range(0, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            batch_inputs = [t[start:end] for t in tensors]
            batch_bases = [b[start:end] for b in base_tensors]
            total_grads = [torch.zeros_like(t) for t in batch_inputs]

            for alpha in alphas:
                scaled_inputs = []
                for t, b in zip(batch_inputs, batch_bases):
                    s = b + alpha * (t - b)
                    s.requires_grad_(True)
                    scaled_inputs.append(s)
                output = forward_fn(*scaled_inputs)
                output = _select_output(output, target)
                grads = torch.autograd.grad(
                    outputs=output,
                    inputs=scaled_inputs,
                    grad_outputs=torch.ones_like(output),
                    create_graph=False,
                    retain_graph=False,
                )
                total_grads = [g_acc + g for g_acc, g in zip(total_grads, grads)]

            for idx, (t, b, g) in enumerate(zip(batch_inputs, batch_bases, total_grads)):
                avg_grad = g / float(steps)
                attr = (t - b) * avg_grad
                outputs[idx][start:end] = attr.detach().cpu().numpy()

    return tuple(outputs)


def summarize_attributions(
    attributions: np.ndarray,
    feature_names: Sequence[str],
    *,
    agg: str = "mean_abs",
) -> pd.Series:
    if attributions.ndim != 2:
        raise ValueError("attributions must be 2d (n_samples, n_features).")
    if len(feature_names) != attributions.shape[1]:
        raise ValueError("feature_names length must match attribution dimension.")

    if agg == "mean_abs":
        scores = np.mean(np.abs(attributions), axis=0)
    elif agg == "mean":
        scores = np.mean(attributions, axis=0)
    elif agg == "sum_abs":
        scores = np.sum(np.abs(attributions), axis=0)
    else:
        raise ValueError("agg must be 'mean_abs', 'mean', or 'sum_abs'.")
    return pd.Series(scores, index=list(feature_names)).sort_values(ascending=False)


def resnet_integrated_gradients(
    model,
    X,
    *,
    baseline=None,
    steps: int = 50,
    batch_size: int = 256,
    target: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """Integrated gradients wrapper for ResNetSklearn."""
    _require_torch("resnet_integrated_gradients")
    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X_np = X.to_numpy(dtype=np.float32, copy=False)
    else:
        X_np = np.asarray(X, dtype=np.float32)
        feature_names = [f"x{i}" for i in range(X_np.shape[1])]

    if device is None:
        try:
            device = next(model.resnet.parameters()).device
        except Exception:
            device = "cpu"
    model.resnet.eval()

    def forward_fn(x):
        out = model.resnet(x)
        if getattr(model, "task_type", None) == "classification":
            out = torch.sigmoid(out)
        return out

    attrs = integrated_gradients_torch(
        forward_fn,
        X_np,
        baseline=baseline,
        steps=steps,
        batch_size=batch_size,
        target=target,
        device=device,
    )
    importance = summarize_attributions(attrs, feature_names)
    return {"attributions": attrs, "importance": importance, "feature_names": feature_names}


def ft_integrated_gradients(
    model,
    X: pd.DataFrame,
    *,
    geo_tokens: Optional[np.ndarray] = None,
    baseline_num=None,
    baseline_geo=None,
    steps: int = 50,
    batch_size: int = 256,
    target: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """Integrated gradients for FTTransformerSklearn (numeric + optional geo tokens).

    Categorical features are held fixed; gradients are computed for numeric/geo inputs.
    """
    _require_torch("ft_integrated_gradients")
    if device is None:
        try:
            device = next(model.ft.parameters()).device
        except Exception:
            device = "cpu"
    model.ft.eval()

    X_num, X_cat, X_geo, _, _, _ = model._tensorize_split(
        X, None, None, geo_tokens=geo_tokens, allow_none=True
    )

    X_num = X_num.to(device)
    X_cat = X_cat.to(device)
    X_geo = X_geo.to(device)

    def forward_fn(num, geo=None):
        if geo is None:
            out = model.ft(num, X_cat, X_geo)
        else:
            out = model.ft(num, X_cat, geo)
        if getattr(model, "task_type", None) == "classification":
            out = torch.sigmoid(out)
        return out

    attrs_num = None
    attrs_geo = None

    if X_geo.shape[1] == 0:
        attrs_num = integrated_gradients_torch(
            lambda num: forward_fn(num, None),
            X_num,
            baseline=baseline_num,
            steps=steps,
            batch_size=batch_size,
            target=target,
            device=device,
        )
    else:
        attrs_num, attrs_geo = integrated_gradients_multi_input_torch(
            forward_fn,
            (X_num, X_geo),
            baselines=(baseline_num, baseline_geo),
            steps=steps,
            batch_size=batch_size,
            target=target,
            device=device,
        )

    num_names = list(getattr(model, "num_cols", []))
    geo_names = [f"geo_{i}" for i in range(X_geo.shape[1])]

    results = {
        "attributions_num": attrs_num,
        "attributions_geo": attrs_geo,
        "num_feature_names": num_names,
        "geo_feature_names": geo_names,
    }

    if attrs_num is not None and num_names:
        results["importance_num"] = summarize_attributions(attrs_num, num_names)
    if attrs_geo is not None and geo_names:
        results["importance_geo"] = summarize_attributions(attrs_geo, geo_names)

    return results
