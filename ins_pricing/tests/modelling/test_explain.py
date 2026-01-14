import numpy as np
import pandas as pd
import pytest


def test_permutation_importance_simple():
    from ins_pricing.explain import permutation_importance

    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=200),
            "x2": rng.normal(size=200),
        }
    )
    y = 3.0 * X["x1"].to_numpy() + rng.normal(scale=0.1, size=200)

    def predict_fn(df):
        return 3.0 * df["x1"].to_numpy()

    imp = permutation_importance(
        predict_fn,
        X,
        y,
        metric="rmse",
        n_repeats=3,
        max_rows=None,
        random_state=0,
    )

    assert imp.loc[0, "feature"] == "x1"
    assert imp["importance_mean"].iloc[0] > imp["importance_mean"].iloc[1]


def test_integrated_gradients_linear():
    torch = pytest.importorskip("torch")
    from ins_pricing.explain import integrated_gradients_torch

    torch.manual_seed(0)
    model = torch.nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        model.weight[:] = torch.tensor([[1.0, 2.0, -1.0]])

    X = torch.tensor(
        [[1.0, 2.0, 3.0], [0.5, -1.0, 0.0]],
        dtype=torch.float32,
    )

    def forward(x):
        return model(x).squeeze(1)

    attrs = integrated_gradients_torch(forward, X, baseline="zeros", steps=10)
    expected = X.numpy() * np.array([1.0, 2.0, -1.0])

    assert attrs.shape == X.shape
    np.testing.assert_allclose(attrs, expected, rtol=1e-2, atol=1e-2)
