import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from ins_pricing.bayesopt.models import GraphNeuralNetSklearn


def test_graph_cache_meta_invalidation(tmp_path):
    X = pd.DataFrame({"a": [0.1, 0.2, 0.3], "b": [1.0, 2.0, 3.0]})
    cache_path = tmp_path / "gnn_cache.pt"

    model = GraphNeuralNetSklearn(
        model_nme="demo",
        input_dim=2,
        k_neighbors=1,
        epochs=1,
        use_approx_knn=False,
        graph_cache_path=str(cache_path),
    )

    adj = model._build_graph_from_df(X)
    assert adj is not None
    assert cache_path.exists()

    cached = model._load_cached_adj(X)
    assert cached is not None

    X_changed = X.copy()
    X_changed.iloc[0, 0] += 1.0
    assert model._load_cached_adj(X_changed) is None
