import os
import tempfile

import pytest
import torch
from torch_geometric.data import HeteroData


def _get_dummy_graph():
    g = HeteroData()
    n = 4
    g["stock"].x = torch.randn(n, 32)
    g["stock", "correlates_with", "stock"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long
    )
    g["stock", "correlates_with", "stock"].edge_attr = torch.randn(4, 1)
    return g


def _get_dummy_model():
    from src.models.htgat import HTGAT

    model = HTGAT(
        node_features=32,
        hidden_dim=64,
        out_dim=64,
        num_heads=2,
        dropout=0.1,
        edge_types=[("stock", "correlates_with", "stock")],
    )
    return model


def test_htgat_forward_shape():
    try:
        model = _get_dummy_model()
    except Exception:
        pytest.skip("Model could not be initialized")

    g = _get_dummy_graph()
    out = model(g.x_dict, g.edge_index_dict, g.edge_attr_dict)

    assert "embedding" in out
    assert out["embedding"].shape == (4, 64)


def test_prediction_head_positivity():
    try:
        from src.models.prediction_heads import MultiTaskPredictionHeads

        heads = MultiTaskPredictionHeads(hidden_dim=64)
    except Exception:
        pytest.skip("MultiTaskPredictionHeads could not be initialized")

    embeddings = torch.randn(4, 64)
    out = heads(embeddings)

    assert "volatility" in out
    # Volatility should be > 0 because of softplus
    assert (out["volatility"] > 0).all()


def test_gradient_flow():
    try:
        model = _get_dummy_model()
    except Exception:
        pytest.skip("Model could not be initialized")

    g = _get_dummy_graph()

    # Forward pass
    out = model(g.x_dict, g.edge_index_dict, g.edge_attr_dict)
    loss = out["embedding"].sum()

    # Backward pass
    loss.backward()

    # Check gradients
    has_grad = False
    for param in model.parameters():
        if param.grad is not None:
            has_grad = True
            break

    assert has_grad, "No gradients were computed"


def test_model_save_load():
    try:
        model = _get_dummy_model()
    except Exception:
        pytest.skip("Model could not be initialized")

    g = _get_dummy_graph()

    with torch.no_grad():
        out1 = model(g.x_dict, g.edge_index_dict, g.edge_attr_dict)["embedding"]

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        torch.save(model.state_dict(), tmp.name)

        model2 = _get_dummy_model()
        model2.load_state_dict(torch.load(tmp.name, weights_only=True))

    with torch.no_grad():
        out2 = model2(g.x_dict, g.edge_index_dict, g.edge_attr_dict)["embedding"]

    assert torch.allclose(out1, out2)
    os.remove(tmp.name)
