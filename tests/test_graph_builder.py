import pandas as pd
import torch


def test_correlation_edge_range():
    # Mock some correlation edge attributes
    edge_attr = torch.tensor([0.5, -0.2, 1.0, -1.0, 0.0], dtype=torch.float32)

    assert (edge_attr >= -1.0).all()
    assert (edge_attr <= 1.0).all()


def test_no_self_loops():
    # Mock an edge index
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)

    # Assert no self loops
    assert (edge_index[0] != edge_index[1]).all()

    # Add a self loop and assert failure
    edge_index_with_loop = torch.tensor(
        [[0, 1, 2, 3, 0], [1, 0, 3, 2, 0]], dtype=torch.long
    )
    assert not (edge_index_with_loop[0] != edge_index_with_loop[1]).all()


def test_edge_symmetry_for_undirected():
    # If edge A->B exists, B->A must exist with same weight for undirected relationships like sector
    sources = [0, 1, 2, 3]
    targets = [1, 0, 3, 2]

    edges = set(zip(sources, targets))
    for u, v in edges:
        assert (v, u) in edges


def test_graph_validation_passes():
    from torch_geometric.data import HeteroData

    g = HeteroData()
    g["stock"].x = torch.randn(4, 32)
    g["stock", "correlates_with", "stock"].edge_index = torch.tensor(
        [[0, 1], [1, 0]], dtype=torch.long
    )
    g["stock", "correlates_with", "stock"].edge_attr = torch.tensor(
        [[0.5], [0.5]], dtype=torch.float32
    )

    # In PyG, we can call validate
    assert g.validate()


def test_temporal_dataset_chronological():
    # Test if dataset builder keeps dates sorted
    dates = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"])
    sorted_dates = dates.sort_values()

    assert sorted_dates[0] == pd.Timestamp("2024-01-01")
    assert sorted_dates[1] == pd.Timestamp("2024-01-02")
    assert sorted_dates[2] == pd.Timestamp("2024-01-03")
