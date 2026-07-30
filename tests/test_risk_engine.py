import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from src.risk_engine.concentration_metrics import ConcentrationMetrics
from src.risk_engine.shock_simulator import ShockSimulator


def test_weight_hhi():
    weights = pd.Series([0.5, 0.5, 0.0])
    embeddings = np.random.randn(3, 10)
    tickers = ["A", "B", "C"]

    metrics = ConcentrationMetrics(weights, embeddings, tickers)
    hhi = metrics.weight_hhi()

    # 0.5^2 + 0.5^2 + 0 = 0.25 + 0.25 = 0.5
    assert np.isclose(hhi, 0.5)


def test_embedding_hhi_weighted():
    # Regression test for embedding_hhi not considering weights
    weights1 = pd.Series([1.0, 0.0])
    weights2 = pd.Series([0.5, 0.5])

    # Large norm difference
    embeddings = np.array([[10.0, 0.0], [1.0, 0.0]])
    tickers = ["A", "B"]

    metrics1 = ConcentrationMetrics(weights1, embeddings, tickers)
    hhi1 = metrics1.embedding_hhi()

    metrics2 = ConcentrationMetrics(weights2, embeddings, tickers)
    hhi2 = metrics2.embedding_hhi()

    # The HHI should change because it now factors in weights, not just raw embeddings
    assert not np.isclose(hhi1, hhi2)


def _get_dummy_heterodata():
    g = HeteroData()
    # 5 nodes
    g["stock"].x = torch.ones(5, 4)
    # 5 correlation edges (fully connected for 2 nodes, plus some)
    g["stock", "correlates_with", "stock"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4], [1, 0, 3, 4, 2]]
    )
    g["stock", "correlates_with", "stock"].edge_attr = torch.ones(5, 1)
    g["stock", "same_sector_as", "stock"].edge_index = torch.tensor([[0, 1], [1, 0]])
    g["stock", "same_sector_as", "stock"].edge_attr = torch.ones(2, 1)
    return g


def test_shock_simulator_liquidity_freeze():
    g = _get_dummy_heterodata()
    import torch.nn as nn

    simulator = ShockSimulator(model=nn.Module(), device="cpu")

    num_edges_before = g["stock", "correlates_with", "stock"].edge_index.size(1)

    g_shocked = simulator.liquidity_freeze(g, removal_pct=0.5)

    num_edges_after = g_shocked["stock", "correlates_with", "stock"].edge_index.size(1)

    assert num_edges_after < num_edges_before


def test_shock_simulator_sector_demand():
    g = _get_dummy_heterodata()
    import torch.nn as nn

    simulator = ShockSimulator(model=nn.Module(), device="cpu")

    num_sector_edges_before = g["stock", "same_sector_as", "stock"].edge_index.size(1)

    g_shocked = simulator.sector_demand_shock(
        g, target_sector_indices=[0, 1], removal_pct=0.5
    )

    num_sector_edges_after = g_shocked[
        "stock", "same_sector_as", "stock"
    ].edge_index.size(1)

    assert num_sector_edges_after <= num_sector_edges_before


def test_shock_simulator_supply_chain():
    g = _get_dummy_heterodata()
    import torch.nn as nn

    simulator = ShockSimulator(model=nn.Module(), device="cpu")

    # Should zero out some features
    features_before = g["stock"].x.clone()

    g_shocked = simulator.supply_chain_failure(g, target_node_idx=0)

    features_after = g_shocked["stock"].x

    # Sum of features should be lower since some are zeroed
    assert features_after.sum() < features_before.sum()
