import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from src.models.htgat import HTGAT
from src.models.prediction_heads import MultiTaskPredictionHeads
from src.rebalancing.cost_aware_optimizer import CostAwareOptimizer
from src.risk_engine.concentration_metrics import ConcentrationMetrics
from src.risk_engine.shock_simulator import ShockSimulator


def test_full_pipeline_offline():
    # 1. Synthetic dataset (tiny 5-stock)
    tickers = ["A", "B", "C", "D", "E"]
    n_stocks = len(tickers)

    # 2. Build HeteroData snapshot
    g = HeteroData()
    g["stock"].x = torch.randn(n_stocks, 32)
    g["stock", "correlates_with", "stock"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4], [1, 0, 3, 4, 2]]
    )
    g["stock", "correlates_with", "stock"].edge_attr = torch.randn(5, 1)

    # 3. HTGAT Forward Pass
    htgat = HTGAT(
        node_features=32,
        hidden_dim=64,
        out_dim=64,
        num_heads=2,
        dropout=0.1,
        edge_types=[("stock", "correlates_with", "stock")],
    )
    htgat.eval()

    with torch.no_grad():
        out_gnn = htgat(g.x_dict, g.edge_index_dict, g.edge_attr_dict)
        embeddings = out_gnn["embedding"]

    assert embeddings.shape == (5, 64)

    # 4. MultiTaskHeads
    heads = MultiTaskPredictionHeads(hidden_dim=64)
    heads.eval()

    with torch.no_grad():
        preds = heads(embeddings)

    # Volatility should be positive
    assert (preds["volatility"] > 0).all()

    # 5. Risk Engine (Concentration Metrics)
    weights = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2], index=tickers)
    metrics = ConcentrationMetrics(weights, embeddings.numpy(), tickers)
    res = metrics.compute_all()

    assert "weight_hhi" in res
    assert "embedding_hhi" in res

    # 6. Shock Simulator
    # We must mock model output to be returned during shock simulator's 'run_scenario'
    class MockModel(torch.nn.Module):
        def forward(self, batch):
            return {
                "return": torch.randn(5),
                "cvar": torch.randn(5),
                "embedding": torch.randn(5, 64),
            }

    mock_model = MockModel()
    simulator = ShockSimulator(mock_model, device="cpu")
    shock_res = simulator.run_scenario(g, "liquidity_freeze", tickers, weights)

    assert shock_res["scenario"] == "liquidity_freeze"

    # 7. CostAwareOptimizer
    expected_returns = pd.Series(preds["return"].numpy(), index=tickers)

    # Create simple PSD covariance matrix
    z_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1).numpy()
    cov_matrix = np.eye(n_stocks) * 0.05
    Sigma_gnn = z_norm @ z_norm.T

    optimizer = CostAwareOptimizer(
        expected_returns,
        cov_matrix,
        weights,
        transaction_cost_rate=0.001,
        max_weight=0.3,
        target_hhi=0.25,
    )

    opt_w = optimizer.optimize(Sigma_gnn=Sigma_gnn, beta=0.5)

    # Validates constraints
    assert opt_w.max() <= 0.3 + 1e-4
    assert np.isclose(opt_w.sum(), 1.0)

    print("Full synthetic pipeline successfully executed!")
