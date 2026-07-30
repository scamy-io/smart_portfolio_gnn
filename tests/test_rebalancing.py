import numpy as np
import pandas as pd

from src.rebalancing.cost_aware_optimizer import CostAwareOptimizer


def test_optimizer_respects_constraints():
    tickers = ["A", "B", "C", "D", "E"]
    expected_returns = pd.Series([0.05, 0.02, 0.08, 0.01, 0.04], index=tickers)
    cov_matrix = np.eye(5) * 0.01
    current_weights = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2], index=tickers)

    # Very strict constraints
    max_weight = 0.3
    target_hhi = 0.25

    optimizer = CostAwareOptimizer(
        expected_returns,
        cov_matrix,
        current_weights,
        max_weight=max_weight,
        target_hhi=target_hhi,
    )

    opt_w = optimizer.optimize()

    # Check max weight constraint
    assert opt_w.max() <= max_weight + 1e-4

    # Check target HHI constraint
    hhi = np.sum(opt_w.values**2)
    assert hhi <= target_hhi + 1e-4

    # Check sum to 1
    assert np.isclose(opt_w.sum(), 1.0)


def test_optimizer_fallback_infeasible():
    tickers = ["A", "B", "C"]
    expected_returns = pd.Series([0.1, -0.1, 0.0], index=tickers)
    cov_matrix = np.eye(3) * 0.05
    current_weights = pd.Series([0.33, 0.33, 0.34], index=tickers)

    # Impossible constraints (max_weight * 3 < 1.0)
    optimizer = CostAwareOptimizer(
        expected_returns,
        cov_matrix,
        current_weights,
        max_weight=0.2,  # 3 * 0.2 = 0.6, cannot sum to 1
    )

    # Should not raise exception, but fallback to equal weight (which violates max_weight, but it's a fallback)
    opt_w = optimizer.optimize()

    # Fallback to equal weight
    assert np.allclose(opt_w.values, np.ones(3) / 3.0)


def test_optimizer_with_sigma_gnn():
    tickers = ["A", "B", "C"]
    expected_returns = pd.Series([0.05, 0.02, 0.08], index=tickers)
    cov_matrix = np.eye(3) * 0.01
    current_weights = pd.Series([0.33, 0.33, 0.34], index=tickers)

    Sigma_gnn = np.eye(3) * 0.02

    optimizer = CostAwareOptimizer(
        expected_returns, cov_matrix, current_weights, max_weight=0.5
    )

    opt_w = optimizer.optimize(Sigma_gnn=Sigma_gnn, beta=0.5)

    assert np.isclose(opt_w.sum(), 1.0)
    assert opt_w.max() <= 0.5 + 1e-4
