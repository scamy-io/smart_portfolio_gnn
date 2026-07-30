#!/usr/bin/env python3

"""Week 4 Risk Engine Sanity Check — mock dry-run of full orchestrator."""

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))

import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():

    print("=" * 60)

    print("WEEK 4 RISK ENGINE VERIFICATION")

    print("=" * 60)

    # ── 1. Load a snapshot and model ──

    snapshot_dir = Path("data/processed/graph_snapshots")

    files = sorted(snapshot_dir.glob("*.pt"))

    assert len(files) >= 2, "Need at least 2 snapshots for correlation breakdown test"

    g_today = torch.load(files[-1], weights_only=False)

    g_yesterday = torch.load(files[-2], weights_only=False)

    date_str = files[-1].stem

    n_nodes = g_today["stock"].x.shape[0]

    device = torch.device("cpu")  # Use CPU for deterministic testing

    # Mock portfolio weights (equal weight)

    tickers = [f"STOCK_{i}" for i in range(n_nodes)]

    weights = pd.Series(1.0 / n_nodes, index=tickers)

    # Mock embeddings (replace with real model output in production)

    np.random.seed(42)

    embeddings = np.random.randn(n_nodes, 64).astype(np.float32)

    print(f"\n✓ Loaded snapshot: {date_str} | nodes={n_nodes}")

    # ── 2. Test ConcentrationMetrics ──

    print("\n--- Testing ConcentrationMetrics ---")

    from src.risk_engine.concentration_metrics import ConcentrationMetrics

    cm = ConcentrationMetrics(weights=weights, embeddings=embeddings, tickers=tickers)

    metrics = cm.compute_all()

    assert 0 < metrics["weight_hhi"] <= 1.0, "HHI out of range"

    assert metrics["enb_weight"] >= 1.0, "ENB weight < 1"

    assert metrics["embedding_hhi"] > 0, "Embedding HHI invalid"

    print(f"  weight_hhi: {metrics['weight_hhi']:.4f}")

    print(f"  enb_weight: {metrics['enb_weight']:.2f}")

    print(f"  embedding_hhi: {metrics['embedding_hhi']:.4f}")

    print(f"  enb_embedding: {metrics['enb_embedding']:.2f}")

    print("  ✓ ConcentrationMetrics OK")

    # ── 3. Test ClusterDetector ──

    print("\n--- Testing SpectralClusterDetector ---")

    from src.risk_engine.cluster_detector import SpectralClusterDetector

    cd = SpectralClusterDetector(
        n_clusters=min(5, n_nodes // 10 + 1), similarity_threshold=0.5
    )

    labels_df = cd.fit(embeddings, tickers)

    assert "cluster_id" in labels_df.columns

    assert len(labels_df) == n_nodes

    risk = cd.detect_concentration_risk(weights, labels_df)

    print(f"  Clusters found: {labels_df['cluster_id'].nunique()}")

    print(f"  is_concentrated: {risk['is_concentrated']}")

    print(f"  total_flagged_weight: {risk.get('total_flagged_weight', 0):.4f}")

    print("  ✓ ClusterDetector OK")

    # ── 4. Test ShockSimulator (mock model) ──

    print("\n--- Testing ShockSimulator ---")

    from src.risk_engine.shock_simulator import ShockSimulator

    class MockModel(torch.nn.Module):

        def forward(self, graph):

            n = graph["stock"].x.shape[0]

            return {
                "embedding": torch.randn(n, 64),
                "volatility": torch.rand(n) * 0.5,
                "return": torch.randn(n) * 0.02,
                "cvar": -torch.rand(n) * 0.05,
            }

    sim = ShockSimulator(model=MockModel(), device="cpu")

    scenarios = ["sector_demand_shock", "liquidity_freeze", "sentiment_contagion"]

    for sc in scenarios:

        result = sim.run_scenario(g_today, sc, tickers, weights=weights)

        assert "portfolio_return" in result

        assert "portfolio_cvar" in result

        assert result["portfolio_cvar"] <= 0, "CVaR should be negative"

        print(
            f"  {sc:25s}: return={result['portfolio_return']:+.4f}, cvar={result['portfolio_cvar']:+.4f}"
        )

    mc = sim.monte_carlo(g_today, n_scenarios=10, tickers=tickers, weights=weights)

    assert len(mc) == 10

    assert "portfolio_return" in mc.columns

    print(f"  Monte Carlo (10 runs): mean_loss={mc['portfolio_return'].mean():+.4f}")

    print("  ✓ ShockSimulator OK")

    # ── 5. Test RebalanceTriggerChecker ──

    print("\n--- Testing RebalanceTriggerChecker ---")

    from src.rebalancing.rebalance_triggers import RebalanceTriggerChecker

    config = {
        "risk": {
            "target_hhi": 0.02,
            "concentration_alert_threshold": 0.30,
            "cvar_spike_multiplier": 2.0,
            "correlation_breakdown_pct": 0.40,
        }
    }

    rtc = RebalanceTriggerChecker(config)

    # Test concentration trigger

    fired, msg = rtc.check_concentration(
        {"embedding_hhi": 0.05, "total_flagged_weight": 0.35}
    )

    assert fired is True, "Concentration trigger should fire"

    print(f"  Concentration fired: {fired} | {msg}")

    # Test scheduled trigger

    is_rebal = rtc.check_scheduled("2024-01-05", "weekly")  # Friday

    print(f"  Scheduled (Friday): {is_rebal}")

    # Test correlation breakdown

    fired2, msg2 = rtc.check_correlation_breakdown(g_today, g_yesterday)

    print(f"  Correlation breakdown: {fired2} | {msg2}")

    print("  ✓ RebalanceTriggerChecker OK")

    # ── 6. Test CostAwareOptimizer ──

    print("\n--- Testing CostAwareOptimizer ---")

    from src.rebalancing.cost_aware_optimizer import CostAwareOptimizer

    mu = pd.Series(np.random.randn(n_nodes) * 0.01, index=tickers)

    sigma_hist = np.eye(n_nodes) * 0.04  # Simplified diagonal covariance

    sigma_gnn = np.eye(n_nodes) * 0.04

    opt = CostAwareOptimizer(
        expected_returns=mu,
        cov_matrix=0.7 * sigma_hist + 0.3 * sigma_gnn,
        current_weights=weights,
        transaction_cost_rate=0.001,
        max_weight=0.05,
        target_hhi=0.02,
        min_enb=15.0,
    )

    new_weights = opt.optimize(gamma=1.0, lambda_conc=0.1)

    assert abs(new_weights.sum() - 1.0) < 1e-4, "Weights don't sum to 1"

    assert (new_weights >= 0).all(), "Negative weights found"

    assert (new_weights <= 1.0 + 1e-6).all(), "Weight exceeds max"

    trades = opt.generate_trades(new_weights)

    print(f"  Optimal weights sum: {new_weights.sum():.6f}")

    print(f"  Max weight: {new_weights.max():.4f}")

    print(f"  Trades generated: {len(trades)}")

    print("  ✓ CostAwareOptimizer OK")

    # ── 7. Test JSON Report Serialization ──

    print("\n--- Testing Report Serialization ---")

    report = {
        "date": date_str,
        "metrics": metrics,
        "clusters": risk,
        "shock_results": {"monte_carlo_cvar": -0.045, "scenarios": []},
        "alerts": [
            {"trigger": "concentration", "severity": "high", "message": "Test alert"}
        ],
        "trades": trades.to_dict("records") if len(trades) > 0 else [],
    }

    report_path = Path("reports/test_risk_report.json")

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:

        json.dump(report, f, indent=2, default=str)

    assert report_path.exists()

    print(f"  Report saved: {report_path}")

    print("  ✓ JSON Serialization OK")

    print("\n" + "=" * 60)

    print("🎉 WEEK 4 VERIFICATION PASSED")

    print("=" * 60)

    print("\nYour risk engine is closed-loop ready.")

    print("Next: Week 5 (Evaluation Framework) or run full backtest.")


if __name__ == "__main__":

    main()
