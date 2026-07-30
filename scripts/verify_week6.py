#!/usr/bin/env python3
"""Week 6 Streaming & Dashboard Verification."""

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).parent.parent))


def main():
    print("=" * 70)
    print("WEEK 6 STREAMING & DASHBOARD VERIFICATION")
    print("=" * 70)

    # ── 1. Test Incremental Updater ──
    print("\n--- Testing IncrementalGraphUpdater ---")
    from src.streaming.incremental_updater import IncrementalGraphUpdater

    snapshot_dir = Path("data/processed/graph_snapshots")
    files = sorted(snapshot_dir.glob("*.pt"))
    assert len(files) >= 2, "Need at least 2 snapshots"

    base = torch.load(files[-2], weights_only=False)
    n_nodes = base["stock"].x.shape[0]
    tickers = [f"T{i}" for i in range(n_nodes)]

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    updater = IncrementalGraphUpdater(
        base_graph=base, window_size=21, device=device_str
    )

    # Simulate 5 days of price ticks
    np.random.seed(42)
    for day in range(5):
        prices = pd.DataFrame(
            {
                "ticker": tickers,
                "open": 100 + np.random.randn(n_nodes),
                "high": 102 + np.random.randn(n_nodes),
                "low": 99 + np.random.randn(n_nodes),
                "close": 101 + np.random.randn(n_nodes),
                "adj_close": 101 + np.random.randn(n_nodes),
                "volume": np.random.randint(1e6, 1e7, n_nodes),
            }
        )
        updater.push_prices(prices)

    # Measure correlation update latency
    t0 = time.time()
    g = updater.update_correlation_edges()
    t1 = time.time()

    assert any("correlates_with" in str(et) for et in g.edge_types)
    assert t1 - t0 < 0.5, f"Correlation update too slow: {t1-t0:.3f}s"
    print(f"  ✓ Correlation update: {(t1-t0)*1000:.1f}ms")

    # Test sentiment update
    news = [
        {
            "timestamp": "2026-07-30T10:00:00Z",
            "tickers_mentioned": ["T0", "T1"],
            "tone": -2.5,
            "is_sector_wide": False,
        },
        {
            "timestamp": "2026-07-30T10:05:00Z",
            "tickers_mentioned": ["T2", "T3", "T4"],
            "tone": -1.2,
            "is_sector_wide": True,
        },
    ]

    # Needs a ticker to index map for updater
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    g2 = updater.update_sentiment_edges(news, ticker_to_idx)
    assert any("sentiment" in str(et) for et in g2.edge_types)
    print("  ✓ Sentiment edges updated")

    # ── 2. Test Alert Service ──
    print("\n--- Testing RealTimeAlertService ---")
    from src.streaming.alert_service import RealTimeAlertService

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, graph):
            n = graph["stock"].x.shape[0]
            return {
                "embedding": torch.randn(n, 64),
                "volatility": torch.ones(n) * 0.2,
                "return": torch.randn(n) * 0.001,
                "cvar": torch.ones(n) * -0.02,
            }

    with tempfile.TemporaryDirectory() as tmpdir:
        alert_path = Path(tmpdir) / "alerts.jsonl"
        service = RealTimeAlertService(
            model=MockModel(),
            updater=updater,
            config={
                "risk": {"target_hhi": 0.02, "concentration_alert_threshold": 0.30}
            },
            alert_log_path=alert_path,
        )

        alerts, _ = service.tick()
        service.log_alerts(alerts)
        assert isinstance(alerts, list)
        print(f"  ✓ Tick processed, alerts: {len(alerts)}")

        if alerts:
            assert alert_path.exists()
            with open(alert_path) as f:
                logged = [json.loads(line) for line in f]
            assert len(logged) == len(alerts)
            assert "timestamp" in logged[0]
            assert "severity" in logged[0]
            print(f"  ✓ Alerts logged to JSONL")

    # ── 3. Test File Polling Consumer ──
    print("\n--- Testing MarketDataConsumer (File Fallback) ---")
    from src.streaming.kafka_consumer import MarketDataConsumer

    with tempfile.TemporaryDirectory() as tmpdir:
        incoming = Path(tmpdir) / "incoming"
        incoming.mkdir()

        # Write test messages
        (incoming / "msg_001.json").write_text(
            json.dumps(
                {
                    "type": "price",
                    "timestamp": "2026-07-30T14:30:00Z",
                    "data": {"ticker": "T0", "close": 150.0, "volume": 1000},
                }
            )
        )

        consumer = MarketDataConsumer(
            bootstrap_servers="localhost:9999",  # Will fail, triggers fallback
            fallback_dir=incoming,
        )
        consumer.start()
        msgs = consumer.poll(timeout_ms=100)
        consumer.stop()

        assert len(msgs) == 1
        assert msgs[0]["type"] == "price"
        assert (incoming / "processed" / "msg_001.json").exists()
        print("  ✓ File polling fallback works")

    # ── 4. Test Dashboard Data Loader ──
    print("\n--- Testing DashboardDataLoader ---")
    from dashboard.data_loader import DashboardDataLoader

    loader = DashboardDataLoader()
    # These will fail gracefully if files don't exist — just test instantiation
    try:
        w = loader.load_portfolio_weights()
        print(f"  ✓ Portfolio weights loaded: {len(w)} rows")
    except FileNotFoundError:
        print("  ⚠ Portfolio weights file not found (expected if backtest not run)")

    # ── 5. Test State Persistence ──
    print("\n--- Testing State Persistence ---")
    state_path = Path("data/streaming/state.pkl")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    updater.save_state(state_path)
    assert state_path.exists()

    new_updater = IncrementalGraphUpdater(base_graph=base, window_size=21)
    new_updater.load_state(state_path)
    print("  ✓ State save/load works")

    print("\n" + "=" * 70)
    print("🎉 WEEK 6 VERIFICATION PASSED")
    print("=" * 70)
    print("\nStart services:")
    print("  Terminal 1: python scripts/run_streaming.py --daemon")
    print("  Terminal 2: python scripts/run_dashboard.py")


if __name__ == "__main__":
    main()
