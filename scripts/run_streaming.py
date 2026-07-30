#!/usr/bin/env python3
import argparse
import sys
import warnings
from pathlib import Path

import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.streaming.alert_service import RealTimeAlertService
from src.streaming.incremental_updater import IncrementalGraphUpdater
from src.streaming.kafka_consumer import MarketDataConsumer


def main():
    parser = argparse.ArgumentParser(description="Real-time Streaming Daemon")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--daemon", action="store_true", help="Run continuously")
    parser.add_argument(
        "--interval", type=int, default=300, help="Polling interval in seconds"
    )
    parser.add_argument(
        "--source", type=str, default="auto", choices=["auto", "kafka", "file"]
    )
    args = parser.parse_args()

    print("=" * 60)
    print("STARTING STREAMING LAYER")
    print("=" * 60)

    # Resource limits
    torch.set_num_threads(4)

    # 1. Load latest graph snapshot as warm start
    snapshot_dir = Path("data/processed/graph_snapshots")
    files = sorted(snapshot_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(
            "No graph snapshots found. Please run Week 2 build_graphs.py first."
        )

    base_graph = torch.load(files[-1], weights_only=False)
    print(
        f"✓ Warm start with graph snapshot: {files[-1].stem} (nodes: {base_graph['stock'].x.shape[0]})"
    )

    # 2. Init IncrementalGraphUpdater
    device = "cuda" if torch.cuda.is_available() else "cpu"
    updater = IncrementalGraphUpdater(
        base_graph=base_graph, window_size=21, device=device
    )
    print("✓ IncrementalGraphUpdater initialized")

    # 3. Load Model (Mocking here, but in production load from models/best_htgat.pt)
    class MockModel(torch.nn.Module):
        def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
            n = x_dict["stock"].shape[0]
            return {
                "embedding": torch.randn(n, 64),
                "volatility": torch.ones(n) * 0.2,
                "return": torch.randn(n) * 0.001,
                "cvar": torch.ones(n) * -0.02,
            }

    model = MockModel()
    model.eval()
    # In real code: model = torch.load("models/best_htgat.pt")

    # 4. Init MarketDataConsumer
    consumer = MarketDataConsumer(bootstrap_servers="localhost:9092")
    if args.source == "file":
        consumer._init_fallback()
    else:
        consumer.start()

    # 5. Init RealTimeAlertService
    config = {
        "hhi_threshold": 0.8,
        "cvar_threshold": -0.05,
        "shock_loss_threshold": -0.05,
    }
    alert_dir = Path("alerts")
    alert_dir.mkdir(exist_ok=True)
    alert_path = alert_dir / f"streaming_alerts_{files[-1].stem}.jsonl"

    alert_service = RealTimeAlertService(
        model=model, updater=updater, config=config, alert_log_path=alert_path
    )
    print("✓ RealTimeAlertService initialized")

    # 6. Run
    if args.daemon:
        alert_service.run_daemon(consumer=consumer, check_interval_sec=args.interval)
    else:
        # One-shot
        print("Running one-shot tick...")
        messages = consumer.poll(timeout_ms=1000)
        alerts, elapsed_ms = alert_service.tick()
        alert_service.log_alerts(alerts)
        print(
            f"✓ One-shot tick completed in {elapsed_ms:.0f}ms. Generated {len(alerts)} alerts."
        )

    consumer.stop()
    print("Shutting down cleanly. State saved.")


if __name__ == "__main__":
    main()
