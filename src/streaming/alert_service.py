import datetime
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn

from src.risk_engine.concentration_metrics import ConcentrationMetrics
from src.risk_engine.shock_simulator import ShockSimulator
from src.streaming.incremental_updater import IncrementalGraphUpdater


class RealTimeAlertService:
    def __init__(
        self,
        model: nn.Module,
        updater: IncrementalGraphUpdater,
        config: Dict,
        alert_log_path: Path,
    ):
        self.model = model
        self.updater = updater
        self.config = config
        self.alert_log_path = alert_log_path

        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.shock_simulator = ShockSimulator(model=self.model, device=str(self.device))

        # We need mock weights for risk engine, assume equal weights
        self.tickers = [f"STOCK_{i}" for i in range(self.updater.num_nodes)]
        import pandas as pd

        self.weights = pd.Series(1.0 / self.updater.num_nodes, index=self.tickers)

    def tick(self) -> List[Dict]:
        start_time = time.time()

        graph = self.updater.get_graph()

        with torch.no_grad():
            if hasattr(self.model, "htgat"):
                x_dict = {"stock": graph["stock"].x}
                edge_index_dict = {et: graph[et].edge_index for et in graph.edge_types}
                edge_attr_dict = {et: graph[et].edge_attr for et in graph.edge_types}
                preds = self.model(x_dict, edge_index_dict, edge_attr_dict)
                embeddings = preds["embedding"]
            else:
                preds = self.model(graph)
                embeddings = preds["embedding"]

        alerts = []

        # 1. Concentration Check
        cm = ConcentrationMetrics(
            weights=self.weights,
            embeddings=embeddings.cpu().numpy(),
            tickers=self.tickers,
        )
        metrics = cm.compute_all()

        # If embedding HHI > threshold (e.g. 0.8)
        if metrics.get("embedding_hhi", 0) > self.config.get("hhi_threshold", 0.8):
            alerts.append(
                {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "severity": "high",
                    "type": "concentration",
                    "message": f"Embedding HHI exceeded threshold: {metrics['embedding_hhi']:.2f}",
                    "data": {"embedding_hhi": metrics["embedding_hhi"]},
                }
            )

        # 2. Predicted CVaR Check
        mean_cvar = preds["cvar"].mean().item()
        if mean_cvar < self.config.get("cvar_threshold", -0.05):
            alerts.append(
                {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "severity": "medium",
                    "type": "risk",
                    "message": f"Average CVaR breached threshold: {mean_cvar:.4f}",
                    "data": {"mean_cvar": mean_cvar},
                }
            )

        # 3. Lightweight Shock Scenarios
        scenarios = ["sector_demand_shock", "liquidity_freeze", "sentiment_contagion"]
        for sc in scenarios:
            res = self.shock_simulator.run_scenario(
                graph, sc, self.tickers, weights=self.weights
            )
            if res["portfolio_return"] < self.config.get("shock_loss_threshold", -0.05):
                alerts.append(
                    {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "severity": "high",
                        "type": "shock",
                        "message": f"Shock scenario '{sc}' causes portfolio loss > threshold: {res['portfolio_return']:.2%}",
                        "data": {
                            "scenario": sc,
                            "portfolio_return": res["portfolio_return"],
                        },
                    }
                )

        elapsed_ms = (time.time() - start_time) * 1000
        return alerts, elapsed_ms

    def log_alerts(self, alerts: List[Dict]):
        if not alerts:
            return

        with open(self.alert_log_path, "a", encoding="utf-8") as f:
            for alert in alerts:
                f.write(json.dumps(alert) + "\n")

    def run_daemon(self, consumer=None, check_interval_sec: int = 300):
        print(
            f"Starting RealTimeAlertService daemon. Checking every {check_interval_sec}s."
        )
        try:
            while True:
                # Polling for data
                if consumer:
                    messages = consumer.poll(timeout_ms=1000)
                    for msg in messages:
                        if msg["type"] == "price":
                            # Mock dataframe for prices
                            df = pd.DataFrame(msg["data"])
                            self.updater.push_prices(df)
                        elif msg["type"] == "news":
                            self.updater.update_sentiment_edges(
                                msg["data"],
                                ticker_to_idx={
                                    t: i for i, t in enumerate(self.tickers)
                                },
                            )
                        elif msg["type"] == "macro":
                            self.updater.update_node_features(msg["data"])

                    if messages:
                        self.updater.update_correlation_edges()

                alerts, elapsed_ms = self.tick()
                self.log_alerts(alerts)

                print(
                    f"Tick at {datetime.datetime.now().strftime('%H:%M:%S')} | {len(alerts)} alerts | graph updated in {elapsed_ms:.0f}ms"
                )
                time.sleep(check_interval_sec)
        except KeyboardInterrupt:
            print("\nShutting down daemon gracefully...")
