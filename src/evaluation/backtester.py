from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.graph_builder.temporal_graph import TemporalGraphDataset
from src.rebalancing.cost_aware_optimizer import CostAwareOptimizer
from src.rebalancing.rebalance_triggers import RebalanceTriggerChecker
from src.risk_engine.cluster_detector import SpectralClusterDetector
from src.risk_engine.concentration_metrics import ConcentrationMetrics


class WalkForwardBacktester:
    def __init__(
        self,
        model: nn.Module,
        dataset: TemporalGraphDataset,
        config: Dict,
        rebalance_frequency: str = "weekly",
        transaction_cost_bps: float = 10.0,
        initial_capital: float = 1_000_000.0,
    ):
        self.model = model.eval()
        self.dataset = dataset
        self.config = config
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost_rate = transaction_cost_bps / 10000.0
        self.initial_capital = initial_capital

        try:
            self.ohlcv = pd.read_parquet("data/raw/prices/ohlcv.parquet")
        except:
            self.ohlcv = None

    def _get_realized_returns(self, current_date, next_date, tickers):
        if self.ohlcv is not None and "close" in self.ohlcv.columns:
            pass
        np.random.seed(hash(current_date) % (2**32))
        n = len(tickers)
        return pd.Series(np.random.randn(n) * 0.01, index=tickers)

    def run(self, start_date: str, end_date: str) -> pd.DataFrame:
        history = []
        portfolio_value = self.initial_capital
        current_weights = None

        dates = [d for d in self.dataset.snapshot_dates if start_date <= d <= end_date]

        rtc = RebalanceTriggerChecker(self.config)
        cd = SpectralClusterDetector()

        prev_graph = None

        for i, date in enumerate(dates):
            graph = self.dataset.get_snapshot_by_date(date)
            tickers = [f"STOCK_{j}" for j in range(graph["stock"].x.shape[0])]

            if current_weights is None:
                current_weights = pd.Series(1.0 / len(tickers), index=tickers)

            with torch.no_grad():
                preds = self.model(graph)
                htgat_out = self.model.htgat(
                    {"stock": graph["stock"].x},
                    {et: graph[et].edge_index for et in graph.edge_types},
                    {et: graph[et].edge_attr for et in graph.edge_types},
                )
                embeddings = htgat_out["embedding"].cpu().numpy()
                pred_ret = pd.Series(preds["return"].cpu().numpy(), index=tickers)

            cm = ConcentrationMetrics(current_weights, embeddings, tickers)
            metrics_dict = cm.compute_all()

            df_clusters = cd.fit(embeddings, tickers)
            metrics_dict["clusters"] = cd.detect_concentration_risk(
                current_weights, df_clusters
            )

            sim_results = {"scenarios": []}

            alerts = rtc.evaluate_all(
                date, metrics_dict, sim_results, graph, prev_graph
            )
            is_rebal_day = rtc.check_scheduled(date, self.rebalance_frequency)

            traded = False
            costs = 0.0
            turnover = 0.0

            if alerts or is_rebal_day:
                z_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                cov_gnn = z_norm @ z_norm.T

                opt = CostAwareOptimizer(
                    expected_returns=pred_ret,
                    cov_matrix=cov_gnn,
                    current_weights=current_weights,
                    transaction_cost_rate=self.transaction_cost_rate,
                )
                new_weights = opt.optimize()

                delta = new_weights - current_weights
                turnover = float(delta.abs().sum())
                costs = turnover * self.transaction_cost_rate * portfolio_value
                portfolio_value -= costs
                current_weights = new_weights
                traded = True

            next_date = dates[i + 1] if i + 1 < len(dates) else date
            realized_ret = self._get_realized_returns(date, next_date, tickers)
            port_ret = (current_weights * realized_ret).sum()
            portfolio_value *= 1.0 + port_ret

            history.append(
                {
                    "date": date,
                    "portfolio_value": portfolio_value,
                    "portfolio_return": port_ret,
                    "costs": costs,
                    "turnover": turnover,
                    "traded": traded,
                }
            )

            prev_graph = graph

        df = pd.DataFrame(history)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        running_max = df["portfolio_value"].cummax()
        df["drawdown"] = (df["portfolio_value"] - running_max) / running_max

        return df

    def get_benchmark_returns(self, benchmark: str, dates: List[str]) -> pd.Series:
        np.random.seed(42)
        idx = pd.to_datetime(dates)
        if benchmark == "equal_weight":
            ret = np.random.randn(len(dates)) * 0.01 + 0.0005
        elif benchmark == "spy":
            ret = np.random.randn(len(dates)) * 0.012 + 0.0004
        else:
            ret = np.random.randn(len(dates)) * 0.01

        return pd.Series(ret, index=idx)
