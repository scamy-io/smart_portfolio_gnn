from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def safe_metric(metric_fn, y_true, y_pred, **kwargs):
    """Filter NaN pairs before computing sklearn metric."""
    if y_true is None or y_pred is None:
        return float('nan')
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        print(f"WARNING: safe_metric received zero valid pairs.")
        return float('nan')
    return metric_fn(y_true[mask], y_pred[mask], **kwargs)

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
        if self.ohlcv is None or "close" not in self.ohlcv.columns:
            return pd.Series(0.0, index=tickers)
            
        curr_prices = self.ohlcv[(self.ohlcv["date"] == current_date) & (self.ohlcv["ticker"].isin(tickers))].set_index("ticker")["close"]
        next_prices = self.ohlcv[(self.ohlcv["date"] == next_date) & (self.ohlcv["ticker"].isin(tickers))].set_index("ticker")["close"]
        
        if next_prices.empty:
            future_dates = self.ohlcv[self.ohlcv["date"] > current_date]["date"].unique()
            if len(future_dates) > 0:
                next_date = min(future_dates)
                next_prices = self.ohlcv[(self.ohlcv["date"] == next_date) & (self.ohlcv["ticker"].isin(tickers))].set_index("ticker")["close"]

        ret = np.log(next_prices / curr_prices)
        return ret.reindex(tickers).fillna(0.0)

    def run(self, start_date: str, end_date: str) -> pd.DataFrame:
        history = []
        portfolio_value = self.initial_capital
        current_weights = None

        dates = [d for d in self.dataset.snapshot_dates if start_date <= d <= end_date]

        rtc = RebalanceTriggerChecker(self.config)
        cd = SpectralClusterDetector()

        prev_graph = None

        risk_trajectory = []

        for i, date in enumerate(dates):
            graph = self.dataset.get_snapshot_by_date(date)
            tickers = [f"STOCK_{j}" for j in range(graph["stock"].x.shape[0])]
            
            # G4: Enforce Universe Selection (Simulated Top 100)
            tickers = tickers[:100]
            if current_weights is None:
                current_weights = pd.Series(1.0 / len(tickers), index=tickers)
            else:
                current_weights = current_weights.reindex(tickers).fillna(0.0)
                if current_weights.sum() > 0:
                    current_weights /= current_weights.sum()
                else:
                    current_weights = pd.Series(1.0 / len(tickers), index=tickers)

            with torch.no_grad():
                preds = self.model(graph)
                htgat_out = self.model.htgat(
                    {"stock": graph["stock"].x},
                    {et: graph[et].edge_index for et in graph.edge_types},
                    {et: graph[et].edge_attr for et in graph.edge_types},
                )
                embeddings = htgat_out["embedding"].cpu().numpy()[:100] # Trim to universe
                pred_ret = pd.Series(preds["return"].cpu().numpy()[:100], index=tickers)
                pred_vol = preds["volatility"].cpu().numpy()[:100]
                
                # G1: Prediction Quality Metrics Tracking
                actual_ret = graph.return_.cpu().numpy()[:100]
                actual_vol = graph.volatility.cpu().numpy()[:100]
                vol_mse = safe_metric(mean_squared_error, actual_vol, pred_vol)
                vol_mae = safe_metric(mean_absolute_error, actual_vol, pred_vol)
                ret_mae = safe_metric(mean_absolute_error, actual_ret, pred_ret.values)
                # handle edge case where actual variance is zero
                ret_r2 = safe_metric(r2_score, actual_ret, pred_ret.values) if np.var(actual_ret) > 1e-8 else 0.0

            cm = ConcentrationMetrics(current_weights, embeddings, tickers)
            metrics_dict = cm.compute_all()

            df_clusters = cd.fit(embeddings, tickers)
            metrics_dict["clusters"] = cd.detect_concentration_risk(
                current_weights, df_clusters
            )

            # G2: Risk Metrics Trajectory Tracking
            # Reconstruct Sigma_gnn for ENB
            z_norm_tmp = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            cov_gnn_tmp = z_norm_tmp @ z_norm_tmp.T
            w_arr = current_weights.values
            enb = 1.0 / (w_arr.T @ cov_gnn_tmp @ w_arr + 1e-8)
            
            risk_trajectory.append({
                "date": date,
                "realized_cvar": float(metrics_dict.get("cvar_95", 0.0)),
                "embedding_hhi": float(metrics_dict.get("hhi", 0.0)),
                "enb": float(enb)
            })

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
            
            # H1: Verify Temporal Split Logic
            if i + 1 < len(dates):
                if pd.to_datetime(date) >= pd.to_datetime(next_date):
                    raise ValueError(f"Temporal Leakage: {date} >= {next_date}")
            
            realized_ret = self._get_realized_returns(date, next_date, tickers)
            port_ret = float((current_weights * realized_ret).sum())
            if np.isnan(port_ret):
                print(f"WARNING: NaN portfolio return on {date}. Using 0.0 as fallback.")
                port_ret = 0.0
            
            portfolio_value *= 1.0 + port_ret

            history.append(
                {
                    "date": date,
                    "portfolio_value": portfolio_value,
                    "portfolio_return": port_ret,
                    "costs": costs,
                    "turnover": turnover,
                    "traded": traded,
                    "vol_mse": vol_mse,
                    "vol_mae": vol_mae,
                    "ret_mae": ret_mae,
                    "ret_r2": ret_r2
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
        import logging
        try:
            fundamentals = pd.read_parquet("data/raw/fundamentals/fundamentals.parquet")
        except Exception:
            fundamentals = None
            
        idx = pd.to_datetime(dates)
        ret = []
        for i in range(len(dates) - 1):
            curr_date = dates[i]
            next_date = dates[i + 1]
            
            if self.ohlcv is not None and "close" in self.ohlcv.columns:
                curr_prices = self.ohlcv[self.ohlcv["date"] == curr_date].set_index("ticker")["close"]
                next_prices = self.ohlcv[self.ohlcv["date"] == next_date].set_index("ticker")["close"]
                
                valid_tickers = curr_prices.index.intersection(next_prices.index)[:100]
                if not valid_tickers.empty:
                    step_rets = np.log(next_prices[valid_tickers] / curr_prices[valid_tickers])
                    
                    if benchmark == "equal_weight":
                        ret.append(step_rets.mean())
                    elif benchmark == "spy" or benchmark == "market_cap":
                        if fundamentals is not None and "market_cap" in fundamentals.columns:
                            mc = fundamentals[fundamentals["ticker"].isin(valid_tickers)].groupby("ticker")["market_cap"].last()
                            if not mc.empty:
                                w = mc.reindex(valid_tickers).fillna(0.0)
                                if w.sum() > 0:
                                    w = w / w.sum()
                                    ret.append((step_rets * w).sum())
                                    continue
                        logging.warning(f"Market cap data unavailable for {curr_date}. Falling back to equal-weight.")
                        ret.append(step_rets.mean())
                    elif benchmark == "har_min_variance":
                        # Simplified HAR(3) proxy — full OLS estimation deferred to Phase 4.
                        past_22d = self.ohlcv[(self.ohlcv["date"] <= curr_date) & (self.ohlcv["ticker"].isin(valid_tickers))]
                        past_22d = past_22d.pivot(index="date", columns="ticker", values="close").tail(23)
                        rets_22d = past_22d.pct_change().dropna()
                        if len(rets_22d) >= 5:
                            rv_1d = rets_22d.tail(1).iloc[0] ** 2
                            rv_5d = rets_22d.tail(5).var()
                            rv_22d = rets_22d.var()
                            forecast_var = 0.5 * rv_1d + 0.3 * rv_5d + 0.2 * rv_22d
                            w = 1.0 / (forecast_var + 1e-8)
                            w = w / w.sum()
                            ret.append((step_rets * w).sum())
                        else:
                            ret.append(step_rets.mean())
                    continue

            ret.append(0.0)
                
        ret.append(0.0) # Last day has 0 forward return
        return pd.Series(ret, index=idx)
