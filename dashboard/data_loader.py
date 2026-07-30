import glob
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch_geometric.data import HeteroData


class DashboardDataLoader:
    def __init__(
        self,
        data_dir: Path = Path("data/processed"),
        reports_dir: Path = Path("reports"),
        alert_dir: Path = Path("alerts"),
    ):
        self.data_dir = data_dir
        self.reports_dir = reports_dir
        self.alert_dir = alert_dir

    @st.cache_data(ttl=60)
    def load_portfolio_weights(_self) -> pd.DataFrame:
        """Mock loader for portfolio weights"""
        try:
            snapshot_dir = _self.data_dir / "graph_snapshots"
            files = sorted(snapshot_dir.glob("*.pt"))
            if not files:
                return pd.DataFrame()
            g = torch.load(files[-1], weights_only=False)
            n = g["stock"].x.shape[0]
            tickers = [f"STOCK_{i}" for i in range(n)]

            df = pd.DataFrame(
                {
                    "ticker": tickers,
                    "weight": [1.0 / n] * n,
                    "sector": [f"Sector_{i%11}" for i in range(n)],
                    "predicted_return": np.random.randn(n) * 0.01,
                    "predicted_volatility": np.random.rand(n) * 0.3,
                }
            )
            return df
        except Exception as e:
            print(f"Error loading portfolio: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=60)
    def load_latest_graph(_self) -> HeteroData:
        try:
            snapshot_dir = _self.data_dir / "graph_snapshots"
            files = sorted(snapshot_dir.glob("*.pt"))
            if not files:
                return None
            return torch.load(files[-1], weights_only=False)
        except Exception:
            return None

    @st.cache_data(ttl=60)
    def load_backtest_summary(_self) -> Dict:
        try:
            files = sorted(_self.reports_dir.glob("backtest_*.json"))
            if not files:
                return {}
            with open(files[-1], "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    @st.cache_data(ttl=60)
    def load_alerts(_self, n_latest: int = 50) -> pd.DataFrame:
        try:
            files = sorted(_self.alert_dir.glob("streaming_alerts_*.jsonl"))
            if not files:
                return pd.DataFrame()

            alerts = []
            with open(files[-1], "r", encoding="utf-8") as f:
                for line in f:
                    alerts.append(json.loads(line.strip()))

            df = pd.DataFrame(alerts[-n_latest:])
            # parse data dict into columns if needed
            return df.iloc[::-1]  # reverse to show newest first
        except Exception:
            return pd.DataFrame()

    @st.cache_data(ttl=60)
    def load_shock_results(_self, scenario: str) -> Dict:
        # Mock shock results for demo purposes
        return {
            "scenario": scenario,
            "portfolio_return": -0.05,
            "portfolio_vol": 0.25,
            "portfolio_cvar": -0.08,
            "worst_ticker": "STOCK_0",
        }

    def refresh(self):
        st.cache_data.clear()
