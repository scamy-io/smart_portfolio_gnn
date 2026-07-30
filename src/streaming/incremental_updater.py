import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


class IncrementalGraphUpdater:
    def __init__(
        self, base_graph: HeteroData, window_size: int = 21, device: str = "cpu"
    ):
        self.graph = base_graph.clone()
        self.window_size = window_size
        self.device = device

        self.num_nodes = self.graph["stock"].x.shape[0]

        # Circular buffer for returns
        self.returns_buffer = np.zeros(
            (self.window_size, self.num_nodes), dtype=np.float32
        )
        self.ptr = 0
        self.is_buffer_full = False

        # Tracking prev adj close for log return calculation
        self.prev_adj_close = np.ones(self.num_nodes, dtype=np.float32)

        # Sentiment edges accumulator
        self.sentiment_edges = pd.DataFrame(
            columns=["source", "target", "weight", "timestamp"]
        )

    def push_prices(self, ohlcv_df: pd.DataFrame) -> None:
        """
        ohlcv_df: DataFrame with [ticker, open, high, low, close, adj_close, volume]
        Must be sorted or aligned such that rows correspond to node indices 0..N-1
        """
        # Assuming ohlcv_df is aligned with node indices
        adj_close = ohlcv_df["adj_close"].values.astype(np.float32)

        # Compute log returns
        # Handle zeros to avoid log(0)
        safe_prev = np.where(self.prev_adj_close == 0, 1e-8, self.prev_adj_close)
        safe_curr = np.where(adj_close == 0, 1e-8, adj_close)

        log_returns = np.log(safe_curr / safe_prev)

        # Update buffer
        self.returns_buffer[self.ptr] = log_returns
        self.ptr += 1
        if self.ptr >= self.window_size:
            self.ptr = 0
            self.is_buffer_full = True

        self.prev_adj_close = adj_close

    def _get_correlation_matrix(self) -> np.ndarray:
        if not self.is_buffer_full and self.ptr == 0:
            return np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)

        valid_len = self.window_size if self.is_buffer_full else self.ptr
        # slice the valid window
        rets = self.returns_buffer[:valid_len]

        # Standardize returns
        rets_mean = rets.mean(axis=0, keepdims=True)
        rets_std = rets.std(axis=0, keepdims=True)
        rets_std[rets_std == 0] = 1e-8
        z = (rets - rets_mean) / rets_std

        # Compute correlation matrix
        corr = (z.T @ z) / max(valid_len - 1, 1)
        return corr

    def update_correlation_edges(self) -> HeteroData:
        corr = self._get_correlation_matrix()

        # Zero out diagonal
        np.fill_diagonal(corr, 0)

        # Apply threshold and top-K
        threshold = 0.3
        top_k = 15

        # We need to construct edge_index and edge_attr
        sources, targets, weights = [], [], []

        for i in range(self.num_nodes):
            row = corr[i]
            # Find indices where abs(corr) >= threshold
            valid_idx = np.where(np.abs(row) >= threshold)[0]
            if len(valid_idx) == 0:
                continue

            # Get top K
            valid_vals = row[valid_idx]
            # argsort sorts ascending, so we take the last K for largest absolute values
            top_k_indices = valid_idx[np.argsort(np.abs(valid_vals))[-top_k:]]

            for j in top_k_indices:
                sources.append(i)
                targets.append(j)
                weights.append(row[j])

        if len(sources) > 0:
            edge_index = torch.tensor([sources, targets], dtype=torch.long)
            edge_attr = torch.tensor(weights, dtype=torch.float32).view(-1, 1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float32)

        self.graph["stock", "correlates_with", "stock"].edge_index = edge_index
        self.graph["stock", "correlates_with", "stock"].edge_attr = edge_attr
        return self.graph

    def update_sentiment_edges(
        self, news_batch: List[Dict], ticker_to_idx: Dict[str, int]
    ) -> HeteroData:
        current_time = time.time()

        # Decay existing edges
        if not self.sentiment_edges.empty:
            decay_factor = np.exp(-np.log(2) / 3)
            self.sentiment_edges["weight"] *= decay_factor

        # Process new batch
        new_edges = []
        for news in news_batch:
            tickers = news.get("tickers_mentioned", [])
            tone = news.get("tone", 0.0)

            # Map tickers to indices
            indices = [ticker_to_idx[t] for t in tickers if t in ticker_to_idx]
            if len(indices) < 2:
                continue

            weight = abs(tone) if tone < 0 else tone * 0.5

            # Create co-mention edges
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    new_edges.append(
                        {
                            "source": indices[i],
                            "target": indices[j],
                            "weight": weight,
                            "timestamp": current_time,
                        }
                    )
                    new_edges.append(
                        {
                            "source": indices[j],
                            "target": indices[i],
                            "weight": weight,
                            "timestamp": current_time,
                        }
                    )

        if new_edges:
            new_df = pd.DataFrame(new_edges)
            self.sentiment_edges = pd.concat(
                [self.sentiment_edges, new_df], ignore_index=True
            )

            # Aggregate by source-target
            self.sentiment_edges = self.sentiment_edges.groupby(
                ["source", "target"], as_index=False
            ).agg({"weight": "sum", "timestamp": "max"})

            # Prune
            self.sentiment_edges = self.sentiment_edges[
                self.sentiment_edges["weight"].abs() >= 0.01
            ]

        if not self.sentiment_edges.empty:
            sources = self.sentiment_edges["source"].values.astype(np.int64)
            targets = self.sentiment_edges["target"].values.astype(np.int64)
            weights = self.sentiment_edges["weight"].values.astype(np.float32)

            self.graph["stock", "sentiment_co_mention", "stock"].edge_index = (
                torch.tensor([sources, targets], dtype=torch.long)
            )
            self.graph["stock", "sentiment_co_mention", "stock"].edge_attr = (
                torch.tensor(weights, dtype=torch.float32).view(-1, 1)
            )
        else:
            self.graph["stock", "sentiment_co_mention", "stock"].edge_index = (
                torch.empty((2, 0), dtype=torch.long)
            )
            self.graph["stock", "sentiment_co_mention", "stock"].edge_attr = (
                torch.empty((0, 1), dtype=torch.float32)
            )

        return self.graph

    def update_node_features(
        self, macro_dict: Dict, feature_stats: Dict = None
    ) -> HeteroData:
        # In a real system, you would incrementally calculate SMA and update self.graph["stock"].x
        # We assume x is updated externally or simplified here.
        # For this implementation, we just mock updating the VIX column (assume it's the last feature)
        vix = macro_dict.get("VIX", 20.0)

        # Assuming VIX is index -1
        # Normalize if feature_stats are provided
        if feature_stats and "VIX_mean" in feature_stats:
            vix = (vix - feature_stats["VIX_mean"]) / feature_stats["VIX_std"]

        self.graph["stock"].x[:, -1] = vix
        return self.graph

    def get_graph(self) -> HeteroData:
        return self.graph.to(self.device)

    def save_state(self, path: Path) -> None:
        import pickle

        state = {
            "returns_buffer": self.returns_buffer,
            "ptr": self.ptr,
            "is_buffer_full": self.is_buffer_full,
            "prev_adj_close": self.prev_adj_close,
            "sentiment_edges": self.sentiment_edges,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load_state(self, path: Path) -> None:
        import pickle

        if not path.exists():
            return
        with open(path, "rb") as f:
            state = pickle.load(f)
            self.returns_buffer = state["returns_buffer"]
            self.ptr = state["ptr"]
            self.is_buffer_full = state["is_buffer_full"]
            self.prev_adj_close = state["prev_adj_close"]
            self.sentiment_edges = state["sentiment_edges"]
