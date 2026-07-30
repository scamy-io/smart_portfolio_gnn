import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


class StockGraph:
    def __init__(
        self,
        date: str,
        tickers: List[str],
        node_features: pd.DataFrame,
        edge_dataframes: Dict[str, pd.DataFrame],
    ):
        self.date = date
        self.tickers = sorted(tickers)
        self.node_features = node_features.fillna(0)
        self.edge_dataframes = edge_dataframes
        self.ticker_to_idx = {ticker: i for i, ticker in enumerate(self.tickers)}
        self.logger = logging.getLogger(__name__)

    def to_pyg(self) -> HeteroData:
        data = HeteroData()

        self.node_features = self.node_features.reindex(self.tickers).fillna(0)
        data["stock"].x = torch.tensor(self.node_features.values, dtype=torch.float32)

        for edge_type, edge_df in self.edge_dataframes.items():
            if edge_df.empty:
                continue

            valid_edges = edge_df[
                edge_df["source"].isin(self.ticker_to_idx)
                & edge_df["target"].isin(self.ticker_to_idx)
            ]
            if valid_edges.empty:
                continue

            src_indices = valid_edges["source"].map(self.ticker_to_idx).values
            tgt_indices = valid_edges["target"].map(self.ticker_to_idx).values
            weights = valid_edges["weight"].values

            edge_index = torch.tensor([src_indices, tgt_indices], dtype=torch.long)
            edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

            data["stock", edge_type, "stock"].edge_index = edge_index
            data["stock", edge_type, "stock"].edge_attr = edge_attr

        data.date = self.date
        return data

    def get_metadata(self) -> Tuple[List[str], List[Tuple]]:
        edge_types = list(self.edge_dataframes.keys())
        return (["stock"], [("stock", et, "stock") for et in edge_types])

    def validate(self) -> bool:
        is_valid = True

        if self.node_features.isna().any().any():
            self.logger.error(f"[{self.date}] NaN found in node features.")
            is_valid = False

        for edge_type, edge_df in self.edge_dataframes.items():
            if edge_df.empty:
                continue

            self_loops = edge_df[edge_df["source"] == edge_df["target"]]
            if not self_loops.empty:
                self.logger.error(f"[{self.date}] Self-loops found in {edge_type}.")
                is_valid = False

            if not pd.api.types.is_numeric_dtype(edge_df["weight"]):
                continue
            if not pd.Series(edge_df["weight"]).apply(np.isfinite).all():
                self.logger.error(
                    f"[{self.date}] Non-finite weights found in {edge_type}."
                )
                is_valid = False

        return is_valid
