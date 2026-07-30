import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from src.graph_builder.base_graph import StockGraph


class TemporalGraphDataset(Dataset):
    def __init__(
        self,
        graph_snapshot_dir: Path,
        node_features_path: Path,
        edge_paths: Dict[str, Path],
        window_size: int = 25,
        prediction_horizon: int = 5,
    ):
        self.graph_snapshot_dir = graph_snapshot_dir
        self.node_features_path = node_features_path
        self.edge_paths = edge_paths
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.logger = logging.getLogger(__name__)

        self.graph_snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots = sorted(list(self.graph_snapshot_dir.glob("*.pt")))

    def build_snapshots(self, dates: List[str]):
        self.logger.info(f"Building snapshots for {len(dates)} dates...")

        nf_df = pd.read_parquet(self.node_features_path)
        nf_df = nf_df.reset_index()
        nf_df["date"] = (
            pd.to_datetime(nf_df["date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
        )

        edges = {}
        for et, path in self.edge_paths.items():
            if path.exists():
                edges[et] = pd.read_parquet(path)
                if "date" in edges[et].columns:
                    edges[et]["date"] = edges[et]["date"].astype(str)
            else:
                edges[et] = pd.DataFrame()

        for d in dates:
            nf_daily = nf_df[nf_df["date"] == d]
            if nf_daily.empty:
                continue

            tickers = nf_daily["ticker"].unique().tolist()
            nf_matrix = nf_daily.set_index("ticker").drop(columns=["date"])

            daily_edges = {}
            for et, edf in edges.items():
                if edf.empty:
                    daily_edges[et] = pd.DataFrame()
                elif "date" in edf.columns:
                    daily_edges[et] = edf[
                        (edf["date"] == d) | (edf["date"] == "static")
                    ]
                else:
                    daily_edges[et] = edf

            sg = StockGraph(d, tickers, nf_matrix, daily_edges)
            if sg.validate():
                pyg_data = sg.to_pyg()
                out_path = self.graph_snapshot_dir / f"{d}.pt"
                torch.save(pyg_data, out_path)
            else:
                self.logger.warning(f"Graph validation failed for {d}. Skipping.")

        self.snapshots = sorted(list(self.graph_snapshot_dir.glob("*.pt")))

    def __len__(self) -> int:
        return len(self.snapshots)

    @property
    def snapshot_dates(self) -> List[str]:
        return [p.stem for p in self.snapshots]

    def get_snapshot_by_date(self, date: str) -> HeteroData:
        try:
            idx = self.snapshot_dates.index(date)
            return self[idx]
        except ValueError:
            raise KeyError(f"No snapshot found for date {date}")

    def __getitem__(self, idx: int) -> HeteroData:
        pyg_data = torch.load(self.snapshots[idx], weights_only=False)

        num_nodes = pyg_data["stock"].x.shape[0]
        v = torch.rand(num_nodes, dtype=torch.float32)
        r = torch.randn(num_nodes, dtype=torch.float32) * 0.01
        c = r - 1.65 * v

        pyg_data.volatility = v
        pyg_data.return_ = r
        pyg_data.cvar = c
        return pyg_data

    def get_loaders(self, batch_size=32, train_ratio=0.7, val_ratio=0.15):
        total = len(self)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        indices = list(range(total))
        train_ds = torch.utils.data.Subset(self, indices[:train_end])
        val_ds = torch.utils.data.Subset(self, indices[train_end:val_end])
        test_ds = torch.utils.data.Subset(self, indices[val_end:])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
