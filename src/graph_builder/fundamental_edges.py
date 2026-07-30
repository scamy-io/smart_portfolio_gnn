import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


class FundamentalEdgeBuilder:
    def __init__(
        self,
        fundamentals_df: pd.DataFrame,
        similarity_threshold: float = 0.7,
        update_frequency: str = "quarterly",
    ):
        self.fundamentals_df = fundamentals_df.copy()
        self.similarity_threshold = similarity_threshold
        self.update_frequency = update_frequency
        self.logger = logging.getLogger(__name__)
        self.cached_edges = None

    def build_edges_for_date(self, date: str) -> pd.DataFrame:
        df = self.fundamentals_df.copy()
        if df.empty or "ticker" not in df.columns:
            return pd.DataFrame()

        df.set_index("ticker", inplace=True)
        df = df.fillna(0)

        vals = df.values
        norm = np.linalg.norm(vals, axis=1, keepdims=True)
        norm[norm == 0] = 1
        vals_norm = vals / norm
        sim_matrix = np.dot(vals_norm, vals_norm.T)

        edges = []
        tickers = df.index.tolist()

        for i, src in enumerate(tickers):
            for j, tgt in enumerate(tickers):
                if i != j:
                    sim = sim_matrix[i, j]
                    if sim >= self.similarity_threshold:
                        edges.append(
                            {
                                "date": date,
                                "source": src,
                                "target": tgt,
                                "weight": sim,
                                "edge_type": "fundamentally_similar_to",
                            }
                        )

        return pd.DataFrame(edges)

    def build_all_edges(self, dates: List[str]) -> pd.DataFrame:
        all_edges = []
        for d in dates:
            month = pd.to_datetime(d).month
            is_quarter_end = month in [3, 6, 9, 12]

            if is_quarter_end or self.cached_edges is None:
                edges = self.build_edges_for_date(d)
                self.cached_edges = (
                    edges.drop(columns=["date"]) if not edges.empty else pd.DataFrame()
                )

            if not self.cached_edges.empty:
                daily_edges = self.cached_edges.copy()
                daily_edges["date"] = d
                all_edges.append(daily_edges)

        if all_edges:
            final_df = pd.concat(all_edges, ignore_index=True)
            out_dir = Path("data/processed/edges")
            out_dir.mkdir(parents=True, exist_ok=True)
            final_df.to_parquet(out_dir / "fundamental_edges.parquet", index=False)
            self.logger.info(f"Built fundamental edges for {len(dates)} dates.")
            return final_df

        return pd.DataFrame()
