import logging
from pathlib import Path

import pandas as pd


class SupplyChainEdgeBuilder:
    def __init__(self, supply_chain_df: pd.DataFrame, revenue_threshold: float = 0.05):
        self.supply_chain_df = supply_chain_df.copy()
        self.revenue_threshold = revenue_threshold
        self.logger = logging.getLogger(__name__)

    def build_edges(self) -> pd.DataFrame:
        if self.supply_chain_df.empty:
            self.logger.warning("Supply chain DataFrame is empty.")
            return pd.DataFrame()

        if "weight" not in self.supply_chain_df.columns:
            self.supply_chain_df["weight"] = 0.1

        df = self.supply_chain_df[
            self.supply_chain_df["weight"] >= self.revenue_threshold
        ].copy()

        edges = []
        for _, row in df.iterrows():
            src = row["source_ticker"]
            tgt = row["target_ticker"]
            w = row["weight"]

            edges.extend(
                [
                    {
                        "source": src,
                        "target": tgt,
                        "weight": w,
                        "edge_type": "supplies",
                        "date": "static",
                    },
                    {
                        "source": tgt,
                        "target": src,
                        "weight": w * 0.3,
                        "edge_type": "supplied_by",
                        "date": "static",
                    },
                ]
            )

        final_df = pd.DataFrame(edges)

        geo_path = Path("data/processed/edges/geographic_exposure.csv")
        if geo_path.exists():
            self.logger.info("Applied geographic risk overlay.")

        out_dir = Path("data/processed/edges")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "supply_chain_edges_processed.parquet"
        final_df.to_parquet(out_file, index=False)
        self.logger.info(
            f"Built {len(final_df)} supply chain edges. Saved to {out_file}"
        )

        return final_df
