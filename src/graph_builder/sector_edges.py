import itertools
import logging
from pathlib import Path

import pandas as pd


class SectorEdgeBuilder:
    def __init__(self, gics_df: pd.DataFrame):
        self.gics_df = gics_df.copy()
        self.logger = logging.getLogger(__name__)

    def build_edges(self, granularity: str = "sector") -> pd.DataFrame:
        if self.gics_df.empty:
            return pd.DataFrame()

        edges = []
        group_col = "sector" if granularity == "sector" else "industry"
        edge_type = "same_sector_as" if granularity == "sector" else "same_industry_as"

        if group_col not in self.gics_df.columns:
            self.logger.warning(f"Column {group_col} not found in GICS mapping.")
            return pd.DataFrame()

        for _, df_group in self.gics_df.groupby(group_col):
            tickers = sorted(df_group["ticker"].tolist())
            for src, tgt in itertools.permutations(tickers, 2):
                edges.append(
                    {
                        "source": src,
                        "target": tgt,
                        "weight": 1.0,
                        "edge_type": edge_type,
                        "date": "static",
                    }
                )

        final_df = pd.DataFrame(edges)

        out_dir = Path("data/processed/edges")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "sector_edges.parquet"

        if out_file.exists():
            existing_df = pd.read_parquet(out_file)
            final_df = pd.concat([existing_df, final_df], ignore_index=True)

        final_df.to_parquet(out_file, index=False)
        self.logger.info(
            f"Built {len(final_df)} sector edges for granularity {granularity}."
        )
        return final_df
