import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


class CorrelationEdgeBuilder:
    def __init__(
        self,
        prices_df: pd.DataFrame,
        window: int = 21,
        threshold: float = 0.3,
        top_k: int = 15,
    ):
        self.prices_df = prices_df.copy()
        if "date" in self.prices_df.columns:
            self.prices_df["date"] = pd.to_datetime(
                self.prices_df["date"]
            ).dt.tz_localize(None)
        self.window = window
        self.threshold = threshold
        self.top_k = top_k
        self.logger = logging.getLogger(__name__)

    def build_edges_for_date(self, date: str) -> pd.DataFrame:
        end_date = pd.to_datetime(date)
        df_up_to_date = self.prices_df[self.prices_df["date"] <= end_date]
        if df_up_to_date.empty:
            return pd.DataFrame()

        df_up_to_date = df_up_to_date.sort_values(by=["ticker", "date"])
        df_up_to_date["log_return"] = df_up_to_date.groupby("ticker")[
            "adj_close"
        ].transform(lambda x: np.log(x / x.shift(1)))

        returns_pivot = df_up_to_date.pivot(
            index="date", columns="ticker", values="log_return"
        )
        returns_window = returns_pivot.loc[:end_date].tail(self.window)

        if len(returns_window) < 2:
            return pd.DataFrame()

        returns_window = returns_window.dropna(axis=1)
        if returns_window.empty:
            return pd.DataFrame()

        corr_matrix = returns_window.corr(method="pearson")

        edges = []
        tickers = sorted(corr_matrix.columns)
        for src in tickers:
            corrs = corr_matrix.loc[src].drop(src)
            valid_corrs = corrs[corrs.abs() >= self.threshold]
            if valid_corrs.empty:
                continue

            valid_corrs = valid_corrs.reset_index()
            valid_corrs.columns = ["target", "weight"]
            valid_corrs["abs_weight"] = valid_corrs["weight"].abs()
            valid_corrs = valid_corrs.sort_values(
                by=["abs_weight", "target"], ascending=[False, True]
            ).head(self.top_k)

            for _, row in valid_corrs.iterrows():
                edges.append(
                    {
                        "date": date,
                        "source": src,
                        "target": row["target"],
                        "weight": row["weight"],
                        "edge_type": "correlates_with",
                        "window": self.window,
                    }
                )

        return pd.DataFrame(edges)

    def build_all_edges(self, dates: List[str]) -> pd.DataFrame:
        all_edges = []
        for d in dates:
            edges = self.build_edges_for_date(d)
            if not edges.empty:
                all_edges.append(edges)

        if all_edges:
            final_df = pd.concat(all_edges, ignore_index=True)
            avg_edges = len(final_df) / len(dates)
            self.logger.info(
                f"Built correlation edges for {len(dates)} dates. Avg edges per date: {avg_edges:.1f}."
            )

            out_dir = Path("data/processed/edges")
            out_dir.mkdir(parents=True, exist_ok=True)
            final_df.to_parquet(out_dir / "correlation_edges.parquet", index=False)
            return final_df

        self.logger.warning("No correlation edges built.")
        return pd.DataFrame()
