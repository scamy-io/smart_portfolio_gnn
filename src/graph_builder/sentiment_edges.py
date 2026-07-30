import itertools
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


class SentimentEdgeBuilder:
    def __init__(self, sentiment_df: pd.DataFrame, co_mention_threshold: float = 0.5):
        self.sentiment_df = sentiment_df.copy()
        if "date" in self.sentiment_df.columns:
            self.sentiment_df["date"] = pd.to_datetime(self.sentiment_df["date"])
        self.co_mention_threshold = co_mention_threshold
        self.logger = logging.getLogger(__name__)

    def build_co_mention_edges(
        self, date: str, gkg_df: pd.DataFrame, ticker_to_name: Dict[str, str]
    ) -> pd.DataFrame:
        if gkg_df.empty:
            return pd.DataFrame()

        edges = []
        decay_factor = np.exp(-np.log(2) / 3)

        for _, row in gkg_df.iterrows():
            orgs = str(row.get("Organizations", "")).lower()
            names = str(row.get("AllNames", "")).lower()
            text = orgs + " " + names

            mentioned_tickers = []
            for ticker, name in ticker_to_name.items():
                if name.lower() in text:
                    mentioned_tickers.append(ticker)

            if len(mentioned_tickers) >= 2:
                tones = str(row.get("V2Tone", "")).split(",")
                try:
                    tone = float(tones[0])
                except (ValueError, IndexError):
                    tone = 0.0

                if abs(tone) < self.co_mention_threshold:
                    continue

                weight = abs(tone) if tone < 0 else tone * 0.5
                weight *= decay_factor

                mentioned_tickers = sorted(mentioned_tickers)
                for src, tgt in itertools.combinations(mentioned_tickers, 2):
                    edges.extend(
                        [
                            {
                                "date": date,
                                "source": src,
                                "target": tgt,
                                "weight": weight,
                                "edge_type": "sentiment_co_mention",
                            },
                            {
                                "date": date,
                                "source": tgt,
                                "target": src,
                                "weight": weight,
                                "edge_type": "sentiment_co_mention",
                            },
                        ]
                    )

        return pd.DataFrame(edges)

    def build_spillover_edges(
        self, date: str, sector_map: Dict[str, str]
    ) -> pd.DataFrame:
        date_dt = pd.to_datetime(date)
        daily_sent = self.sentiment_df[self.sentiment_df["date"] == date_dt]
        if daily_sent.empty:
            return pd.DataFrame()

        edges = []
        for _, row in daily_sent.iterrows():
            ticker = row["ticker"]
            tone = row.get("avg_tone", 0.0)
            if abs(tone) > 2.0:
                sector = sector_map.get(ticker)
                if not sector:
                    continue
                same_sector_tickers = [
                    t for t, s in sector_map.items() if s == sector and t != ticker
                ]
                weight = abs(tone) / 10.0
                for tgt in sorted(same_sector_tickers):
                    edges.append(
                        {
                            "date": date,
                            "source": ticker,
                            "target": tgt,
                            "weight": weight,
                            "edge_type": "sentiment_spillover",
                        }
                    )

        return pd.DataFrame(edges)

    def build_all_edges(
        self,
        dates: List[str],
        gkg_dfs: Dict[str, pd.DataFrame],
        ticker_to_name: Dict[str, str],
        sector_map: Dict[str, str],
    ) -> pd.DataFrame:
        all_edges = []
        for d in dates:
            co_edges = self.build_co_mention_edges(
                d, gkg_dfs.get(d, pd.DataFrame()), ticker_to_name
            )
            spill_edges = self.build_spillover_edges(d, sector_map)

            daily_edges = pd.concat([co_edges, spill_edges], ignore_index=True)
            if not daily_edges.empty:
                daily_edges = (
                    daily_edges.groupby(["date", "source", "target", "edge_type"])[
                        "weight"
                    ]
                    .mean()
                    .reset_index()
                )
                all_edges.append(daily_edges)

        if all_edges:
            final_df = pd.concat(all_edges, ignore_index=True)
            self.logger.info(f"Built sentiment edges for {len(dates)} dates.")
            out_dir = Path("data/processed/edges")
            out_dir.mkdir(parents=True, exist_ok=True)
            final_df.to_parquet(out_dir / "sentiment_edges.parquet", index=False)
            return final_df

        self.logger.warning("No sentiment edges built.")
        return pd.DataFrame()
