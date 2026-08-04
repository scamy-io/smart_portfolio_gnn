"""GDELT Processor for sentiment data."""

import logging
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm


class GDELTProcessor:
    """Processor for GDELT Global Knowledge Graph data."""

    def __init__(
        self, tickers: List[str], company_name_map: Dict[str, str], output_dir: Path,
        config: Dict = None,
    ):
        """
        Initialize the GDELT processor.

        Args:
            tickers (List[str]): List of tickers.
            company_name_map (Dict[str, str]): Mapping from ticker to company name.
            output_dir (Path): Directory to save processed sentiment data.
            config (Dict, optional): YAML config dict for GDELT-specific settings.
        """
        self.tickers = tickers
        self.company_name_map = company_name_map
        self.output_dir = output_dir
        self.config = config or {}
        self.cache_dir = output_dir.parent.parent / "raw" / "gdelt" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def fetch_gkg_for_date(self, date: datetime.date) -> pd.DataFrame:
        """
        Download GDELT GKG v2.1 for a single date.

        Args:
            date (datetime.date): Date to download data for.

        Returns:
            pd.DataFrame: Downloaded GKG data or empty DataFrame on failure.
        """
        date_str = date.strftime("%Y%m%d")
        url = f"http://data.gdeltproject.org/gkg/{date_str}.gkg.csv.zip"
        zip_path = self.cache_dir / f"{date_str}.gkg.csv.zip"
        csv_name = f"{date_str}.gkg.csv"

        expected_cols = [
            "GKGRECORDID",
            "DATE",
            "SourceCommonName",
            "DocumentIdentifier",
            "Organizations",
            "V2Organizations",
            "V2Tone",
            "AllNames",
            "Themes",
        ]

        try:
            if not zip_path.exists():
                urllib.request.urlretrieve(url, zip_path)

            with zipfile.ZipFile(zip_path, "r") as z:
                with z.open(csv_name) as f:
                    df = pd.read_csv(f, sep="\t", engine="python", on_bad_lines="skip")

            for col in expected_cols:
                if col not in df.columns:
                    df[col] = pd.NA

            return df[expected_cols]
        except Exception as e:
            self.logger.debug(f"Failed to fetch GDELT for {date_str}: {e}")
            return pd.DataFrame(columns=expected_cols)

    def extract_stock_sentiment(
        self, gkg_df: pd.DataFrame, date: datetime.date
    ) -> pd.DataFrame:
        """
        Extract sentiment for tickers from GKG data.

        Args:
            gkg_df (pd.DataFrame): GKG dataframe for a specific date.
            date (datetime.date): The date of the data.

        Returns:
            pd.DataFrame: Sentiment dataframe.
        """
        if gkg_df.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "ticker",
                    "avg_tone",
                    "ToneDispersion",
                    "NumMentions",
                    "avg_positive",
                    "avg_negative",
                    "polarity",
                ]
            )

        results = []

        for ticker in self.tickers:
            company_name = self.company_name_map.get(ticker, "").lower()
            if not company_name:
                continue

            # Find mentions
            mask = gkg_df["Organizations"].str.lower().str.contains(
                company_name, na=False
            ) | gkg_df["AllNames"].str.lower().str.contains(company_name, na=False)

            mentions = gkg_df[mask]
            if mentions.empty:
                continue

            # Parse V2Tone
            tones = mentions["V2Tone"].dropna().str.split(",", expand=True)
            if tones.empty or tones.shape[1] < 4:
                continue

            for i in range(4):
                tones[i] = pd.to_numeric(tones[i], errors="coerce")

            avg_tone = tones[0].mean()
            tone_std = tones[0].std()
            avg_pos = tones[1].mean()
            avg_neg = tones[2].mean()
            polarity = tones[3].mean()

            results.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "avg_tone": avg_tone,
                    "ToneDispersion": tone_std if not pd.isna(tone_std) else 0.0,
                    "NumMentions": len(mentions),
                    "avg_positive": avg_pos,
                    "avg_negative": avg_neg,
                    "polarity": polarity,
                }
            )

        return pd.DataFrame(results)

    def build_sentiment_timeseries(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Build sentiment timeseries over a date range.

        Args:
            start_date (str): Start date YYYY-MM-DD.
            end_date (str): End date YYYY-MM-DD.

        Returns:
            pd.DataFrame: Full sentiment panel.
        """
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        date_list = [start + timedelta(days=x) for x in range((end - start).days + 1)]

        all_sentiments = []

        def process_date(date):
            df = self.fetch_gkg_for_date(date)
            return self.extract_stock_sentiment(df, date)

        self.logger.info(
            f"Processing GDELT from {start_date} to {end_date} ({len(date_list)} days)"
        )

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                tqdm(executor.map(process_date, date_list), total=len(date_list))
            )

        for i, res in enumerate(results):
            if not res.empty:
                all_sentiments.append(res)
            if (i + 1) % 50 == 0:
                self.logger.info(f"Processed {i + 1} dates...")

        if not all_sentiments:
            return pd.DataFrame()

        final_df = pd.concat(all_sentiments, ignore_index=True)
        final_df["date"] = pd.to_datetime(final_df["date"])

        halflife = float(self.config.get("gdelt", {}).get("decay_halflife_days", 3.0))
        dfs = []
        for ticker, group in final_df.groupby("ticker"):
            group = (
                group.set_index("date").reindex(pd.date_range(start, end)).reset_index()
            )
            group["ticker"] = ticker
            group = group.rename(columns={"index": "date"})

            is_na = group["avg_tone"].isna()
            group = group.ffill()

            decay_factor = np.exp(-np.log(2) / halflife)
            for col in ["avg_tone", "avg_positive", "avg_negative", "polarity"]:
                mask = group[col].notna() & is_na
                blocks = is_na.groupby((~is_na).cumsum()).cumcount()
                group.loc[mask, col] = group.loc[mask, col] * (
                    decay_factor ** blocks[mask]
                )

            group["NumMentions"] = group["NumMentions"].fillna(0)
            group["ToneDispersion"] = group["ToneDispersion"].fillna(0)
            dfs.append(group)

        result_df = pd.concat(dfs, ignore_index=True)

        # Save to the path that FeatureEngineer reads from
        out_file = self.output_dir.parent / "sentiment_node_features.parquet"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_parquet(out_file, index=False)
        self.logger.info(f"Saved sentiment timeseries to {out_file}")

        return result_df
