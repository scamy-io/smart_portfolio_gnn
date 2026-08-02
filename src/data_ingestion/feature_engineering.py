"""Feature Engineering module for technical indicators and fundamentals."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Class to engineer features from price and fundamental data."""

    def __init__(self, prices_df: pd.DataFrame, output_dir: Path):
        """
        Initialize FeatureEngineer.

        Args:
            prices_df (pd.DataFrame): Price dataframe.
            output_dir (Path): Output directory.
        """
        self.prices_df = prices_df.copy()
        if "date" in self.prices_df.columns:
            self.prices_df["date"] = pd.to_datetime(self.prices_df["date"])
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)

    def compute_technical_indicators(self) -> pd.DataFrame:
        """
        Compute technical indicators.

        Returns:
            pd.DataFrame: DataFrame with added indicators.
        """
        self.logger.info("Computing technical indicators...")
        df = self.prices_df.sort_values(by=["ticker", "date"])

        def compute_group(g):
            g["sma_5"] = g["adj_close"].rolling(window=5, min_periods=1).mean()
            g["sma_20"] = g["adj_close"].rolling(window=20, min_periods=1).mean()
            g["ema_12"] = g["adj_close"].ewm(span=12, adjust=False).mean()
            g["ema_26"] = g["adj_close"].ewm(span=26, adjust=False).mean()
            g["macd"] = g["ema_12"] - g["ema_26"]
            g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()

            # RSI
            delta = g["adj_close"].diff()
            gain = delta.clip(lower=0)
            loss = -1 * delta.clip(upper=0)
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            g["rsi_14"] = 100 - (100 / (1 + rs))
            g["rsi_14"] = g["rsi_14"].fillna(50)

            # Bollinger Bands
            std_20 = g["adj_close"].rolling(window=20, min_periods=1).std()
            g["bb_upper"] = g["sma_20"] + 2 * std_20
            g["bb_lower"] = g["sma_20"] - 2 * std_20

            # Volatility
            g["log_return"] = np.log(g["adj_close"] / g["adj_close"].shift(1))
            g["volatility_21d"] = (
                g["log_return"].rolling(window=21, min_periods=1).std()
            )

            # Momentum
            g["momentum_12m"] = (g["adj_close"] / g["adj_close"].shift(252)) - 1

            # Volume
            g["volume_sma_20"] = g["volume"].rolling(window=20, min_periods=1).mean()
            g["price_to_sma20"] = g["adj_close"] / g["sma_20"].replace(0, np.nan)

            return g

        self.prices_df = df.groupby("ticker", group_keys=False).apply(compute_group)
        return self.prices_df

    def compute_log_returns(self) -> pd.DataFrame:
        """
        Compute log and simple returns.

        Returns:
            pd.DataFrame: DataFrame with added return columns.
        """
        self.logger.info("Computing returns...")
        df = self.prices_df.sort_values(by=["ticker", "date"])

        def compute_returns(g):
            g["log_return"] = np.log(g["adj_close"] / g["adj_close"].shift(1))
            g["simple_return"] = (g["adj_close"] / g["adj_close"].shift(1)) - 1
            return g

        self.prices_df = df.groupby("ticker", group_keys=False).apply(compute_returns)
        return self.prices_df

    def merge_fundamentals(self, fundamentals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge fundamentals into price data.

        Args:
            fundamentals_df (pd.DataFrame): Fundamental data.

        Returns:
            pd.DataFrame: Merged DataFrame.
        """
        self.logger.info("Merging fundamentals...")
        if fundamentals_df.empty:
            return self.prices_df

        df = pd.merge(self.prices_df, fundamentals_df, on="ticker", how="left")

        fund_cols = [c for c in fundamentals_df.columns if c != "ticker"]
        for col in fund_cols:
            if col in df.columns:
                df[col] = df.groupby("ticker")[col].transform(
                    lambda x: x.fillna(x.median())
                )

        self.prices_df = df
        return self.prices_df

    def build_node_features(
        self, fundamentals_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Build complete node features and normalize.

        Args:
            fundamentals_df (Optional[pd.DataFrame]): Fundamental data to merge.

        Returns:
            pd.DataFrame: Normalized node features.
        """
        self.compute_technical_indicators()
        self.compute_log_returns()
        if fundamentals_df is not None:
            self.merge_fundamentals(fundamentals_df)

        vix_path = self.output_dir.parent.parent / "raw/macro/VIXCLS.csv"
        import os
        if vix_path.exists():
            vix_df = pd.read_csv(vix_path)
            vix_df["DATE"] = pd.to_datetime(vix_df["DATE"])
            vix_df.rename(columns={"DATE": "date", "VIXCLS": "vix_level"}, inplace=True)
            self.prices_df = pd.merge(self.prices_df, vix_df, on="date", how="left")
            self.prices_df["vix_level"] = self.prices_df["vix_level"].ffill().bfill()
            
        if "vix_level" not in self.prices_df.columns or self.prices_df["vix_level"].isna().all():
            if os.environ.get("SMART_PORTFOLIO_OFFLINE_MODE") == "1":
                self.prices_df["vix_level"] = 20.0
            else:
                raise ValueError("VIX data missing after macro merge. Run macro_fetcher first or set SMART_PORTFOLIO_OFFLINE_MODE=1")
                
        if self.prices_df["vix_level"].isna().any():
            if os.environ.get("SMART_PORTFOLIO_OFFLINE_MODE") == "1":
                self.prices_df["vix_level"] = self.prices_df["vix_level"].fillna(20.0)
            else:
                raise ValueError("VIX contains NaN after forward-fill")

        sentiment_features_path = self.output_dir.parent / "sentiment_node_features.parquet"
        if sentiment_features_path.exists():
            self.logger.info("Merging sentiment node features...")
            sentiment_df = pd.read_parquet(sentiment_features_path)
            sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
            self.prices_df = pd.merge(self.prices_df, sentiment_df, on=["date", "ticker"], how="left")
            self.prices_df["NumMentions"] = self.prices_df["NumMentions"].fillna(0.0)
            self.prices_df["ToneDispersion"] = self.prices_df["ToneDispersion"].fillna(0.0)
        else:
            self.prices_df["NumMentions"] = 0.0
            self.prices_df["ToneDispersion"] = 0.0

        self.logger.info("Normalizing features...")
        exclude_cols = [
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "log_return",
            "simple_return",
        ]
        feature_cols = [c for c in self.prices_df.columns if c not in exclude_cols]

        def normalize_group(g):
            for col in feature_cols:
                mean_252 = g[col].rolling(window=252, min_periods=1).mean()
                std_252 = (
                    g[col].rolling(window=252, min_periods=1).std().replace(0, 1e-8)
                )
                z_score = (g[col] - mean_252) / std_252
                g[col] = z_score.clip(-5, 5)
            return g

        df = self.prices_df.sort_values(by=["ticker", "date"])
        df = df.groupby("ticker", group_keys=False).apply(normalize_group)

        df.set_index(["date", "ticker"], inplace=True)

        out_file = self.output_dir / "node_features.parquet"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_file)
        self.logger.info(f"Saved node features to {out_file}")

        return df
