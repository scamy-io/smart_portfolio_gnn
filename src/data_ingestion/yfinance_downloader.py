"""YFinance downloader for OHLCV and fundamental data."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yfinance as yf


class DownloadError(Exception):
    """Custom exception for network failures during download."""

    pass


class YFinanceDownloader:
    """Downloader for yfinance data."""

    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        output_dir: Path,
        config: Dict[str, Any],
    ):
        """
        Initialize the downloader.

        Args:
            tickers (List[str]): List of tickers to download.
            start_date (str): Start date in YYYY-MM-DD.
            end_date (str): End date in YYYY-MM-DD.
            output_dir (Path): Directory to save raw data.
            config (Dict[str, Any]): Configuration dictionary.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = output_dir
        self.config = config
        self.logger = logging.getLogger(__name__)

    def download_prices(self) -> pd.DataFrame:
        """
        Download OHLCV + Adjusted Close + Volume for all tickers.

        Returns:
            pd.DataFrame: Downloaded price data.
        """
        batch_size = self.config["yfinance"]["batch_size"]
        sleep_time = self.config["yfinance"]["sleep_between_batches"]
        retries = self.config["yfinance"]["retry_attempts"]

        all_data = []
        success_count = 0
        fail_count = 0

        for i in range(0, len(self.tickers), batch_size):
            batch = self.tickers[i : i + batch_size]
            self.logger.info(f"Downloading batch {i//batch_size + 1}, tickers: {batch}")

            for ticker in batch:
                attempt = 0
                while attempt < retries:
                    try:
                        ticker_obj = yf.Ticker(ticker)
                        df = ticker_obj.history(
                            start=self.start_date, end=self.end_date, auto_adjust=False
                        )
                        if df.empty:
                            raise DownloadError(f"No price data found for {ticker}")

                        df.reset_index(inplace=True)
                        df["ticker"] = ticker
                        # Standardize columns
                        df = df.rename(
                            columns={
                                "Date": "date",
                                "Open": "open",
                                "High": "high",
                                "Low": "low",
                                "Close": "close",
                                "Adj Close": "adj_close",
                                "Volume": "volume",
                            }
                        )
                        # Keep only relevant columns
                        cols_to_keep = [
                            "date",
                            "ticker",
                            "open",
                            "high",
                            "low",
                            "close",
                            "adj_close",
                            "volume",
                        ]
                        df = df[[c for c in cols_to_keep if c in df.columns]]
                        all_data.append(df)
                        success_count += 1
                        break
                    except Exception as e:
                        attempt += 1
                        self.logger.warning(
                            f"Error downloading {ticker} (Attempt {attempt}/{retries}): {e}"
                        )
                        if attempt == retries:
                            self.logger.error(
                                f"Failed to download {ticker} after {retries} attempts."
                            )
                            fail_count += 1
                        time.sleep(2**attempt)  # Exponential backoff

            time.sleep(sleep_time)

        self.logger.info(
            f"Price download complete. Success: {success_count}, Failed: {fail_count}"
        )
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            # Ensure output dir exists
            prices_dir = self.output_dir / "prices"
            prices_dir.mkdir(parents=True, exist_ok=True)

            output_file = prices_dir / "ohlcv.parquet"
            final_df.to_parquet(output_file, index=False)
            self.logger.info(f"Saved price data to {output_file}")
            return final_df
        return pd.DataFrame()

    def download_fundamentals(self) -> pd.DataFrame:
        """
        Download fundamental data for all tickers.

        Returns:
            pd.DataFrame: Downloaded fundamental data.
        """
        metrics = [
            "trailingPE",
            "forwardPE",
            "debtToEquity",
            "returnOnEquity",
            "returnOnAssets",
            "currentRatio",
            "quickRatio",
            "marketCap",
            "beta",
            "dividendYield",
        ]
        data = []

        for ticker in self.tickers:
            try:
                info = yf.Ticker(ticker).info
                row = {"ticker": ticker}
                for m in metrics:
                    row[m] = info.get(m, float("nan"))
                data.append(row)
            except Exception as e:
                self.logger.warning(f"Failed to fetch fundamentals for {ticker}: {e}")

        df = pd.DataFrame(data)
        if not df.empty:
            fund_dir = self.output_dir / "fundamentals"
            fund_dir.mkdir(parents=True, exist_ok=True)
            output_file = fund_dir / "fundamentals.parquet"
            df.to_parquet(output_file, index=False)
            self.logger.info(f"Saved fundamental data to {output_file}")
        return df

    def get_sp500_tickers(self) -> List[str]:
        """
        Get S&P 500 tickers from Wikipedia, fallback to local CSV.

        Returns:
            List[str]: List of ticker symbols.
        """
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            df = tables[0]
            tickers = df["Symbol"].tolist()
            # Clean up tickers (e.g., BRK.B -> BRK-B)
            tickers = [t.replace(".", "-") for t in tickers]
            self.logger.info(f"Successfully scraped {len(tickers)} S&P 500 tickers.")
            return tickers
        except Exception as e:
            self.logger.error(f"Failed to scrape S&P 500 tickers: {e}")
            fallback_path = Path("data/external/sp500_constituents.csv")
            if fallback_path.exists():
                df = pd.read_csv(fallback_path)
                tickers = df["Symbol"].tolist()
                self.logger.info(f"Loaded {len(tickers)} tickers from fallback CSV.")
                return tickers
            self.logger.error(
                "No fallback CSV found. Returning hardcoded subset of tickers."
            )
            return [
                "AAPL",
                "MSFT",
                "GOOG",
                "AMZN",
                "NVDA",
                "META",
                "TSLA",
                "JPM",
                "V",
                "JNJ",
            ]
