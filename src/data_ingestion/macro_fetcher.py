import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


import os

class MacroFetcher:
    """
    Fetches macro data (e.g. from FRED). Falls back to synthetic data
    if API keys or network are not available.
    """

    def __init__(self):
        self.api_key = os.environ.get("FRED_API_KEY")
        if not self.api_key:
            logger.warning("FRED_API_KEY not found. Set it in .env or export FRED_API_KEY=your_key")
            logger.warning("Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        else:
            logger.info("FRED API key loaded successfully.")

    def fetch_series_fred(self, series_id: str, start_date: str, end_date: str) -> pd.Series:
        import pandas_datareader.data as web
        import os
        cache_file = f"data/raw/macro/{series_id}.csv"
        os.makedirs("data/raw/macro", exist_ok=True)
        
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, index_col="DATE", parse_dates=True)
            if df.index.min() <= pd.to_datetime(start_date) and df.index.max() >= pd.to_datetime(end_date):
                s = df[series_id]
                return s.loc[start_date:end_date].ffill()
        
        try:
            df = web.DataReader(series_id, 'fred', start_date, end_date)
            df.to_csv(cache_file)
            return df[series_id].ffill()
        except Exception as e:
            import os
            if os.environ.get("SMART_PORTFOLIO_OFFLINE_MODE") == "1":
                logger.warning(f"OFFLINE MODE: Using zeros for {series_id}")
                dates = pd.date_range(start=start_date, end=end_date, freq="B")
                return pd.Series(0.0, index=dates, name=series_id)
            raise RuntimeError(f"FRED API failed for {series_id}: {e}") from e

    def fetch_vix(self, start_date: str, end_date: str) -> pd.Series:
        """Fetch VIX from FRED or cache."""
        logger.info(f"Fetching VIX from {start_date} to {end_date}")
        return self.fetch_series_fred("VIXCLS", start_date, end_date)

    def fetch_interest_rates(self, start_date: str, end_date: str) -> pd.Series:
        """Fetch 10Y Treasury from FRED or cache."""
        logger.info(f"Fetching Interest Rates from {start_date} to {end_date}")
        return self.fetch_series_fred("DGS10", start_date, end_date)
        
    def fetch_corporate_spread(self, start_date: str, end_date: str) -> pd.Series:
        """Fetch BAA Corporate Spread from FRED or cache."""
        logger.info(f"Fetching BAA Spread from {start_date} to {end_date}")
        return self.fetch_series_fred("BAA10Y", start_date, end_date)
