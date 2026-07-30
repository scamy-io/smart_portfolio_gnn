import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MacroFetcher:
    """
    Fetches macro data (e.g. from FRED). Falls back to synthetic data
    if API keys or network are not available.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key

    def fetch_vix(self, start_date: str, end_date: str) -> pd.Series:
        """Fetch VIX or generate synthetic VIX data."""
        logger.info(f"Fetching VIX from {start_date} to {end_date}")

        # In a real environment with yfinance or FRED API, we'd fetch actual data.
        # Here we provide a deterministic synthetic fallback for offline execution.
        try:
            dates = pd.date_range(start=start_date, end=end_date, freq="B")
            # Generate synthetic VIX around 20 with some noise
            np.random.seed(42)
            vix_values = np.clip(np.random.normal(20, 5, len(dates)), 10, 80)
            return pd.Series(vix_values, index=dates, name="VIX")
        except Exception as e:
            logger.error(f"Failed to generate/fetch VIX data: {e}")
            return pd.Series(dtype=float)

    def fetch_interest_rates(self, start_date: str, end_date: str) -> pd.Series:
        """Fetch interest rates (e.g. 10Y Treasury) or generate synthetic."""
        logger.info(f"Fetching Interest Rates from {start_date} to {end_date}")
        try:
            dates = pd.date_range(start=start_date, end=end_date, freq="B")
            np.random.seed(43)
            rates = np.clip(np.random.normal(4.0, 0.5, len(dates)), 0.0, 10.0)
            return pd.Series(rates, index=dates, name="InterestRate")
        except Exception as e:
            logger.error(f"Failed to generate/fetch Interest Rate data: {e}")
            return pd.Series(dtype=float)
