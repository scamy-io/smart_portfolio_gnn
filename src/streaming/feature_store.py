import logging
import time
from typing import Dict, Optional

import pandas as pd


class FeatureStore:
    """
    Lightweight in-memory feature cache with TTL for technical indicators and fundamentals.
    Serves as a precursor to a production system like Feast or Tecton.
    """
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
        # Dict mapping (ticker, date_str) to (timestamp, dataframe_row)
        self.cache: Dict[tuple, tuple] = {}
        self.logger = logging.getLogger(__name__)

    def _get_key(self, ticker: str, date: str) -> tuple:
        return (ticker, date)

    def update_features(self, ticker: str, date: str, features: pd.Series) -> None:
        """
        Upsert features into the store with current timestamp.
        """
        key = self._get_key(ticker, date)
        self.cache[key] = (time.time(), features)
        self.logger.debug(f"Updated features in store for {ticker} on {date}")

    def get_features(self, ticker: str, date: str) -> Optional[pd.Series]:
        """
        Retrieve features if they exist and haven't expired.
        """
        key = self._get_key(ticker, date)
        if key in self.cache:
            timestamp, features = self.cache[key]
            if time.time() - timestamp <= self.ttl_seconds:
                return features
            else:
                self.logger.debug(f"Features expired for {ticker} on {date}")
                del self.cache[key]
        return None

    def evict_expired(self) -> None:
        """
        Clean up all expired entries.
        """
        current_time = time.time()
        expired_keys = [
            k for k, (timestamp, _) in self.cache.items()
            if current_time - timestamp > self.ttl_seconds
        ]
        for k in expired_keys:
            del self.cache[k]
        if expired_keys:
            self.logger.info(f"Evicted {len(expired_keys)} expired feature entries.")
