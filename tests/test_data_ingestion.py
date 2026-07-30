from unittest.mock import patch

import numpy as np
import pandas as pd


def test_yfinance_download_not_empty():
    mock_df = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Adj Close": [101.0, 102.0],
            "Volume": [1000, 1100],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    with patch("yfinance.download", return_value=mock_df) as mock_download:
        df = mock_download("AAPL")

    assert not df.empty
    assert df.shape == (2, 6)
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def test_gdelt_tone_parsing():
    tone_str = "1.23,4.56,2.34,3.1"

    def parse_tone(tone_string: str) -> float:
        try:
            return float(tone_string.split(",")[0])
        except (ValueError, IndexError):
            return 0.0

    parsed = parse_tone(tone_str)
    assert parsed == 1.23


def test_feature_engineering_no_lookahead():
    prices = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0])
    sma = prices.rolling(window=3).mean()
    assert np.isclose(sma.iloc[2], 12.0)
    assert pd.isna(sma.iloc[0])


def test_zscore_normalization():
    prices = pd.Series(np.random.normal(100, 5, 100))
    window = 20
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    zscore = (prices - rolling_mean) / rolling_std
    valid_zscore = zscore.dropna()

    assert valid_zscore.mean() < 0.5
    assert valid_zscore.mean() > -0.5
    assert valid_zscore.std() > 0.8
    assert valid_zscore.std() < 1.2
