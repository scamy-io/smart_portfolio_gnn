import pandas as pd


def check_node_features():
    print("--- 1. Checking node_features.parquet ---")
    try:
        df = pd.read_parquet("data/processed/node_features.parquet")
    except Exception as e:
        print(f"Error reading node_features.parquet: {e}")
        return

    # Check 1: No look-ahead bias
    max_date = str(df.index.get_level_values("date").max())
    print(f"Max date in data: {max_date}")
    assert max_date < "2026-07-30", f"Look-ahead bias detected! Max date: {max_date}"
    print("Check 1 (No look-ahead bias): PASS")

    # Check 2: Z-scores are actually centered
    # In our implementation, sma_5 is normalized in-place
    print("Z-score stats (mean should be ~0):")
    print(df.groupby("ticker")["sma_5"].mean().describe())
    print("Check 2 (Z-scores centered): DONE")

    # Check 3: No NaN leakage
    print("NaN counts per column (Top 10):")
    print(df.isnull().sum().sort_values(ascending=False).head(10))
    print("Check 3 (NaN leakage): DONE")

    # Check 4: All tickers have same date range
    print("Counts per ticker:")
    print(df.groupby("ticker").size().describe())
    print("Check 4 (Consistent date range): DONE")
    print()


def check_sentiment():
    print("--- 2. Checking sentiment_daily.parquet ---")
    try:
        sent = pd.read_parquet("data/processed/sentiment/sentiment_daily.parquet")
    except Exception as e:
        print(f"Error reading sentiment_daily.parquet: {e}")
        return

    print(f"Sentiment data shape: {sent.shape}")
    print("First few rows of AAPL (if present):")
    if "AAPL" in sent["ticker"].values:
        print(sent[sent["ticker"] == "AAPL"][["date", "avg_tone"]].head(10))
    else:
        print("AAPL not found, showing first ticker:")
        first_ticker = sent["ticker"].iloc[0]
        print(sent[sent["ticker"] == first_ticker][["date", "avg_tone"]].head(10))
    print("Check (Decay is working): DONE")
    print()


def check_ohlcv():
    print("--- 3. Checking ohlcv.parquet ---")
    try:
        prices = pd.read_parquet("data/raw/prices/ohlcv.parquet")
    except Exception as e:
        print(f"Error reading ohlcv.parquet: {e}")
        return

    # Check: Adjusted close <= Close (dividends/adjustments)
    valid_adj = (prices["adj_close"] <= prices["close"] + 0.01).all()
    print(f"Is Adj Close <= Close? {valid_adj}")
    if not valid_adj:
        print(prices[prices["adj_close"] > prices["close"] + 0.01].head())
    print("Check (Adj Close): DONE")
    print()


if __name__ == "__main__":
    check_node_features()
    check_sentiment()
    check_ohlcv()
