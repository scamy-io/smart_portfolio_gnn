"""Orchestrator script for data ingestion pipeline."""

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

# Add src to path so we can import from it
sys.path.append(str(Path(__file__).parent.parent))

from src.data_ingestion.feature_engineering import FeatureEngineer
from src.data_ingestion.gdelt_processor import GDELTProcessor
from src.data_ingestion.sec_parser import SECParser
from src.data_ingestion.yfinance_downloader import YFinanceDownloader


def setup_logging(level_str: str, file_path: str):
    """Set up logging to file and console."""
    level = getattr(logging, level_str.upper(), logging.INFO)

    log_file = Path(file_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def load_config(config_path: Path) -> dict:
    """Load YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Data Ingestion Pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data_config.yaml"),
        help="Path to config YAML",
    )
    parser.add_argument(
        "--skip-gdelt", action="store_true", help="Skip GDELT processing"
    )
    parser.add_argument(
        "--tickers",
        type=str,
        help="Comma-separated list of tickers to override universe",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config["logging"]["level"], config["logging"]["file"])
    logger = logging.getLogger(__name__)

    start_time = time.time()

    raw_dir = Path(config["paths"]["raw_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])

    # 2. Get S&P 500 tickers
    logger.info("Initializing YFinance downloader to fetch universe...")
    yf_dl = YFinanceDownloader(
        tickers=[],
        start_date=config["date_range"]["start"],
        end_date=config["date_range"]["end"],
        output_dir=raw_dir,
        config=config,
    )

    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",")]
    else:
        tickers = yf_dl.get_sp500_tickers()
        max_stocks = config["universe"].get("max_stocks")
        if max_stocks and len(tickers) > max_stocks:
            tickers = tickers[:max_stocks]

    logger.info(f"Targeting {len(tickers)} tickers.")

    # Update downloader with real tickers
    yf_dl.tickers = tickers

    # 3. Run YFinanceDownloader
    t0 = time.time()
    prices_df = yf_dl.download_prices()
    fundamentals_df = yf_dl.download_fundamentals()
    logger.info(f"YFinance download took {time.time() - t0:.2f} seconds")

    # 4. Run GDELTProcessor
    if not args.skip_gdelt:
        t0 = time.time()
        # Create a dummy company name map for now
        company_map = {t: t for t in tickers}
        gdelt = GDELTProcessor(
            tickers=tickers, company_name_map=company_map, output_dir=processed_dir
        )
        gdelt.build_sentiment_timeseries(
            config["date_range"]["start"], config["date_range"]["end"]
        )
        logger.info(f"GDELT processing took {time.time() - t0:.2f} seconds")
    else:
        logger.info("Skipping GDELT processing as requested.")

    # 5. Run FeatureEngineer
    t0 = time.time()
    if not prices_df.empty:
        fe = FeatureEngineer(prices_df=prices_df, output_dir=processed_dir)
        features_df = fe.build_node_features(fundamentals_df)
        logger.info(f"Feature engineering took {time.time() - t0:.2f} seconds")
    else:
        logger.error("Price data is empty, skipping feature engineering.")
        features_df = None

    # 6. Run SECParser
    t0 = time.time()
    sec = SECParser(output_dir=processed_dir / "edges")
    sec_df = sec.build_supply_chain_graph()
    logger.info(f"SEC parsing took {time.time() - t0:.2f} seconds")

    # 7. Print summary report
    logger.info("\n--- PIPELINE SUMMARY ---")
    logger.info(f"Tickers processed: {len(tickers)}")
    logger.info(
        f"Date range: {config['date_range']['start']} to {config['date_range']['end']}"
    )

    if features_df is not None:
        missing_pct = features_df.isna().mean().mean() * 100
        logger.info(f"Missing data percentage in features: {missing_pct:.2f}%")

    logger.info("Output files generated:")
    logger.info(f"- {raw_dir / 'prices/ohlcv.parquet'}")
    logger.info(f"- {raw_dir / 'fundamentals/fundamentals.parquet'}")
    if not args.skip_gdelt:
        logger.info(f"- {processed_dir / 'sentiment/sentiment_daily.parquet'}")
    logger.info(f"- {processed_dir / 'node_features.parquet'}")
    logger.info(f"- {processed_dir / 'edges/supply_chain_edges.parquet'}")
    logger.info(f"Total pipeline time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
