from dotenv import load_dotenv
load_dotenv()
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from src.graph_builder.correlation_edges import CorrelationEdgeBuilder
from src.graph_builder.fundamental_edges import FundamentalEdgeBuilder
from src.graph_builder.sector_edges import SectorEdgeBuilder
from src.graph_builder.sentiment_edges import SentimentEdgeBuilder
from src.graph_builder.supply_chain_edges import SupplyChainEdgeBuilder
from src.graph_builder.temporal_graph import TemporalGraphDataset


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def main():
    parser = argparse.ArgumentParser(description="Graph Builder Pipeline")
    parser.add_argument("--config", type=Path, default=Path("configs/data_config.yaml"))
    parser.add_argument("--start-date", type=str)
    parser.add_argument("--end-date", type=str)
    parser.add_argument("--skip-sentiment", action="store_true")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    start_date = args.start_date or config["date_range"]["start"]
    end_date = args.end_date or config["date_range"]["end"]

    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(start_date, end_date)]

    try:
        prices_df = pd.read_parquet("data/raw/prices/ohlcv.parquet")
        if "tickers" in config:
            prices_df = prices_df[prices_df["ticker"].isin(config["tickers"])]
    except Exception as e:
        logger.error(f"Failed to load prices: {e}")
        return

    try:
        fundamentals_df = pd.read_parquet("data/raw/fundamentals/fundamentals.parquet")
        if "tickers" in config and not fundamentals_df.empty:
            fundamentals_df = fundamentals_df[fundamentals_df["ticker"].isin(config["tickers"])]
    except Exception:
        fundamentals_df = pd.DataFrame()

    try:
        supply_chain_df = pd.read_parquet(
            "data/processed/edges/supply_chain_edges.parquet"
        )
    except Exception:
        supply_chain_df = pd.DataFrame(
            columns=[
                "source_ticker",
                "target_ticker",
                "relationship_type",
                "weight",
                "data_source",
            ]
        )

    try:
        gics_df = pd.read_csv("data/external/gics_mapping.csv")
    except Exception:
        gics_df = pd.DataFrame(
            columns=["ticker", "sector", "industry_group", "industry"]
        )

    try:
        nf_df = pd.read_parquet("data/processed/node_features.parquet").reset_index()
        if "tickers" in config:
            nf_df = nf_df[nf_df["ticker"].isin(config["tickers"])]
        nf_df.set_index(["date", "ticker"]).to_parquet("data/processed/node_features_dry_run.parquet")
        nf_path = Path("data/processed/node_features_dry_run.parquet")
    except Exception:
        nf_df = pd.DataFrame()
        nf_path = Path("data/processed/node_features.parquet")

    logger.info("Building correlation edges...")
    vix_series = None
    if not nf_df.empty and "vix_level" in nf_df.columns:
        vix_series = nf_df.groupby("date")["vix_level"].first()
    corr_builder = CorrelationEdgeBuilder(prices_df, vix_series=vix_series)
    corr_builder.build_all_edges(dates)

    logger.info("Building supply chain edges...")
    sc_builder = SupplyChainEdgeBuilder(supply_chain_df)
    sc_builder.build_edges()

    logger.info("Building sector edges...")
    sec_builder = SectorEdgeBuilder(gics_df)
    sec_builder.build_edges(granularity="sector")
    sec_builder.build_edges(granularity="industry")

    logger.info("Building fundamental edges...")
    fund_builder = FundamentalEdgeBuilder(fundamentals_df)
    fund_builder.build_all_edges(dates)

    if not args.skip_sentiment:
        logger.info("Building sentiment edges...")
        try:
            sent_df = pd.read_parquet(
                "data/processed/sentiment/sentiment_daily.parquet"
            )
            sent_builder = SentimentEdgeBuilder(sent_df)
            sent_builder.build_all_edges(dates, {}, {}, {})
        except Exception as e:
            logger.error(f"Failed to build sentiment edges: {e}")

    logger.info("Building TemporalGraphDataset snapshots...")

    edge_paths = {
        "correlates_with": Path("data/processed/edges/correlation_edges.parquet"),
        "sentiment_co_mention": Path("data/processed/edges/sentiment_edges.parquet"),
        "supplies": Path("data/processed/edges/supply_chain_edges_processed.parquet"),
        "same_sector_as": Path("data/processed/edges/sector_edges.parquet"),
        "fundamentally_similar_to": Path(
            "data/processed/edges/fundamental_edges.parquet"
        ),
    }

    dataset = TemporalGraphDataset(
        graph_snapshot_dir=Path("data/processed/graph_snapshots"),
        node_features_path=nf_path,
        edge_paths=edge_paths,
    )
    dataset.build_snapshots(dates)

    train_loader, val_loader, test_loader = dataset.get_loaders()

    logger.info("\n--- GRAPH BUILDER SUMMARY ---")
    logger.info(f"Total dates processed: {len(dates)}")
    logger.info(
        f"Train/Val/Test split sizes: {len(train_loader.dataset)} / {len(val_loader.dataset)} / {len(test_loader.dataset)}"
    )
    logger.info("Snapshots saved to data/processed/graph_snapshots/")


if __name__ == "__main__":
    main()
