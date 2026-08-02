from dotenv import load_dotenv
load_dotenv()
import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from scripts.train_model import FullModel
from src.evaluation.ablation_study import AblationStudy
from src.evaluation.backtester import WalkForwardBacktester
from src.evaluation.portfolio_metrics import compute_all_metrics
from src.evaluation.visualization import (
    plot_cumulative_returns,
    plot_drawdown,
    plot_rolling_sharpe,
)
from src.graph_builder.temporal_graph import TemporalGraphDataset


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/data_config.yaml"))
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--benchmark", type=str, default="equal_weight")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    args.output_dir.mkdir(exist_ok=True, parents=True)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading graph dataset...")
    edge_paths = {
        "correlates_with": Path("data/processed/edges/correlation_edges.parquet"),
        "sentiment_co_mention": Path("data/processed/edges/sentiment_edges.parquet"),
        "supplies": Path("data/processed/edges/supply_chain_edges_processed.parquet"),
        "same_sector_as": Path("data/processed/edges/sector_edges.parquet"),
        "fundamentally_similar_to": Path(
            "data/processed/edges/fundamental_edges.parquet"
        ),
    }

    nf_path = Path("data/processed/node_features_dry_run.parquet") if "tickers" in config else Path("data/processed/node_features.parquet")
    dataset = TemporalGraphDataset(
        graph_snapshot_dir=Path("data/processed/graph_snapshots"),
        node_features_path=nf_path,
        edge_paths=edge_paths,
    )

    logger.info("Loading model...")
    model_path = Path("models/best_htgat.pt")
    if len(dataset) > 0:
        graph = dataset[0]
        in_channels = graph["stock"].x.shape[1]
        model = FullModel(graph.metadata(), in_channels).to(device)
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
    else:
        logger.error("Dataset empty")
        model.eval()

    # DRY RUN CONFIG
    start_date = config.get("date_range", {}).get("start", args.start)
    end_date = config.get("date_range", {}).get("end", args.end)
    transaction_cost_bps = config.get("backtest", {}).get("transaction_cost_bps", 10.0)
    rebalance_frequency = config.get("backtest", {}).get("rebalance_frequency", "weekly")

    logger.info(f"Running backtest from {start_date} to {end_date}...")
    backtester = WalkForwardBacktester(
        model=model,
        dataset=dataset,
        config=config,
        rebalance_frequency=rebalance_frequency,
        transaction_cost_bps=transaction_cost_bps,
    )

    df_history = backtester.run(start_date=start_date, end_date=end_date)
    if df_history is None or df_history.empty:
        logger.error("Backtest returned no history. Check dates and data availability.")
        return

    logger.info("Computing metrics...")
    metrics = compute_all_metrics(df_history)

    # Calculate Benchmark Returns
    benchmark_rets = {}
    for b_name in ["equal_weight", "market_cap", "har_min_variance"]:
        b_ret = backtester.get_benchmark_returns(b_name, list(df_history.index.astype(str)))
        benchmark_rets[b_name] = b_ret.values

    df_history["benchmark_ew"] = benchmark_rets["equal_weight"]
    df_history["benchmark_mc"] = benchmark_rets["market_cap"]
    df_history["benchmark_har"] = benchmark_rets["har_min_variance"]

    logger.info("\n--- BACKTEST RESULTS ---")

    if args.ablation:
        logger.info("Running ablation studies...")
        ab_study = AblationStudy(config, backtester, start_date, end_date)
        ab_df = ab_study.run_all()

    logger.info("Generating plots...")
    plot_rolling_sharpe(df_history, output_path=args.output_dir / "rolling_sharpe.png")

    report = {"start_date": start_date, "end_date": end_date, "metrics": metrics}

    with open(args.output_dir / f"backtest_{start_date}_{end_date}.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 50)
    print("BACKTEST REPORT")
    print("=" * 50)
    print(f"Annualized Return:  {metrics['annualized_return']:.2%}")
    print(f"Sharpe Ratio:       {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:       {metrics['max_drawdown']:.2%}")
    print(f"Calmar Ratio:       {metrics['calmar_ratio']:.2f}")
    print(f"Turnover:           {metrics['turnover_pct']:.2%}")
    if "information_ratio" in metrics:
        print(
            f"Info Ratio vs {args.benchmark.upper()}:  {metrics['information_ratio']:.2f}"
        )
    print("=" * 50)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
