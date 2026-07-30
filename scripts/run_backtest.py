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

    dataset = TemporalGraphDataset(
        graph_snapshot_dir=Path("data/processed/graph_snapshots"),
        node_features_path=Path("data/processed/node_features.parquet"),
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
        return

    logger.info(f"Running backtest from {args.start} to {args.end}...")
    bt = WalkForwardBacktester(model, dataset, config)
    port_df = bt.run(args.start, args.end)

    bm_ret = bt.get_benchmark_returns(
        args.benchmark, port_df.index.astype(str).tolist()
    )
    bm_df = pd.DataFrame({"portfolio_return": bm_ret})
    bm_df.index = port_df.index

    metrics = compute_all_metrics(port_df, bm_df)

    if args.ablation:
        logger.info("Running ablation studies...")
        ab_study = AblationStudy(config, bt, args.start, args.end)
        ab_df = ab_study.run_all()

    logger.info("Generating plots...")
    plot_cumulative_returns(port_df, bm_df, args.output_dir / "cumulative_returns.png")
    plot_drawdown(port_df, args.output_dir / "drawdown.png")
    plot_rolling_sharpe(port_df, output_path=args.output_dir / "rolling_sharpe.png")

    report = {"start_date": args.start, "end_date": args.end, "metrics": metrics}

    with open(args.output_dir / f"backtest_{args.start}_{args.end}.json", "w") as f:
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
