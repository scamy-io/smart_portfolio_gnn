import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from scripts.train_model import FullModel
from src.rebalancing.cost_aware_optimizer import CostAwareOptimizer
from src.rebalancing.rebalance_triggers import RebalanceTriggerChecker
from src.risk_engine.cluster_detector import SpectralClusterDetector
from src.risk_engine.concentration_metrics import ConcentrationMetrics
from src.risk_engine.shock_simulator import ShockSimulator


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str)
    parser.add_argument("--portfolio", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/data_config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    args.output_dir.mkdir(exist_ok=True)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    snapshot_dir = Path("data/processed/graph_snapshots")
    snapshots = sorted(list(snapshot_dir.glob("*.pt")))

    if not snapshots:
        logger.error("No graph snapshots found.")
        return

    current_path = snapshots[-1]
    date_str = current_path.stem
    if args.date:
        current_path = snapshot_dir / f"{args.date}.pt"
        if not current_path.exists():
            logger.error(f"Snapshot for {args.date} not found.")
            return
        date_str = args.date

    logger.info(f"Using snapshot: {date_str}")
    graph = torch.load(current_path, weights_only=False).to(device)

    nf_df = pd.read_parquet("data/processed/node_features.parquet")
    tickers = nf_df.index.get_level_values("ticker").unique().tolist()
    tickers = sorted(tickers)[: graph["stock"].x.shape[0]]

    if args.portfolio and args.portfolio.exists():
        weights_df = pd.read_csv(args.portfolio, index_col=0)
        weights = weights_df["weight"]
    else:
        w = 1.0 / len(tickers)
        weights = pd.Series([w] * len(tickers), index=tickers)

    model_path = Path("models/best_htgat.pt")
    in_channels = graph["stock"].x.shape[1]
    model = FullModel(graph.metadata(), in_channels).to(device)
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        logger.warning("Trained model not found. Using untrained model.")

    model.eval()

    with torch.no_grad():
        preds = model(graph)
        htgat_out = model.htgat(
            {"stock": graph["stock"].x},
            {et: graph[et].edge_index for et in graph.edge_types},
            {et: graph[et].edge_attr for et in graph.edge_types},
        )
        embeddings = htgat_out["embedding"].cpu().numpy()
        pred_rets = pd.Series(preds["return"].cpu().numpy(), index=tickers)

    cm = ConcentrationMetrics(weights, embeddings, tickers)
    metrics_dict = cm.compute_all()

    cd = SpectralClusterDetector()
    df_clusters = cd.fit(embeddings, tickers)
    cluster_res = cd.detect_concentration_risk(weights, df_clusters)
    metrics_dict["clusters"] = cluster_res

    sim = ShockSimulator(model, str(device))
    sim_res_mc = sim.monte_carlo(
        graph, n_scenarios=10, tickers=tickers, weights=weights
    )
    mc_cvar = (
        float(sim_res_mc["portfolio_return"].quantile(0.05))
        if not sim_res_mc.empty
        else 0.0
    )

    single_shock = sim.run_scenario(
        graph, "sector_demand_shock", tickers, weights, target_indices=[0, 1]
    )

    sim_results = {"monte_carlo_cvar": mc_cvar, "scenarios": [single_shock]}

    rtc = RebalanceTriggerChecker(config)
    alerts = rtc.evaluate_all(date_str, metrics_dict, sim_results, graph, None)

    trades = []
    if alerts:
        logger.info("Alerts triggered! Running optimizer...")
        z_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        cov_gnn = z_norm @ z_norm.T

        opt = CostAwareOptimizer(pred_rets, cov_gnn, weights)
        opt_w = opt.optimize()
        df_trades = opt.generate_trades(opt_w)
        trades = df_trades.to_dict(orient="records")

    report = {
        "date": date_str,
        "metrics": metrics_dict,
        "shock_results": sim_results,
        "alerts": alerts,
        "trades": trades,
    }

    report_path = args.output_dir / f"daily_risk_report_{date_str}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
