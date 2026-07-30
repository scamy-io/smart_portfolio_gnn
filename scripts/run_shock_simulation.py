import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from src.risk_engine.shock_simulator import ShockSimulator


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def main():
    parser = argparse.ArgumentParser(description="Run Shock Simulations")
    parser.add_argument(
        "--scenario",
        type=str,
        default="liquidity_freeze",
        choices=[
            "liquidity_freeze",
            "sector_demand_shock",
            "supply_chain_failure",
            "sentiment_contagion",
            "macro_regime_shift",
        ],
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info(f"Running shock simulation for scenario: {args.scenario}")

    # Synthetic initialization for offline script execution
    g = HeteroData()
    g["stock"].x = torch.ones(5, 32)
    g["stock", "correlates_with", "stock"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4], [1, 0, 3, 4, 2]]
    )
    g["stock", "correlates_with", "stock"].edge_attr = torch.ones(5, 1)

    # Dummy model
    class DummyModel(nn.Module):
        def forward(self, x):
            return {
                "return": torch.zeros(5),
                "cvar": torch.zeros(5),
                "embedding": torch.zeros(5, 64),
            }

    model = DummyModel()
    simulator = ShockSimulator(model, device=args.device)

    tickers = ["A", "B", "C", "D", "E"]
    weights = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2], index=tickers)

    res = simulator.run_scenario(
        g, args.scenario, tickers, weights, target_indices=[0, 1], target_idx=0
    )

    logger.info("Simulation Results:")
    for k, v in res.items():
        logger.info(f"{k}: {v}")


if __name__ == "__main__":
    main()
