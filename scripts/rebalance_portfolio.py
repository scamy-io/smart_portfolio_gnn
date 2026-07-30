import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.rebalancing.cost_aware_optimizer import CostAwareOptimizer
from src.rebalancing.trade_generator import TradeGenerator


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def main():
    parser = argparse.ArgumentParser(description="Rebalance Portfolio")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Loading latest portfolio state...")

    # In a real pipeline, we'd load these from a DB or feature store.
    # We create synthetic data here to satisfy the script execution offline.
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    n = len(tickers)
    np.random.seed(42)
    expected_returns = pd.Series(np.random.normal(0.01, 0.05, n), index=tickers)
    cov_matrix = np.cov(np.random.normal(0, 0.02, (n, 100)))

    current_weights = pd.Series(np.ones(n) / n, index=tickers)

    optimizer = CostAwareOptimizer(
        expected_returns,
        cov_matrix,
        current_weights,
        transaction_cost_rate=config["rebalancing"]["transaction_cost_rate"],
        max_weight=config["rebalancing"]["max_weight"],
        target_hhi=config["rebalancing"]["target_hhi"],
        min_enb=config["rebalancing"]["min_enb"],
    )

    logger.info("Running optimizer...")
    opt_w = optimizer.optimize()

    generator = TradeGenerator(
        transaction_cost_rate=config["rebalancing"]["transaction_cost_rate"]
    )
    trades = generator.generate_trades(current_weights, opt_w)

    logger.info("Optimization complete. Recommended trades:")
    logger.info("\n" + trades.to_string())


if __name__ == "__main__":
    main()
