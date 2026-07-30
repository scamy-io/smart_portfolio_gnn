import numpy as np
import pandas as pd


class TradeGenerator:
    """
    Generates actual trade orders (BUY/SELL) from current and target weights.
    """

    def __init__(self, transaction_cost_rate: float = 0.001):
        self.transaction_cost_rate = transaction_cost_rate

    def generate_trades(
        self, current_weights: pd.Series, optimal_weights: pd.Series
    ) -> pd.DataFrame:
        df = pd.DataFrame(
            {"current_weight": current_weights, "target_weight": optimal_weights}
        ).fillna(0.0)

        df["delta"] = df["target_weight"] - df["current_weight"]
        df["trade_direction"] = np.where(df["delta"] > 0, "BUY", "SELL")
        df["trade_direction"] = np.where(
            df["delta"] == 0, "HOLD", df["trade_direction"]
        )
        df["estimated_cost"] = df["delta"].abs() * self.transaction_cost_rate

        # Filter out negligible trades
        df = df[df["delta"].abs() > 0.001]

        return df.reset_index().rename(columns={"index": "ticker"})
