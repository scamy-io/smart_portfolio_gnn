from typing import Dict, Tuple

import numpy as np
import pandas as pd


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.04, periods_per_year: int = 252
) -> float:
    excess = returns - risk_free_rate / periods_per_year
    std = excess.std()
    if std == 0:
        return 0.0
    return float((excess.mean() / std) * np.sqrt(periods_per_year))


def maximum_drawdown(
    portfolio_values: pd.Series,
) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    running_peak = portfolio_values.cummax()
    drawdown = (portfolio_values - running_peak) / running_peak
    max_dd = float(drawdown.min())

    trough_date = drawdown.idxmin()
    peak_date = portfolio_values.loc[:trough_date].idxmax()

    return max_dd, peak_date, trough_date


def calmar_ratio(
    returns: pd.Series, max_dd: float, periods_per_year: int = 252
) -> float:
    annual_return = float(returns.mean() * periods_per_year)
    if max_dd == 0:
        return float("inf")
    return annual_return / abs(max_dd)


def turnover(trades_df: pd.DataFrame, periods_per_year: int = 252) -> float:
    if "turnover" not in trades_df.columns:
        return 0.0
    total_turnover = trades_df["turnover"].sum()
    num_rebalances = trades_df["traded"].sum()
    if num_rebalances == 0:
        return 0.0
    annual_turnover = float(
        total_turnover * (periods_per_year / max(1, len(trades_df)))
    )
    return annual_turnover * 100.0


def information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    diff = strategy_returns - benchmark_returns
    tracking_error = diff.std()
    if tracking_error == 0:
        return 0.0
    return float((diff.mean() / tracking_error) * np.sqrt(periods_per_year))


def compute_all_metrics(
    portfolio_df: pd.DataFrame, benchmark_df: pd.DataFrame = None
) -> Dict:
    rets = portfolio_df["portfolio_return"]
    max_dd, peak, trough = maximum_drawdown(portfolio_df["portfolio_value"])

    ann_ret = float(rets.mean() * 252)
    ann_vol = float(rets.std() * np.sqrt(252))

    neg_rets = rets[rets < 0]
    downside_std = neg_rets.std() * np.sqrt(252)
    sortino = (ann_ret - 0.04) / downside_std if downside_std > 0 else float("inf")

    metrics = {
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe_ratio(rets),
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar_ratio(rets, max_dd),
        "turnover_pct": turnover(portfolio_df),
    }

    if benchmark_df is not None:
        metrics["information_ratio"] = information_ratio(
            rets, benchmark_df["portfolio_return"]
        )

    return metrics
