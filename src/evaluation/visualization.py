from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_cumulative_returns(
    strategy_df: pd.DataFrame, benchmark_df: pd.DataFrame, output_path: Path
):
    plt.figure(figsize=(12, 6))

    s_ret = strategy_df["portfolio_return"]
    b_ret = benchmark_df["portfolio_return"]

    s_cum = (1 + s_ret).cumprod()
    b_cum = (1 + b_ret).cumprod()

    plt.plot(
        s_cum.index,
        s_cum.values,
        label="Strategy (HT-GAT)",
        color="#2e86ab",
        linewidth=2,
    )
    plt.plot(
        b_cum.index,
        b_cum.values,
        label="Benchmark (SPY)",
        color="#f18f01",
        linewidth=2,
        linestyle="--",
    )

    plt.title("Cumulative Returns: Strategy vs Benchmark")
    plt.ylabel("Cumulative Return (Base = 1.0)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_drawdown(strategy_df: pd.DataFrame, output_path: Path):
    plt.figure(figsize=(12, 4))
    dd = strategy_df["drawdown"] * 100

    plt.fill_between(dd.index, dd.values, 0, color="#d90429", alpha=0.5)
    plt.plot(dd.index, dd.values, color="#d90429", linewidth=1)

    plt.title("Portfolio Drawdown")
    plt.ylabel("Drawdown (%)")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_rolling_sharpe(strategy_df: pd.DataFrame, output_path: Path, window: int = 63):
    plt.figure(figsize=(12, 4))
    rets = strategy_df["portfolio_return"]

    rolling_mean = rets.rolling(window).mean()
    rolling_std = rets.rolling(window).std()

    rolling_sharpe = (rolling_mean / rolling_std) * (252**0.5)

    plt.plot(rolling_sharpe.index, rolling_sharpe.values, color="#02c39a", linewidth=2)
    plt.axhline(0, color="black", linestyle="--", alpha=0.5)

    plt.title(f"{window}-Day Rolling Sharpe Ratio")
    plt.ylabel("Sharpe Ratio")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_weight_heatmap(weights_history: pd.DataFrame, output_path: Path):
    if weights_history is None or weights_history.empty:
        return

    plt.figure(figsize=(12, 8))
    mean_weights = weights_history.mean().sort_values(ascending=False)
    top_cols = mean_weights.head(20).index

    sns.heatmap(weights_history[top_cols].T, cmap="viridis", xticklabels=False)
    plt.title("Portfolio Weight Allocations Over Time (Top 20)")
    plt.ylabel("Ticker")
    plt.xlabel("Time")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_ablation_bars(ablation_df: pd.DataFrame, metric: str, output_path: Path):
    if metric not in ablation_df.index:
        return

    plt.figure(figsize=(10, 6))
    data = ablation_df.loc[metric]

    colors = ["#2e86ab" if x == "base" else "#83d475" for x in data.index]
    data.plot(kind="bar", color=colors)

    plt.title(f"Ablation Study: {metric}")
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
