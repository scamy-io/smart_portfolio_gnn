import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from data_loader import DashboardDataLoader

# Initialize Data Loader
dl = DashboardDataLoader()

st.set_page_config(page_title="Live Portfolio Risk Dashboard", layout="wide")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    [
        "Portfolio Overview",
        "Graph Explorer",
        "Shock Simulator",
        "Backtest Results",
        "Rebalancing",
    ],
)

# Global actions
if st.sidebar.button("Refresh Data"):
    dl.refresh()
    st.rerun()

# ── PAGE 1: Portfolio Overview ──
if page == "Portfolio Overview":
    st.title("Live Portfolio Risk Dashboard")

    alerts_df = dl.load_alerts(1)
    if not alerts_df.empty and alerts_df.iloc[0].get("severity") == "high":
        st.error(f"High Severity Alert: {alerts_df.iloc[0].get('message', '')}")

    portfolio_df = dl.load_portfolio_weights()
    if portfolio_df.empty:
        st.warning("No portfolio data available.")
    else:
        # KPI Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Value", "$1,000,000", "0.5%")
        col2.metric("Sharpe YTD", "1.85", "0.02")
        col3.metric("Max DD", "-8.2%", "1.1%")

        # Calculate HHI (mock)
        hhi = (portfolio_df["weight"] ** 2).sum()
        enb = 1.0 / hhi if hhi > 0 else 0
        col4.metric("Current HHI", f"{hhi:.4f}")
        col5.metric("ENB", f"{enb:.2f}")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Sector Allocation")
            fig = px.pie(portfolio_df, values="weight", names="sector", hole=0.3)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Top 10 Holdings")
            top_10 = portfolio_df.nlargest(10, "weight")
            st.dataframe(
                top_10[
                    ["ticker", "weight", "predicted_return", "predicted_volatility"]
                ],
                use_container_width=True,
            )

# ── PAGE 2: Graph Explorer ──
elif page == "Graph Explorer":
    st.title("Graph Explorer")
    g = dl.load_latest_graph()

    if g is None:
        st.warning("No graph snapshot available.")
    else:
        edge_types = [et for et in g.edge_types]
        selected_edge_type = st.selectbox(
            "Select Edge Type to Visualize",
            ["correlates_with", "sentiment_co_mention", "same_sector_as"],
        )

        st.info(
            "Graph Visualization Placeholder: Imagine a beautiful NetworkX + Plotly 3D scatter plot here. Nodes colored by sector, edges representing "
            + selected_edge_type
        )
        # To actually render this, we'd build a networkx graph, get spring_layout, and plot as Plotly Scatter.
        # This is omitted for brevity but would use Plotly go.Scatter(mode='markers+lines')

# ── PAGE 3: Shock Simulator ──
elif page == "Shock Simulator":
    st.title("Shock Simulator")

    scenario = st.selectbox(
        "Scenario",
        [
            "Sector Demand Shock",
            "Supply Chain Failure",
            "Sentiment Contagion",
            "Liquidity Freeze",
            "Macro Regime Shift",
        ],
    )
    target = st.selectbox("Target Sector/Ticker", ["All", "Tech", "AAPL", "NVDA"])

    if st.button("Run Simulation"):
        with st.spinner("Running shock scenario..."):
            res = dl.load_shock_results(scenario)

            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Portfolio Return", f"{res['portfolio_return']:.2%}")
            col2.metric("Predicted Portfolio Vol", f"{res['portfolio_vol']:.2%}")
            col3.metric("Predicted Portfolio CVaR", f"{res['portfolio_cvar']:.2%}")

            st.subheader(f"Worst Hit Ticker: {res['worst_ticker']}")

            # Histogram mock
            st.subheader("Monte Carlo Loss Distribution")
            mock_losses = np.random.normal(
                loc=res["portfolio_return"], scale=0.02, size=1000
            )
            fig = px.histogram(x=mock_losses, nbins=50, labels={"x": "Return"})
            st.plotly_chart(fig, use_container_width=True)

# ── PAGE 4: Backtest Results ──
elif page == "Backtest Results":
    st.title("Backtest Results")
    summary = dl.load_backtest_summary()

    if not summary:
        st.warning("No backtest summary found.")
    else:
        # Mock cumulative returns plot
        dates = pd.date_range(start="2024-01-01", periods=100)
        strat_returns = np.cumsum(np.random.normal(0.001, 0.01, 100))
        bench_returns = np.cumsum(np.random.normal(0.0005, 0.01, 100))

        df = pd.DataFrame(
            {"Date": dates, "Strategy": strat_returns, "Benchmark": bench_returns}
        )
        fig = px.line(
            df, x="Date", y=["Strategy", "Benchmark"], title="Cumulative Returns"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Metrics Table
        st.subheader("Key Metrics")
        metrics_df = pd.DataFrame(
            [
                {
                    "Sharpe Ratio": summary.get("sharpe", 1.5),
                    "Max Drawdown": summary.get("max_drawdown", -0.1),
                    "Calmar Ratio": summary.get("calmar", 2.0),
                    "Turnover": summary.get("turnover", 0.5),
                }
            ]
        )
        st.dataframe(metrics_df, use_container_width=True)

# ── PAGE 5: Rebalancing ──
elif page == "Rebalancing":
    st.title("Rebalancing & Trade Execution")
    portfolio_df = dl.load_portfolio_weights()

    if portfolio_df.empty:
        st.warning("No portfolio data.")
    else:
        # Mock recommended weights
        portfolio_df["recommended_weight"] = portfolio_df["weight"] + np.random.normal(
            0, 0.01, len(portfolio_df)
        )
        portfolio_df["recommended_weight"] = portfolio_df["recommended_weight"].clip(
            0, 1
        )
        portfolio_df["recommended_weight"] /= portfolio_df["recommended_weight"].sum()

        portfolio_df["delta"] = (
            portfolio_df["recommended_weight"] - portfolio_df["weight"]
        )
        portfolio_df["direction"] = np.where(portfolio_df["delta"] > 0, "Buy", "Sell")

        trades = portfolio_df[portfolio_df["delta"].abs() > 0.001].copy()

        st.subheader("Recommended Trades")
        st.dataframe(
            trades[["ticker", "weight", "recommended_weight", "delta", "direction"]],
            use_container_width=True,
        )

        csv = trades.to_csv(index=False).encode("utf-8")
        st.download_button("Export Trades to CSV", csv, "trades.csv", "text/csv")
