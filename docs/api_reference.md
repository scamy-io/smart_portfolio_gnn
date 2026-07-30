# API Reference

## src.models.htgat.HTGAT
The core Heterogeneous Temporal Graph Attention Network implementation matching §6.2.

- **`__init__(node_features: int, hidden_dim: int, out_dim: int, num_heads: int, dropout: float, edge_types: List[Tuple])`**: Initializes the heterogeneous GAT model.
- **`forward(x_dict, edge_index_dict, edge_attr_dict)`**: Performs the forward pass over the graph snapshot.

## src.rebalancing.cost_aware_optimizer.CostAwareOptimizer
Implements the cost-aware, constraint-driven portfolio optimization algorithm (§9.1).

- **`__init__(expected_returns, cov_matrix, current_weights, ...)`**: Sets up the optimization constraints and objectives.
- **`optimize(gamma, lambda_conc, Sigma_gnn, beta)`**: Solves the portfolio weights maximizing risk-adjusted return minus transaction costs and concentration penalty. Falls back to equal-weight if infeasible.

## src.risk_engine.shock_simulator.ShockSimulator
Simulates macroeconomic and market shocks (§8).

- **`liquidity_freeze(graph, removal_pct)`**: Drops a percentage of correlation edges.
- **`sector_demand_shock(graph, target_sector_indices)`**: Drops intra-sector edges and boosts inter-sector correlations.
- **`supply_chain_failure(graph, target_node_idx)`**: Zeroes out features for a targeted node.
