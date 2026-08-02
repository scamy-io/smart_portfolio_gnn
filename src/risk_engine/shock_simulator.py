import copy
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData


class ShockSimulator:
    def __init__(self, model: nn.Module, device: str, shock_horizon: int = 21):
        self.model = model.eval()
        self.device = device
        self.shock_horizon = shock_horizon

    def sector_demand_shock(
        self,
        graph: HeteroData,
        target_sector_indices: List[int],
        removal_pct: float = 0.20,
    ) -> HeteroData:
        g = copy.deepcopy(graph)
        target_set = set(target_sector_indices)

        for et in [
            ("stock", "correlates_with", "stock"),
            ("stock", "same_sector_as", "stock"),
        ]:
            if et not in g.edge_types:
                continue
            ei = g[et].edge_index
            ea = g[et].edge_attr

            intra_mask = torch.tensor(
                [
                    (src.item() in target_set and tgt.item() in target_set)
                    for src, tgt in ei.t()
                ]
            )
            inter_mask = torch.tensor(
                [
                    (src.item() in target_set) ^ (tgt.item() in target_set)
                    for src, tgt in ei.t()
                ]
            )

            if intra_mask.any():
                keep_mask = torch.rand(intra_mask.sum()) > removal_pct
                new_ei = []
                new_ea = []
                intra_idx = 0
                for i in range(ei.shape[1]):
                    if intra_mask[i]:
                        if keep_mask[intra_idx]:
                            new_ei.append(ei[:, i])
                            new_ea.append(ea[i])
                        intra_idx += 1
                    else:
                        new_ei.append(ei[:, i])
                        if inter_mask[i] and et[1] == "correlates_with":
                            new_ea.append(ea[i] * 1.3)
                        else:
                            new_ea.append(ea[i])
                if new_ei:
                    g[et].edge_index = torch.stack(new_ei, dim=1)
                    g[et].edge_attr = torch.stack(new_ea)
                else:
                    g[et].edge_index = torch.empty((2, 0), dtype=torch.long)
                    g[et].edge_attr = torch.empty((0, 1), dtype=torch.float32)
        return g

    def supply_chain_failure(
        self, graph: HeteroData, target_node_idx: int
    ) -> HeteroData:
        g = copy.deepcopy(graph)
        g["stock"].x[target_node_idx] = 0.0

        for et in g.edge_types:
            ei = g[et].edge_index
            mask = (ei[0] != target_node_idx) & (ei[1] != target_node_idx)
            g[et].edge_index = ei[:, mask]
            if g[et].edge_attr is not None:
                g[et].edge_attr = g[et].edge_attr[mask]
        return g

    def sentiment_contagion(
        self,
        graph: HeteroData,
        target_sector_indices: List[int],
        noise_intensity: float = 0.2,
    ) -> HeteroData:
        g = copy.deepcopy(graph)
        for idx in target_sector_indices:
            noise = torch.randn_like(g["stock"].x[idx]) * noise_intensity
            g["stock"].x[idx] += noise

        for et in [
            ("stock", "sentiment_co_mention", "stock"),
            ("stock", "sentiment_spillover", "stock"),
        ]:
            if et in g.edge_types and g[et].edge_attr is not None:
                g[et].edge_attr = g[et].edge_attr * 1.5
        return g

    def liquidity_freeze(
        self, graph: HeteroData, removal_pct: float = 0.20
    ) -> HeteroData:
        g = copy.deepcopy(graph)
        et = ("stock", "correlates_with", "stock")
        if et in g.edge_types:
            ei = g[et].edge_index
            ea = g[et].edge_attr
            num_edges = ei.shape[1]
            if num_edges > 0:
                keep = torch.rand(num_edges) > removal_pct
                g[et].edge_index = ei[:, keep]
                g[et].edge_attr = ea[keep]
        return g

    def macro_regime_shift(
        self, graph: HeteroData, vix_multiplier: float = 2.0
    ) -> HeteroData:
        g = copy.deepcopy(graph)
        et = ("stock", "correlates_with", "stock")
        if et in g.edge_types and g[et].edge_attr is not None:
            g[et].edge_attr = g[et].edge_attr / vix_multiplier
        return g

    @torch.no_grad()
    def run_scenario(
        self,
        graph: HeteroData,
        scenario: str,
        tickers: List[str],
        weights: pd.Series,
        **kwargs
    ) -> Dict:
        if scenario == "sector_demand_shock":
            g = self.sector_demand_shock(graph, kwargs.get("target_indices", []))
        elif scenario == "supply_chain_failure":
            g = self.supply_chain_failure(graph, kwargs.get("target_idx", 0))
        elif scenario == "sentiment_contagion":
            g = self.sentiment_contagion(graph, kwargs.get("target_indices", []))
        elif scenario == "liquidity_freeze":
            g = self.liquidity_freeze(graph)
        elif scenario == "macro_regime_shift":
            g = self.macro_regime_shift(graph)
        else:
            g = graph

        g = g.to(self.device)

        preds = self.model(g)

        w_tensor = torch.tensor(
            weights.reindex(tickers).fillna(0).values,
            dtype=torch.float32,
            device=self.device,
        )

        pred_ret = preds["return"]
        port_ret = (w_tensor * pred_ret).sum().item()

        if hasattr(self.model, "htgat"):
            x_dict = {"stock": g["stock"].x}
            edge_index_dict = {et: g[et].edge_index for et in g.edge_types}
            edge_attr_dict = {et: g[et].edge_attr for et in g.edge_types}
            htgat_out = self.model.htgat(x_dict, edge_index_dict, edge_attr_dict)
            z = htgat_out["embedding"]
        else:
            z = preds["embedding"]

        z_norm = torch.nn.functional.normalize(z, p=2, dim=1)
        cov = torch.matmul(z_norm, z_norm.t())
        port_vol = torch.sqrt(
            torch.clamp(torch.matmul(w_tensor, torch.matmul(cov, w_tensor)), min=0.0)
        ).item()

        port_cvar = (w_tensor * preds["cvar"]).sum().item()
        worst_idx = torch.argmin(pred_ret).item()
        worst_ticker = tickers[worst_idx]

        if hasattr(self.model, "htgat"):
            z_base = self.model.htgat(
                {"stock": graph["stock"].x.to(self.device)},
                {et: graph[et].edge_index.to(self.device) for et in graph.edge_types},
                {et: graph[et].edge_attr.to(self.device) for et in graph.edge_types},
            )["embedding"]
        else:
            z_base = self.model(graph.to(self.device))["embedding"]

        # Simulate Geometric Brownian Motion to estimate Max Drawdown and Recovery Time
        horizon = self.shock_horizon
        dt = 1.0 / 252.0
        # Generate paths
        torch.manual_seed(42)
        random_shocks = torch.randn(horizon)
        # Assuming port_ret is annualized expected return, port_vol is annualized volatility
        drift = (port_ret - 0.5 * port_vol ** 2) * dt
        diffusion = port_vol * np.sqrt(dt) * random_shocks
        log_returns = drift + diffusion
        cum_returns = torch.exp(torch.cumsum(log_returns, dim=0))
        
        # Max Drawdown
        running_max = torch.cummax(cum_returns, dim=0)[0]
        drawdowns = (cum_returns - running_max) / running_max
        max_drawdown = float(torch.min(drawdowns).item())
        
        # Recovery Time
        # The trough is where max drawdown occurs
        trough_idx = torch.argmin(drawdowns).item()
        recovery_time = np.inf
        # Find first day after trough where cum_returns crosses running_max at trough
        if trough_idx < horizon - 1:
            trough_val = running_max[trough_idx]
            for t in range(trough_idx + 1, horizon):
                if cum_returns[t] >= trough_val:
                    recovery_time = float(t - trough_idx)
                    break
                    
        return {
            "scenario": scenario,
            "portfolio_return": float(port_ret),
            "portfolio_vol": float(port_vol),
            "portfolio_cvar": float(port_cvar),
            "max_drawdown": max_drawdown,
            "recovery_time_days": float(recovery_time),
            "worst_ticker": worst_ticker,
            "embedding_shift": float(torch.mean(torch.norm(z - z_base, dim=1)).item()),
        }

    def monte_carlo(
        self,
        graph: HeteroData,
        n_scenarios: int,
        tickers: List[str],
        weights: pd.Series,
    ) -> pd.DataFrame:
        scenarios = ["liquidity_freeze", "macro_regime_shift", "sector_demand_shock"]
        results = []
        for _ in range(n_scenarios):
            s = np.random.choice(scenarios)
            res = self.run_scenario(graph, s, tickers, weights, target_indices=[0, 1])
            results.append(res)

        df = pd.DataFrame(results)
        return df
