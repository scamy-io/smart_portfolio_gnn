from typing import Dict, List, Tuple

import pandas as pd
from torch_geometric.data import HeteroData


class RebalanceTriggerChecker:
    def __init__(self, config: Dict):
        self.config = config

    def check_concentration(self, metrics: Dict) -> Tuple[bool, str]:
        target_hhi = self.config.get("risk", {}).get("target_hhi", 0.05)
        alert_thresh = self.config.get("risk", {}).get(
            "concentration_alert_threshold", 0.20
        )

        if metrics.get("embedding_hhi", 0.0) > target_hhi:
            return True, "Embedding HHI exceeded"

        cluster_info = metrics.get("clusters", {})
        if cluster_info.get("total_flagged_weight", 0.0) > alert_thresh:
            return True, "Hidden cluster concentration"

        return False, ""

    def check_shock_warning(self, sim_results: Dict) -> Tuple[bool, str]:
        hist_cvar = self.config.get("risk", {}).get("historical_avg_cvar", 0.02)

        for res in sim_results.get("scenarios", []):
            if res.get("portfolio_cvar", 0.0) > 2 * hist_cvar:
                return True, "CVaR spike detected in scenario: " + res["scenario"]

        return False, ""

    def check_correlation_breakdown(
        self, current_graph: HeteroData, prev_graph: HeteroData
    ) -> Tuple[bool, str]:
        if prev_graph is None:
            return False, ""

        et = ("stock", "correlates_with", "stock")
        if et in current_graph.edge_types and et in prev_graph.edge_types:
            cur_ea = current_graph[et].edge_attr
            prev_ea = prev_graph[et].edge_attr

            if (
                cur_ea is not None
                and prev_ea is not None
                and len(cur_ea) > 0
                and len(prev_ea) > 0
            ):
                cur_mean = cur_ea.mean().item()
                prev_mean = prev_ea.mean().item()

                if prev_mean != 0:
                    change = abs(cur_mean - prev_mean) / abs(prev_mean)
                    if change > 0.40:
                        return True, f"Correlation regime shift ({change:.1%} change)"

        return False, ""

    def check_scheduled(self, date: str, frequency: str = "weekly") -> bool:
        dt = pd.to_datetime(date)
        if frequency == "weekly":
            return dt.weekday() == 4
        return False

    def evaluate_all(
        self,
        date: str,
        metrics: Dict,
        sim_results: Dict,
        current_graph: HeteroData,
        prev_graph: HeteroData = None,
    ) -> List[Dict]:
        alerts = []

        conc, msg = self.check_concentration(metrics)
        if conc:
            alerts.append(
                {"trigger": "concentration", "severity": "high", "message": msg}
            )

        shock, msg = self.check_shock_warning(sim_results)
        if shock:
            alerts.append({"trigger": "shock", "severity": "high", "message": msg})

        corr, msg = self.check_correlation_breakdown(current_graph, prev_graph)
        if corr:
            alerts.append(
                {
                    "trigger": "correlation_breakdown",
                    "severity": "medium",
                    "message": msg,
                }
            )

        if self.check_scheduled(date):
            alerts.append(
                {
                    "trigger": "scheduled",
                    "severity": "low",
                    "message": "Scheduled rebalance",
                }
            )

        return alerts
