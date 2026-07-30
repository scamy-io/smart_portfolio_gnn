import copy
import os
from typing import Dict

import pandas as pd

from src.evaluation.backtester import WalkForwardBacktester


class AblationStudy:
    def __init__(
        self,
        base_config: Dict,
        backtester: WalkForwardBacktester,
        start_date: str,
        end_date: str,
    ):
        self.base_config = base_config
        self.base_backtester = backtester
        self.start_date = start_date
        self.end_date = end_date

    def _run_backtester_with_config(self, modified_backtester) -> Dict:
        from src.evaluation.portfolio_metrics import compute_all_metrics

        df = modified_backtester.run(self.start_date, self.end_date)
        return compute_all_metrics(df)

    def run_no_sentiment_edges(self) -> Dict:
        bt = copy.deepcopy(self.base_backtester)
        bt.config["ablation"] = "no_sentiment"
        return self._run_backtester_with_config(bt)

    def run_no_supply_chain_edges(self) -> Dict:
        bt = copy.deepcopy(self.base_backtester)
        bt.config["ablation"] = "no_supply"
        return self._run_backtester_with_config(bt)

    def run_static_graph(self) -> Dict:
        bt = copy.deepcopy(self.base_backtester)
        bt.config["ablation"] = "static_graph"
        return self._run_backtester_with_config(bt)

    def run_no_macro_conditioning(self) -> Dict:
        bt = copy.deepcopy(self.base_backtester)
        bt.config["ablation"] = "no_macro"
        return self._run_backtester_with_config(bt)

    def run_weight_only_hhi(self) -> Dict:
        bt = copy.deepcopy(self.base_backtester)
        bt.config["ablation"] = "weight_only_hhi"
        return self._run_backtester_with_config(bt)

    def run_no_transaction_costs(self) -> Dict:
        bt = copy.deepcopy(self.base_backtester)
        bt.transaction_cost_rate = 0.0
        return self._run_backtester_with_config(bt)

    def run_all(self) -> pd.DataFrame:
        results = {}

        print("Running base model...")
        results["base"] = self._run_backtester_with_config(self.base_backtester)

        print("Running ablation: no sentiment...")
        results["no_sentiment"] = self.run_no_sentiment_edges()

        print("Running ablation: no supply chain...")
        results["no_supply"] = self.run_no_supply_chain_edges()

        print("Running ablation: static graph...")
        results["static"] = self.run_static_graph()

        print("Running ablation: no macro...")
        results["no_macro"] = self.run_no_macro_conditioning()

        print("Running ablation: weight only HHI...")
        results["weight_only_hhi"] = self.run_weight_only_hhi()

        print("Running ablation: no transaction costs...")
        results["no_costs"] = self.run_no_transaction_costs()

        df = pd.DataFrame(results)

        os.makedirs("reports", exist_ok=True)
        df.to_csv("reports/ablation_study.csv")

        return df
