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
        bt.dataset.ablation_config = {"no_sentiment": True}
        return self._run_backtester_with_config(bt)

    def run_no_supply_chain_edges(self) -> Dict:
        bt = copy.deepcopy(self.base_backtester)
        bt.dataset.ablation_config = {"no_supply": True}
        return self._run_backtester_with_config(bt)

    def run_static_graph(self) -> Dict:
        bt = copy.deepcopy(self.base_backtester)
        bt.dataset.ablation_config = {"static_graph": True}
        return self._run_backtester_with_config(bt)

    def run_no_macro_conditioning(self) -> Dict:
        bt = copy.deepcopy(self.base_backtester)
        bt.dataset.ablation_config = {"no_macro": True}
        return self._run_backtester_with_config(bt)

    def run_no_embedding_concentration_penalty(self) -> Dict:
        bt = copy.deepcopy(self.base_backtester)
        if "risk" not in bt.config:
            bt.config["risk"] = {}
        bt.config["risk"]["lambda_conc"] = 0.0
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
        results["no_embedding_concentration_penalty"] = self.run_no_embedding_concentration_penalty()

        print("Running ablation: no transaction costs...")
        results["no_costs"] = self.run_no_transaction_costs()

        df = pd.DataFrame(results)

        os.makedirs("reports", exist_ok=True)
        df.to_csv("reports/ablation_study.csv")

        return df
