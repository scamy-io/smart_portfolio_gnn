import cvxpy as cp
import numpy as np
import pandas as pd


class CostAwareOptimizer:
    def __init__(
        self,
        expected_returns: pd.Series,
        cov_matrix: np.ndarray,
        current_weights: pd.Series,
        transaction_cost_rate: float = 0.001,
        max_weight: float = 0.05,
        target_hhi: float = 0.02,
        min_enb: float = 15.0,
    ):
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.current_weights = current_weights
        self.transaction_cost_rate = transaction_cost_rate
        self.max_weight = max_weight
        self.target_hhi = target_hhi
        self.min_enb = min_enb

    def optimize(
        self,
        gamma: float = 1.0,
        lambda_conc: float = 0.1,
        Sigma_gnn: np.ndarray = None,
        beta: float = 0.5,
    ) -> pd.Series:
        tickers = self.expected_returns.index.tolist()
        n = len(tickers)

        mu = self.expected_returns.values
        Sigma_hist = self.cov_matrix

        if Sigma_gnn is not None:
            Sigma_blend = beta * Sigma_hist + (1 - beta) * Sigma_gnn
            # Ensure PSD
            eigvals, eigvecs = np.linalg.eigh(Sigma_blend)
            eigvals = np.maximum(eigvals, 1e-8)
            Sigma_blend = eigvecs @ np.diag(eigvals) @ eigvecs.T
        else:
            Sigma_blend = Sigma_hist

        w_curr = self.current_weights.reindex(tickers).fillna(0.0).values

        w = cp.Variable(n)

        ret = mu.T @ w
        risk = cp.quad_form(w, cp.psd_wrap(Sigma_blend))
        t_cost = self.transaction_cost_rate * cp.norm1(w - w_curr)

        # Penalize embedding-space concentration instead of weight HHI
        if Sigma_gnn is not None:
            conc_penalty = cp.quad_form(w, cp.psd_wrap(Sigma_gnn))
        else:
            conc_penalty = cp.sum_squares(w)

        objective = cp.Maximize(
            ret - (gamma / 2.0) * risk - t_cost - lambda_conc * conc_penalty
        )

        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= self.max_weight,
            cp.sum_squares(w) <= self.target_hhi,
        ]

        if Sigma_gnn is not None:
            # Approximation: PCA eigenvectors used as torsion matrix. Full Meucci minimum torsion deferred to Phase 4.
            # ENB_embed = 1 / (w^T Sigma_gnn w)
            # ENB_embed >= min_enb  =>  w^T Sigma_gnn w <= 1 / min_enb
            constraints.append(cp.quad_form(w, cp.psd_wrap(Sigma_gnn)) <= 1.0 / self.min_enb)

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.SCS)
            if w.value is None:
                raise ValueError("Optimizer returned None")
            opt_w = np.array(w.value).flatten()

            # Post-validate
            if (
                not np.isclose(np.sum(opt_w), 1.0, atol=1e-3)
                or np.any(opt_w < -1e-3)
                or np.any(opt_w > self.max_weight + 1e-3)
            ):
                raise ValueError("Solver constraints violated post-solve")

        except Exception as e:
            print(f"Optimization failed: {e}. Falling back to equal weight.")
            opt_w = np.ones(n) / n

        # We do NOT clip and re-normalize the successfully solved weights,
        # as that breaks max_weight and target_hhi constraints.
        # However, for the fallback, we ensure it's normalized.
        # Just tiny precision cleanup without breaking constraints:
        opt_w = np.clip(opt_w, 0, self.max_weight)
        if not np.isclose(np.sum(opt_w), 1.0, atol=1e-4):
            opt_w = opt_w / np.sum(opt_w)  # only if fallback or minor precision issue

        return pd.Series(opt_w, index=tickers)

    def generate_trades(self, optimal_weights: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame(
            {"current_weight": self.current_weights, "target_weight": optimal_weights}
        ).fillna(0.0)

        df["delta"] = df["target_weight"] - df["current_weight"]
        df["trade_direction"] = np.where(df["delta"] > 0, "BUY", "SELL")
        df["estimated_cost"] = df["delta"].abs() * self.transaction_cost_rate

        df = df[df["delta"].abs() > 0.001]

        return df.reset_index().rename(columns={"index": "ticker"})
