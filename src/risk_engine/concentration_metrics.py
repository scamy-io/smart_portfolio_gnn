from typing import Dict, List

import numpy as np
import pandas as pd


class ConcentrationMetrics:
    def __init__(self, weights: pd.Series, embeddings: np.ndarray, tickers: List[str]):
        self.weights = weights
        self.embeddings = embeddings
        self.tickers = tickers

    def weight_hhi(self) -> float:
        return float(np.sum(self.weights.values**2))

    def effective_number_of_bets_weight(self) -> float:
        hhi = self.weight_hhi()
        return 1.0 / hhi if hhi > 0 else float("inf")

    def embedding_hhi(self) -> float:
        norms = np.linalg.norm(self.embeddings, axis=1)
        w = self.weights.values
        weighted_norms = w * norms
        total_weighted_norm = np.sum(weighted_norms)
        if total_weighted_norm == 0:
            return 0.0
        m_i = weighted_norms / total_weighted_norm
        held_mask = w > 0
        return float(np.sum(m_i[held_mask] ** 2))

    def cross_asset_embedding_correlation(self) -> pd.DataFrame:
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        z_norm = self.embeddings / norms
        cos_sim = z_norm @ z_norm.T
        return pd.DataFrame(cos_sim, index=self.tickers, columns=self.tickers)

    def graph_centrality_concentration(self, centrality_scores: pd.Series) -> float:
        c_norm = centrality_scores / centrality_scores.sum()
        w = self.weights.reindex(c_norm.index).fillna(0)
        return float(np.sum((c_norm.values * w.values) ** 2))

    def effective_number_of_bets_embedding(self) -> float:
        z = self.embeddings
        z_mean = np.mean(z, axis=0, keepdims=True)
        z_centered = z - z_mean
        cov = (z_centered.T @ z_centered) / max(z.shape[0] - 1, 1)

        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 0)

        sum_eig = np.sum(eigvals)
        sum_sq_eig = np.sum(eigvals**2)
        if sum_sq_eig == 0:
            return 0.0
        return float((sum_eig**2) / sum_sq_eig)

    def compute_all(self, centrality_scores: pd.Series = None) -> Dict[str, float]:
        res = {
            "weight_hhi": self.weight_hhi(),
            "enb_weight": self.effective_number_of_bets_weight(),
            "embedding_hhi": self.embedding_hhi(),
            "enb_embedding": self.effective_number_of_bets_embedding(),
        }
        if centrality_scores is not None:
            res["graph_centrality_concentration"] = self.graph_centrality_concentration(
                centrality_scores
            )
        return res
