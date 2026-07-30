from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel


class SpectralClusterDetector:
    def __init__(self, n_clusters: int = 10, similarity_threshold: float = 0.7):
        self.n_clusters = n_clusters
        self.similarity_threshold = similarity_threshold

    def fit(self, embeddings: np.ndarray, tickers: List[str]) -> pd.DataFrame:
        n_samples = embeddings.shape[0]
        n_clusters = min(self.n_clusters, max(n_samples // 2, 2))

        affinity_matrix = rbf_kernel(embeddings, gamma=1.0)

        sc = SpectralClustering(
            n_clusters=n_clusters, affinity="precomputed", random_state=42
        )
        labels = sc.fit_predict(affinity_matrix)

        return pd.DataFrame(
            {"ticker": tickers, "cluster_id": labels, "embedding": list(embeddings)}
        )

    def detect_concentration_risk(self, weights: pd.Series, df: pd.DataFrame) -> Dict:
        flagged = []
        total_flagged_weight = 0.0

        embs = np.vstack(df["embedding"].values)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        z_norm = embs / norms
        cos_sim_matrix = z_norm @ z_norm.T

        for cluster_id, group in df.groupby("cluster_id"):
            cluster_tickers = group["ticker"].tolist()
            cluster_weight = weights.reindex(cluster_tickers).sum()

            idx = group.index.values
            if len(idx) > 1:
                sub_sim = cos_sim_matrix[np.ix_(idx, idx)]
                n_el = len(idx)
                avg_sim = (sub_sim.sum() - n_el) / (n_el * (n_el - 1))
            else:
                avg_sim = 1.0

            if cluster_weight > 0.30 and avg_sim > 0.70:
                flagged.append(
                    {
                        "cluster_id": int(cluster_id),
                        "weight": float(cluster_weight),
                        "avg_similarity": float(avg_sim),
                        "tickers": cluster_tickers,
                    }
                )
                total_flagged_weight += float(cluster_weight)

        return {
            "flagged_clusters": flagged,
            "total_flagged_weight": total_flagged_weight,
            "is_concentrated": len(flagged) > 0,
        }

    def visualize_clusters(
        self, df: pd.DataFrame, weights: pd.Series, date: str
    ) -> None:
        embs = np.vstack(df["embedding"].values)
        pca = PCA(n_components=2)
        embs_2d = pca.fit_transform(embs)

        sizes = weights.reindex(df["ticker"]).fillna(0.01).values * 1000

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            embs_2d[:, 0],
            embs_2d[:, 1],
            c=df["cluster_id"],
            s=sizes,
            cmap="tab10",
            alpha=0.7,
        )
        plt.colorbar(scatter, label="Cluster ID")
        plt.title(f"Concentration Clusters ({date})")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")

        out_dir = Path("reports")
        out_dir.mkdir(exist_ok=True)
        plt.savefig(out_dir / f"concentration_cluster_plot_{date}.png")
        plt.close()
