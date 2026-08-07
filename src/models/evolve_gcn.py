import torch
import torch.nn as nn


class EvolveGCNLayer(nn.Module):
    """Memory-Efficient EvolveGCN Layer (EvolveGCN-H style).

    Evolves a weight matrix dynamically using a GRU based on graph-level
    summaries. Operates row-wise in parallel to keep parameter complexity at
    O(d^2) instead of O(d^4).
    """

    def __init__(self, node_emb_dim: int, out_channels: int):
        super().__init__()
        self.node_emb_dim = node_emb_dim
        self.out_channels = out_channels

        # Input: Graph-level summary vector [node_emb_dim]
        # Hidden State: Individual row vector of the weight matrix [out_channels]
        # Batch Dimension: in_channels (number of rows in the weight matrix)
        self.gru = nn.GRUCell(input_size=node_emb_dim, hidden_size=out_channels)

    def forward(
        self, node_features: torch.Tensor, weight_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Args:

            node_features: [num_nodes, node_emb_dim] weight_matrix:
            [in_channels, out_channels]

        Returns:
            evolved_weight: [in_channels, out_channels]
        """
        # Defensive check for empty node tensors
        if node_features.numel() == 0:
            return weight_matrix

        # 1. Compute graph-level summary via mean pooling: shape [1, node_emb_dim]
        graph_summary = node_features.mean(dim=0, keepdim=True)

        # 2. Expand summary across all rows (in_channels) of the weight matrix
        # Shape: [in_channels, node_emb_dim]
        in_channels = weight_matrix.shape[0]
        summary_expanded = graph_summary.expand(in_channels, -1)

        # 3. Evolve weight matrix row-by-row in parallel
        # input: [in_channels, node_emb_dim], hidden: [in_channels, out_channels]
        evolved_weight = self.gru(summary_expanded, weight_matrix)

        return evolved_weight