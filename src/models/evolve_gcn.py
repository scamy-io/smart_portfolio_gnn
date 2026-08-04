import torch
import torch.nn as nn

class EvolveGCNLayer(nn.Module):
    """
    True EvolveGCN implementation.
    Evolves a weight matrix dynamically using a GRU based on graph-level summaries.
    """
    def __init__(self, node_emb_dim: int, weight_dim: int):
        super().__init__()
        self.node_emb_dim = node_emb_dim
        self.weight_dim = weight_dim
        
        # Input: Graph-level node summary (node_emb_dim)
        # Hidden State: Flattened weight matrix (weight_dim)
        self.gru = nn.GRUCell(input_size=node_emb_dim, hidden_size=weight_dim)

    def forward(self, node_features: torch.Tensor, weight_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [num_nodes, node_emb_dim]
            weight_matrix: [in_channels, out_channels]
        Returns:
            evolved_weight: [in_channels, out_channels]
        """
        # Graph-level summary via mean pooling
        graph_summary = node_features.mean(dim=0, keepdim=True)  # [1, node_emb_dim]
        
        original_shape = weight_matrix.shape
        flat_weight = weight_matrix.view(1, -1)  # [1, in_channels * out_channels]
        
        if self.gru.hidden_size != flat_weight.shape[1]:
            raise ValueError(f"GRU hidden_size ({self.gru.hidden_size}) must match flattened weight size ({flat_weight.shape[1]})")
            
        evolved_flat_weight = self.gru(graph_summary, flat_weight)
        return evolved_flat_weight.view(original_shape)
