from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, HeteroConv
import torch.nn.functional as F

from src.models.temporal_encoder import TemporalNodeEncoder
from src.models.evolve_gcn import EvolveGCNLayer

class HTGAT(nn.Module):
    """
    Heterogeneous Temporal Graph Attention Network.
    Matches the test_model.py contract and implements §6.2.
    """

    def __init__(
        self,
        node_features: int,
        hidden_dim: int,
        out_dim: int,
        num_heads: int,
        dropout: float,
        edge_types: List[Tuple[str, str, str]],
        num_layers: int = 3,
    ):
        super().__init__()
        
        # Ensure 63-day edge type is covered
        if ("stock", "correlates_with_63d", "stock") not in edge_types:
            edge_types.append(("stock", "correlates_with_63d", "stock"))
            
        self.num_layers = num_layers

        # 1. Temporal Node Encoder (handles 3D sequential input)
        self.temporal_encoder = TemporalNodeEncoder(
            in_channels=node_features, 
            hidden_channels=hidden_dim, 
            window_size=25
        )

        self.evolve_layers = nn.ModuleList()
        self.evolve_weights = nn.ParameterList()
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()

        for i in range(num_layers):
            # Output of TemporalNodeEncoder is always hidden_dim
            in_dim = hidden_dim  
            current_out_dim = out_dim if i == num_layers - 1 else hidden_dim

            # Initialize an evolvable weight matrix for this layer
            w_init = nn.Parameter(torch.randn(in_dim, in_dim))
            nn.init.xavier_uniform_(w_init)
            self.evolve_weights.append(w_init)
            
            # EvolveGCN layer to evolve the above matrix
            self.evolve_layers.append(EvolveGCNLayer(node_emb_dim=in_dim, weight_dim=in_dim * in_dim))

            # Build specific GATConv for each edge type (Heterogeneous Attention)
            conv_dict = {}
            for edge_type in edge_types:
                conv_dict[edge_type] = GATConv(
                    in_channels=in_dim,
                    out_channels=current_out_dim,
                    heads=num_heads,
                    concat=False,  # We average the heads
                    dropout=dropout,
                    edge_dim=1,  # Edge attributes exist
                    add_self_loops=False,
                )

            self.convs.append(HeteroConv(conv_dict, aggr="mean"))
            self.layer_norms.append(nn.LayerNorm(current_out_dim))

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        # 1. Temporal Encoding
        # x_dict["stock"] has shape [num_nodes, window_size, features]
        x_stock = self.temporal_encoder(x_dict["stock"])
        x_dict["stock"] = x_stock
        
        hidden_states = []

        # 2. EvolveGCN + HeteroConv
        for i in range(self.num_layers):
            # Evolve the weight matrix using GRU
            w_evolved = self.evolve_layers[i](x_dict["stock"], self.evolve_weights[i])
            
            # Apply evolved transformation matrix
            x_dict["stock"] = x_dict["stock"] @ w_evolved
            
            # Pass through GAT layers
            x_dict = self.convs[i](
                x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict
            )
            x_stock = x_dict["stock"]

            x_stock = self.elu(x_stock)
            x_stock = self.layer_norms[i](x_stock)
            x_stock = F.dropout(x_stock, p=self.dropout.p, training=self.training)

            x_dict["stock"] = x_stock
            hidden_states.append(x_stock)

        return {"embedding": x_dict["stock"], "hidden_states": hidden_states}
