from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv

from src.models.evolve_gcn import EvolveGCNLayer
from src.models.temporal_encoder import TemporalNodeEncoder


class HTGAT(nn.Module):
    """Heterogeneous Temporal Graph Attention Network.

    Features gradient stabilization, residual connections, and defensive shape
    handling to prevent NaN losses and over-smoothing.
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

        # Copy edge_types to avoid mutating the caller's list
        self.edge_types = list(edge_types)
        if ("stock", "correlates_with_63d", "stock") not in self.edge_types:
            self.edge_types.append(("stock", "correlates_with_63d", "stock"))

        self.num_layers = num_layers
        self.dropout_rate = dropout

        # 1. Temporal Node Encoder (handles 3D sequential input)
        self.temporal_encoder = TemporalNodeEncoder(
            in_channels=node_features,
            hidden_channels=hidden_dim,
            window_size=25,
        )

        self.evolve_layers = nn.ModuleList()
        self.evolve_weights = nn.ParameterList()
        self.convs = nn.ModuleList()
        self.layer_norms_evolve = nn.ModuleList()
        self.layer_norms_gat = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()

        for i in range(num_layers):
            in_dim = hidden_dim
            current_out_dim = out_dim if i == num_layers - 1 else hidden_dim

            # Initialize an evolvable weight matrix for this layer
            w_init = nn.Parameter(torch.randn(in_dim, in_dim))
            nn.init.xavier_uniform_(w_init)
            self.evolve_weights.append(w_init)

            # EvolveGCN layer to evolve weight matrix dynamically
            # EvolveGCN layer to evolve weight matrix dynamically
            self.evolve_layers.append(
                EvolveGCNLayer(
                    node_emb_dim=in_dim, out_channels=in_dim
                )
            )
            self.layer_norms_evolve.append(nn.LayerNorm(in_dim))

            # Build specific GATConv for each edge type
            conv_dict = {}
            for edge_type in self.edge_types:
                conv_dict[edge_type] = GATConv(
                    in_channels=in_dim,
                    out_channels=current_out_dim,
                    heads=num_heads,
                    concat=False,  # Average attention heads
                    dropout=dropout,
                    edge_dim=1,
                    add_self_loops=False,
                )

            self.convs.append(HeteroConv(conv_dict, aggr="mean"))
            self.layer_norms_gat.append(nn.LayerNorm(current_out_dim))

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        # 1. Temporal Encoding: [num_nodes, 25, features] -> [num_nodes, hidden_dim]
        x_stock = self.temporal_encoder(x_dict["stock"])
        x_dict["stock"] = x_stock

        # Ensure edge attributes are 2D [num_edges, 1] defensively
        formatted_edge_attr = {}
        if edge_attr_dict is not None:
            for k, v in edge_attr_dict.items():
                if v is not None and v.dim() == 1:
                    formatted_edge_attr[k] = v.unsqueeze(-1)
                else:
                    formatted_edge_attr[k] = v
        else:
            formatted_edge_attr = edge_attr_dict

        hidden_states = []

        # 2. Message Passing Loop
        for i in range(self.num_layers):
            residual = x_dict["stock"]

            # Evolve weight matrix via GRU
            w_evolved = self.evolve_layers[i](
                x_dict["stock"], self.evolve_weights[i]
            )

            # Evolved transformation with LayerNorm to stabilize gradients
            x_transformed = x_dict["stock"] @ w_evolved
            x_dict["stock"] = self.layer_norms_evolve[i](x_transformed)

            # Heterogeneous Message Passing
            x_dict = self.convs[i](
                x_dict, edge_index_dict, edge_attr_dict=formatted_edge_attr
            )
            x_stock = x_dict["stock"]

            # Activation, LayerNorm, and Dropout
            x_stock = self.elu(x_stock)
            x_stock = self.layer_norms_gat[i](x_stock)
            x_stock = self.dropout(x_stock)

            # Apply Residual (Skip) connection if dimensions match
            if residual.shape == x_stock.shape:
                x_stock = x_stock + residual

            x_dict["stock"] = x_stock
            hidden_states.append(x_stock)

        return {"embedding": x_dict["stock"], "hidden_states": hidden_states}