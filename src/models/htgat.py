from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, HeteroConv


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
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()

        for i in range(num_layers):
            in_dim = node_features if i == 0 else hidden_dim
            current_out_dim = out_dim if i == num_layers - 1 else hidden_dim

            # Build specific GATConv for each edge type (Heterogeneous Attention)
            conv_dict = {}
            for edge_type in edge_types:
                conv_dict[edge_type] = GATConv(
                    in_channels=in_dim,
                    out_channels=current_out_dim,
                    heads=num_heads,
                    concat=False,  # We average the heads
                    dropout=dropout,
                    edge_dim=1,  # Edge attributes exist (correlation weight, sentiment score, etc.)
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

        was_training = self.training
        is_training = was_training and torch.is_grad_enabled()
        if was_training and not is_training:
            self.eval()

        hidden_states = []

        for i in range(self.num_layers):
            x_dict = self.convs[i](
                x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict
            )
            x_stock = x_dict["stock"]

            x_stock = self.elu(x_stock)
            x_stock = self.layer_norms[i](x_stock)
            # Use F.dropout so we have explicit control, although self.eval() should handle it
            import torch.nn.functional as F

            x_stock = F.dropout(x_stock, p=self.dropout.p, training=self.training)

            x_dict["stock"] = x_stock
            hidden_states.append(x_stock)

        if was_training and not is_training:
            self.train()

        return {"embedding": x_dict["stock"], "hidden_states": hidden_states}
