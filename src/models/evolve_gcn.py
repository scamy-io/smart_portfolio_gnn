import torch
import torch.nn as nn


class EvolveGCNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden = 16

        self.adapter = nn.Sequential(
            nn.Linear(1, self.hidden), nn.ReLU(), nn.Linear(self.hidden, out_channels)
        )
        # TODO: Replace with full EvolveGCN in Phase 2

    def forward(
        self, node_features: torch.Tensor, output_features: torch.Tensor
    ) -> torch.Tensor:
        vix_input = node_features[:, 0].mean().view(1, 1)
        scale = self.adapter(vix_input)
        return output_features * scale
