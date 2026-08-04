import torch
import torch.nn as nn


class TemporalNodeEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        window_size: int = 25,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.window_size = window_size

        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU()
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, window_size, hidden_channels))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=num_heads,
            dim_feedforward=hidden_channels * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x_history: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x_history)
        x = x + self.pos_embedding
        out = self.transformer(x)
        return out[:, -1, :]
