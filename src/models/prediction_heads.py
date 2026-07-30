from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskPredictionHeads(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

        self.vol_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.ret_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.cvar_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        vol = self.vol_head(z).squeeze(-1)
        vol = F.softplus(vol)

        ret = self.ret_head(z).squeeze(-1)
        cvar = self.cvar_head(z).squeeze(-1)

        return {"volatility": vol, "return": ret, "cvar": cvar}
