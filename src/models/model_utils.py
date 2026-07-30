from typing import Dict

import torch
import torch.nn as nn


class QuantileLoss(nn.Module):
    def __init__(self, quantile: float = 0.05):
        super().__init__()
        self.quantile = quantile

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = target - pred
        loss = torch.max(self.quantile * error, (self.quantile - 1.0) * error)
        return loss.mean()


class PortfolioLoss(nn.Module):
    def __init__(
        self,
        lambda_vol: float = 1.0,
        lambda_ret: float = 1.0,
        lambda_cvar: float = 1.0,
        lambda_reg: float = 1e-4,
    ):
        super().__init__()
        self.lambda_vol = lambda_vol
        self.lambda_ret = lambda_ret
        self.lambda_cvar = lambda_cvar
        self.lambda_reg = lambda_reg

        self.vol_loss = nn.MSELoss()
        self.ret_loss = nn.HuberLoss(delta=0.1)
        self.cvar_loss = QuantileLoss(quantile=0.05)

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        model: nn.Module = None,
    ) -> torch.Tensor:
        l_vol = self.vol_loss(preds["volatility"], targets["volatility"])
        l_ret = self.ret_loss(preds["return"], targets["return"])
        l_cvar = self.cvar_loss(preds["cvar"], targets["cvar"])

        total_loss = (
            self.lambda_vol * l_vol
            + self.lambda_ret * l_ret
            + self.lambda_cvar * l_cvar
        )

        if model is not None and self.lambda_reg > 0:
            l2_reg = sum(p.norm(2) for p in model.parameters())
            total_loss += self.lambda_reg * l2_reg

        return total_loss


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        batch_targets = {
            "volatility": batch.volatility,
            "return": batch.return_,
            "cvar": batch.cvar,
        }

        if isinstance(batch_targets, dict):
            batch_targets = {k: v.to(device) for k, v in batch_targets.items()}

        optimizer.zero_grad()

        preds = model(batch)
        loss = criterion(preds, batch_targets, model=model)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    vol_mse = 0.0
    ret_mae = 0.0
    cvar_mae = 0.0
    correct_dir = 0
    total_samples = 0

    ret_loss_fn = nn.L1Loss(reduction="sum")
    vol_loss_fn = nn.MSELoss(reduction="sum")
    cvar_loss_fn = nn.L1Loss(reduction="sum")

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch_targets = {
                "volatility": batch.volatility,
                "return": batch.return_,
                "cvar": batch.cvar,
            }

            if isinstance(batch_targets, dict):
                batch_targets = {k: v.to(device) for k, v in batch_targets.items()}

            preds = model(batch)
            loss = criterion(preds, batch_targets)
            total_loss += loss.item()

            N = preds["return"].size(0)
            total_samples += N

            vol_mse += vol_loss_fn(
                preds["volatility"], batch_targets["volatility"]
            ).item()
            ret_mae += ret_loss_fn(preds["return"], batch_targets["return"]).item()
            cvar_mae += cvar_loss_fn(preds["cvar"], batch_targets["cvar"]).item()

            sign_match = (
                (torch.sign(preds["return"]) == torch.sign(batch_targets["return"]))
                .sum()
                .item()
            )
            correct_dir += sign_match

    return {
        "total_loss": total_loss / len(loader),
        "vol_mse": vol_mse / total_samples,
        "ret_mae": ret_mae / total_samples,
        "cvar_mae": cvar_mae / total_samples,
        "directional_accuracy": correct_dir / total_samples,
    }
