from typing import Dict

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def directional_accuracy(pred_returns: np.ndarray, true_returns: np.ndarray) -> float:
    mask = true_returns != 0
    if mask.sum() == 0:
        return 0.0
    matches = (np.sign(pred_returns[mask]) == np.sign(true_returns[mask])).sum()
    return float(matches / mask.sum())


def mse_volatility(pred_vol: np.ndarray, true_vol: np.ndarray) -> float:
    return float(mean_squared_error(true_vol, pred_vol))


def mae_returns(pred_ret: np.ndarray, true_ret: np.ndarray) -> float:
    return float(mean_absolute_error(true_ret, pred_ret))


def r2_score_returns(pred_ret: np.ndarray, true_ret: np.ndarray) -> float:
    return float(r2_score(true_ret, pred_ret))


@torch.no_grad()
def compute_all_prediction_metrics(model, loader, device) -> Dict:
    model.eval()

    all_pred_ret, all_true_ret = [], []
    all_pred_vol, all_true_vol = [], []

    for batch in loader:
        batch = batch.to(device)
        preds = model(batch)

        batch_targets = batch.targets if hasattr(batch, "targets") else batch[1]
        if isinstance(batch_targets, dict):
            t_ret = batch_targets["return"]
            t_vol = batch_targets["volatility"]
        else:
            t_ret = batch_targets[:, 1] if batch_targets.dim() > 1 else batch_targets
            t_vol = torch.zeros_like(t_ret)

        all_pred_ret.append(preds["return"].cpu().numpy())
        all_pred_vol.append(preds["volatility"].cpu().numpy())
        all_true_ret.append(t_ret.cpu().numpy())
        all_true_vol.append(t_vol.cpu().numpy())

    pred_ret = np.concatenate(all_pred_ret)
    true_ret = np.concatenate(all_true_ret)
    pred_vol = np.concatenate(all_pred_vol)
    true_vol = np.concatenate(all_true_vol)

    return {
        "directional_accuracy": directional_accuracy(pred_ret, true_ret),
        "mae_returns": mae_returns(pred_ret, true_ret),
        "r2_score_returns": r2_score_returns(pred_ret, true_ret),
        "mse_volatility": mse_volatility(pred_vol, true_vol),
    }
