#!/usr/bin/env python3

"""Week 3 Model Sanity Check — verify forward/backward pass before full training."""

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))

from pathlib import Path

import numpy as np

from src.models.htgat import HTGAT
from src.models.model_utils import PortfolioLoss
from src.models.prediction_heads import MultiTaskPredictionHeads


def main():

    # ── Load one snapshot ──

    from src.graph_builder.temporal_graph import TemporalGraphDataset
    dataset = TemporalGraphDataset(
        graph_snapshot_dir=Path("data/processed/graph_snapshots"),
        node_features_path=Path("data/processed/node_features.parquet"),
        edge_paths={
            "correlates_with": Path("data/processed/edges/correlation_edges.parquet"),
            "sentiment_co_mention": Path("data/processed/edges/sentiment_edges.parquet"),
            "supplies": Path("data/processed/edges/supply_chain_edges_processed.parquet"),
            "same_sector_as": Path("data/processed/edges/sector_edges.parquet"),
            "fundamentally_similar_to": Path("data/processed/edges/fundamental_edges.parquet"),
        }
    )
    assert len(dataset) > 0, "No snapshots found!"
    g = dataset[0]
    metadata = g.metadata()
    num_features = g["stock"].x.shape[-1]
    num_nodes = g["stock"].x.shape[0]

    print(f"Loaded graph: {g.date if hasattr(g, 'date') else 'unknown'}")

    print(f"  Nodes: {num_nodes}, Features: {num_features}")

    print(f"  Edge types: {len(g.edge_types)}")

    # ── Init model ──

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HTGAT(
        node_features=num_features,
        hidden_dim=64,
        out_dim=64,
        num_heads=4,
        dropout=0.2,
        edge_types=metadata[1],
    ).to(device)

    heads = MultiTaskPredictionHeads(hidden_dim=64).to(device)

    criterion = PortfolioLoss(lambda_vol=1.0, lambda_ret=1.0, lambda_cvar=0.5).to(
        device
    )

    # Move graph to device

    g = g.to(device)

    x_dict = {"stock": g["stock"].x}

    edge_index_dict = {et: g[et].edge_index for et in g.edge_types}

    edge_attr_dict = {et: g[et].edge_attr for et in g.edge_types}

    # ── Forward pass ──

    out = model(x_dict, edge_index_dict, edge_attr_dict)

    z = out["embedding"]

    print(f"\n✓ Forward pass OK | embedding shape: {z.shape}")

    assert z.shape == (num_nodes, 64), f"Expected ({num_nodes}, 64), got {z.shape}"

    preds = heads(z)

    print(f"✓ Prediction heads OK")

    print(
        f"  volatility: {preds['volatility'].shape} | min={preds['volatility'].min():.4f}"
    )

    print(f"  return:     {preds['return'].shape}")

    print(f"  cvar:       {preds['cvar'].shape}")

    # Volatility must be positive (softplus check)

    assert (preds["volatility"] >= 0).all(), "Volatility predictions are negative!"

    # ── Fake targets ──

    target_vol = torch.rand(num_nodes, device=device) * 0.5  # 0-50% vol

    target_ret = torch.randn(num_nodes, device=device) * 0.02  # -2% to +2%

    target_cvar = -torch.rand(num_nodes, device=device) * 0.05  # negative CVaR

    # ── Backward pass ──

    targets = {"volatility": target_vol, "return": target_ret, "cvar": target_cvar}
    loss = criterion(preds, targets)
    print(f"  Loss: {loss.item():.4f}")

    loss.backward()

    print(f"\n✓ Backward pass OK | total_loss: {loss.item():.6f}")

    # ── Gradient check ──

    has_grad = 0

    total_params = 0

    for name, p in list(model.named_parameters()) + list(heads.named_parameters()):

        if p.requires_grad:

            total_params += 1

            if p.grad is not None and p.grad.abs().sum() > 0:

                has_grad += 1

            else:

                print(f"  ⚠️ No gradient: {name}")

    print(
        f"\n✓ Gradients: {has_grad}/{total_params} parameters have non-zero gradients"
    )

    assert has_grad > 0, "No parameters receiving gradients!"

    # ── Parameter count ──

    total = sum(p.numel() for p in model.parameters()) + sum(
        p.numel() for p in heads.parameters()
    )

    print(f"\n✓ Total parameters: {total:,} (~{total/1e6:.2f}M)")

    print("\n🎉 WEEK 3 VERIFICATION PASSED — safe to train")


if __name__ == "__main__":

    main()
