from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


def verify_graphs():
    snapshot_dir = Path("data/processed/graph_snapshots")
    files = sorted(snapshot_dir.glob("*.pt"))
    print(f"Found {len(files)} graph snapshots")

    if not files:
        print("No graph snapshots found!")
        return

    # Check 1: Load first, middle, last
    for f in [files[0], files[len(files) // 2], files[-1]]:
        g = torch.load(
            f, weights_only=False
        )  # Adding weights_only=False to avoid warnings in newer torch
        print(f"\n=== {f.stem} ===")
        print(f"  Nodes: {g['stock'].x.shape}")
        print(f"  Node features NaN: {torch.isnan(g['stock'].x).sum().item()}")
        print(f"  Edge types: {g.edge_types}")
        for et in g.edge_types:
            ei = g[et].edge_index
            ea = g[et].edge_attr
            print(
                f"    {et}: {ei.shape[1]} edges, weight range [{ea.min():.3f}, {ea.max():.3f}]"
            )

            # Check no self-loops
            self_loops = (ei[0] == ei[1]).sum().item()
            assert self_loops == 0, f"Self-loops found in {et}!"

            # Check indices valid
            if ei.shape[1] > 0:
                assert ei.max() < g["stock"].x.shape[0], "Edge index out of bounds!"

    # Check 2: Temporal alignment
    dates = [f.stem for f in files]
    assert dates == sorted(dates), "Snapshots not in chronological order!"

    # Check 3: Edge type consistency
    first_g = torch.load(files[0], weights_only=False)
    expected_edge_types = set(first_g.edge_types)
    for f in files[::10]:  # sample every 10th
        g = torch.load(f, weights_only=False)
        assert (
            set(g.edge_types) == expected_edge_types
        ), f"Edge type mismatch at {f.stem}"

    # Check 4: Correlation edge weights in valid range
    corr_files = sorted(Path("data/processed/edges").glob("correlation_edges.parquet"))
    if corr_files:
        df = pd.read_parquet(corr_files[0])
        if not df.empty:
            assert (
                df["weight"].abs().max() <= 1.0001
            ), f"Correlation > 1.0! Max was {df['weight'].abs().max()}"
            assert (
                df["weight"].abs().min() >= 0.2999
            ), f"Correlation below threshold! Min was {df['weight'].abs().min()}"

    print("\n✅ ALL CHECKS PASSED")


if __name__ == "__main__":
    verify_graphs()
