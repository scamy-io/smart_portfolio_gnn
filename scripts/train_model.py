import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(str(Path(__file__).parent.parent))

from src.graph_builder.temporal_graph import TemporalGraphDataset
from src.models.htgat import HTGAT
from src.models.model_utils import PortfolioLoss, evaluate, train_epoch
from src.models.prediction_heads import MultiTaskPredictionHeads


class FullModel(nn.Module):
    def __init__(self, metadata, in_channels, hidden_channels=64):
        super().__init__()
        self.htgat = HTGAT(
            node_features=in_channels,
            hidden_dim=hidden_channels,
            out_dim=hidden_channels,
            num_heads=4,
            dropout=0.2,
            edge_types=metadata[1],
        )
        self.heads = MultiTaskPredictionHeads(hidden_dim=hidden_channels)

    def forward(self, batch_data):
        out = self.htgat(
            batch_data.x_dict, batch_data.edge_index_dict, batch_data.edge_attr_dict
        )
        return self.heads(out["embedding"])


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def main():
    parser = argparse.ArgumentParser(description="Train HTGAT Model")
    parser.add_argument("--config", type=Path, default=Path("configs/data_config.yaml"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    torch.manual_seed(42)
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    edge_paths = {
        "correlates_with": Path("data/processed/edges/correlation_edges.parquet"),
        "sentiment_co_mention": Path("data/processed/edges/sentiment_edges.parquet"),
        "supplies": Path("data/processed/edges/supply_chain_edges_processed.parquet"),
        "same_sector_as": Path("data/processed/edges/sector_edges.parquet"),
        "fundamentally_similar_to": Path(
            "data/processed/edges/fundamental_edges.parquet"
        ),
    }

    dataset = TemporalGraphDataset(
        graph_snapshot_dir=Path("data/processed/graph_snapshots"),
        node_features_path=Path("data/processed/node_features.parquet"),
        edge_paths=edge_paths,
    )

    if len(dataset) == 0:
        logger.error("No graph snapshots found. Please run build_graphs.py first.")
        return

    train_loader, val_loader, test_loader = dataset.get_loaders(batch_size=32)

    sample_data = dataset[0]
    in_channels = sample_data["stock"].x.shape[1]
    metadata = sample_data.metadata()

    model = FullModel(metadata, in_channels).to(device)
    logger.info(
        f"Initialized HTGAT with parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    criterion = PortfolioLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        logger.info(f"Resumed from {args.resume} at epoch {start_epoch}")

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 10

    train_losses = []
    val_losses = []

    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    logger.info("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics["total_loss"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logger.info(
            f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Dir Acc: {val_metrics['directional_accuracy']:.2%}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_val_loss,
                },
                out_dir / "best_htgat.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curves")
    plt.legend()
    plt.savefig(logs_dir / "training_curves.png")
    logger.info("Saved training curves to logs/training_curves.png")

    if len(test_loader) > 0:
        logger.info("Evaluating on Test Set...")
        model.load_state_dict(torch.load(out_dir / "best_htgat.pt")["model_state_dict"])
        test_metrics = evaluate(model, test_loader, criterion, device)

        logger.info("--- TEST METRICS ---")
        for k, v in test_metrics.items():
            if "accuracy" in k:
                logger.info(f"  {k}: {v:.2%}")
            else:
                logger.info(f"  {k}: {v:.4f}")
    else:
        logger.info("No test set available for evaluation.")


if __name__ == "__main__":
    main()
