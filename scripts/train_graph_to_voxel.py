#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molsim.data import DatasetManager
from molsim.models.graph_to_voxel import GraphToVoxelNet
from molsim.spatial import VoxelConfig
from molsim.training.voxel import VoxelTrainer, VoxelTrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train graph-to-voxel model on QM9 (optional mol2 supervision).")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--resolution", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--mol2-dir",
        type=str,
        default="",
        help="Optional directory containing <qm9_name>.mol2 files for target supervision.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "graph_to_voxel_qm9.pt"),
        help="Path to save trained graph-to-voxel model checkpoint.",
    )
    parser.add_argument(
        "--artifact-path",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "graph_to_voxel_qm9.json"),
        help="Path to save training artifact JSON.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_three_way(items: list, train_frac: float = 0.8, val_frac: float = 0.1):
    n = len(items)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]
    return train, val, test


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    manager = DatasetManager(project_root=PROJECT_ROOT)
    dataset = manager.load("qm9")

    n = min(args.max_samples, len(dataset))
    items = [dataset[i] for i in range(n)]
    random.shuffle(items)

    train_data, val_data, test_data = split_three_way(items)

    in_channels = int(items[0].x.shape[-1])
    model = GraphToVoxelNet(
        in_channels=in_channels,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        grid_size=args.grid_size,
        dropout=args.dropout,
    )
    voxel_cfg = VoxelConfig(
        grid_size=args.grid_size,
        resolution=args.resolution,
        sigma=args.sigma,
        use_atomic_weights=True,
    )

    trainer = VoxelTrainer(
        model=model,
        train_config=VoxelTrainingConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            device=args.device,
        ),
        voxel_config=voxel_cfg,
        mol2_dir=args.mol2_dir if args.mol2_dir else None,
    )

    history = trainer.fit(train_data, val_data)

    from torch_geometric.loader import DataLoader as GeoDataLoader

    test_loader = GeoDataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    test_metrics = trainer.evaluate(test_loader)

    checkpoint_path = Path(args.checkpoint_path).resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "in_channels": in_channels,
                "hidden_dim": args.hidden_dim,
                "latent_dim": args.latent_dim,
                "grid_size": args.grid_size,
                "dropout": args.dropout,
            },
            "voxel_config": {
                "grid_size": args.grid_size,
                "resolution": args.resolution,
                "sigma": args.sigma,
                "use_atomic_weights": True,
            },
            "dataset": {
                "dataset_id": "qm9",
                "max_samples": n,
            },
        },
        checkpoint_path,
    )

    out_path = Path(args.artifact_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_samples": n,
        "grid_size": args.grid_size,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "dropout": args.dropout,
        "resolution": args.resolution,
        "sigma": args.sigma,
        "mol2_dir": str(Path(args.mol2_dir).resolve()) if args.mol2_dir else None,
        "history": history,
        "test_metrics": test_metrics,
        "checkpoint_path": str(checkpoint_path),
    }
    out_path.write_text(json.dumps(out, indent=2) + "\n")

    print(f"Saved voxel checkpoint: {checkpoint_path}")
    print(f"Saved voxel artifact: {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
