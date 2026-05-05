#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molsim.data import Mol2VoxelDataset, list_mol2_files
from molsim.models import VoxelAutoencoder
from molsim.spatial import VoxelConfig
from molsim.training import AutoencoderTrainingConfig, VoxelAutoencoderTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train voxel autoencoder from mol2 files for DegradeMaster integration.")
    parser.add_argument("--mol2-dir", type=str, default=str(PROJECT_ROOT / "data" / "QM9_mol2"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])

    parser.add_argument("--grid-size", type=int, default=24)
    parser.add_argument("--resolution", type=float, default=0.45)
    parser.add_argument("--sigma", type=float, default=0.5)

    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "voxel_autoencoder_qm9.pt"),
    )
    parser.add_argument(
        "--artifact-path",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "voxel_autoencoder_qm9.json"),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_three_way(items: list[Path], val_fraction: float, test_fraction: float) -> tuple[list[Path], list[Path], list[Path]]:
    if val_fraction < 0 or test_fraction < 0 or (val_fraction + test_fraction) >= 1:
        raise ValueError("val_fraction and test_fraction must be >=0 and sum to < 1")

    n = len(items)
    n_val = int(n * val_fraction)
    n_test = int(n * test_fraction)
    n_train = n - n_val - n_test

    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]
    return train, val, test


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    files = list_mol2_files(args.mol2_dir)
    if not files:
        raise FileNotFoundError(f"No mol2 files found in {Path(args.mol2_dir).resolve()}")

    n = min(len(files), args.max_samples)
    files = files[:n]
    random.shuffle(files)

    train_files, val_files, test_files = split_three_way(files, args.val_fraction, args.test_fraction)

    voxel_cfg = VoxelConfig(
        grid_size=args.grid_size,
        resolution=args.resolution,
        sigma=args.sigma,
        use_atomic_weights=True,
    )

    train_ds = Mol2VoxelDataset(train_files, voxel_cfg)
    val_ds = Mol2VoxelDataset(val_files, voxel_cfg)
    test_ds = Mol2VoxelDataset(test_files, voxel_cfg)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = VoxelAutoencoder(
        grid_size=args.grid_size,
        embedding_dim=args.embedding_dim,
        base_channels=args.base_channels,
        dropout=args.dropout,
    )
    trainer = VoxelAutoencoderTrainer(
        model=model,
        config=AutoencoderTrainingConfig(
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
            device=args.device,
        ),
    )

    history = trainer.fit(train_loader, val_loader)
    test_metrics = trainer.evaluate(test_loader)

    checkpoint_path = Path(args.checkpoint_path).resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "grid_size": args.grid_size,
                "embedding_dim": args.embedding_dim,
                "base_channels": args.base_channels,
                "dropout": args.dropout,
            },
            "voxel_config": {
                "grid_size": args.grid_size,
                "resolution": args.resolution,
                "sigma": args.sigma,
                "use_atomic_weights": True,
            },
        },
        checkpoint_path,
    )

    artifact = {
        "mol2_dir": str(Path(args.mol2_dir).resolve()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_samples": n,
        "splits": {
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        },
        "model": {
            "grid_size": args.grid_size,
            "embedding_dim": args.embedding_dim,
            "base_channels": args.base_channels,
            "dropout": args.dropout,
        },
        "voxel": {
            "resolution": args.resolution,
            "sigma": args.sigma,
            "use_atomic_weights": True,
        },
        "history": history,
        "test_metrics": test_metrics,
        "checkpoint_path": str(checkpoint_path),
    }

    artifact_path = Path(args.artifact_path).resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved artifact:   {artifact_path}")
    print(json.dumps({"test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
