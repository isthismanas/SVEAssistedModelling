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

from molsim.data import DatasetManager, QM9TargetAdapter
from molsim.models import GCNRegressor
from molsim.training import RegressionTrainer, TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 2 QM9 baseline model.")
    parser.add_argument("--target", type=str, default="gap", help="QM9 target name.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for training.",
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
    raw_ds = manager.load("qm9")

    n = min(args.max_samples, len(raw_ds))
    adapter = QM9TargetAdapter(target_name=args.target)
    items = [adapter.apply(raw_ds[i]) for i in range(n)]
    random.shuffle(items)

    train_data, val_data, test_data = split_three_way(items)

    model = GCNRegressor(in_channels=items[0].x.shape[-1])
    trainer = RegressionTrainer(
        model=model,
        config=TrainingConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            device=args.device,
        ),
    )

    history = trainer.fit(train_data, val_data)

    from torch_geometric.loader import DataLoader as GeoDataLoader

    test_loader = GeoDataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    test_metrics = trainer.evaluate(test_loader)

    artifact_dir = PROJECT_ROOT / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_dir / f"baseline_qm9_{args.target}.json"
    out = {
        "target": args.target,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_samples": n,
        "history": history,
        "test_metrics": test_metrics,
    }
    out_path.write_text(json.dumps(out, indent=2) + "\n")

    print(f"Saved baseline artifact: {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
