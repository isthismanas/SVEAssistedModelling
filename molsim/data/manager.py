from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 42


class DatasetManager:
    def __init__(self, project_root: str | Path = ".") -> None:
        self.project_root = Path(project_root).resolve()

    def load(self, dataset_id: str):
        dataset_id = dataset_id.lower()

        if dataset_id == "qm9":
            from torch_geometric.datasets import QM9

            return QM9(root=str(self.project_root / "data" / "QM9"))

        if dataset_id == "zinc":
            from torch_geometric.datasets import ZINC

            return ZINC(root=str(self.project_root / "data" / "ZINC"), subset=True, split="train")

        raise NotImplementedError(f"Dataset loader not implemented for '{dataset_id}'")

    def split_indices(self, dataset_size: int, config: SplitConfig | None = None) -> dict[str, list[int]]:
        cfg = config or SplitConfig()

        total = cfg.train_fraction + cfg.val_fraction + cfg.test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Train/val/test fractions must sum to 1.0")

        g = torch.Generator().manual_seed(cfg.seed)
        perm = torch.randperm(dataset_size, generator=g).tolist()

        n_train = int(dataset_size * cfg.train_fraction)
        n_val = int(dataset_size * cfg.val_fraction)

        train_idx = perm[:n_train]
        val_idx = perm[n_train : n_train + n_val]
        test_idx = perm[n_train + n_val :]

        return {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }
