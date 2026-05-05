from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from molsim.spatial import VoxelConfig, parse_mol2_structure, voxelize_positions


class Mol2VoxelDataset(Dataset):
    """Lazy mol2->voxel dataset for spatial autoencoder training."""

    def __init__(self, mol2_paths: Iterable[str | Path], voxel_config: VoxelConfig) -> None:
        self.paths = [Path(p).resolve() for p in mol2_paths]
        self.voxel_config = voxel_config

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        coords, atomic_nums, _ = parse_mol2_structure(path)
        voxel = voxelize_positions(coords, atomic_nums, self.voxel_config)
        return voxel


def list_mol2_files(mol2_dir: str | Path, max_samples: int | None = None) -> list[Path]:
    root = Path(mol2_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"mol2 directory not found: {root}")

    files = sorted(root.glob("*.mol2"))
    if max_samples is not None:
        return files[: max(0, int(max_samples))]
    return files
