from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VoxelConfig:
    grid_size: int = 16
    resolution: float = 0.5
    sigma: float = 0.5
    use_atomic_weights: bool = True


def voxelize_positions(
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor | None,
    config: VoxelConfig,
    center: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert atom coordinates into a dense voxel field using Gaussian splatting."""
    if positions.numel() == 0:
        return torch.zeros((1, config.grid_size, config.grid_size, config.grid_size), dtype=torch.float32)

    pos = positions.float()
    ctr = pos.mean(dim=0) if center is None else center.float()
    pos = pos - ctr

    half_extent = (config.grid_size * config.resolution) / 2.0
    coords = torch.linspace(-half_extent, half_extent, config.grid_size)
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    grid = torch.stack([gx, gy, gz], dim=-1).float()

    pos_exp = pos.view(-1, 1, 1, 1, 3)
    grid_exp = grid.unsqueeze(0)
    dist_sq = torch.sum((grid_exp - pos_exp) ** 2, dim=-1)

    if config.use_atomic_weights and atomic_numbers is not None:
        weights = atomic_numbers.float().view(-1, 1, 1, 1) / 6.0
    else:
        weights = 1.0

    gauss = weights * torch.exp(-dist_sq / (2.0 * (config.sigma ** 2)))
    voxel = torch.sum(gauss, dim=0)
    return voxel.unsqueeze(0)


def voxelize_data(data, config: VoxelConfig) -> torch.Tensor:
    pos = data.pos
    z = data.z if hasattr(data, "z") else None
    return voxelize_positions(pos, z, config)
