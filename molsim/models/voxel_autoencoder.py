from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxelAutoencoder(nn.Module):
    """3D convolutional autoencoder for mol2-derived voxel tensors."""

    def __init__(
        self,
        grid_size: int = 24,
        embedding_dim: int = 256,
        base_channels: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if grid_size % 8 != 0:
            raise ValueError("grid_size must be divisible by 8 for current encoder/decoder strides")

        self.grid_size = int(grid_size)
        self.embedding_dim = int(embedding_dim)
        self.base_channels = int(base_channels)
        self.bottleneck_size = grid_size // 8

        c = self.base_channels

        self.enc1 = nn.Conv3d(1, c, kernel_size=4, stride=2, padding=1)
        self.enc2 = nn.Conv3d(c, c * 2, kernel_size=4, stride=2, padding=1)
        self.enc3 = nn.Conv3d(c * 2, c * 4, kernel_size=4, stride=2, padding=1)
        self.enc_dropout = nn.Dropout3d(dropout)

        flat_dim = c * 4 * (self.bottleneck_size ** 3)
        self.fc_embed = nn.Linear(flat_dim, self.embedding_dim)

        self.fc_decode = nn.Linear(self.embedding_dim, flat_dim)
        self.dec1 = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose3d(c * 2, c, kernel_size=4, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose3d(c, 1, kernel_size=4, stride=2, padding=1)

    def encode(self, voxel: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.enc1(voxel))
        h = self.enc_dropout(h)
        h = F.relu(self.enc2(h))
        h = self.enc_dropout(h)
        h = F.relu(self.enc3(h))
        h = h.view(h.shape[0], -1)
        return self.fc_embed(h)

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        c = self.base_channels
        h = self.fc_decode(embedding)
        h = h.view(-1, c * 4, self.bottleneck_size, self.bottleneck_size, self.bottleneck_size)
        h = F.relu(self.dec1(h))
        h = F.relu(self.dec2(h))
        return F.softplus(self.dec3(h))

    def forward(self, voxel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(voxel)
        recon = self.decode(embedding)
        return recon, embedding
