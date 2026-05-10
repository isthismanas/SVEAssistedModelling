from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout3d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VoxelAutoencoder(nn.Module):
    """3D U-Net style autoencoder for mol2-derived voxel tensors."""

    def __init__(
        self,
        grid_size: int = 24,
        embedding_dim: int = 256,
        base_channels: int = 32,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if grid_size % 8 != 0:
            raise ValueError("grid_size must be divisible by 8 for current encoder/decoder strides")

        self.grid_size = int(grid_size)
        self.embedding_dim = int(embedding_dim)
        self.base_channels = int(base_channels)
        self.bottleneck_size = grid_size // 8

        c = self.base_channels

        self.enc1 = ConvBlock3D(1, c, dropout=dropout)
        self.down1 = nn.Conv3d(c, c * 2, kernel_size=4, stride=2, padding=1)
        self.enc2 = ConvBlock3D(c * 2, c * 2, dropout=dropout)
        self.down2 = nn.Conv3d(c * 2, c * 4, kernel_size=4, stride=2, padding=1)
        self.enc3 = ConvBlock3D(c * 4, c * 4, dropout=dropout)
        self.down3 = nn.Conv3d(c * 4, c * 8, kernel_size=4, stride=2, padding=1)
        self.bottleneck = ConvBlock3D(c * 8, c * 8, dropout=dropout)

        flat_dim = c * 8 * (self.bottleneck_size ** 3)
        self.fc_embed = nn.Linear(flat_dim, self.embedding_dim)
        self.fc_decode = nn.Linear(self.embedding_dim, flat_dim)

        self.up1 = nn.ConvTranspose3d(c * 8, c * 4, kernel_size=4, stride=2, padding=1)
        self.dec1 = ConvBlock3D(c * 8, c * 4, dropout=dropout)
        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = ConvBlock3D(c * 4, c * 2, dropout=dropout)
        self.up3 = nn.ConvTranspose3d(c * 2, c, kernel_size=4, stride=2, padding=1)
        self.dec3 = ConvBlock3D(c * 2, c, dropout=dropout)
        self.out_conv = nn.Conv3d(c, 1, kernel_size=1)

    def _encode_with_skips(self, voxel: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        x1 = self.enc1(voxel)
        x2 = self.enc2(self.down1(x1))
        x3 = self.enc3(self.down2(x2))
        b = self.bottleneck(self.down3(x3))
        z = self.fc_embed(b.reshape(b.shape[0], -1))
        return z, (x1, x2, x3)

    def encode(self, voxel: torch.Tensor) -> torch.Tensor:
        z, _ = self._encode_with_skips(voxel)
        return z

    def _decode_with_skips(
        self,
        embedding: torch.Tensor,
        skips: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        x1, x2, x3 = skips
        c = self.base_channels
        h = self.fc_decode(embedding)
        h = h.view(-1, c * 8, self.bottleneck_size, self.bottleneck_size, self.bottleneck_size)

        h = self.up1(h)
        h = self.dec1(torch.cat((h, x3), dim=1))
        h = self.up2(h)
        h = self.dec2(torch.cat((h, x2), dim=1))
        h = self.up3(h)
        h = self.dec3(torch.cat((h, x1), dim=1))
        return torch.sigmoid(self.out_conv(h))

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        c = self.base_channels
        h = self.fc_decode(embedding)
        h = h.view(-1, c * 8, self.bottleneck_size, self.bottleneck_size, self.bottleneck_size)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        return torch.sigmoid(self.out_conv(h))

    def forward(self, voxel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding, skips = self._encode_with_skips(voxel)
        recon = self._decode_with_skips(embedding, skips)
        return recon, embedding
