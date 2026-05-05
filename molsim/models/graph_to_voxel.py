from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GraphToVoxelNet(nn.Module):
    """Encode molecular graphs and decode into dense voxel fields."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        latent_dim: int = 256,
        grid_size: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if grid_size % 4 != 0:
            raise ValueError("grid_size must be divisible by 4 for the current decoder design")

        self.grid_size = grid_size
        self.init_size = grid_size // 4

        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.fc_latent = nn.Linear(hidden_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 64 * (self.init_size ** 3))

        self.deconv1 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1)

    def encode(self, x, edge_index, batch_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        h = F.relu(self.conv2(h, edge_index))
        h = self.dropout(h)
        h = F.relu(self.conv3(h, edge_index))
        h = global_mean_pool(h, batch_index)
        return self.fc_latent(h)

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, 64, self.init_size, self.init_size, self.init_size)
        h = F.relu(self.deconv1(h))
        return F.softplus(self.deconv2(h))

    def forward(self, x, edge_index, batch_index):
        latent = self.encode(x, edge_index, batch_index)
        voxels = self.decode(latent)
        return voxels, latent
