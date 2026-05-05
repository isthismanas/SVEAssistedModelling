from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from molsim.metrics import compute_voxel_mse


@dataclass(frozen=True)
class AutoencoderTrainingConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-6
    batch_size: int = 32
    epochs: int = 10
    device: str = "cpu"


class VoxelAutoencoderTrainer:
    def __init__(self, model: torch.nn.Module, config: AutoencoderTrainingConfig) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.loss_fn = torch.nn.MSELoss()

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            row = {
                "epoch": float(epoch),
                "train_recon_mse": float(train_loss),
                **val_metrics,
            }
            history.append(row)
        return history

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for voxel in loader:
            voxel = voxel.to(self.device).float()
            recon, _ = self.model(voxel)

            self.optimizer.zero_grad()
            loss = self.loss_fn(recon, voxel)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

        return total_loss / max(total_batches, 1)

    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        recon_batches: list[torch.Tensor] = []
        target_batches: list[torch.Tensor] = []

        with torch.no_grad():
            for voxel in loader:
                voxel = voxel.to(self.device).float()
                recon, _ = self.model(voxel)
                recon_batches.append(recon.cpu())
                target_batches.append(voxel.cpu())

        if not recon_batches:
            return {"val_recon_mse": float("nan")}

        recon_all = torch.cat(recon_batches, dim=0).numpy()
        target_all = torch.cat(target_batches, dim=0).numpy()
        mse = compute_voxel_mse(target_all, recon_all)
        return {"val_recon_mse": float(mse)}
