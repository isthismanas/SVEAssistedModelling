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
    occupied_weight: float = 8.0
    occupancy_threshold: float = 0.1
    sparsity_weight: float = 1e-3
    l1_weight: float = 0.5
    dice_weight: float = 0.5


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

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base = torch.nn.functional.mse_loss(pred, target)
        l1 = torch.nn.functional.l1_loss(pred, target)

        occ_mask = target >= self.config.occupancy_threshold
        if torch.any(occ_mask):
            occ_loss = torch.nn.functional.smooth_l1_loss(pred[occ_mask], target[occ_mask])
        else:
            occ_loss = pred.new_tensor(0.0)

        target_occ = (target >= self.config.occupancy_threshold).float()
        pred_occ = pred.clamp(0.0, 1.0)
        intersection = torch.sum(pred_occ * target_occ)
        denom = torch.sum(pred_occ) + torch.sum(target_occ) + 1e-8
        dice_loss = 1.0 - ((2.0 * intersection + 1e-8) / denom)

        sparsity = pred.mean()
        return (
            base
            + (self.config.l1_weight * l1)
            + (self.config.occupied_weight * occ_loss)
            + (self.config.dice_weight * dice_loss)
            + (self.config.sparsity_weight * sparsity)
        )

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
            loss = self._loss(recon, voxel)
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
