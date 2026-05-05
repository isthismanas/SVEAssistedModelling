from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch_geometric.loader import DataLoader as GeoDataLoader

from molsim.metrics import compute_voxel_mse, compute_voxel_overlap
from molsim.spatial import VoxelConfig, parse_mol2_atoms, voxelize_data, voxelize_positions


@dataclass(frozen=True)
class VoxelTrainingConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-6
    batch_size: int = 16
    epochs: int = 5
    device: str = "cpu"


class VoxelTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_config: VoxelTrainingConfig,
        voxel_config: VoxelConfig,
        mol2_dir: str | None = None,
    ) -> None:
        self.model = model
        self.train_config = train_config
        self.voxel_config = voxel_config
        self.mol2_dir = Path(mol2_dir).resolve() if mol2_dir else None
        self.device = torch.device(train_config.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
        self.loss_fn = torch.nn.MSELoss()

    def fit(self, train_data: list[Any], val_data: list[Any]) -> list[dict[str, float]]:
        train_loader = GeoDataLoader(train_data, batch_size=self.train_config.batch_size, shuffle=True)
        val_loader = GeoDataLoader(val_data, batch_size=self.train_config.batch_size, shuffle=False)

        history: list[dict[str, float]] = []
        for epoch in range(1, self.train_config.epochs + 1):
            train_mse = self._train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            row = {"epoch": float(epoch), "train_voxel_mse": train_mse, **val_metrics}
            history.append(row)
        return history

    def _target_for_data(self, data) -> torch.Tensor:
        if self.mol2_dir is not None and hasattr(data, "name"):
            mol2_path = self.mol2_dir / f"{data.name}.mol2"
            if mol2_path.exists():
                coords, atomic_nums = parse_mol2_atoms(mol2_path)
                return voxelize_positions(coords, atomic_nums, self.voxel_config)

        return voxelize_data(data, self.voxel_config)

    def _targets_from_batch(self, batch) -> torch.Tensor:
        targets = [self._target_for_data(d) for d in batch.to_data_list()]
        return torch.stack(targets).to(self.device)

    def _train_epoch(self, loader: GeoDataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for batch in loader:
            batch = batch.to(self.device)
            target = self._targets_from_batch(batch)
            pred, _ = self.model(batch.x.float(), batch.edge_index, batch.batch)

            self.optimizer.zero_grad()
            loss = self.loss_fn(pred, target)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

        return total_loss / max(total_batches, 1)

    def evaluate(self, loader: GeoDataLoader) -> dict[str, float]:
        self.model.eval()
        pred_batches: list[torch.Tensor] = []
        target_batches: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                target = self._targets_from_batch(batch)
                pred, _ = self.model(batch.x.float(), batch.edge_index, batch.batch)
                pred_batches.append(pred.cpu())
                target_batches.append(target.cpu())

        pred_all = torch.cat(pred_batches, dim=0).numpy()
        target_all = torch.cat(target_batches, dim=0).numpy()

        mse = compute_voxel_mse(target_all, pred_all)
        overlap = compute_voxel_overlap(target_all, pred_all)
        return {
            "val_voxel_mse": mse,
            "val_voxel_overlap": overlap,
        }
