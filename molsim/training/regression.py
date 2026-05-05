from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch_geometric.loader import DataLoader as GeoDataLoader

from molsim.metrics import compute_regression_metrics


@dataclass(frozen=True)
class TrainingConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-6
    batch_size: int = 64
    epochs: int = 5
    device: str = "cpu"


class RegressionTrainer:
    def __init__(self, model: torch.nn.Module, config: TrainingConfig) -> None:
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

    def fit(self, train_data: list[Any], val_data: list[Any]) -> list[dict[str, float]]:
        train_loader = GeoDataLoader(train_data, batch_size=self.config.batch_size, shuffle=True)
        val_loader = GeoDataLoader(val_data, batch_size=self.config.batch_size, shuffle=False)

        history: list[dict[str, float]] = []
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            row = {"epoch": float(epoch), "train_mse": train_loss, **val_metrics}
            history.append(row)
        return history

    def _train_epoch(self, loader: GeoDataLoader) -> float:
        self.model.train()
        running_loss = 0.0
        total_batches = 0

        for batch in loader:
            batch = batch.to(self.device)
            pred = self.model(batch.x.float(), batch.edge_index, batch.batch)
            target = batch.y.view(-1).float()

            self.optimizer.zero_grad()
            loss = self.loss_fn(pred, target)
            loss.backward()
            self.optimizer.step()

            running_loss += float(loss.item())
            total_batches += 1

        return running_loss / max(total_batches, 1)

    def evaluate(self, loader: GeoDataLoader) -> dict[str, float]:
        self.model.eval()
        ys_true: list[np.ndarray] = []
        ys_pred: list[np.ndarray] = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                pred = self.model(batch.x.float(), batch.edge_index, batch.batch)
                target = batch.y.view(-1).float()
                ys_true.append(target.cpu().numpy())
                ys_pred.append(pred.cpu().numpy())

        y_true = np.concatenate(ys_true, axis=0)
        y_pred = np.concatenate(ys_pred, axis=0)
        m = compute_regression_metrics(y_true, y_pred)
        return {"val_rmse": m.rmse, "val_mae": m.mae, "val_r2": m.r2}
