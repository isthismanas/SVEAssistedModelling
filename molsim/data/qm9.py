from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


QM9_TARGET_INDEX = {
    "mu": 0,
    "alpha": 1,
    "homo": 2,
    "lumo": 3,
    "gap": 4,
    "r2": 5,
    "zpve": 6,
    "u0": 7,
    "u298": 8,
    "h298": 9,
    "g298": 10,
    "cv": 11,
    "u0_atom": 12,
    "u298_atom": 13,
    "h298_atom": 14,
    "g298_atom": 15,
    "a": 16,
    "b": 17,
    "c": 18,
}


@dataclass(frozen=True)
class QM9TargetAdapter:
    target_name: str

    def target_index(self) -> int:
        name = self.target_name.lower()
        if name not in QM9_TARGET_INDEX:
            known = ", ".join(sorted(QM9_TARGET_INDEX.keys()))
            raise KeyError(f"Unknown QM9 target '{self.target_name}'. Known targets: {known}")
        return QM9_TARGET_INDEX[name]

    def apply(self, data):
        idx = self.target_index()
        out = data.clone()
        # QM9 y shape can be [1, 19]; make scalar regression target.
        out.y = out.y.view(-1)[idx].reshape(1)
        return out

    def transform_batch(self, items: Sequence):
        return [self.apply(item) for item in items]

    def get_target_tensor(self, batch) -> torch.Tensor:
        y = batch.y
        return y.view(-1)
