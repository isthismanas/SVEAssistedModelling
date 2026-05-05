from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RegressionMetrics:
    rmse: float
    mae: float
    r2: float


@dataclass
class BinaryMetrics:
    roc_auc: float
    pr_auc: float
    f1: float
    balanced_accuracy: float


def _as_np(values: np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def compute_regression_metrics(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> RegressionMetrics:
    true = _as_np(y_true)
    pred = _as_np(y_pred)

    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have same shape")

    err = pred - true
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    var = float(np.sum((true - np.mean(true)) ** 2))
    sse = float(np.sum(err ** 2))
    r2 = float(1.0 - (sse / var)) if var > 0 else 0.0

    return RegressionMetrics(rmse=rmse, mae=mae, r2=r2)


def compute_binary_metrics(
    y_true: np.ndarray | list[float],
    y_score: np.ndarray | list[float],
    threshold: float = 0.5,
) -> BinaryMetrics:
    true = _as_np(y_true).astype(int)
    score = _as_np(y_score)

    if true.shape != score.shape:
        raise ValueError("y_true and y_score must have same shape")

    try:
        from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
    except ImportError as exc:
        raise ImportError("scikit-learn is required for binary metrics") from exc

    pred = (score >= threshold).astype(int)

    roc_auc = float(roc_auc_score(true, score))
    pr_auc = float(average_precision_score(true, score))
    f1 = float(f1_score(true, pred, zero_division=0))
    bal_acc = float(balanced_accuracy_score(true, pred))

    return BinaryMetrics(roc_auc=roc_auc, pr_auc=pr_auc, f1=f1, balanced_accuracy=bal_acc)


def compute_voxel_mse(target: np.ndarray, recon: np.ndarray) -> float:
    target_np = np.asarray(target, dtype=float)
    recon_np = np.asarray(recon, dtype=float)

    if target_np.shape != recon_np.shape:
        raise ValueError("target and recon must have same shape")

    return float(np.mean((target_np - recon_np) ** 2))


def compute_voxel_overlap(
    target: np.ndarray,
    recon: np.ndarray,
    threshold: float = 0.1,
    eps: float = 1e-8,
) -> float:
    """Soft overlap score over thresholded occupancy."""
    target_np = np.asarray(target, dtype=float)
    recon_np = np.asarray(recon, dtype=float)

    if target_np.shape != recon_np.shape:
        raise ValueError("target and recon must have same shape")

    target_mask = target_np >= threshold
    recon_mask = recon_np >= threshold

    intersection = np.logical_and(target_mask, recon_mask).sum()
    union = np.logical_or(target_mask, recon_mask).sum()
    return float((intersection + eps) / (union + eps))
