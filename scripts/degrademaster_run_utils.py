#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "results" / "generated" / "command_logs"
CURVE_DIR = PROJECT_ROOT / "results" / "generated" / "degrademaster_curves"


CLASS_TRAIN_RE = re.compile(
    r"Epoch:\s*(?P<epoch>\d+)/(?:\d+)\s+Train Loss:\s*(?P<train_loss>[-+0-9.eE]+)\s*\|\s*Train acc:\s*(?P<train_acc>[-+0-9.eE]+)\|\s*Train auc:\s*(?P<train_auc>[-+0-9.eE]+)\|Valid f1:\s*(?P<train_f1>[-+0-9.eE]+)\|\s*Train pre:\s*(?P<train_precision>[-+0-9.eE]+)\|\s*Train rec:\s*(?P<train_recall>[-+0-9.eE]+)"
)
CLASS_VALID_RE = re.compile(
    r"Valid loss:\s*(?P<val_loss>[-+0-9.eE]+)\s*\|\s*Valid acc:\s*(?P<val_acc>[-+0-9.eE]+)\|\s*Valid auc:\s*(?P<val_auc>[-+0-9.eE]+)\|Valid f1:\s*(?P<val_f1>[-+0-9.eE]+)\|\s*Valid pre:\s*(?P<val_precision>[-+0-9.eE]+)\|\s*Valid rec:\s*(?P<val_recall>[-+0-9.eE]+)"
)
REG_TRAIN_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)/(?:\d+)\s*\|\s*Train Loss\(scaled\):\s*(?P<train_loss>[-+0-9.eE]+)\s*\|\s*LR:\s*(?P<lr>[-+0-9.eE]+)\s*\|\s*Mean RMSE:\s*(?P<mean_rmse>[-+0-9.eE]+)"
)
REG_VALID_RE = re.compile(
    r"Test\s*\|\s*Loss\(scaled\):\s*(?P<val_loss>[-+0-9.eE]+)\s*\|\s*RMSE\(DC50\):\s*(?P<rmse_dc50>[-+0-9.eE]+)\s*\|\s*RMSE\(Dmax\):\s*(?P<rmse_dmax>[-+0-9.eE]+)\s*\|\s*MAE\(DC50\):\s*(?P<mae_dc50>[-+0-9.eE]+)\s*\|\s*MAE\(Dmax\):\s*(?P<mae_dmax>[-+0-9.eE]+)\s*\|\s*R2\(DC50\):\s*(?P<r2_dc50>[-+0-9.eE]+)\s*\|\s*R2\(Dmax\):\s*(?P<r2_dmax>[-+0-9.eE]+)"
)


def stream_command(command: list[str], workdir: Path, log_name: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{log_name}.log"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        proc = subprocess.Popen(
            command,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            handle.write(line)
        returncode = proc.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, command)
    return log_path


def parse_classification_log(log_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    pending: dict[str, float] | None = None
    for raw in log_path.read_text(encoding="utf-8").splitlines():
        match = CLASS_TRAIN_RE.search(raw)
        if match:
            pending = {"epoch": float(match.group("epoch"))}
            pending.update({
                "train_loss": float(match.group("train_loss")),
                "train_acc": float(match.group("train_acc")),
                "train_auc": float(match.group("train_auc")),
                "train_f1": float(match.group("train_f1")),
                "train_precision": float(match.group("train_precision")),
                "train_recall": float(match.group("train_recall")),
            })
            continue
        match = CLASS_VALID_RE.search(raw)
        if match and pending is not None:
            pending.update({
                "val_loss": float(match.group("val_loss")),
                "val_acc": float(match.group("val_acc")),
                "val_auc": float(match.group("val_auc")),
                "val_f1": float(match.group("val_f1")),
                "val_precision": float(match.group("val_precision")),
                "val_recall": float(match.group("val_recall")),
            })
            rows.append(pending)
            pending = None
    return rows


def parse_regression_log(log_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    pending: dict[str, float] | None = None
    for raw in log_path.read_text(encoding="utf-8").splitlines():
        match = REG_TRAIN_RE.search(raw)
        if match:
            pending = {
                "epoch": float(match.group("epoch")),
                "train_loss_scaled": float(match.group("train_loss")),
                "lr": float(match.group("lr")),
                "mean_rmse": float(match.group("mean_rmse")),
            }
            continue
        match = REG_VALID_RE.search(raw)
        if match and pending is not None:
            pending.update({
                "val_loss_scaled": float(match.group("val_loss")),
                "rmse_dc50_nm": float(match.group("rmse_dc50")),
                "rmse_dmax_pct": float(match.group("rmse_dmax")),
                "mae_dc50_nm": float(match.group("mae_dc50")),
                "mae_dmax_pct": float(match.group("mae_dmax")),
                "r2_dc50_nm": float(match.group("r2_dc50")),
                "r2_dmax_pct": float(match.group("r2_dmax")),
            })
            rows.append(pending)
            pending = None
    return rows


def write_curve_json(task: str, run_name: str, rows: list[dict[str, float]], log_path: Path) -> Path:
    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CURVE_DIR / f"{run_name}.json"
    payload = {
        "task": task,
        "run_name": run_name,
        "log_path": str(log_path),
        "history": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path
