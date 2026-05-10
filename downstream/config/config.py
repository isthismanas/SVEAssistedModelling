from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default="./config/config.yml")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    known, _ = parser.parse_known_args()

    cfg_path = Path(known.config)
    payload = _load_yaml(cfg_path)

    if known.dataset_root is not None:
        payload["dataset_root"] = known.dataset_root
    if known.save_dir is not None:
        payload["save_dir"] = known.save_dir
    if known.run_name is not None:
        payload["run_name"] = known.run_name
    if known.mode is not None:
        payload["mode"] = known.mode
    if known.checkpoint_path is not None:
        payload["checkpoint_path"] = known.checkpoint_path

    payload.setdefault("hidden_size", 100)
    payload.setdefault("conv_name", "EGNN")
    payload.setdefault("n_layers", 1)
    payload.setdefault("attention", False)
    payload.setdefault("no_block_embedding", False)
    payload.setdefault("e3_dim", 30)
    payload.setdefault("tar_dim", 30)
    payload.setdefault("protac_dim", 167)
    payload.setdefault("feature", True)
    payload.setdefault("mode", "Train")
    payload.setdefault("epoch_pre", 30)
    payload.setdefault("epoch", 2000)
    payload.setdefault("lr", 1e-4)
    payload.setdefault("lr_l", 1e-3)
    payload.setdefault("batch_size", 40)
    payload.setdefault("train_rate", 0.8)
    payload.setdefault("class_num", 2)
    payload.setdefault("seed", 111)
    payload.setdefault("dataset_type", "name")
    payload.setdefault("dataset_root", "data/PROTAC")
    payload.setdefault("save_dir", "runs")
    payload.setdefault("run_name", "default_run")
    payload.setdefault("select_pocket_war", 10)
    payload.setdefault("select_pocket_e3", 10)
    payload.setdefault("k_neighbors", 9)
    payload.setdefault("w_consistent", 0.0)
    payload.setdefault("w_ent", 0.0)
    payload.setdefault("margin", 0.4)
    payload.setdefault("num_k", 1000)
    payload.setdefault("T", 0.5)
    payload.setdefault("start_epoch", 0)
    payload.setdefault("end_epoch", 20)
    payload.setdefault("regression_variant", "base")
    payload.setdefault("weight_decay", 0.0)
    payload.setdefault("lr_patience", 20)
    payload.setdefault("log_every", 10)
    payload.setdefault("normalize_targets", True)
    payload.setdefault("log_dc50_target", True)
    payload.setdefault("dmax_upper", 100.0)
    payload.setdefault("save_predictions", True)
    payload.setdefault("num_workers", 0)
    payload.setdefault("tabular_backend", "hist_gb")
    payload.setdefault("tabular_learning_rate", 0.05)
    payload.setdefault("tabular_max_depth", 6)
    payload.setdefault("tabular_max_iter", 500)
    payload.setdefault("tabular_n_estimators", 500)
    payload.setdefault("tabular_n_jobs", -1)

    return argparse.Namespace(**payload)
