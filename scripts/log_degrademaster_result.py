#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "results" / "degrademaster_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record one DegradeMaster run into results/degrademaster_results.json")
    parser.add_argument("--task", type=str, choices=["classification", "regression"], required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--metrics", type=str, required=True)
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {"classification": {}, "regression": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    metrics_path = Path(args.metrics)
    if not metrics_path.is_absolute():
        metrics_path = (PROJECT_ROOT / metrics_path).resolve()

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    store = _load_json(RESULTS_PATH)
    key = str(args.tag or cfg.get("run_name") or metrics_path.parent.name)
    entry = {
        "task": args.task,
        "tag": key,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "metrics_path": str(metrics_path),
        "run_name": str(cfg.get("run_name", metrics_path.parent.name)),
        "dataset_root": str(cfg.get("dataset_root", "")),
        "mode": str(cfg.get("mode", "")),
        "conv_name": str(cfg.get("conv_name", "tabular" if cfg.get("regression_variant") == "tabular" else "unknown")),
        "hidden_size": cfg.get("hidden_size"),
        "batch_size": cfg.get("batch_size"),
        "epochs": cfg.get("epoch"),
        "feature": cfg.get("feature"),
        "regression_variant": cfg.get("regression_variant"),
        "metrics": metrics,
    }
    store.setdefault(args.task, {})[key] = entry
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(store, indent=2) + "\n", encoding="utf-8")
    print(f"Updated results registry: {RESULTS_PATH}")
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
