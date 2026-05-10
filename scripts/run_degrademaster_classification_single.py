#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from degrademaster_run_utils import parse_classification_log, stream_command, write_curve_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = str(PROJECT_ROOT / "molinsim" / "bin" / "python")
CONFIG_MAP = {
    "baseline": PROJECT_ROOT / "downstream" / "config" / "config.yml",
    "with_sve": PROJECT_ROOT / "downstream" / "config" / "config_with_sve.yml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one DegradeMaster classification job and log results.")
    parser.add_argument("--mode", choices=["baseline", "with_sve"], required=True)
    parser.add_argument("--python", type=str, default=DEFAULT_PYTHON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = CONFIG_MAP[args.mode]
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    run_name = str(cfg.get("run_name", args.mode))
    log_path = stream_command([
        args.python,
        str(PROJECT_ROOT / "scripts" / "run_degrademaster_classification.py"),
        "--config",
        "config/" + config_path.name,
        "--python",
        args.python,
    ], workdir=PROJECT_ROOT, log_name=run_name)

    curve_rows = parse_classification_log(log_path)
    write_curve_json("classification", run_name, curve_rows, log_path)
    metrics_path = PROJECT_ROOT / "downstream" / str(cfg.get("save_dir", "runs_classification")) / str(cfg.get("run_name", args.mode)) / "final_metrics.json"
    subprocess.run([
        args.python,
        str(PROJECT_ROOT / "scripts" / "log_degrademaster_result.py"),
        "--task",
        "classification",
        "--config",
        str(config_path),
        "--metrics",
        str(metrics_path),
        "--tag",
        str(cfg.get("run_name", args.mode)),
    ], cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
