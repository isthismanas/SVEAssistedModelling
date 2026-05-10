#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml

from degrademaster_run_utils import parse_regression_log, stream_command, write_curve_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = str(PROJECT_ROOT / "molinsim" / "bin" / "python")
BASELINE_CONFIGS = {
    "base": "config_regression_train.yml",
    "two_head": "config_regression_two_head.yml",
    "cross_attention": "config_regression_cross_attention.yml",
    "pdc50_bounded": "config_regression_pdc50_bounded.yml",
    "tabular": "config_regression_tabular.yml",
}
SVE_CONFIGS = {
    "base": "config_regression_train_with_sve.yml",
    "two_head": "config_regression_two_head_with_sve.yml",
    "cross_attention": "config_regression_cross_attention_with_sve.yml",
    "pdc50_bounded": "config_regression_pdc50_bounded_with_sve.yml",
    "tabular": "config_regression_tabular_with_sve.yml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one DegradeMaster regression job and log results.")
    parser.add_argument("--mode", choices=["baseline", "with_sve"], required=True)
    parser.add_argument("--variant", choices=["base", "two_head", "cross_attention", "pdc50_bounded", "tabular"], required=True)
    parser.add_argument("--python", type=str, default=DEFAULT_PYTHON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_name = BASELINE_CONFIGS[args.variant] if args.mode == "baseline" else SVE_CONFIGS[args.variant]
    config_path = PROJECT_ROOT / "downstream" / "config" / config_name
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    run_name = str(cfg.get("run_name", f"{args.variant}_{args.mode}"))

    log_path = stream_command([
        args.python,
        str(PROJECT_ROOT / "downstream" / "train_regression.py"),
        "--config",
        str(config_path),
    ], workdir=PROJECT_ROOT, log_name=run_name)

    curve_rows = parse_regression_log(log_path)
    write_curve_json("regression", run_name, curve_rows, log_path)
    metrics_path = PROJECT_ROOT / "downstream" / str(cfg.get("save_dir", "runs_regression")) / str(cfg.get("run_name", f"{args.variant}_{args.mode}")) / "final_metrics.json"
    subprocess.run([
        args.python,
        str(PROJECT_ROOT / "scripts" / "log_degrademaster_result.py"),
        "--task",
        "regression",
        "--config",
        str(config_path),
        "--metrics",
        str(metrics_path),
        "--tag",
        str(cfg.get("run_name", f"{args.variant}_{args.mode}")),
    ], cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
