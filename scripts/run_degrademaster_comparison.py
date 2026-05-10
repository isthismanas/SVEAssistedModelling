#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGRESSION_BASELINE_CONFIGS = {
    "base": "config_regression_train.yml",
    "two_head": "config_regression_two_head.yml",
    "cross_attention": "config_regression_cross_attention.yml",
    "pdc50_bounded": "config_regression_pdc50_bounded.yml",
    "tabular": "config_regression_tabular.yml",
}

REGRESSION_SVE_CONFIGS = {
    "base": "config_regression_train_with_sve.yml",
    "two_head": "config_regression_two_head_with_sve.yml",
    "cross_attention": "config_regression_cross_attention_with_sve.yml",
    "pdc50_bounded": "config_regression_pdc50_bounded_with_sve.yml",
    "tabular": "config_regression_tabular_with_sve.yml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DegradeMaster baseline and with-SVE comparisons.")
    parser.add_argument("--skip-classification", action="store_true")
    parser.add_argument("--skip-regression", action="store_true")
    parser.add_argument("--python", type=str, default=sys.executable)
    return parser.parse_args()


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    if not args.skip_classification:
        _run([args.python, str(PROJECT_ROOT / "scripts" / "run_degrademaster_classification.py"), "--config", "config/config.yml"])
        _run([args.python, str(PROJECT_ROOT / "scripts" / "run_degrademaster_classification.py"), "--config", "config/config_with_sve.yml"])

    if not args.skip_regression:
        for variant, config_name in REGRESSION_BASELINE_CONFIGS.items():
            _run([
                args.python,
                str(PROJECT_ROOT / "downstream" / "train_regression.py"),
                "--config",
                str(PROJECT_ROOT / "downstream" / "config" / config_name),
            ])
            _run([
                args.python,
                str(PROJECT_ROOT / "downstream" / "train_regression.py"),
                "--config",
                str(PROJECT_ROOT / "downstream" / "config" / REGRESSION_SVE_CONFIGS[variant]),
            ])


if __name__ == "__main__":
    main()
