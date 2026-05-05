#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOWNSTREAM_ROOT = PROJECT_ROOT / "downstream"

VARIANT_TO_CONFIG = {
    "base": "config_regression_train.yml",
    "two_head": "config_regression_two_head.yml",
    "cross_attention": "config_regression_cross_attention.yml",
    "pdc50_bounded": "config_regression_pdc50_bounded.yml",
    "tabular": "config_regression_tabular.yml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run downstream DegradeMaster regression variants.")
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["all", *VARIANT_TO_CONFIG.keys()],
        help="Regression variant to run.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter to use for the downstream run.",
    )
    return parser.parse_args()


def run_one(python_exe: str, config_name: str) -> None:
    command = [
        python_exe,
        str(DOWNSTREAM_ROOT / "train_regression.py"),
        "--config",
        str(DOWNSTREAM_ROOT / "config" / config_name),
    ]
    subprocess.run(command, cwd=DOWNSTREAM_ROOT, check=True)


def main() -> None:
    args = parse_args()
    variants = list(VARIANT_TO_CONFIG) if args.variant == "all" else [args.variant]
    for variant in variants:
        run_one(args.python, VARIANT_TO_CONFIG[variant])


if __name__ == "__main__":
    main()
