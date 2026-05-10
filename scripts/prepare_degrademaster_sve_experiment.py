#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DegradeMaster baseline+SVE data roots.")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--source-data-root", type=str, default="data/PROTAC")
    parser.add_argument("--output-sve-root", type=str, default="data/PROTAC_sve")
    parser.add_argument("--output-regression-root", type=str, default="data/PROTAC_regression_sve")
    parser.add_argument("--protac-csv", type=str, default="data/_downloads/PROTAC-8K_extracted/PROTAC-8K/protac.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = [
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "export_degrademaster_protac_embeddings.py"),
            "--device", args.device,
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_degrademaster_sve_dataset.py"),
            "--source-data-root", args.source_data_root,
            "--output-data-root", args.output_sve_root,
        ],
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_degrademaster_regression_preprocess.py"),
            "--source-data-root", Path(args.output_sve_root).relative_to(PROJECT_ROOT).as_posix() if Path(args.output_sve_root).is_absolute() else args.output_sve_root,
            "--regression-data-root", Path(args.output_regression_root).relative_to(PROJECT_ROOT).as_posix() if Path(args.output_regression_root).is_absolute() else args.output_regression_root,
            "--name-json", (Path(args.output_sve_root) / "name.json").relative_to(PROJECT_ROOT).as_posix() if Path(args.output_sve_root).is_absolute() else f"{args.output_sve_root}/name.json",
            "--protac-csv", args.protac_csv,
        ],
    ]
    for command in commands:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
