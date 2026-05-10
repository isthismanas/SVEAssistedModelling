#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOWNSTREAM_ROOT = PROJECT_ROOT / "downstream"
DEFAULT_PYTHON = str(PROJECT_ROOT / "molinsim" / "bin" / "python")


def _to_downstream_rel(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return path.as_posix()
    abs_path = (PROJECT_ROOT / path).resolve()
    try:
        return abs_path.relative_to(DOWNSTREAM_ROOT).as_posix()
    except ValueError:
        return abs_path.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare downstream DegradeMaster regression data workspace.")
    parser.add_argument("--source-data-root", type=str, default="data/PROTAC")
    parser.add_argument("--regression-data-root", type=str, default="data/PROTAC_regression")
    parser.add_argument("--name-json", type=str, default="data/PROTAC/name.json")
    parser.add_argument(
        "--protac-csv",
        type=str,
        default="data/_downloads/PROTAC-8K_extracted/PROTAC-8K/protac.csv",
    )
    parser.add_argument("--skip-graph-preprocess", action="store_true")
    parser.add_argument(
        "--python",
        type=str,
        default=DEFAULT_PYTHON,
        help="Python interpreter to use for the downstream run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [
        args.python,
        str(DOWNSTREAM_ROOT / "regression_preprocess.py"),
        "--project-root",
        ".",
        "--source-data-root",
        _to_downstream_rel(args.source_data_root),
        "--regression-data-root",
        _to_downstream_rel(args.regression_data_root),
        "--name-json",
        _to_downstream_rel(args.name_json),
        "--protac-csv",
        _to_downstream_rel(args.protac_csv),
    ]
    if args.skip_graph_preprocess:
        command.append("--skip-graph-preprocess")
    subprocess.run(command, cwd=DOWNSTREAM_ROOT, check=True)


if __name__ == "__main__":
    main()
