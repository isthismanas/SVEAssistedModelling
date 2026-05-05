#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOWNSTREAM_ROOT = PROJECT_ROOT / "downstream"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run downstream DegradeMaster classification training/eval.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DOWNSTREAM_ROOT / "config" / "config.yml"),
        help="Path to downstream classification config file.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter to use for the downstream run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    (DOWNSTREAM_ROOT / "latent").mkdir(exist_ok=True)
    command = [args.python, str(DOWNSTREAM_ROOT / "main.py"), "--config", args.config]
    subprocess.run(command, cwd=DOWNSTREAM_ROOT, check=True)


if __name__ == "__main__":
    main()
