from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a notebook using the active Python environment.")
    parser.add_argument("notebook", type=str, help="Path to the .ipynb file.")
    parser.add_argument("--to", type=str, default="html", choices=["html", "pdf"], help="Export format.")
    parser.add_argument("--output", type=str, default=None, help="Optional output filename.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notebook_path = Path(args.notebook)
    if not notebook_path.is_absolute():
        notebook_path = (PROJECT_ROOT / notebook_path).resolve()
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    if args.to == "pdf" and shutil.which("xelatex") is None:
        raise SystemExit(
            "PDF export requires `xelatex`, which is not installed or not on PATH. "
            "Use `--to html` for a portable export, or install a TeX distribution first."
        )

    command = [sys.executable, "-m", "nbconvert", str(notebook_path), "--to", args.to]
    if args.output:
        command.extend(["--output", args.output])

    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
