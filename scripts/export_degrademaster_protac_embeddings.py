#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SVE embeddings for downstream PROTAC mol2 files.")
    parser.add_argument("--mol2-dir", type=str, default="data/PROTAC/protac")
    parser.add_argument("--checkpoint-path", type=str, default=str(PROJECT_ROOT / "artifacts" / "voxel_autoencoder_qm9.pt"))
    parser.add_argument("--output-npz", type=str, default=str(PROJECT_ROOT / "artifacts" / "degrademaster_protac_embeddings.npz"))
    parser.add_argument("--output-json", type=str, default=str(PROJECT_ROOT / "artifacts" / "degrademaster_protac_embeddings_meta.json"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="mps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "export_degrademaster_embeddings.py"),
        "--mol2-dir", args.mol2_dir,
        "--checkpoint-path", args.checkpoint_path,
        "--output-npz", args.output_npz,
        "--output-json", args.output_json,
        "--batch-size", str(args.batch_size),
        "--device", args.device,
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
