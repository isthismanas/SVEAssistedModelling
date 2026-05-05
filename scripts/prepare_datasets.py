#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molsim.data import DatasetManager, Mol2ExportConfig, Mol2Exporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download datasets under data/ and optionally export mol2 files from 3D graph data."
    )
    parser.add_argument(
        "--dataset-ids",
        nargs="+",
        default=["qm9"],
        help="Dataset ids to prepare (currently supported by DatasetManager: qm9, zinc).",
    )
    parser.add_argument(
        "--export-mol2",
        action="store_true",
        help="Export mol2 files when dataset samples include pos/z fields.",
    )
    parser.add_argument(
        "--mol2-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "QM9_mol2"),
        help="Output directory for mol2 files.",
    )
    parser.add_argument(
        "--max-mol2",
        type=int,
        default=0,
        help="Maximum mol2 files to export (0 means all samples).",
    )
    parser.add_argument(
        "--overwrite-mol2",
        action="store_true",
        help="Overwrite existing mol2 files if present.",
    )
    return parser.parse_args()


def _supports_mol2_export(dataset) -> bool:
    if len(dataset) == 0:
        return False
    first = dataset[0]
    return getattr(first, "pos", None) is not None and getattr(first, "z", None) is not None


def main() -> None:
    args = parse_args()
    manager = DatasetManager(project_root=PROJECT_ROOT)

    for dataset_id_raw in args.dataset_ids:
        dataset_id = dataset_id_raw.lower()
        print(f"\nPreparing dataset: {dataset_id}")
        dataset = manager.load(dataset_id)
        print(f"- local root ready: data/{dataset_id.upper()}")
        print(f"- sample count: {len(dataset)}")

        if not args.export_mol2:
            continue

        if not _supports_mol2_export(dataset):
            print("- mol2 export skipped: dataset does not expose both pos and z")
            continue

        output_dir = Path(args.mol2_dir).resolve()
        if dataset_id != "qm9":
            output_dir = output_dir.parent / f"{dataset_id.upper()}_mol2"

        max_samples = None if args.max_mol2 <= 0 else args.max_mol2
        exporter = Mol2Exporter(
            Mol2ExportConfig(output_dir=str(output_dir), overwrite=args.overwrite_mol2)
        )

        n_to_export = len(dataset) if max_samples is None else min(len(dataset), max_samples)
        print(f"- exporting mol2 to: {output_dir}")
        print(f"- mol2 files to export: {n_to_export}")

        for idx in range(n_to_export):
            exporter.export_one(dataset[idx], idx)
            if (idx + 1) % 1000 == 0 or (idx + 1) == n_to_export:
                print(f"  exported {idx + 1}/{n_to_export}")


if __name__ == "__main__":
    main()
