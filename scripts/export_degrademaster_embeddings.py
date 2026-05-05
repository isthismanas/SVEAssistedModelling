#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molsim.data import list_mol2_files
from molsim.models import VoxelAutoencoder
from molsim.spatial import VoxelConfig, parse_mol2_structure, voxelize_positions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export mol2 spatial embeddings for DegradeMaster consumption.")
    parser.add_argument("--mol2-dir", type=str, required=True, help="Directory containing mol2 files.")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "voxel_autoencoder_qm9.pt"),
        help="Trained voxel autoencoder checkpoint.",
    )
    parser.add_argument(
        "--output-npz",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "degrademaster_mol2_embeddings.npz"),
        help="Output NPZ file with arrays: names, embeddings.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "degrademaster_mol2_embeddings_meta.json"),
        help="Output metadata JSON.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all mol2 files.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    return parser.parse_args()


def _load_checkpoint(path: Path, device: torch.device) -> tuple[VoxelAutoencoder, VoxelConfig]:
    ckpt = torch.load(path, map_location=device)
    model_cfg = ckpt["model_config"]
    voxel_cfg = ckpt["voxel_config"]

    model = VoxelAutoencoder(
        grid_size=int(model_cfg["grid_size"]),
        embedding_dim=int(model_cfg["embedding_dim"]),
        base_channels=int(model_cfg["base_channels"]),
        dropout=float(model_cfg["dropout"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, VoxelConfig(
        grid_size=int(voxel_cfg["grid_size"]),
        resolution=float(voxel_cfg["resolution"]),
        sigma=float(voxel_cfg["sigma"]),
        use_atomic_weights=bool(voxel_cfg["use_atomic_weights"]),
    )


def _voxel_from_mol2(path: Path, voxel_cfg: VoxelConfig) -> torch.Tensor:
    coords, atomic_nums, _ = parse_mol2_structure(path)
    return voxelize_positions(coords, atomic_nums, voxel_cfg)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    checkpoint_path = Path(args.checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, voxel_cfg = _load_checkpoint(checkpoint_path, device)

    files = list_mol2_files(args.mol2_dir)
    if args.max_samples > 0:
        files = files[: args.max_samples]

    if not files:
        raise FileNotFoundError(f"No mol2 files found in {Path(args.mol2_dir).resolve()}")

    names: list[str] = []
    embs: list[np.ndarray] = []

    batch: list[torch.Tensor] = []
    batch_names: list[str] = []

    with torch.no_grad():
        for path in files:
            voxel = _voxel_from_mol2(path, voxel_cfg)
            batch.append(voxel)
            batch_names.append(path.stem)

            if len(batch) >= args.batch_size:
                x = torch.stack(batch, dim=0).to(device).float()
                z = model.encode(x).cpu().numpy()
                for n, e in zip(batch_names, z):
                    names.append(n)
                    embs.append(e)
                batch.clear()
                batch_names.clear()

        if batch:
            x = torch.stack(batch, dim=0).to(device).float()
            z = model.encode(x).cpu().numpy()
            for n, e in zip(batch_names, z):
                names.append(n)
                embs.append(e)

    emb_arr = np.stack(embs, axis=0)
    names_arr = np.asarray(names, dtype=object)

    output_npz = Path(args.output_npz).resolve()
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, names=names_arr, embeddings=emb_arr)

    meta = {
        "mol2_dir": str(Path(args.mol2_dir).resolve()),
        "checkpoint_path": str(checkpoint_path),
        "num_samples": int(len(names)),
        "embedding_dim": int(emb_arr.shape[1]),
        "output_npz": str(output_npz),
    }

    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"Saved embeddings: {output_npz}")
    print(f"Saved metadata:   {output_json}")
    print(json.dumps({"num_samples": len(names), "embedding_dim": int(emb_arr.shape[1])}, indent=2))


if __name__ == "__main__":
    main()
