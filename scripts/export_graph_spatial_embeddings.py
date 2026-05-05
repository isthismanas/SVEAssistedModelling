#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader as GeoDataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molsim.data.manager import DatasetManager, SplitConfig
from molsim.models import GraphToVoxelNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export graph-derived spatial embeddings for DegradeMaster."
    )
    parser.add_argument("--dataset-id", type=str, default="qm9", choices=["qm9"])
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "graph_to_voxel_qm9.pt"),
        help="Path to graph-to-voxel model checkpoint.",
    )
    parser.add_argument(
        "--output-npz",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "degrademaster_graph_spatial_embeddings.npz"),
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "degrademaster_graph_spatial_embeddings.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "degrademaster_graph_spatial_embeddings_meta.json"),
    )
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    return parser.parse_args()


def _load_model(checkpoint_path: Path, device: torch.device) -> GraphToVoxelNet:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = checkpoint["model_config"]
    model = GraphToVoxelNet(
        in_channels=int(model_cfg["in_channels"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        latent_dim=int(model_cfg["latent_dim"]),
        grid_size=int(model_cfg["grid_size"]),
        dropout=float(model_cfg["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _select_indices(dataset_size: int, args: argparse.Namespace) -> list[int]:
    if args.split == "all":
        indices = list(range(dataset_size))
    else:
        manager = DatasetManager(project_root=PROJECT_ROOT)
        split_cfg = SplitConfig(seed=args.seed)
        split_map = manager.split_indices(dataset_size=dataset_size, config=split_cfg)
        indices = split_map[args.split]

    if args.max_samples > 0:
        return indices[: args.max_samples]
    return indices


def _write_csv(path: Path, names: list[str], indices: list[int], embeddings: np.ndarray) -> None:
    header = ["molecule_name", "source_index", *[f"spatial_emb_{i:04d}" for i in range(embeddings.shape[1])]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row_idx, name in enumerate(names):
            writer.writerow([name, int(indices[row_idx]), *embeddings[row_idx].tolist()])


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    checkpoint_path = Path(args.checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = _load_model(checkpoint_path, device)

    manager = DatasetManager(project_root=PROJECT_ROOT)
    dataset = manager.load(args.dataset_id)
    selected_indices = _select_indices(len(dataset), args)
    if not selected_indices:
        raise RuntimeError("No samples selected for embedding export.")

    items = [dataset[idx] for idx in selected_indices]
    names = [str(getattr(item, "name", f"{args.dataset_id}_{idx}")) for idx, item in zip(selected_indices, items)]

    loader = GeoDataLoader(items, batch_size=args.batch_size, shuffle=False)
    batches: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z = model.encode(batch.x.float(), batch.edge_index, batch.batch)
            batches.append(z.cpu().numpy())

    embeddings = np.concatenate(batches, axis=0)
    if embeddings.shape[0] != len(names):
        raise RuntimeError("Embedding count mismatch after export.")

    output_npz = Path(args.output_npz).resolve()
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        names=np.asarray(names, dtype=object),
        source_indices=np.asarray(selected_indices, dtype=np.int64),
        embeddings=embeddings,
    )

    output_csv = Path(args.output_csv).resolve()
    _write_csv(output_csv, names, selected_indices, embeddings)

    meta = {
        "dataset_id": args.dataset_id,
        "split": args.split,
        "num_samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "checkpoint_path": str(checkpoint_path),
        "output_npz": str(output_npz),
        "output_csv": str(output_csv),
        "seed": int(args.seed),
    }

    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"Saved embeddings NPZ: {output_npz}")
    print(f"Saved embeddings CSV: {output_csv}")
    print(f"Saved metadata JSON:  {output_json}")
    print(json.dumps({"num_samples": meta["num_samples"], "embedding_dim": meta["embedding_dim"]}, indent=2))


if __name__ == "__main__":
    main()
