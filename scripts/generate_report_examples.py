#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-report-assets")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molsim.data.manager import DatasetManager
from molsim.spatial import VoxelConfig, parse_mol2_structure, voxelize_positions


ASSETS_DIR = PROJECT_ROOT / "docs" / "report_assets"
VOXEL_FIGURE = ASSETS_DIR / "sample_voxel_gdb_1.png"
PANEL_FIGURE = ASSETS_DIR / "sample_datapoint_panels_gdb_1.png"
EXAMPLE_JSON = ASSETS_DIR / "sample_data_example.json"


def _load_sample():
    manager = DatasetManager(project_root=PROJECT_ROOT)
    dataset = manager.load("qm9")
    item = dataset[0]
    if str(getattr(item, "name", "")) != "gdb_1":
        raise RuntimeError("Expected QM9 sample 0 to be gdb_1 for the report example.")
    return item


def _load_smoke_embedding() -> dict[str, object]:
    path = PROJECT_ROOT / "artifacts" / "degrademaster_graph_spatial_embeddings_smoke.npz"
    arr = np.load(path, allow_pickle=True)
    full_embedding = [round(float(x), 4) for x in arr["embeddings"][0].tolist()]
    return {
        "name": str(arr["names"][0]),
        "source_index": int(arr["source_indices"][0]),
        "embedding_dim": int(arr["embeddings"].shape[1]),
        "full": full_embedding,
        "head12": [round(float(x), 4) for x in arr["embeddings"][0][:12]],
    }


def _render_voxel(sample, embedding: dict[str, object]) -> dict[str, object]:
    mol2_path = PROJECT_ROOT / "data" / "QM9_mol2" / "gdb_1.mol2"
    coords, atomic_nums, _ = parse_mol2_structure(mol2_path)
    voxel_cfg = VoxelConfig(grid_size=24, resolution=0.45, sigma=0.5, use_atomic_weights=True)
    voxel = voxelize_positions(coords, atomic_nums, voxel_cfg).squeeze(0).numpy()

    threshold = float(np.quantile(voxel, 0.965))
    occupied = np.argwhere(voxel >= threshold)
    values = voxel[voxel >= threshold]

    fig = plt.figure(figsize=(10, 4.4))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(
        occupied[:, 0],
        occupied[:, 1],
        occupied[:, 2],
        c=values,
        cmap="viridis",
        s=18,
        alpha=0.85,
        linewidths=0,
    )
    ax1.set_title("Occupied voxel render")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.view_init(elev=24, azim=38)

    mid = voxel.shape[0] // 2
    ax2 = fig.add_subplot(1, 2, 2)
    slice_img = voxel[:, :, mid]
    im = ax2.imshow(slice_img, cmap="magma", origin="lower")
    ax2.set_title(f"Central z-slice (index={mid})")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle("Example voxelized geometry for QM9 sample gdb_1")
    fig.tight_layout()
    VOXEL_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(VOXEL_FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)

    graph = nx.Graph()
    z_values = [int(v) for v in sample.z.tolist()]
    labels = {idx: ("C" if z == 6 else "H" if z == 1 else str(z)) for idx, z in enumerate(z_values)}
    positions_2d = {idx: (float(sample.pos[idx, 0]), float(sample.pos[idx, 1])) for idx in range(sample.pos.shape[0])}
    graph.add_nodes_from(range(sample.x.shape[0]))
    edges = set()
    for src, dst in zip(sample.edge_index[0].tolist(), sample.edge_index[1].tolist()):
        if src != dst:
            a, b = sorted((int(src), int(dst)))
            edges.add((a, b))
    graph.add_edges_from(sorted(edges))

    emb_arr = np.asarray(embedding["full"], dtype=float)

    fig = plt.figure(figsize=(11, 8.2))
    ax1 = fig.add_subplot(2, 2, 1)
    x_mat = sample.x.numpy()
    im1 = ax1.imshow(x_mat, cmap="Blues", aspect="auto")
    ax1.set_title("Graph tensor x")
    ax1.set_xlabel("feature index")
    ax1.set_ylabel("node index")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(2, 2, 2)
    node_colors = ["#1f77b4" if z == 6 else "#ff7f0e" for z in z_values]
    nx.draw_networkx(
        graph,
        pos=positions_2d,
        labels=labels,
        node_color=node_colors,
        edge_color="#444444",
        node_size=700,
        font_size=10,
        width=2.0,
        ax=ax2,
    )
    ax2.set_title("NetworkX graph view")
    ax2.set_axis_off()

    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax3.scatter(
        occupied[:, 0],
        occupied[:, 1],
        occupied[:, 2],
        c=values,
        cmap="viridis",
        s=16,
        alpha=0.85,
        linewidths=0,
    )
    ax3.set_title("3D voxel representation")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.view_init(elev=24, azim=38)

    ax4 = fig.add_subplot(2, 2, 4)
    emb_img = emb_arr.reshape(1, -1)
    im4 = ax4.imshow(emb_img, cmap="coolwarm", aspect="auto")
    ax4.set_title("Final embedding vector")
    ax4.set_xlabel("embedding dimension")
    ax4.set_yticks([])
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    fig.suptitle("Single datapoint visualisation for QM9 sample gdb_1")
    fig.tight_layout()
    fig.savefig(PANEL_FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "grid_size": voxel_cfg.grid_size,
        "resolution": voxel_cfg.resolution,
        "sigma": voxel_cfg.sigma,
        "occupancy_threshold_for_render": round(threshold, 4),
        "voxel_shape": list(voxel.shape),
        "voxel_min": round(float(voxel.min()), 4),
        "voxel_max": round(float(voxel.max()), 4),
    }


def main() -> None:
    sample = _load_sample()
    embedding = _load_smoke_embedding()
    voxel_meta = _render_voxel(sample, embedding)

    example = {
        "sample_name": str(sample.name),
        "graph_tensor": {
            "x_shape": list(sample.x.shape),
            "edge_index_shape": list(sample.edge_index.shape),
            "pos_shape": list(sample.pos.shape),
            "z_shape": list(sample.z.shape),
            "x": [[round(float(v), 4) for v in row] for row in sample.x.tolist()],
            "edge_index": sample.edge_index.tolist(),
            "pos": [[round(float(v), 4) for v in row] for row in sample.pos.tolist()],
            "z": [int(v) for v in sample.z.tolist()],
        },
        "voxel_render": voxel_meta,
        "downstream_embedding_example": embedding,
    }

    EXAMPLE_JSON.write_text(json.dumps(example, indent=2) + "\n")
    print(f"Wrote example JSON: {EXAMPLE_JSON}")
    print(f"Wrote voxel figure: {VOXEL_FIGURE}")
    print(f"Wrote panel figure: {PANEL_FIGURE}")


if __name__ == "__main__":
    main()
