#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molsim.data import DatasetManager, Mol2ExportConfig, Mol2Exporter
from molsim.spatial import VoxelConfig, parse_mol2_structure, voxelize_positions

_Z_TO_SYMBOL = {
    1: "H",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Plotly 3D comparison: graph voxelization vs mol2 voxelization.")
    parser.add_argument("--dataset-id", type=str, default="qm9")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--mol2-dir", type=str, default=str(PROJECT_ROOT / "data" / "QM9_mol2"))
    parser.add_argument("--auto-export-mol2", action="store_true", help="Export one mol2 file if missing.")
    parser.add_argument("--grid-size", type=int, default=24)
    parser.add_argument("--resolution", type=float, default=0.45)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--iso-threshold", type=float, default=0.10)
    parser.add_argument("--diff-threshold", type=float, default=0.05)
    parser.add_argument(
        "--output-html",
        type=str,
        default="",
        help="Output HTML path. Default: artifacts/visualizations/voxel_vs_mol2_<dataset>_<idx>.html",
    )
    return parser.parse_args()


def _unique_bonds(edge_index: torch.Tensor | None) -> list[tuple[int, int]]:
    if edge_index is None or edge_index.numel() == 0:
        return []
    out: set[tuple[int, int]] = set()
    for k in range(edge_index.shape[1]):
        i = int(edge_index[0, k])
        j = int(edge_index[1, k])
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        out.add((a, b))
    return sorted(out)


def _name_for_sample(data, idx: int) -> str:
    name = getattr(data, "name", None)
    if isinstance(name, str):
        return name
    if hasattr(name, "item"):
        try:
            return str(name.item())
        except Exception:
            pass
    if name is not None:
        return str(name)
    return f"sample_{idx:06d}"


def _smiles_for_sample(data) -> str | None:
    smiles = getattr(data, "smiles", None)
    if isinstance(smiles, str):
        return smiles
    if smiles is not None:
        return str(smiles)
    return None


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _formula_from_z(z_values: torch.Tensor) -> str:
    counts: dict[str, int] = {}
    for z in z_values.detach().cpu().view(-1).tolist():
        sym = _Z_TO_SYMBOL.get(int(round(float(z))), "X")
        counts[sym] = counts.get(sym, 0) + 1

    order: list[str] = []
    if "C" in counts:
        order.append("C")
    if "H" in counts:
        order.append("H")
    for sym in sorted(k for k in counts.keys() if k not in {"C", "H"}):
        order.append(sym)

    parts: list[str] = []
    for sym in order:
        n = counts[sym]
        parts.append(f"{sym}{n if n > 1 else ''}")
    return "".join(parts) if parts else "unknown"


def _build_isosurface(go, xx, yy, zz, values, threshold: float, scene: str, name: str, colorscale: str):
    return go.Isosurface(
        x=xx.ravel(),
        y=yy.ravel(),
        z=zz.ravel(),
        value=values.ravel(),
        isomin=float(threshold),
        isomax=float(values.max()),
        opacity=0.35,
        surface_count=3,
        colorscale=colorscale,
        name=name,
        showscale=False,
        caps=dict(x_show=False, y_show=False, z_show=False),
        scene=scene,
    )


def _build_atom_scatter(go, pos: np.ndarray, z: np.ndarray, scene: str, name: str):
    return go.Scatter3d(
        x=pos[:, 0],
        y=pos[:, 1],
        z=pos[:, 2],
        mode="markers",
        marker=dict(size=4, color=z, colorscale="Viridis", opacity=0.95, colorbar=None),
        name=name,
        scene=scene,
    )


def _build_bond_lines(
    go,
    pos: np.ndarray,
    bonds: list[tuple[int, int]],
    scene: str,
    name: str,
    color: str = "rgba(80,80,80,0.7)",
):
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []

    for i, j in bonds:
        xs.extend([float(pos[i, 0]), float(pos[j, 0]), None])
        ys.extend([float(pos[i, 1]), float(pos[j, 1]), None])
        zs.extend([float(pos[i, 2]), float(pos[j, 2]), None])

    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(width=2, color=color),
        name=name,
        scene=scene,
    )


def main() -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError("plotly is required: pip install plotly") from exc

    args = parse_args()

    manager = DatasetManager(project_root=PROJECT_ROOT)
    dataset = manager.load(args.dataset_id)

    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(f"sample index {args.sample_index} out of range for dataset size {len(dataset)}")

    data = dataset[args.sample_index]
    sample_name = _name_for_sample(data, args.sample_index)

    graph_pos = data.pos.detach().cpu().float()
    graph_z = data.z.detach().cpu().float()
    smiles = _smiles_for_sample(data)
    formula = _formula_from_z(graph_z)

    mol2_dir = Path(args.mol2_dir).resolve()
    mol2_dir.mkdir(parents=True, exist_ok=True)
    mol2_path = mol2_dir / f"{sample_name}.mol2"

    if not mol2_path.exists() and args.auto_export_mol2:
        exporter = Mol2Exporter(Mol2ExportConfig(output_dir=str(mol2_dir), overwrite=False))
        exporter.export_one(data, args.sample_index)

    if mol2_path.exists():
        mol2_pos, mol2_z, mol2_bonds = parse_mol2_structure(mol2_path)
    else:
        mol2_pos, mol2_z = graph_pos, graph_z
        mol2_bonds = _unique_bonds(getattr(data, "edge_index", None))

    voxel_cfg = VoxelConfig(
        grid_size=args.grid_size,
        resolution=args.resolution,
        sigma=args.sigma,
        use_atomic_weights=True,
    )

    graph_voxel = voxelize_positions(graph_pos, graph_z, voxel_cfg).squeeze(0).cpu().numpy()
    mol2_voxel = voxelize_positions(mol2_pos, mol2_z, voxel_cfg).squeeze(0).cpu().numpy()
    diff_voxel = np.abs(graph_voxel - mol2_voxel)
    diff_max = float(diff_voxel.max())
    eff_diff_threshold = args.diff_threshold
    if diff_max > 0.0:
        eff_diff_threshold = min(args.diff_threshold, max(diff_max * 0.25, 1e-6))

    half_extent = (args.grid_size * args.resolution) / 2.0
    coords = np.linspace(-half_extent, half_extent, args.grid_size)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            "Graph Positions -> Voxels",
            "mol2 Positions -> Voxels",
            "|GraphVoxel - Mol2Voxel|",
        ),
        horizontal_spacing=0.02,
    )

    graph_pos_np = graph_pos.cpu().numpy()
    mol2_pos_np = mol2_pos.cpu().numpy()
    graph_z_np = graph_z.cpu().numpy()
    mol2_z_np = mol2_z.cpu().numpy()

    fig.add_trace(
        _build_isosurface(go, xx, yy, zz, graph_voxel, args.iso_threshold, "scene", "graph voxel", "Blues"),
        row=1,
        col=1,
    )
    fig.add_trace(_build_atom_scatter(go, graph_pos_np, graph_z_np, "scene", "graph atoms"), row=1, col=1)
    fig.add_trace(_build_bond_lines(go, graph_pos_np, _unique_bonds(getattr(data, "edge_index", None)), "scene", "graph bonds"), row=1, col=1)

    fig.add_trace(
        _build_isosurface(go, xx, yy, zz, mol2_voxel, args.iso_threshold, "scene2", "mol2 voxel", "Greens"),
        row=1,
        col=2,
    )
    fig.add_trace(_build_atom_scatter(go, mol2_pos_np, mol2_z_np, "scene2", "mol2 atoms"), row=1, col=2)
    if mol2_bonds:
        fig.add_trace(
            _build_bond_lines(
                go,
                mol2_pos_np,
                mol2_bonds,
                "scene2",
                "mol2 bonds",
                color="rgba(0,100,0,0.75)",
            ),
            row=1,
            col=2,
        )

    if diff_max > 0.0:
        fig.add_trace(
            _build_isosurface(go, xx, yy, zz, diff_voxel, eff_diff_threshold, "scene3", "absolute diff", "Reds"),
            row=1,
            col=3,
        )

    scene_layout = dict(
        xaxis=dict(range=[-half_extent, half_extent], title="x"),
        yaxis=dict(range=[-half_extent, half_extent], title="y"),
        zaxis=dict(range=[-half_extent, half_extent], title="z"),
        aspectmode="cube",
    )

    title_parts = [
        f"Voxel vs mol2 spatial comparison | sample={sample_name} (index={args.sample_index})",
        f"formula={formula}",
    ]
    if smiles:
        title_parts.append(f"smiles={smiles}")

    fig.update_layout(
        title=" | ".join(title_parts),
        scene=scene_layout,
        scene2=scene_layout,
        scene3=scene_layout,
        width=1800,
        height=700,
        showlegend=False,
    )
    if diff_max <= 0.0:
        fig.add_annotation(
            x=0.84,
            y=1.03,
            xref="paper",
            yref="paper",
            text="Diff panel empty: graph and mol2 voxels are numerically identical for this sample.",
            showarrow=False,
            font=dict(size=12, color="#8b0000"),
        )
    elif eff_diff_threshold < args.diff_threshold:
        fig.add_annotation(
            x=0.84,
            y=1.03,
            xref="paper",
            yref="paper",
            text=f"Diff threshold auto-adjusted: {args.diff_threshold:.4g} -> {eff_diff_threshold:.4g}",
            showarrow=False,
            font=dict(size=12, color="#8b0000"),
        )

    if args.output_html:
        out_path = Path(args.output_html).resolve()
    else:
        out_dir = (PROJECT_ROOT / "artifacts" / "visualizations").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"voxel_vs_mol2_{args.dataset_id.lower()}_{args.sample_index}_{_safe_name(sample_name)}.html"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Wrote visualization: {out_path}")


if __name__ == "__main__":
    main()
