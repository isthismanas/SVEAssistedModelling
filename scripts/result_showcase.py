#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molsim.data import DatasetManager, Mol2ExportConfig, Mol2Exporter
from molsim.metrics import compute_voxel_mse, compute_voxel_overlap
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
    parser = argparse.ArgumentParser(
        description="Build consolidated Plotly showcase(s): 3D voxel vs mol2, slices, and training metrics."
    )
    parser.add_argument("--dataset-id", type=str, default="qm9")
    parser.add_argument("--sample-index", type=int, default=0, help="Single sample index mode.")
    parser.add_argument(
        "--sample-indices",
        type=str,
        default="",
        help="Batch mode. Comma-separated indices, e.g. '0,100,1000'. Overrides --sample-index when provided.",
    )
    parser.add_argument("--mol2-dir", type=str, default=str(PROJECT_ROOT / "data" / "QM9_mol2"))
    parser.add_argument("--auto-export-mol2", action="store_true")

    parser.add_argument("--grid-size", type=int, default=24)
    parser.add_argument("--resolution", type=float, default=0.45)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--iso-threshold", type=float, default=0.10)
    parser.add_argument("--diff-threshold", type=float, default=0.05)

    parser.add_argument(
        "--baseline-artifact",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "baseline_qm9_gap.json"),
    )
    parser.add_argument(
        "--voxel-artifact",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "graph_to_voxel_qm9.json"),
    )
    parser.add_argument(
        "--output-html",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "visualizations" / "result_showcase.html"),
        help="Output file for single-sample mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "visualizations"),
        help="Output directory for batch mode. Ignored in single-sample mode unless --sample-indices is set.",
    )
    return parser.parse_args()


def _parse_sample_indices(args: argparse.Namespace) -> list[int]:
    raw = args.sample_indices.strip()
    if not raw:
        return [int(args.sample_index)]

    chunks = [c.strip() for c in raw.split(",") if c.strip()]
    if not chunks:
        raise ValueError("--sample-indices provided but no valid integers were parsed")

    out: list[int] = []
    for item in chunks:
        out.append(int(item))
    return out


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
        marker=dict(size=4, color=z, colorscale="Viridis", opacity=0.95),
        name=name,
        scene=scene,
    )


def _build_bond_lines(go, pos: np.ndarray, bonds: list[tuple[int, int]], scene: str, name: str):
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
        line=dict(width=2, color="rgba(80,80,80,0.7)"),
        name=name,
        scene=scene,
    )


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _resolve_output_path(args: argparse.Namespace, dataset_id: str, sample_index: int, sample_name: str, batch_mode: bool) -> Path:
    if not batch_mode:
        return Path(args.output_html).resolve()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_sample = _safe_name(sample_name)
    return out_dir / f"result_showcase_{dataset_id.lower()}_{sample_index}_{safe_sample}.html"


def _render_showcase_for_sample(
    go,
    make_subplots,
    *,
    data,
    sample_index: int,
    dataset_id: str,
    args: argparse.Namespace,
    baseline: dict | None,
    voxel: dict | None,
) -> tuple[str, Path]:
    sample_name = _name_for_sample(data, sample_index)

    graph_pos = data.pos.detach().cpu().float()
    graph_z = data.z.detach().cpu().float()
    smiles = _smiles_for_sample(data)
    formula = _formula_from_z(graph_z)

    mol2_dir = Path(args.mol2_dir).resolve()
    mol2_dir.mkdir(parents=True, exist_ok=True)
    mol2_path = mol2_dir / f"{sample_name}.mol2"

    if not mol2_path.exists() and args.auto_export_mol2:
        exporter = Mol2Exporter(Mol2ExportConfig(output_dir=str(mol2_dir), overwrite=False))
        exporter.export_one(data, sample_index)

    if mol2_path.exists():
        mol2_pos, mol2_z, mol2_bonds = parse_mol2_structure(mol2_path)
        mol2_source = str(mol2_path)
    else:
        mol2_pos, mol2_z = graph_pos, graph_z
        mol2_bonds = _unique_bonds(getattr(data, "edge_index", None))
        mol2_source = "fallback_to_graph_pos"

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

    sample_voxel_mse = compute_voxel_mse(mol2_voxel, graph_voxel)
    sample_voxel_overlap = compute_voxel_overlap(mol2_voxel, graph_voxel, threshold=args.iso_threshold)

    half_extent = (args.grid_size * args.resolution) / 2.0
    coords = np.linspace(-half_extent, half_extent, args.grid_size)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")

    fig = make_subplots(
        rows=3,
        cols=3,
        specs=[
            [{"type": "scene"}, {"type": "scene"}, {"type": "scene"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=(
            "3D: Graph -> Voxel",
            "3D: mol2 -> Voxel",
            "3D: |Graph - mol2|",
            "Slice z-mid: Graph Voxel",
            "Slice z-mid: mol2 Voxel",
            "Slice z-mid: |Diff|",
            "Baseline Epoch Curves",
            "Voxel Epoch Curves",
            "Test Metrics Comparison",
        ),
        horizontal_spacing=0.03,
        vertical_spacing=0.08,
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
    fig.add_trace(
        _build_bond_lines(go, graph_pos_np, _unique_bonds(getattr(data, "edge_index", None)), "scene", "graph bonds"),
        row=1,
        col=1,
    )

    fig.add_trace(
        _build_isosurface(go, xx, yy, zz, mol2_voxel, args.iso_threshold, "scene2", "mol2 voxel", "Greens"),
        row=1,
        col=2,
    )
    fig.add_trace(_build_atom_scatter(go, mol2_pos_np, mol2_z_np, "scene2", "mol2 atoms"), row=1, col=2)
    if mol2_bonds:
        fig.add_trace(
            go.Scatter3d(
                x=[coord for i, j in mol2_bonds for coord in (float(mol2_pos_np[i, 0]), float(mol2_pos_np[j, 0]), None)],
                y=[coord for i, j in mol2_bonds for coord in (float(mol2_pos_np[i, 1]), float(mol2_pos_np[j, 1]), None)],
                z=[coord for i, j in mol2_bonds for coord in (float(mol2_pos_np[i, 2]), float(mol2_pos_np[j, 2]), None)],
                mode="lines",
                line=dict(width=2, color="rgba(0,100,0,0.75)"),
                name="mol2 bonds",
                scene="scene2",
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

    mid = args.grid_size // 2
    fig.add_trace(go.Heatmap(z=graph_voxel[:, :, mid], colorscale="Blues", showscale=False, name="graph_slice"), row=2, col=1)
    fig.add_trace(go.Heatmap(z=mol2_voxel[:, :, mid], colorscale="Greens", showscale=False, name="mol2_slice"), row=2, col=2)
    fig.add_trace(go.Heatmap(z=diff_voxel[:, :, mid], colorscale="Reds", showscale=False, name="diff_slice"), row=2, col=3)

    if baseline and baseline.get("history"):
        h = baseline["history"]
        epochs = [row.get("epoch") for row in h]
        fig.add_trace(go.Scatter(x=epochs, y=[row.get("train_mse") for row in h], mode="lines+markers", name="baseline train_mse"), row=3, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=[row.get("val_rmse") for row in h], mode="lines+markers", name="baseline val_rmse"), row=3, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=[row.get("val_mae") for row in h], mode="lines+markers", name="baseline val_mae"), row=3, col=1)

    if voxel and voxel.get("history"):
        h = voxel["history"]
        epochs = [row.get("epoch") for row in h]
        fig.add_trace(go.Scatter(x=epochs, y=[row.get("train_voxel_mse") for row in h], mode="lines+markers", name="voxel train_mse"), row=3, col=2)
        fig.add_trace(go.Scatter(x=epochs, y=[row.get("val_voxel_mse") for row in h], mode="lines+markers", name="voxel val_mse"), row=3, col=2)
        fig.add_trace(go.Scatter(x=epochs, y=[row.get("val_voxel_overlap") for row in h], mode="lines+markers", name="voxel overlap"), row=3, col=2)

    baseline_test = baseline.get("test_metrics", {}) if baseline else {}
    voxel_test = voxel.get("test_metrics", {}) if voxel else {}
    names = sorted(set(baseline_test.keys()) | set(voxel_test.keys()))
    if names:
        fig.add_trace(
            go.Bar(x=names, y=[baseline_test.get(k, None) for k in names], name="baseline test", marker_color="#1f77b4"),
            row=3,
            col=3,
        )
        fig.add_trace(
            go.Bar(x=names, y=[voxel_test.get(k, None) for k in names], name="voxel test", marker_color="#2ca02c"),
            row=3,
            col=3,
        )

    scene_layout = dict(
        xaxis=dict(range=[-half_extent, half_extent], title="x"),
        yaxis=dict(range=[-half_extent, half_extent], title="y"),
        zaxis=dict(range=[-half_extent, half_extent], title="z"),
        aspectmode="cube",
    )

    summary_parts = [
        f"sample={sample_name}",
        f"index={sample_index}",
        f"formula={formula}",
    ]
    if smiles:
        summary_parts.append(f"smiles={smiles}")
    summary_parts.append(f"source={mol2_source}")
    summary_text = " | ".join(summary_parts) + "<br>" + (
        f"sample_voxel_mse={sample_voxel_mse:.6f} | sample_voxel_overlap={sample_voxel_overlap:.6f}"
    )

    fig.update_layout(
        title=f"Result Showcase | {summary_text}",
        scene=scene_layout,
        scene2=scene_layout,
        scene3=scene_layout,
        barmode="group",
        width=1900,
        height=1450,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.01),
    )
    if diff_max <= 0.0:
        fig.add_annotation(
            x=0.84,
            y=0.965,
            xref="paper",
            yref="paper",
            text="Diff panel empty: graph and mol2 voxels are numerically identical for this sample.",
            showarrow=False,
            font=dict(size=11, color="#8b0000"),
        )
    elif eff_diff_threshold < args.diff_threshold:
        fig.add_annotation(
            x=0.84,
            y=0.965,
            xref="paper",
            yref="paper",
            text=f"Diff threshold auto-adjusted: {args.diff_threshold:.4g} -> {eff_diff_threshold:.4g}",
            showarrow=False,
            font=dict(size=11, color="#8b0000"),
        )

    sample_indices = _parse_sample_indices(args)
    batch_mode = len(sample_indices) > 1 or bool(args.sample_indices.strip())
    out_path = _resolve_output_path(args, dataset_id, sample_index, sample_name, batch_mode=batch_mode)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return sample_name, out_path


def main() -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError("plotly is required: pip install plotly") from exc

    args = parse_args()
    sample_indices = _parse_sample_indices(args)

    manager = DatasetManager(project_root=PROJECT_ROOT)
    dataset = manager.load(args.dataset_id)

    baseline = _load_json(Path(args.baseline_artifact).resolve())
    voxel = _load_json(Path(args.voxel_artifact).resolve())

    written: list[Path] = []
    for sample_index in sample_indices:
        if sample_index < 0 or sample_index >= len(dataset):
            raise IndexError(f"sample index {sample_index} out of range for dataset size {len(dataset)}")

        sample_name, out_path = _render_showcase_for_sample(
            go,
            make_subplots,
            data=dataset[sample_index],
            sample_index=sample_index,
            dataset_id=args.dataset_id,
            args=args,
            baseline=baseline,
            voxel=voxel,
        )
        written.append(out_path)
        print(f"Wrote result showcase for sample {sample_index} ({sample_name}): {out_path}")

    if len(written) > 1:
        print(f"Generated {len(written)} showcase files in {written[0].parent}")


if __name__ == "__main__":
    main()
