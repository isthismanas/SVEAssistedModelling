#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import patches
from torch_geometric.nn import global_mean_pool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molsim.data import DatasetManager, QM9TargetAdapter
from molsim.metrics import compute_regression_metrics, compute_voxel_mse, compute_voxel_overlap
from molsim.models import GCNRegressor, GraphToVoxelNet, VoxelAutoencoder
from molsim.spatial import VoxelConfig, normalize_voxel, parse_mol2_structure, voxelize_positions


REPORT_DIR = PROJECT_ROOT / "reports"
ASSET_DIR = REPORT_DIR / "assets"

MOLSIM_FILES = [
    "molsim/__init__.py",
    "molsim/metrics.py",
    "molsim/data/__init__.py",
    "molsim/data/manager.py",
    "molsim/data/qm9.py",
    "molsim/data/mol2_export.py",
    "molsim/data/mol2_voxel_dataset.py",
    "molsim/spatial/__init__.py",
    "molsim/spatial/mol2.py",
    "molsim/spatial/voxelization.py",
    "molsim/models/__init__.py",
    "molsim/models/baselines.py",
    "molsim/models/graph_to_voxel.py",
    "molsim/models/voxel_autoencoder.py",
    "molsim/training/__init__.py",
    "molsim/training/regression.py",
    "molsim/training/voxel.py",
    "molsim/training/autoencoder.py",
]

WORKFLOW_SCRIPTS = [
    "scripts/prepare_datasets.py",
    "scripts/train_qm9_baseline.py",
    "scripts/train_graph_to_voxel.py",
    "scripts/train_mol2_spatial_encoder.py",
    "scripts/export_graph_spatial_embeddings.py",
    "scripts/export_degrademaster_embeddings.py",
    "scripts/export_degrademaster_protac_embeddings.py",
    "scripts/build_degrademaster_sve_dataset.py",
]

FILE_SUMMARIES = {
    "molsim/__init__.py": "Root package file. It carries almost no runtime logic, but it matters architecturally because it marks the namespace boundary and keeps the public top-level API deliberately small.",
    "molsim/metrics.py": "Metric layer. It translates raw arrays into interpretable scientific quantities such as RMSE, MAE, $R^2$, ROC-AUC, and voxel overlap.",
    "molsim/data/__init__.py": "Data-layer export surface. It exists so user code can import key data utilities from one place without knowing submodule paths.",
    "molsim/data/manager.py": "Dataset orchestration layer. It knows where datasets live on disk and how to produce reproducible split indices.",
    "molsim/data/qm9.py": "QM9 adaptation layer. It converts QM9's native 19-target vector into a scalar prediction problem suitable for the baseline trainer.",
    "molsim/data/mol2_export.py": "Graph-to-mol2 bridge. It serializes PyG graph data into Tripos mol2 text so geometric supervision can be reused outside the original graph object.",
    "molsim/data/mol2_voxel_dataset.py": "Mol2-to-voxel dataset layer. It lazily materializes spatial tensors from mol2 files at sample access time.",
    "molsim/spatial/__init__.py": "Spatial export surface. It exposes the geometric primitives that the rest of the repository depends on.",
    "molsim/spatial/mol2.py": "Mol2 parser. It recovers coordinates, atom types, atomic numbers, and bond relationships from Tripos text.",
    "molsim/spatial/voxelization.py": "Core geometry transform. It converts atom-centered 3D coordinates into a dense volumetric occupancy field using Gaussian splatting.",
    "molsim/models/__init__.py": "Model export surface. It keeps downstream imports short and predictable.",
    "molsim/models/baselines.py": "Graph-only scalar baseline. It provides the non-spatial reference model used to measure the added value of geometric representations.",
    "molsim/models/graph_to_voxel.py": "Graph-to-voxel model. It asks whether graph topology and node features are enough to reconstruct a 3D molecular occupancy field.",
    "molsim/models/voxel_autoencoder.py": "Primary SVE model. It is the current best geometric encoder in the repository and produces the latent embeddings used downstream.",
    "molsim/training/__init__.py": "Training export surface. It aggregates trainer classes and config dataclasses into a single import point.",
    "molsim/training/regression.py": "Regression training engine for graph-only scalar prediction.",
    "molsim/training/voxel.py": "Graph-to-voxel training engine with occupancy-weighted reconstruction loss.",
    "molsim/training/autoencoder.py": "Voxel autoencoder training engine with multi-term reconstruction loss.",
    "scripts/prepare_datasets.py": "Entrypoint for dataset preparation and optional mol2 export.",
    "scripts/train_qm9_baseline.py": "End-to-end graph-only baseline training script.",
    "scripts/train_graph_to_voxel.py": "End-to-end graph-to-voxel training script.",
    "scripts/train_mol2_spatial_encoder.py": "End-to-end voxel autoencoder training script.",
    "scripts/export_graph_spatial_embeddings.py": "Entrypoint for exporting graph-derived latent vectors from the graph-to-voxel encoder.",
    "scripts/export_degrademaster_embeddings.py": "Entrypoint for exporting mol2-derived latent vectors from the trained voxel autoencoder.",
    "scripts/export_degrademaster_protac_embeddings.py": "Thin orchestration wrapper that specializes the general mol2 embedding exporter for PROTAC files.",
    "scripts/build_degrademaster_sve_dataset.py": "Feature-fusion script that appends the learned SVE block to downstream PROTAC feature matrices.",
}

BLOCK_ROLES = {
    "SplitConfig": "This dataclass is the reproducibility contract for index splitting. Rather than scattering fractions and seeds through scripts, the code wraps them into one immutable object.",
    "DatasetManager": "This class is the data gateway. All high-level training and export scripts rely on it to resolve dataset roots and generate consistent train/validation/test partitions.",
    "QM9TargetAdapter": "This adapter converts a multi-target chemistry dataset into a single-task regression problem by slicing the correct target column and reshaping it into a scalar tensor.",
    "Mol2ExportConfig": "This dataclass carries file-system intent: where mol2 exports go and whether existing files may be overwritten.",
    "Mol2Exporter": "This class performs structural serialization. It takes graph data with coordinates and writes human-readable Tripos mol2 text that preserves atoms and bonds.",
    "Mol2VoxelDataset": "This dataset object is the lazy bridge from on-disk mol2 files to in-memory dense voxel tensors.",
    "list_mol2_files": "This helper resolves the mol2 corpus and optionally truncates it for smoke tests or smaller experiments.",
    "RegressionMetrics": "This dataclass groups the three scalar regression metrics used throughout the project.",
    "BinaryMetrics": "This dataclass groups binary classification metrics into one strongly-typed return object.",
    "_as_np": "This helper normalizes incoming arrays and lists into a consistent NumPy shape so metric formulas are simple and safe.",
    "compute_regression_metrics": "This function turns prediction error into interpretable scalar regression quality numbers.",
    "compute_binary_metrics": "This function delegates robust binary ranking and classification metric computation to scikit-learn.",
    "compute_voxel_mse": "This function measures dense voxel reconstruction fidelity with elementwise MSE.",
    "compute_voxel_overlap": "This function measures how well thresholded occupied regions overlap between target and reconstruction.",
    "VoxelConfig": "This dataclass defines the geometry of the voxelizer: cube size, physical spacing, Gaussian width, and atomic weighting behavior.",
    "voxelize_positions": "This is the core geometry operator of the whole package. It transforms a molecular point cloud into a volumetric tensor by evaluating Gaussian kernels on a 3D grid.",
    "voxelize_data": "This helper adapts a PyG sample directly into voxelization inputs.",
    "normalize_voxel": "This helper rescales a voxel field by its own maximum intensity, creating a consistent dynamic range for the autoencoder.",
    "GCNRegressor": "This model defines the graph-only baseline. It produces a scalar property estimate from node features and edges without using explicit 3D voxel structure.",
    "GraphToVoxelNet": "This model predicts a full 3D occupancy field from graph structure. It is the model-level test of whether graph information alone can reconstruct geometry.",
    "ConvBlock3D": "This is the reusable convolutional micro-block in the autoencoder: two convolutions, instance normalization, nonlinearities, and optional dropout.",
    "VoxelAutoencoder": "This is the main learned spatial encoder in the repository. Its bottleneck vector is the actual Spatial Voxel Embedding used downstream.",
    "TrainingConfig": "This dataclass collects optimizer and hardware settings for scalar regression.",
    "RegressionTrainer": "This trainer implements the standard supervised learning loop for graph-only regression.",
    "VoxelTrainingConfig": "This dataclass collects optimizer, occupancy, and sparsity settings for graph-to-voxel training.",
    "VoxelTrainer": "This trainer pairs graph inputs with voxel targets and optimizes the graph-to-voxel model.",
    "AutoencoderTrainingConfig": "This dataclass collects reconstruction-loss hyperparameters for voxel autoencoding.",
    "VoxelAutoencoderTrainer": "This trainer computes the multi-term autoencoder loss and performs training/evaluation loops.",
}

TRACE_REFERENCE_MAP = {
    "DatasetManager": ["split_indices_example.png", "qm9_node_features.png"],
    "QM9TargetAdapter": ["qm9_target_vector.png"],
    "Mol2Exporter": ["mol2_text_excerpt.png"],
    "Mol2VoxelDataset": ["voxel_raw_vs_normalized.png"],
    "parse_mol2_structure": ["mol2_point_cloud.png", "atom_coordinate_matrix.png", "centered_coordinate_matrix.png"],
    "voxelize_positions": ["gaussian_center_slice.png", "voxel_3d.png", "voxel_raw_vs_normalized.png"],
    "normalize_voxel": ["voxel_raw_vs_normalized.png"],
    "GCNRegressor": ["baseline_gcn_layers.png", "qm9_node_features.png", "qm9_edge_index.png"],
    "GraphToVoxelNet": ["graph_to_voxel_layers.png", "graph_to_voxel_output_slices.png"],
    "VoxelAutoencoder": ["autoencoder_layers.png", "autoencoder_reconstruction_slices.png", "latent_embedding_vector.png"],
    "RegressionTrainer": ["baseline_prediction_example.png"],
    "VoxelTrainer": ["graph_to_voxel_output_slices.png", "graph_to_voxel_loss_components.png"],
    "VoxelAutoencoderTrainer": ["autoencoder_reconstruction_slices.png", "autoencoder_loss_components.png"],
}


@dataclass
class CodeBlock:
    name: str
    kind: str
    code: str


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)


def read_text(rel_path: str) -> str:
    return (PROJECT_ROOT / rel_path).read_text(encoding="utf-8")


def extract_blocks(rel_path: str) -> list[CodeBlock]:
    source = read_text(rel_path)
    lines = source.splitlines()
    tree = ast.parse(source)
    blocks: list[CodeBlock] = []
    top_nodes = list(tree.body)

    if top_nodes:
        first_top_line = top_nodes[0].lineno - 1
        if first_top_line > 0:
            preamble = "\n".join(lines[:first_top_line]).rstrip()
            if preamble:
                blocks.append(CodeBlock("module_preamble", "module", preamble))
    elif source.strip():
        blocks.append(CodeBlock("module_preamble", "module", source.rstrip()))

    for node in top_nodes:
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign, ast.Expr)):
            continue
        snippet = "\n".join(lines[node.lineno - 1 : node.end_lineno]).rstrip()
        if isinstance(node, ast.ClassDef):
            blocks.append(CodeBlock(node.name, "class", snippet))
        elif isinstance(node, ast.FunctionDef):
            blocks.append(CodeBlock(node.name, "function", snippet))
    return blocks


def explain_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return "spacing-only line that separates logical regions of the block"
    if stripped.startswith("#"):
        return "comment line that documents intent rather than changing runtime state"
    if stripped.startswith("from ") or stripped.startswith("import "):
        return "imports a dependency required by the later runtime logic"
    if stripped.startswith("@dataclass"):
        return "applies the dataclass decorator so the following class becomes a lightweight structured container"
    if stripped.startswith("class "):
        return "declares the class and establishes the abstraction boundary for the code that follows; from this point on, every indented line becomes part of the object definition rather than immediate script execution"
    if stripped.startswith("def "):
        return "declares a new function or method entrypoint; Python stores this callable on the surrounding module or class and does not execute the body yet"
    if stripped.startswith("return "):
        return "returns the computed value to the caller and ends the current control path; mathematically this line says 'the function's output is exactly this expression'"
    if stripped.startswith("if "):
        return "opens a guard branch that handles a special case or validates an assumption; the lines inside this block only matter if the condition evaluates to True"
    if stripped.startswith("elif "):
        return "opens an alternative guard branch after a prior conditional failed"
    if stripped == "else:":
        return "opens the fallback branch when the earlier condition is not met"
    if stripped.startswith("for "):
        return "iterates over a collection and repeats the enclosed logic for each element"
    if stripped.startswith("with "):
        return "enters a managed context so setup and cleanup happen automatically"
    if stripped.startswith("try:"):
        return "starts an exception-handling region for an operation that may fail"
    if stripped.startswith("except "):
        return "handles the exceptional case if the preceding risky operation raises"
    if stripped.startswith("raise "):
        return "throws an explicit exception because the current state violates an assumption"
    if "parser.add_argument" in stripped:
        return "registers one command-line argument so users can configure the workflow without editing code"
    if stripped.startswith("parser = argparse.ArgumentParser"):
        return "creates the command-line parser object for this script"
    if "Path(" in stripped or ".resolve()" in stripped:
        return "normalizes or constructs a filesystem path so later file I/O is deterministic"
    if "torch.load(" in stripped:
        return "loads a serialized PyTorch checkpoint or tensor object from disk"
    if "torch.save(" in stripped:
        return "serializes the current training artifact or checkpoint to disk"
    if "torch.manual_seed" in stripped or "np.random.seed" in stripped or "random.seed" in stripped:
        return "sets a random seed to improve reproducibility"
    if "DataLoader(" in stripped or "GeoDataLoader(" in stripped:
        return "constructs a mini-batch iterator for the subsequent training or evaluation loop"
    if ".append(" in stripped:
        return "accumulates an item into a list for later aggregation"
    if "json.dumps" in stripped or "write_text" in stripped:
        return "materializes a textual artifact on disk so the experiment can be inspected later"
    if stripped.startswith("out.") or stripped.startswith("self."):
        return "updates object state or attributes used by later calls"
    if "=" in stripped:
        return "assigns the result of the expression on the right-hand side to a new or existing variable; conceptually this creates the named intermediate that later lines build on"
    return "participates in the local control flow or expression evaluation of the current block"


def line_math_intuition(line: str) -> tuple[str, str]:
    stripped = line.strip()
    if not stripped:
        return ("No mathematical update occurs on this line.", "This line is visual structure for human readers.")
    if stripped.startswith("class "):
        return ("No tensor math is executed yet.", "The class is a template describing what math will happen later when an instance is created or called.")
    if stripped.startswith("def "):
        return ("No tensor math is executed yet.", "The function signature names the inputs and output contract before any runtime values flow through it.")
    if stripped.startswith("return self.block(x)"):
        return ("The output tensor is defined as the composition of all layers stored in `self.block` applied to input `x`.", "The whole block behaves like one macro-layer: feed the tensor in, get the transformed tensor out.")
    if "nn.Conv3d" in stripped:
        return ("A learned 3D convolution computes weighted local sums over a $3\\times3\\times3$ neighborhood and mixes channels into a new feature basis.", "This line tells the model what local volumetric patterns to look for, such as blobs, edges, or occupancy gradients.")
    if "nn.InstanceNorm3d" in stripped:
        return ("For each sample and channel, activations are normalized by that sample's own mean and variance, then optionally rescaled and shifted by learnable affine parameters.", "This stabilizes training by preventing one channel from dominating simply because it has a larger raw numeric scale.")
    if "nn.LeakyReLU" in stripped:
        return ("The activation is transformed by $f(x)=x$ for positive values and $f(x)=0.1x$ for negative values.", "Negative responses are not fully killed, so the network preserves some gradient flow even when a feature is currently 'off'.")
    if "nn.Dropout3d" in stripped:
        return ("Whole feature responses are randomly zeroed during training with probability equal to the dropout rate.", "This discourages the model from depending too heavily on a single channel and improves robustness.")
    if stripped.startswith("layers:"):
        return ("A Python list is created to hold a symbolic sequence of layer objects.", "The model is being assembled as an ordered pipeline before any actual tensor is sent through it.")
    if stripped.startswith("self.block = nn.Sequential"):
        return ("The ordered layers are wrapped into a function composition $f_n(\dots f_2(f_1(x))\dots)$.", "This turns a list of separate layer objects into one callable block.")
    if stripped.startswith("if dropout > 0"):
        return ("This is a hyperparameter-controlled structural branch rather than a tensor update.", "The architecture itself changes depending on whether regularization is requested.")
    if stripped.startswith("layers.append"):
        return ("A new transformation is appended to the end of the composition pipeline.", "The block becomes deeper only when dropout is enabled.")
    if stripped.startswith("super().__init__()"):
        return ("No tensor math occurs; the parent `nn.Module` state is initialized.", "Without this call, PyTorch would not correctly register parameters and submodules.")
    if stripped.startswith("return "):
        return ("The function output is identified explicitly by the returned expression.", "This is where the computation becomes externally observable to the caller.")
    if "=" in stripped:
        return ("This line defines an intermediate object or value used by later steps.", "The code is naming a piece of the computation so the next lines can build on it.")
    return ("This line contributes to the local computation or control flow.", "Its main role is to support the surrounding algorithmic structure.")


def line_sample_effect(line: str, block_name: str) -> str:
    stripped = line.strip()
    if block_name == "ConvBlock3D":
        if "nn.Conv3d(in_channels, out_channels" in stripped:
            return "For the worked voxel sample, this is the first true learned spatial transform: the single-channel occupancy cube is converted into `out_channels` learned feature maps while preserving spatial resolution because padding is 1."
        if "nn.InstanceNorm3d" in stripped:
            return "For the worked sample, each output channel is recentered and rescaled using that sample's own statistics, so bright channels and dim channels enter the next stage on a comparable numeric footing."
        if "nn.LeakyReLU" in stripped:
            return "For the worked sample, negative responses become small negative values rather than exact zeros, so subtle 'anti-features' remain visible in the activation maps."
        if "nn.Conv3d(out_channels, out_channels" in stripped:
            return "For the worked sample, the second convolution remixes the first set of learned volumetric features into a refined representation with the same spatial size."
        if stripped.startswith("return self.block(x)"):
            return "For the worked sample, the input voxel tensor is pushed through every layer in sequence; the report's `convblock3d_internal.png` figure visualizes those intermediate tensors."
    if block_name == "voxelize_positions":
        return "For the worked molecule, this line contributes to turning a small atom-coordinate matrix into the dense voxel cube shown in the voxelization figures."
    if block_name in {"GraphToVoxelNet", "VoxelAutoencoder", "GCNRegressor"}:
        return "For the worked sample, this line participates in the actual forward-pass tensor trace shown in the execution figures for this block."
    if stripped.startswith("return "):
        return "This is the point where the current block hands its transformed version of the worked sample to the caller."
    return "This line either changes how the worked sample will be processed later or documents the control structure that governs that processing."


def markdown_line_table(code: str, block_name: str) -> str:
    rows = ["| Code Line | Framework Meaning | Mathematical Meaning | Effect On Worked Sample |", "|---|---|---|---|"]
    for raw in code.splitlines():
        safe_code = raw.replace("|", "\\|")
        framework = explain_line(raw).replace("|", "\\|")
        math_text, intuition = line_math_intuition(raw)
        math_text = (math_text + " " + intuition).replace("|", "\\|")
        sample_text = line_sample_effect(raw, block_name).replace("|", "\\|")
        rows.append(f"| `{safe_code}` | {framework} | {math_text} | {sample_text} |")
    return "\n".join(rows)


def fig_save(filename: str) -> str:
    path = ASSET_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    return f"assets/{filename}"


def fig_heatmap(matrix: np.ndarray, title: str, filename: str, xlabels: list[str] | None = None, ylabels: list[str] | None = None, cmap: str = "coolwarm") -> str:
    fig, ax = plt.subplots(figsize=(max(6, matrix.shape[1] * 0.35), max(3.5, matrix.shape[0] * 0.28)))
    image = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_title(title)
    if xlabels is not None:
        ax.set_xticks(range(len(xlabels)), labels=xlabels, rotation=45, ha="right")
    if ylabels is not None:
        ax.set_yticks(range(len(ylabels)), labels=ylabels)
    fig.colorbar(image, ax=ax, fraction=0.046)
    return fig_save(filename)


def fig_bar(values: list[float], labels: list[str], title: str, filename: str, horizontal: bool = False) -> str:
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.8), 4.5))
    if horizontal:
        ax.barh(range(len(labels)), values, color="#93c5fd")
        ax.set_yticks(range(len(labels)), labels=labels)
        ax.invert_yaxis()
    else:
        ax.bar(range(len(labels)), values, color="#93c5fd")
        ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_title(title)
    return fig_save(filename)


def fig_text_block(lines: list[str], title: str, filename: str) -> str:
    fig, ax = plt.subplots(figsize=(11, max(3, 0.45 * len(lines))))
    ax.axis("off")
    ax.text(0.01, 0.98, title, va="top", ha="left", fontsize=14, weight="bold")
    ax.text(0.01, 0.90, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)
    return fig_save(filename)


def fig_point_cloud(coords: np.ndarray, atomic_numbers: np.ndarray, title: str, filename: str) -> str:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=atomic_numbers, cmap="viridis", s=32 + 4 * atomic_numbers)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.colorbar(scatter, ax=ax, pad=0.08, shrink=0.8, label="atomic number")
    return fig_save(filename)


def fig_voxel_points(voxel: np.ndarray, title: str, filename: str, threshold_ratio: float = 0.20) -> str:
    volume = voxel[0]
    threshold = max(0.10, float(volume.max()) * threshold_ratio)
    xs, ys, zs = np.where(volume >= threshold)
    values = volume[xs, ys, zs]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(xs, ys, zs, c=values, cmap="magma", s=12 + 30 * values)
    ax.set_title(title)
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    ax.set_zlabel("k")
    fig.colorbar(scatter, ax=ax, pad=0.08, shrink=0.8, label="occupancy")
    return fig_save(filename)


def fig_slice_grid(volumes: list[np.ndarray], titles: list[str], filename: str) -> str:
    cols = 3
    rows = len(volumes)
    fig, axes = plt.subplots(rows, cols, figsize=(12, max(3.5, 2.8 * rows)))
    axes = np.atleast_2d(axes)
    for row_idx, (volume, row_title) in enumerate(zip(volumes, titles)):
        mid = volume.shape[0] // 2
        panels = [volume[mid], volume[:, mid, :], volume[:, :, mid]]
        labels = ["xy", "xz", "yz"]
        for col_idx, panel in enumerate(panels):
            image = axes[row_idx, col_idx].imshow(panel, cmap="viridis")
            axes[row_idx, col_idx].set_title(f"{row_title} {labels[col_idx]}")
            fig.colorbar(image, ax=axes[row_idx, col_idx], fraction=0.046)
    return fig_save(filename)


def fig_stage_heatmaps(stages: list[np.ndarray], titles: list[str], filename: str) -> str:
    fig, axes = plt.subplots(len(stages), 1, figsize=(12, max(3.5, 2.6 * len(stages))))
    axes = np.atleast_1d(axes)
    for ax, mat, title in zip(axes, stages, titles):
        image = ax.imshow(mat, aspect="auto", cmap="coolwarm")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.046)
    return fig_save(filename)


def fig_pipeline() -> str:
    fig, ax = plt.subplots(figsize=(15, 4.2))
    ax.axis("off")
    labels = [
        "QM9 graph / mol2 text",
        "parse coordinates + targets",
        "voxelize with Gaussians",
        "3D encoder / decoder",
        "latent SVE",
        "downstream feature fusion",
    ]
    xs = np.linspace(0.03, 0.84, len(labels))
    for x, label in zip(xs, labels):
        box = patches.FancyBboxPatch((x, 0.35), 0.12, 0.24, boxstyle="round,pad=0.02", facecolor="#dbeafe")
        ax.add_patch(box)
        ax.text(x + 0.06, 0.47, label, ha="center", va="center", fontsize=11)
    for i in range(len(labels) - 1):
        ax.annotate("", xy=(xs[i + 1], 0.47), xytext=(xs[i] + 0.12, 0.47), arrowprops={"arrowstyle": "->", "lw": 2})
    ax.text(0.5, 0.86, "molsim End-to-End Execution Pipeline", ha="center", va="center", fontsize=17, weight="bold")
    ax.text(0.5, 0.15, "The report below traces a single real QM9 molecule through this chain using actual tensors and trained checkpoints whenever available.", ha="center", va="center", fontsize=10)
    return fig_save("pipeline_overview.png")


def fig_package_map() -> str:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    groups = [
        ("data", ["manager", "qm9", "mol2_export", "mol2_voxel_dataset"], "#dcfce7"),
        ("spatial", ["mol2", "voxelization"], "#ede9fe"),
        ("models", ["baselines", "graph_to_voxel", "voxel_autoencoder"], "#fee2e2"),
        ("training", ["regression", "voxel", "autoencoder"], "#fef3c7"),
        ("metrics", ["metrics"], "#dbeafe"),
    ]

    label_x = 0.05
    label_w = 0.16
    item_w = 0.13
    item_gap = 0.03
    start_x = 0.30
    row_gap = 0.17
    top_y = 0.83
    box_h = 0.09

    ax.text(0.5, 0.96, "molsim Package Structure", ha="center", va="center", fontsize=19, weight="bold")

    max_items = max(len(items) for _, items, _ in groups)
    table_left = start_x - 0.02
    table_right = start_x + max_items * item_w + (max_items - 1) * item_gap + 0.02
    table_top = top_y + 0.06
    table_bottom = top_y - (len(groups) - 1) * row_gap - 0.08
    ax.add_patch(
        patches.FancyBboxPatch(
            (table_left, table_bottom),
            table_right - table_left,
            table_top - table_bottom,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            facecolor="#f8fafc",
            edgecolor="#cbd5e1",
            linewidth=1.0,
        )
    )

    for row_idx, (group, items, color) in enumerate(groups):
        y_center = top_y - row_idx * row_gap
        label_y = y_center - box_h / 2

        label_box = patches.FancyBboxPatch(
            (label_x, label_y),
            label_w,
            box_h,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor=color,
            edgecolor="#111827",
            linewidth=1.6,
        )
        ax.add_patch(label_box)
        ax.text(label_x + label_w / 2, y_center, group, ha="center", va="center", fontsize=14, weight="bold")

        line_start = label_x + label_w + 0.015
        line_end = start_x - 0.025
        ax.plot([line_start, line_end], [y_center, y_center], color="#94a3b8", linewidth=1.6)

        for item_idx, item in enumerate(items):
            x = start_x + item_idx * (item_w + item_gap)
            item_box = patches.FancyBboxPatch(
                (x, label_y),
                item_w,
                box_h,
                boxstyle="round,pad=0.015,rounding_size=0.02",
                facecolor="#ffffff",
                edgecolor="#374151",
                linewidth=1.3,
            )
            ax.add_patch(item_box)
            ax.text(x + item_w / 2, y_center, item, ha="center", va="center", fontsize=11, color="#111827")

        for item_idx in range(1, len(items)):
            prev_x = start_x + (item_idx - 1) * (item_w + item_gap)
            curr_x = start_x + item_idx * (item_w + item_gap)
            mid_y = y_center
            ax.plot(
                [prev_x + item_w + 0.01, curr_x - 0.01],
                [mid_y, mid_y],
                color="#cbd5e1",
                linewidth=1.0,
                linestyle="--",
            )

    return fig_save("molsim_package_map.png")


def load_qm9_sample() -> dict[str, Any]:
    manager = DatasetManager(project_root=PROJECT_ROOT)
    dataset = manager.load("qm9")
    sample = dataset[0]
    gap_adapter = QM9TargetAdapter("gap")
    sample_gap = gap_adapter.apply(sample)
    return {
        "dataset": dataset,
        "sample": sample,
        "sample_gap": sample_gap,
        "name": str(getattr(sample, "name", "gdb_1")),
    }


def find_sample_mol2(name: str) -> Path:
    preferred = PROJECT_ROOT / "data" / "QM9_mol2" / f"{name}.mol2"
    if preferred.exists():
        return preferred
    for root in [PROJECT_ROOT / "data" / "QM9_mol2", PROJECT_ROOT / "data" / "PROTAC" / "ligase_pocket"]:
        if root.exists():
            matches = sorted(root.glob("*.mol2"))
            if matches:
                return matches[0]
    raise FileNotFoundError("No mol2 sample found.")


def load_autoencoder() -> tuple[VoxelAutoencoder, VoxelConfig, str]:
    path = PROJECT_ROOT / "artifacts" / "voxel_autoencoder_qm9.pt"
    checkpoint = torch.load(path, map_location="cpu")
    model_cfg = checkpoint["model_config"]
    voxel_cfg = checkpoint["voxel_config"]
    model = VoxelAutoencoder(
        grid_size=int(model_cfg["grid_size"]),
        embedding_dim=int(model_cfg["embedding_dim"]),
        base_channels=int(model_cfg["base_channels"]),
        dropout=float(model_cfg["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    cfg = VoxelConfig(
        grid_size=int(voxel_cfg["grid_size"]),
        resolution=float(voxel_cfg["resolution"]),
        sigma=float(voxel_cfg["sigma"]),
        use_atomic_weights=bool(voxel_cfg["use_atomic_weights"]),
    )
    return model, cfg, str(voxel_cfg.get("input_normalization", "none"))


def load_graph_to_voxel() -> tuple[GraphToVoxelNet, VoxelConfig]:
    path = PROJECT_ROOT / "artifacts" / "graph_to_voxel_qm9.pt"
    checkpoint = torch.load(path, map_location="cpu")
    model_cfg = checkpoint["model_config"]
    voxel_cfg = checkpoint["voxel_config"]
    model = GraphToVoxelNet(
        in_channels=int(model_cfg["in_channels"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        latent_dim=int(model_cfg["latent_dim"]),
        grid_size=int(model_cfg["grid_size"]),
        dropout=float(model_cfg["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    cfg = VoxelConfig(
        grid_size=int(voxel_cfg["grid_size"]),
        resolution=float(voxel_cfg["resolution"]),
        sigma=float(voxel_cfg["sigma"]),
        use_atomic_weights=bool(voxel_cfg["use_atomic_weights"]),
    )
    return model, cfg


def node_matrix_preview(tensor: torch.Tensor, max_features: int = 24) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    return arr[:, : min(arr.shape[1], max_features)]


def volume_mean_slice(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 5:
        arr = arr[0].mean(axis=0)
    elif arr.ndim == 4:
        arr = arr.mean(axis=0)
    mid = arr.shape[0] // 2
    return arr[mid]


def trace_baseline(sample_gap) -> dict[str, Any]:
    model = GCNRegressor(in_channels=int(sample_gap.x.shape[-1]))
    x = sample_gap.x.float()
    edge_index = sample_gap.edge_index
    batch = torch.zeros(x.shape[0], dtype=torch.long)
    with torch.no_grad():
        h1 = F.relu(model.conv1(x, edge_index))
        h2 = F.relu(model.conv2(model.dropout(h1), edge_index))
        h3 = F.relu(model.conv3(model.dropout(h2), edge_index))
        pooled = global_mean_pool(h3, batch)
        pred = model.head(pooled).view(-1)
    return {
        "model": model,
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "pooled": pooled,
        "pred": pred,
        "target": sample_gap.y.view(-1).float(),
    }


def trace_graph_to_voxel(sample) -> dict[str, Any]:
    model, voxel_cfg = load_graph_to_voxel()
    x = sample.x.float()
    edge_index = sample.edge_index
    batch = torch.zeros(x.shape[0], dtype=torch.long)
    with torch.no_grad():
        h1 = F.relu(model.conv1(x, edge_index))
        h2 = F.relu(model.conv2(model.dropout(h1), edge_index))
        h3 = F.relu(model.conv3(model.dropout(h2), edge_index))
        pooled = global_mean_pool(h3, batch)
        latent = model.fc_latent(pooled)
        decode_fc = model.fc_decode(latent).view(-1, 64, model.init_size, model.init_size, model.init_size)
        deconv1 = F.relu(model.deconv1(decode_fc))
        out = F.softplus(model.deconv2(deconv1)) - math.log(2.0)
        out = torch.clamp_min(out, 0.0)
        target = voxelize_positions(sample.pos, sample.z, voxel_cfg).unsqueeze(0)
    occ_mask = target >= 0.1
    base = F.mse_loss(out, target)
    occ_loss = F.mse_loss(out[occ_mask], target[occ_mask]) if torch.any(occ_mask) else out.new_tensor(0.0)
    sparsity = out.mean()
    return {
        "voxel_cfg": voxel_cfg,
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "pooled": pooled,
        "latent": latent,
        "decode_fc": decode_fc,
        "deconv1": deconv1,
        "out": out,
        "target": target,
        "loss_components": {
            "base_mse": float(base.item()),
            "occupied_mse": float(occ_loss.item()),
            "sparsity": float(sparsity.item()),
        },
    }


def trace_autoencoder(mol2_path: Path) -> dict[str, Any]:
    model, voxel_cfg, normalization = load_autoencoder()
    coords_t, atomic_t, bonds = parse_mol2_structure(mol2_path)
    voxel = voxelize_positions(coords_t, atomic_t, voxel_cfg)
    raw_voxel = voxel.clone()
    if normalization == "per_sample_max":
        voxel = normalize_voxel(voxel)
    x = voxel.unsqueeze(0).float()
    with torch.no_grad():
        x1 = model.enc1(x)
        down1 = model.down1(x1)
        x2 = model.enc2(down1)
        down2 = model.down2(x2)
        x3 = model.enc3(down2)
        down3 = model.down3(x3)
        bottleneck = model.bottleneck(down3)
        flat = bottleneck.reshape(bottleneck.shape[0], -1)
        embedding = model.fc_embed(flat)
        decoded_flat = model.fc_decode(embedding).view_as(bottleneck)
        up1 = model.up1(decoded_flat)
        dec1 = model.dec1(torch.cat((up1, x3), dim=1))
        up2 = model.up2(dec1)
        dec2 = model.dec2(torch.cat((up2, x2), dim=1))
        up3 = model.up3(dec2)
        dec3 = model.dec3(torch.cat((up3, x1), dim=1))
        recon = torch.sigmoid(model.out_conv(dec3))
    occ_mask = x >= 0.1
    base = F.mse_loss(recon, x)
    l1 = F.l1_loss(recon, x)
    occ = F.smooth_l1_loss(recon[occ_mask], x[occ_mask]) if torch.any(occ_mask) else recon.new_tensor(0.0)
    target_occ = (x >= 0.1).float()
    pred_occ = recon.clamp(0.0, 1.0)
    intersection = torch.sum(pred_occ * target_occ)
    denom = torch.sum(pred_occ) + torch.sum(target_occ) + 1e-8
    dice = 1.0 - ((2.0 * intersection + 1e-8) / denom)
    sparsity = recon.mean()
    return {
        "mol2_path": mol2_path,
        "coords": coords_t,
        "atomic": atomic_t,
        "bonds": bonds,
        "voxel_cfg": voxel_cfg,
        "normalization": normalization,
        "raw_voxel": raw_voxel,
        "voxel": voxel,
        "x1": x1,
        "down1": down1,
        "x2": x2,
        "down2": down2,
        "x3": x3,
        "down3": down3,
        "bottleneck": bottleneck,
        "embedding": embedding,
        "decoded_flat": decoded_flat,
        "up1": up1,
        "dec1": dec1,
        "up2": up2,
        "dec2": dec2,
        "up3": up3,
        "dec3": dec3,
        "recon": recon,
        "loss_components": {
            "mse": float(base.item()),
            "l1": float(l1.item()),
            "occupied": float(occ.item()),
            "dice": float(dice.item()),
            "sparsity": float(sparsity.item()),
        },
    }


def trace_convblock3d(ae_ctx: dict[str, Any]) -> dict[str, Any]:
    model, _, _ = load_autoencoder()
    x = ae_ctx["voxel"].unsqueeze(0).float()
    layers = list(model.enc1.block.children())
    outputs = []
    current = x
    with torch.no_grad():
        outputs.append(("input", current.clone()))
        for idx, layer in enumerate(layers, start=1):
            current = layer(current)
            outputs.append((f"layer_{idx}_{layer.__class__.__name__}", current.clone()))
    return {
        "input_shape": tuple(x.shape),
        "outputs": outputs,
        "out_channels": int(outputs[-1][1].shape[1]),
    }


def generate_assets(qm9_ctx: dict[str, Any], baseline_ctx: dict[str, Any], g2v_ctx: dict[str, Any], ae_ctx: dict[str, Any], convblock_ctx: dict[str, Any]) -> dict[str, str]:
    assets: dict[str, str] = {}
    sample = qm9_ctx["sample"]
    sample_gap = qm9_ctx["sample_gap"]
    sample_name = qm9_ctx["name"]
    coords = ae_ctx["coords"].numpy()
    atomic = ae_ctx["atomic"].numpy()
    centered = coords - coords.mean(axis=0, keepdims=True)
    ylabels_atoms = [f"atom {i}" for i in range(coords.shape[0])]

    assets["pipeline"] = fig_pipeline()
    assets["package_map"] = fig_package_map()
    assets["qm9_node_features"] = fig_heatmap(sample.x.numpy(), f"QM9 Node Feature Matrix for {sample_name}", "qm9_node_features.png")
    assets["qm9_edge_index"] = fig_heatmap(sample.edge_index.numpy(), f"QM9 Edge Index Matrix for {sample_name}", "qm9_edge_index.png", cmap="Blues")
    assets["qm9_positions"] = fig_heatmap(sample.pos.numpy(), f"QM9 Position Matrix for {sample_name}", "qm9_positions.png", xlabels=["x", "y", "z"], ylabels=ylabels_atoms)
    assets["qm9_target_vector"] = fig_bar(sample.y.view(-1).tolist(), [str(i) for i in range(sample.y.numel())], "QM9 Raw Target Vector (19 tasks)", "qm9_target_vector.png")
    assets["split_indices"] = fig_bar([16, 2, 2], ["train", "val", "test"], "Example DatasetManager.split_indices(20)", "split_indices_example.png")
    assets["point_cloud"] = fig_point_cloud(coords, atomic, f"mol2 Point Cloud: {ae_ctx['mol2_path'].stem}", "mol2_point_cloud.png")
    assets["atom_matrix"] = fig_heatmap(np.concatenate([coords, atomic[:, None]], axis=1), "Parsed Atom Matrix [x, y, z, Z]", "atom_coordinate_matrix.png", xlabels=["x", "y", "z", "Z"], ylabels=ylabels_atoms)
    assets["centered_matrix"] = fig_heatmap(np.concatenate([centered, atomic[:, None]], axis=1), "Centered Atom Matrix [x-μx, y-μy, z-μz, Z]", "centered_coordinate_matrix.png", xlabels=["cx", "cy", "cz", "Z"], ylabels=ylabels_atoms)
    mol2_lines = ae_ctx["mol2_path"].read_text().splitlines()[:18]
    assets["mol2_excerpt"] = fig_text_block(mol2_lines, "mol2 Text Excerpt", "mol2_text_excerpt.png")
    assets["voxel_3d"] = fig_voxel_points(ae_ctx["voxel"].numpy(), "Voxel Occupancy from mol2 Coordinates", "voxel_3d.png")
    assets["gaussian_slice"] = fig_heatmap(ae_ctx["raw_voxel"][0].numpy()[ae_ctx["raw_voxel"].shape[-1] // 2], "Central Slice of Raw Gaussian-Splatted Voxel Field", "gaussian_center_slice.png", cmap="magma")
    raw_norm_panels = [ae_ctx["raw_voxel"][0].numpy(), ae_ctx["voxel"][0].numpy()]
    assets["voxel_raw_vs_normalized"] = fig_slice_grid(raw_norm_panels, ["raw voxel", "normalized voxel"], "voxel_raw_vs_normalized.png")
    assets["baseline_layers"] = fig_stage_heatmaps(
        [sample.x.numpy(), node_matrix_preview(baseline_ctx["h1"]), node_matrix_preview(baseline_ctx["h2"]), node_matrix_preview(baseline_ctx["h3"]), baseline_ctx["pooled"].numpy()],
        ["input node features", "after conv1 + relu", "after conv2 + relu", "after conv3 + relu", "global mean pooled vector"],
        "baseline_gcn_layers.png",
    )
    assets["baseline_prediction"] = fig_bar(
        [float(baseline_ctx["target"].item()), float(baseline_ctx["pred"].item())],
        ["true gap", "baseline prediction"],
        "Single-Sample Baseline Prediction Example",
        "baseline_prediction_example.png",
    )
    assets["g2v_layers"] = fig_stage_heatmaps(
        [
            sample.x.numpy(),
            node_matrix_preview(g2v_ctx["h1"]),
            node_matrix_preview(g2v_ctx["h2"]),
            node_matrix_preview(g2v_ctx["h3"]),
            g2v_ctx["pooled"].numpy(),
            g2v_ctx["latent"].numpy(),
        ],
        ["graph input", "conv1 activations", "conv2 activations", "conv3 activations", "pooled molecular vector", "latent vector"],
        "graph_to_voxel_layers.png",
    )
    assets["g2v_output_slices"] = fig_slice_grid([g2v_ctx["target"][0, 0].numpy(), g2v_ctx["out"][0, 0].numpy()], ["target voxel", "predicted voxel"], "graph_to_voxel_output_slices.png")
    assets["g2v_loss"] = fig_bar(list(g2v_ctx["loss_components"].values()), list(g2v_ctx["loss_components"].keys()), "Graph-to-Voxel Loss Components on Worked Sample", "graph_to_voxel_loss_components.png")
    assets["ae_layers"] = fig_slice_grid(
        [
            ae_ctx["voxel"][0].numpy(),
            volume_mean_slice(ae_ctx["x1"]).reshape(1, *volume_mean_slice(ae_ctx["x1"]).shape),
            volume_mean_slice(ae_ctx["x2"]).reshape(1, *volume_mean_slice(ae_ctx["x2"]).shape),
            volume_mean_slice(ae_ctx["x3"]).reshape(1, *volume_mean_slice(ae_ctx["x3"]).shape),
            volume_mean_slice(ae_ctx["bottleneck"]).reshape(1, *volume_mean_slice(ae_ctx["bottleneck"]).shape),
        ],
        ["input voxel", "enc1 mean-channel", "enc2 mean-channel", "enc3 mean-channel", "bottleneck mean-channel"],
        "autoencoder_layers.png",
    )
    assets["ae_reconstruction"] = fig_slice_grid([ae_ctx["voxel"][0].numpy(), ae_ctx["recon"][0, 0].numpy()], ["input voxel", "reconstruction"], "autoencoder_reconstruction_slices.png")
    assets["ae_embedding"] = fig_heatmap(ae_ctx["embedding"].numpy(), "Voxel Autoencoder Latent Embedding", "latent_embedding_vector.png", cmap="coolwarm")
    assets["ae_loss"] = fig_bar(list(ae_ctx["loss_components"].values()), list(ae_ctx["loss_components"].keys()), "Voxel Autoencoder Loss Components on Worked Sample", "autoencoder_loss_components.png")
    convblock_stage_mats = []
    convblock_stage_titles = []
    for label, tensor in convblock_ctx["outputs"]:
        arr = tensor.detach().cpu().numpy()
        if arr.ndim == 5:
            arr = arr[0].mean(axis=0)
        mid = arr.shape[0] // 2
        convblock_stage_mats.append(arr[mid])
        convblock_stage_titles.append(label.replace("_", " "))
    assets["convblock3d_internal"] = fig_stage_heatmaps(convblock_stage_mats, convblock_stage_titles, "convblock3d_internal.png")
    results_path = PROJECT_ROOT / "results" / "degrademaster_results.json"
    if results_path.exists():
        payload = json.loads(results_path.read_text())
        entries = payload if isinstance(payload, list) else payload.get("runs", [])
        rows: list[tuple[str, float]] = []
        for entry in entries:
            name = str(entry.get("name", ""))
            metrics = entry.get("metrics", {})
            if name.startswith("regression_") and "mean_rmse" in metrics:
                rows.append((name.replace("regression_", ""), float(metrics["mean_rmse"])))
        if rows:
            labels, values = zip(*rows[:12])
            assets["results_summary"] = fig_bar(list(values), list(labels), "Downstream Regression mean RMSE Summary", "regression_results_summary.png", horizontal=True)
    sve_meta_path = PROJECT_ROOT / "data" / "PROTAC_sve" / "sve_merge_metadata.json"
    if sve_meta_path.exists():
        meta = json.loads(sve_meta_path.read_text())
        assets["sve_merge"] = fig_bar(
            [meta["original_protac_dim"], meta["embedding_dim"], meta["new_protac_dim"]],
            ["original", "SVE", "merged"],
            "Downstream SVE Feature Concatenation",
            "downstream_sve_feature_merge.png",
        )
    return assets


def metrics_and_math_sections(qm9_ctx: dict[str, Any], baseline_ctx: dict[str, Any], g2v_ctx: dict[str, Any], ae_ctx: dict[str, Any]) -> str:
    centroid = ae_ctx["coords"].mean(dim=0).numpy()
    centered = ae_ctx["coords"].numpy() - centroid[None, :]
    atom_idx = int(np.argmin(np.linalg.norm(centered, axis=1)))
    ref_atom = centered[atom_idx]
    cfg = ae_ctx["voxel_cfg"]
    half_extent = (cfg.grid_size * cfg.resolution) / 2.0
    axis_values = np.linspace(-half_extent, half_extent, cfg.grid_size)
    center_coord = np.array([axis_values[cfg.grid_size // 2]] * 3)
    dist_sq = float(np.sum((center_coord - ref_atom) ** 2))
    weight = float(ae_ctx["atomic"][atom_idx].item() / 6.0)
    gaussian = weight * math.exp(-dist_sq / (2.0 * cfg.sigma**2))
    voxel_mse = compute_voxel_mse(ae_ctx["voxel"].numpy(), ae_ctx["recon"].squeeze(0).numpy())
    voxel_overlap = compute_voxel_overlap(ae_ctx["voxel"].numpy(), ae_ctx["recon"].squeeze(0).numpy())
    reg_metrics = compute_regression_metrics([float(baseline_ctx["target"].item())], [float(baseline_ctx["pred"].item())])
    return "\n".join(
        [
            "## Worked Sample Mathematics",
            "",
            f"The worked molecule is `{qm9_ctx['name']}`. Its centered coordinate matrix is obtained by subtracting the centroid",
            "",
            f"$$\\mu = ({centroid[0]:.6f}, {centroid[1]:.6f}, {centroid[2]:.6f}).$$",
            "",
            "from every atom position. If one representative centered atom is",
            "",
            f"$$r_a = ({ref_atom[0]:.6f}, {ref_atom[1]:.6f}, {ref_atom[2]:.6f}),$$",
            "",
            "and the center voxel coordinate is",
            "",
            f"$$c_{'{mid}'} = ({center_coord[0]:.6f}, {center_coord[1]:.6f}, {center_coord[2]:.6f}),$$",
            "",
            "then the squared distance used by the Gaussian kernel is",
            "",
            f"$$\\lVert c_{'{mid}'} - r_a \\rVert_2^2 = {dist_sq:.6f}. $$",
            "",
            "With atomic weighting enabled, the worked contribution is",
            "",
            f"$$w_a = Z_a / 6 = {weight:.6f},$$",
            "",
            f"$$w_a \\exp\\left(-\\frac{{{dist_sq:.6f}}}{{2({cfg.sigma})^2}}\\right) = {gaussian:.6f}. $$",
            "",
            "For the trained voxel autoencoder on this one sample, the observed reconstruction metrics are:",
            "",
            f"- voxel MSE: `{voxel_mse:.8f}`",
            f"- voxel overlap: `{voxel_overlap:.8f}`",
            "",
            "For the graph-only baseline trace shown in this report, only a single worked sample is forwarded. That is enough to explain tensor flow but not enough to produce meaningful aggregate evaluation. Still, the scalar formulas used by `compute_regression_metrics()` reduce here to:",
            "",
            f"- sample target: `{float(baseline_ctx['target'].item()):.8f}`",
            f"- sample prediction: `{float(baseline_ctx['pred'].item()):.8f}`",
            f"- resulting one-sample RMSE: `{reg_metrics.rmse:.8f}`",
            f"- resulting one-sample MAE: `{reg_metrics.mae:.8f}`",
            "",
        ]
    )


def block_details(block: CodeBlock) -> str:
    role = BLOCK_ROLES.get(block.name, "This block participates in the framework structure of the file and is included because understanding its exact source is part of understanding the full module.")
    refs = TRACE_REFERENCE_MAP.get(block.name, [])
    ref_text = ""
    if refs:
        linked = ", ".join(f"`reports/assets/{ref}`" for ref in refs)
        ref_text = f"Relevant execution-trace figures: {linked}."
    return "\n".join(
        [
            role,
            "",
            "From a code-framework perspective, this block is reproduced in full so the reader can connect the abstract algorithm to the exact control flow, path handling, tensor construction, and serialization behavior that the repository actually executes.",
            "",
            ref_text,
        ]
    ).strip()


def custom_block_walkthrough(
    rel_path: str,
    block: CodeBlock,
    qm9_ctx: dict[str, Any],
    baseline_ctx: dict[str, Any],
    g2v_ctx: dict[str, Any],
    ae_ctx: dict[str, Any],
    convblock_ctx: dict[str, Any],
    assets: dict[str, str],
) -> str:
    if block.name == "ConvBlock3D":
        lines = [
            "#### Deep Worked-Sample Walkthrough",
            "",
            "`ConvBlock3D` is not just a container of layers. On the worked sample it is the first learned volumetric feature extractor in the autoencoder encoder path. The specific trace shown here uses the **actual input voxel tensor of `gdb_1` entering `enc1`**.",
            "",
            f"Worked input tensor shape: `{convblock_ctx['input_shape']}`. Because `enc1` is constructed as `ConvBlock3D(1, 32, dropout=0.05)`, the block transforms one occupancy channel into 32 learned channels while preserving spatial size `24 x 24 x 24`.",
            "",
            f"![ConvBlock3D internal trace]({assets['convblock3d_internal']})",
            "",
            "Reading this figure row by row:",
            "",
            "1. `input`: the central slice of the normalized voxel cube before any learnable filtering.",
            "2. `layer 1 Conv3d`: local 3D kernels convert raw occupancy intensities into 32 learned response maps.",
            "3. `layer 2 InstanceNorm3d`: each channel is standardized per sample so channel scale does not destabilize later stages.",
            "4. `layer 3 LeakyReLU`: negative responses are attenuated, not erased.",
            "5. `layer 4 Conv3d`: a second local mixing step refines the learned features.",
            "6. `layer 5 InstanceNorm3d`: the refined channels are normalized again.",
            "7. `layer 6 LeakyReLU`: the final block output becomes the activation volume consumed by `down1` in the full autoencoder.",
            "",
            "Line-by-line mathematical reading of the core layer declarations:",
            "",
            "- `nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)`: for each output channel $c'$ and voxel location $(i,j,k)$, the block computes a weighted sum over a local $3\\times3\\times3$ neighborhood from the input channels. Padding 1 preserves spatial size.",
            "- `nn.InstanceNorm3d(out_channels, affine=True)`: for each channel independently, the block subtracts that sample's channel mean and divides by its channel standard deviation, then applies learnable scale and shift.",
            "- `nn.LeakyReLU(negative_slope=0.1)`: the activation law is $f(x)=x$ for $x>0$ and $0.1x$ for $x \le 0$.",
            "- the second convolution-normalization-activation triplet repeats the same logic at a higher feature level.",
            "- `self.block = nn.Sequential(*layers)` turns the list into one composed function $f(x)=f_6(f_5(f_4(f_3(f_2(f_1(x))))))$.",
            "",
            "Intuition: the first convolution discovers local volumetric motifs; normalization makes those motifs numerically comparable; the activation selectively preserves useful responses; the second convolution then builds more abstract motifs on top of the first. This is why `ConvBlock3D` is the reusable microscopic unit of the entire voxel autoencoder.",
            "",
        ]
        return "\n".join(lines)
    if block.name == "voxelize_positions":
        return "\n".join(
            [
                "#### Deep Worked-Sample Walkthrough",
                "",
                "This function is the geometric heart of the repository. On the worked sample, it receives a `5 x 3` coordinate matrix and a 5-element atomic-number vector and expands them into a dense `1 x 24 x 24 x 24` voxel tensor.",
                "",
                f"![Centered atom matrix]({assets['centered_matrix']})",
                "",
                f"![Gaussian center slice]({assets['gaussian_slice']})",
                "",
                f"![Raw vs normalized voxel]({assets['voxel_raw_vs_normalized']})",
                "",
                "Mathematically, the function implements Gaussian splatting. Each atom contributes a smooth bump to every nearby voxel center, and the final occupancy field is the sum of all those bumps. The centered atom matrix is the exact pre-voxelization representation. The Gaussian center-slice figure shows the resulting density pattern in one plane of the cube.",
                "",
                "Per-line intuition for the main mathematical steps:",
                "",
                "- the empty-position branch returns a zero cube so downstream code never crashes on degenerate molecules;",
                "- `ctr = pos.mean(dim=0)` computes the molecule centroid, making the representation translation-invariant;",
                "- `torch.linspace` plus `torch.meshgrid` constructs the physical voxel lattice;",
                "- `dist_sq = ...` computes all atom-to-voxel squared distances in one broadcasted tensor operation;",
                "- `weights = atomic_numbers / 6.0` scales heavier atoms to contribute more intensity;",
                "- `torch.exp(-dist_sq / (2 sigma^2))` is the actual Gaussian kernel evaluation;",
                "- `torch.sum(gauss, dim=0)` collapses the atom axis and produces one occupancy field.",
                "",
            ]
        )
    if block.name == "parse_mol2_structure":
        return "\n".join(
            [
                "#### Deep Worked-Sample Walkthrough",
                "",
                "For `gdb_1`, this parser reads mol2 text and constructs the exact coordinate matrix and bond list later used by the voxelizer.",
                "",
                f"![mol2 text excerpt]({assets['mol2_excerpt']})",
                "",
                f"![mol2 point cloud]({assets['point_cloud']})",
                "",
                "The parser does three logically separate tasks: detect which section of the mol2 file is currently being read, parse atom lines into numeric coordinates and atom types, and parse bond lines into index pairs. The point-cloud figure is the numeric result of this block after text parsing is finished.",
                "",
                "Intuition: this block is the transition from symbolic chemistry text to numerical geometry. Everything that follows in the voxel pipeline depends on this conversion being correct.",
                "",
            ]
        )
    if block.name == "GCNRegressor":
        return "\n".join(
            [
                "#### Deep Worked-Sample Walkthrough",
                "",
                f"![Baseline GCN layer trace]({assets['baseline_layers']})",
                "",
                f"![Baseline prediction example]({assets['baseline_prediction']})",
                "",
                "The figure shows the actual node-feature matrix of `gdb_1`, followed by the hidden node matrices after each graph convolution. Each graph convolution mixes information from neighboring atoms according to `edge_index`, so each row gradually becomes less 'local atom descriptor' and more 'context-aware atom descriptor'. Global mean pooling then averages those node embeddings into one molecular vector. The MLP head converts that pooled vector into a scalar property prediction.",
                "",
                "Mathematically, if $H^{(0)}=X$ is the input node matrix, each graph convolution produces a new matrix $H^{(l+1)} = \\sigma(\\hat{A} H^{(l)} W^{(l)})$ up to the exact normalization convention used by `GCNConv`.",
                "",
            ]
        )
    if block.name == "GraphToVoxelNet":
        return "\n".join(
            [
                "#### Deep Worked-Sample Walkthrough",
                "",
                f"![Graph-to-voxel layers]({assets['g2v_layers']})",
                "",
                f"![Graph-to-voxel output slices]({assets['g2v_output_slices']})",
                "",
                f"![Graph-to-voxel loss components]({assets['g2v_loss']})",
                "",
                "This block combines graph encoding and 3D decoding. The top half of the model behaves like the baseline GCN: node features become graph-aware node embeddings and then one pooled molecular vector. The difference is what happens next: instead of directly predicting a scalar, the vector is projected to a latent code, reshaped into a small 3D seed volume, and upsampled with transposed convolutions until it becomes a full voxel cube.",
                "",
                "The output-slice figure compares the predicted voxel cube to the target voxel cube produced from the same molecule's coordinates. The loss-component figure visualizes how the trainer balances overall MSE, occupied-region MSE, and sparsity pressure.",
                "",
            ]
        )
    if block.name == "VoxelAutoencoder":
        return "\n".join(
            [
                "#### Deep Worked-Sample Walkthrough",
                "",
                f"![Autoencoder layers]({assets['ae_layers']})",
                "",
                f"![Autoencoder reconstruction]({assets['ae_reconstruction']})",
                "",
                f"![Latent embedding]({assets['ae_embedding']})",
                "",
                f"![Autoencoder loss components]({assets['ae_loss']})",
                "",
                "This is the main learned spatial encoder of the project. The input voxel tensor is gradually contracted in spatial resolution while expanding in channel dimension, producing a bottleneck tensor that is flattened and projected into the latent SVE vector. The decoder then inverts that process with transposed convolutions and skip connections. The reconstruction figure shows what the trained model remembers and what detail it loses on the worked sample.",
                "",
            ]
        )
    if block.name == "VoxelAutoencoderTrainer":
        return "\n".join(
            [
                "#### Deep Worked-Sample Walkthrough",
                "",
                f"![Autoencoder loss components]({assets['ae_loss']})",
                "",
                "The trainer is where the abstract reconstruction objective becomes actual scalar numbers. On the worked sample, MSE measures dense fidelity, L1 measures absolute error, the occupied-region term focuses learning on non-empty voxels, the Dice-like term encourages overlap of occupied regions, and the sparsity term discourages diffuse overly filled reconstructions.",
                "",
            ]
        )
    return ""


def render_file_section(
    rel_path: str,
    section_label: str,
    qm9_ctx: dict[str, Any],
    baseline_ctx: dict[str, Any],
    g2v_ctx: dict[str, Any],
    ae_ctx: dict[str, Any],
    convblock_ctx: dict[str, Any],
    assets: dict[str, str],
) -> str:
    parts = [f"## {section_label}: `{rel_path}`", "", FILE_SUMMARIES[rel_path], ""]
    for block in extract_blocks(rel_path):
        deep_walkthrough = custom_block_walkthrough(rel_path, block, qm9_ctx, baseline_ctx, g2v_ctx, ae_ctx, convblock_ctx, assets)
        parts.extend(
            [
                f"### Block: `{block.name}`",
                "",
                "```python",
                block.code,
                "```",
                "",
                block_details(block),
                "",
                deep_walkthrough,
                "",
                "#### Line-By-Line Framework Notes",
                "",
                markdown_line_table(block.code, block.name),
                "",
            ]
        )
    return "\n".join(parts)


def write_markdown(qm9_ctx: dict[str, Any], baseline_ctx: dict[str, Any], g2v_ctx: dict[str, Any], ae_ctx: dict[str, Any], convblock_ctx: dict[str, Any], assets: dict[str, str]) -> Path:
    sample = qm9_ctx["sample"]
    sample_gap = qm9_ctx["sample_gap"]
    coords = ae_ctx["coords"].numpy()
    atomic = ae_ctx["atomic"].numpy()
    centered = coords - coords.mean(axis=0, keepdims=True)
    xlabels_node = [f"f{i}" for i in range(sample.x.shape[1])]
    lines: list[str] = [
        "# molsim Deep Academic Walkthrough",
        "",
        "This report is intentionally large and explicit. It is designed to explain the `molsim` codebase to researchers at the level of real data tensors, control flow, mathematics, and trained model execution traces.",
        "",
        "## Why `molsim` Exists",
        "",
        "`molsim` is the spatial representation-learning core of the repository. It exists to answer a concrete research question: can explicit 3D molecular structure be transformed into a compact learned representation that helps downstream molecular prediction?",
        "",
        "The package does this by supporting three interlocking scientific paths:",
        "",
        "1. a graph-only molecular baseline,",
        "2. a graph-to-voxel reconstruction path, and",
        "3. a mol2-to-voxel autoencoding path whose bottleneck vector is the Spatial Voxel Embedding (SVE).",
        "",
        "## Overall Pipeline",
        "",
        f"![Pipeline overview]({assets['pipeline']})",
        "",
        f"![Package map]({assets['package_map']})",
        "",
        "## Real Worked Data Specimen",
        "",
        f"The report traces a real QM9 sample named `{qm9_ctx['name']}`. The PyG graph object has:",
        "",
        f"- node feature shape: `{tuple(sample.x.shape)}`",
        f"- edge index shape: `{tuple(sample.edge_index.shape)}`",
        f"- position shape: `{tuple(sample.pos.shape)}`",
        f"- raw target shape: `{tuple(sample.y.shape)}`",
        f"- scalar `gap` target shape after adaptation: `{tuple(sample_gap.y.shape)}`",
        "",
        f"![QM9 node features]({assets['qm9_node_features']})",
        "",
        f"![QM9 edge index]({assets['qm9_edge_index']})",
        "",
        f"![QM9 positions]({assets['qm9_positions']})",
        "",
        f"![QM9 target vector]({assets['qm9_target_vector']})",
        "",
        "These are the actual matrices entering the graph-learning parts of the project. The edge index matrix is especially important because PyG message passing uses it to decide which node pairs exchange information during graph convolution.",
        "",
        "## Mol2 Parsing Trace",
        "",
        f"The corresponding mol2 file is `{ae_ctx['mol2_path']}`. Parsing it produces a coordinate matrix and an atomic-number vector.",
        "",
        f"![mol2 text excerpt]({assets['mol2_excerpt']})",
        "",
        f"![mol2 point cloud]({assets['point_cloud']})",
        "",
        f"![atom matrix]({assets['atom_matrix']})",
        "",
        f"![centered atom matrix]({assets['centered_matrix']})",
        "",
        "The parser first reads atom lines from the Tripos text, converts atom types into approximate atomic numbers, and then optionally reads bond lines. The centered coordinate matrix is important because the voxelizer shifts the molecule to its own centroid before placing Gaussian densities on the grid.",
        "",
        "## Voxelization Trace",
        "",
        "The voxelizer takes the centered point cloud and converts it into a dense cube using Gaussian splatting:",
        "",
        "$$",
        "V_{ijk} = \\sum_{a=1}^{N} w_a \\exp\\left(-\\frac{\\lVert c_{ijk} - r_a \\rVert_2^2}{2\\sigma^2}\\right).",
        "$$",
        "",
        f"For the trained autoencoder configuration in this repository, the voxelization hyperparameters are: grid size `{ae_ctx['voxel_cfg'].grid_size}`, resolution `{ae_ctx['voxel_cfg'].resolution}`, sigma `{ae_ctx['voxel_cfg'].sigma}`, and normalization mode `{ae_ctx['normalization']}`.",
        "",
        f"![gaussian center slice]({assets['gaussian_slice']})",
        "",
        f"![voxel 3d]({assets['voxel_3d']})",
        "",
        f"![raw vs normalized voxel]({assets['voxel_raw_vs_normalized']})",
        "",
        "The central Gaussian slice shows the local intensity structure created by the atoms. The raw-vs-normalized figure shows how `normalize_voxel()` preserves the shape while compressing the intensity range into a stable `[0, 1]` interval.",
        "",
        metrics_and_math_sections(qm9_ctx, baseline_ctx, g2v_ctx, ae_ctx),
        "## Graph Baseline Execution Trace",
        "",
        "There is no saved baseline checkpoint in the repository, so the baseline forward pass below uses the exact current architecture with fresh initialization. The value here is structural understanding: what matrices and tensor shapes the code produces, not the scientific quality of this single initialized pass.",
        "",
        f"![baseline layer trace]({assets['baseline_layers']})",
        "",
        f"![baseline prediction]({assets['baseline_prediction']})",
        "",
        "The baseline path performs three graph convolutions, then global mean pooling compresses the node set into one molecular vector, and finally the MLP head emits one scalar. In other words, the baseline is a pure graph-to-scalar pathway with no explicit 3D volumetric reconstruction stage.",
        "",
        "## Graph-To-Voxel Execution Trace",
        "",
        "The graph-to-voxel model uses a trained checkpoint, so the activations below are from a learned model rather than random weights.",
        "",
        f"![graph-to-voxel layers]({assets['g2v_layers']})",
        "",
        f"![graph-to-voxel output slices]({assets['g2v_output_slices']})",
        "",
        f"![graph-to-voxel loss components]({assets['g2v_loss']})",
        "",
        "The first four matrices show node-level hidden features across successive graph convolution layers. The pooled vector and latent vector then collapse graph structure into one global representation, which is reshaped and decoded into a dense 3D voxel cube.",
        "",
        "## Voxel Autoencoder Execution Trace",
        "",
        "The voxel autoencoder is the current best SVE generator in the repository. The following figures trace a real normalized voxel tensor through its learned 3D encoder-decoder path.",
        "",
        f"![autoencoder layer trace]({assets['ae_layers']})",
        "",
        f"![autoencoder reconstruction]({assets['ae_reconstruction']})",
        "",
        f"![latent embedding]({assets['ae_embedding']})",
        "",
        f"![autoencoder loss components]({assets['ae_loss']})",
        "",
        "The layer-trace figure uses central slices of channel-mean activations to visualize how the spatial field contracts and transforms through the encoder. The reconstruction figure shows the final decoded voxel field, and the embedding heatmap shows the learned bottleneck vector that later becomes the downstream SVE feature block.",
        "",
        "## ConvBlock3D Micro-Trace",
        "",
        "Because `ConvBlock3D` is the repeated microscopic building block of the autoencoder, it deserves its own worked-sample explanation. The figure below traces the actual `enc1` block on the real normalized voxel input.",
        "",
        f"![ConvBlock3D micro trace]({assets['convblock3d_internal']})",
        "",
        f"This specific invocation receives tensor shape `{convblock_ctx['input_shape']}` and emits `{convblock_ctx['out_channels']}` output channels while keeping the spatial grid size fixed.",
        "",
    ]

    if "sve_merge" in assets:
        lines.extend(
            [
                "## Downstream Feature Fusion Context",
                "",
                f"![SVE merge]({assets['sve_merge']})",
                "",
                "This is the final downstream handoff of `molsim`: the latent vector is concatenated onto existing PROTAC features so downstream architectures can decide whether and how to exploit the spatial signal.",
                "",
            ]
        )

    if "results_summary" in assets:
        lines.extend(
            [
                "## Downstream Result Context",
                "",
                f"![results summary]({assets['results_summary']})",
                "",
                "The downstream registry confirms the current project-level interpretation: spatial embeddings are not universally beneficial, but they can help stronger downstream architectures, especially cross-attention regression.",
                "",
            ]
        )

    lines.extend(["## Full Module Walkthrough", "", "The sections below reproduce every top-level class/function block from every file in `molsim/`, then explain the code framework and give line-by-line notes.", ""])

    for rel_path in MOLSIM_FILES:
        lines.append(render_file_section(rel_path, "Module", qm9_ctx, baseline_ctx, g2v_ctx, ae_ctx, convblock_ctx, assets))
        lines.append("")

    lines.extend(["## Workflow Scripts Using `molsim`", "", "These scripts are outside the package, but they are required to understand how the package is actually exercised in experiments.", ""])

    for rel_path in WORKFLOW_SCRIPTS:
        lines.append(render_file_section(rel_path, "Script", qm9_ctx, baseline_ctx, g2v_ctx, ae_ctx, convblock_ctx, assets))
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "At the code level, `molsim` is a cleanly layered research scaffold: data ingestion, spatial conversion, neural modeling, training, and evaluation are separated enough to inspect independently. At the scientific level, its strongest current contribution is the voxel autoencoder and the exported SVE bottleneck. At the empirical level, the repository results suggest that explicit 3D geometric information is useful, but only if the downstream architecture is expressive enough to exploit it.",
            "",
            "## Rebuild Command",
            "",
            "```bash",
            "python3 scripts/build_molsim_module_report.py",
            "```",
            "",
        ]
    )

    md_path = REPORT_DIR / "molsim_module_academic_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def maybe_export_docx(md_path: Path) -> Path | None:
    docx_path = REPORT_DIR / "molsim_module_academic_report.docx"
    try:
        subprocess.run(["pandoc", str(md_path), "-o", str(docx_path)], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
        return docx_path
    except Exception:
        return None


def main() -> None:
    ensure_dirs()
    qm9_ctx = load_qm9_sample()
    mol2_path = find_sample_mol2(qm9_ctx["name"])
    baseline_ctx = trace_baseline(qm9_ctx["sample_gap"])
    g2v_ctx = trace_graph_to_voxel(qm9_ctx["sample"])
    ae_ctx = trace_autoencoder(mol2_path)
    convblock_ctx = trace_convblock3d(ae_ctx)
    assets = generate_assets(qm9_ctx, baseline_ctx, g2v_ctx, ae_ctx, convblock_ctx)
    md_path = write_markdown(qm9_ctx, baseline_ctx, g2v_ctx, ae_ctx, convblock_ctx, assets)
    docx_path = maybe_export_docx(md_path)
    print(
        json.dumps(
            {
                "report_path": str(md_path),
                "docx_path": str(docx_path) if docx_path else None,
                "asset_dir": str(ASSET_DIR),
                "sample_name": qm9_ctx["name"],
                "mol2_path": str(mol2_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
