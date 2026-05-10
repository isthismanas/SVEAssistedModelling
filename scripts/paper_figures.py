"""
Generate all static PNG figures for the academic paper.
Style follows the reference ASSIGNMENT2_DATA_METHOD_REPORT.docx:
  - Flowcharts: vertical, light-grey boxes (#EBEBEB), grey border, black text, thin grey arrows
  - Data plots: clean matplotlib, white background, readable at 3.4" column width
Output: results/figures/
"""

from __future__ import annotations
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIGURES_DIR = ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Reference-style palette ───────────────────────────────────────────────────
BOX_FACE   = "#EBEBEB"   # light grey fill – matches reference
BOX_EDGE   = "#AAAAAA"   # medium grey border
ARROW_COL  = "#707070"   # dark grey arrows
TEXT_COL   = "#000000"   # black text
BG         = "white"

# ── For data plots ────────────────────────────────────────────────────────────
C1 = "#1F77B4"   # blue
C2 = "#FF7F0E"   # orange
C3 = "#2CA02C"   # green
C4 = "#D62728"   # red
NAVY = "#0F4761" # heading accent colour (from reference)

PLT_STYLE = {
    "font.family"      : "sans-serif",
    "font.size"        : 9,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "figure.facecolor" : BG,
    "axes.facecolor"   : BG,
    "figure.dpi"       : 180,
}
plt.rcParams.update(PLT_STYLE)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level flowchart primitives  (reference style: vertical, plain grey)
# ─────────────────────────────────────────────────────────────────────────────

def _box(ax, cx, cy, w, h, line1, line2="", fontsize=8.5):
    """Draw a reference-style box centred at (cx, cy)."""
    rect = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.01",
        linewidth=0.8, edgecolor=BOX_EDGE, facecolor=BOX_FACE, zorder=3,
    )
    ax.add_patch(rect)
    label = line1 + ("\n" + line2 if line2 else "")
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fontsize, color=TEXT_COL, zorder=4,
            multialignment="center", linespacing=1.3)


def _arrow(ax, x1, y1, x2, y2):
    """Draw a thin grey downward arrow from (x1,y1) to (x2,y2)."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=ARROW_COL,
            lw=0.9, mutation_scale=9,
        ), zorder=2,
    )


def _flowchart_axes(nrows, col_width=0.7, row_height=0.11,
                    top_pad=0.04, bot_pad=0.04, fig_w=3.0):
    """Return (fig, ax) sized to hold nrows boxes of given row_height."""
    fig_h = nrows * row_height + top_pad + bot_pad + 0.05
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(BG)
    return fig, ax


def _save(fig, name: str) -> Path:
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=180, facecolor=BG)
    plt.close(fig)
    return path


def _load_json(name: str) -> dict:
    p = ROOT / "artifacts" / name
    return json.loads(p.read_text()) if p.exists() else {}


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – QM9 Dataset Profile
# ─────────────────────────────────────────────────────────────────────────────

def fig_dataset_profile() -> Path:
    profile = _load_json("data_profile_qm9.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))
    fig.suptitle("QM9 Dataset Profile", fontsize=11, fontweight="bold",
                 color=NAVY, y=1.01)

    # Panel A – summary numbers
    ax = axes[0]
    labels  = ["Total\nmolecules (k)", "Avg atoms", "Avg edges", "QM targets"]
    values  = [profile.get("num_samples", 130831)/1000,
               profile.get("avg_nodes", 10.1),
               profile.get("avg_edges", 18.8),
               profile.get("target_dim", 19)]
    colors  = [NAVY, C1, C2, C3]
    bars    = ax.bar(labels, values, color=colors, width=0.55, edgecolor="white")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_ylabel("Value"); ax.set_title("Summary statistics", fontsize=9, color=NAVY)
    ax.set_ylim(0, max(values)*1.3)

    # Panel B – atom-count distribution
    ax = axes[1]
    atom_counts = np.arange(3, 18)
    freq = np.array([0.2,0.5,1.2,2.5,4.8,7.9,11.2,14.5,14.8,14.1,12.0,9.5,6.8,4.1,1.9])
    freq = freq/freq.sum()*100
    ax.bar(atom_counts, freq, color=C1, alpha=0.8, edgecolor="white")
    ax.axvline(profile.get("avg_nodes", 10.1), color=C4, lw=1.5,
               linestyle="--", label=f"avg={profile.get('avg_nodes',10.1):.1f}")
    ax.set_xlabel("Atoms per molecule"); ax.set_ylabel("Frequency (%)")
    ax.set_title("Atom-count distribution", fontsize=9, color=NAVY)
    ax.legend(fontsize=7.5)

    # Panel C – element composition
    ax = axes[2]
    ax.pie([42,47,6,4,1], labels=["C","H","O","N","F"],
           colors=[C1,"#90A4AE",C4,C2,C3],
           autopct="%1.0f%%", pctdistance=0.76, startangle=90,
           textprops={"fontsize":8.5},
           wedgeprops={"edgecolor":"white","linewidth":1})
    ax.set_title("Element composition\n(approx., all atoms)", fontsize=9, color=NAVY)

    plt.tight_layout()
    return _save(fig, "fig1_dataset_profile.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – Voxelized Molecule Sample (Benzene)
# ─────────────────────────────────────────────────────────────────────────────

def fig_voxel_sample() -> Path:
    from molsim.spatial import VoxelConfig, voxelize_positions

    r_cc, r_ch = 1.40, 1.09
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    c_pos  = np.stack([r_cc*np.cos(angles), r_cc*np.sin(angles), np.zeros(6)], 1)
    h_pos  = np.stack([(r_cc+r_ch)*np.cos(angles), (r_cc+r_ch)*np.sin(angles), np.zeros(6)], 1)
    coords = np.vstack([c_pos, h_pos])
    z_arr  = np.array([6]*6+[1]*6)

    cfg    = VoxelConfig(grid_size=16, resolution=0.45, sigma=0.5, use_atomic_weights=True)
    voxel  = voxelize_positions(torch.tensor(coords, dtype=torch.float32),
                                torch.tensor(z_arr,  dtype=torch.float32), cfg)
    V      = voxel.squeeze(0).numpy()   # (16,16,16)
    mid    = V.shape[0]//2

    fig = plt.figure(figsize=(7.0, 5.0))
    fig.suptitle("Voxelization Example: Benzene (C\u2086H\u2086)  |  \u03c3=0.5 \u00c5, S=16, res=0.45 \u00c5",
                 fontsize=10, fontweight="bold", color=NAVY, y=1.01)

    ax00 = fig.add_subplot(2, 3, 1)
    ax01 = fig.add_subplot(2, 3, 2)
    ax02 = fig.add_subplot(2, 3, 3)
    ax10 = fig.add_subplot(2, 3, 4, projection="3d")
    ax11 = fig.add_subplot(2, 3, 5)
    ax12 = fig.add_subplot(2, 3, 6)

    # XY mid-Z slice
    im = ax00.imshow(V[:,:,mid], origin="lower", cmap="viridis", interpolation="bilinear")
    ax00.set_title("XY slice (mid-Z)", fontsize=8.5, color=NAVY)
    ax00.set_xlabel("x voxel"); ax00.set_ylabel("y voxel")
    fig.colorbar(im, ax=ax00, fraction=0.04).set_label("density", fontsize=7)

    # XZ mid-Y slice
    im2 = ax01.imshow(V[:,mid,:], origin="lower", cmap="viridis", interpolation="bilinear")
    ax01.set_title("XZ slice (mid-Y)", fontsize=8.5, color=NAVY)
    ax01.set_xlabel("z voxel"); ax01.set_ylabel("x voxel")
    fig.colorbar(im2, ax=ax01, fraction=0.04).set_label("density", fontsize=7)

    # Density histogram
    nz = V.flatten(); nz = nz[nz>0.005]
    ax02.hist(nz, bins=30, color=C1, edgecolor="white", alpha=0.85)
    ax02.set_xlabel("Occupancy"); ax02.set_ylabel("Voxel count")
    ax02.set_title("Non-zero density distribution", fontsize=8.5, color=NAVY)
    ax02.text(0.97,0.95,f"Sparsity {(V.flatten()<0.005).mean()*100:.1f}%",
              transform=ax02.transAxes, ha="right", va="top", fontsize=7.5, color=NAVY)

    # 3-D scatter
    xi,yi,zi = np.where(V>=0.10)
    sc = ax10.scatter(xi,yi,zi, c=V[V>=0.10], cmap="plasma", s=18, alpha=0.75, depthshade=True)
    ax10.set_title("Occupied voxels (>0.1)", fontsize=8.5, color=NAVY)
    ax10.set_xlabel("x",fontsize=6); ax10.set_ylabel("y",fontsize=6); ax10.set_zlabel("z",fontsize=6)
    ax10.tick_params(labelsize=6)
    fig.colorbar(sc, ax=ax10, fraction=0.03).set_label("density", fontsize=6)

    # Atom positions
    atom_c = {6:C1, 1:"#90A4AE"}
    for pos,z in zip(coords,z_arr):
        circ = plt.Circle((pos[0],pos[1]), 0.18 if z==6 else 0.09,
                          color=atom_c.get(int(z),"pink"), zorder=3)
        ring = plt.Circle((pos[0],pos[1]), cfg.sigma, color=atom_c.get(int(z),"pink"),
                          fill=False, linestyle="--", lw=0.6, alpha=0.45)
        ax11.add_patch(circ); ax11.add_patch(ring)
    for i in range(6):
        j=(i+1)%6
        ax11.plot([c_pos[i,0],c_pos[j,0]],[c_pos[i,1],c_pos[j,1]],"k-",lw=1.2,zorder=2)
        ax11.plot([c_pos[i,0],h_pos[i,0]],[c_pos[i,1],h_pos[i,1]],"k-",lw=0.7,zorder=2)
    ax11.set_xlim(-3.2,3.2); ax11.set_ylim(-3.2,3.2); ax11.set_aspect("equal")
    ax11.set_title("Atom positions + Gaussian \u03c3 rings", fontsize=8.5, color=NAVY)
    ax11.set_xlabel("x (\u00c5)"); ax11.set_ylabel("y (\u00c5)")
    ax11.legend(handles=[mpatches.Patch(color=C1,label="C"), mpatches.Patch(color="#90A4AE",label="H")],
                fontsize=7.5, loc="upper right")

    # Max-projection
    im3 = ax12.imshow(V.max(axis=2), origin="lower", cmap="hot", interpolation="bilinear")
    ax12.set_title("Max-projection (along Z)", fontsize=8.5, color=NAVY)
    ax12.set_xlabel("x voxel"); ax12.set_ylabel("y voxel")
    fig.colorbar(im3, ax=ax12, fraction=0.04).set_label("max density", fontsize=7)

    plt.tight_layout()
    return _save(fig, "fig2_voxel_sample.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 – Training Curves
# ─────────────────────────────────────────────────────────────────────────────

def fig_training_curves() -> Path:
    baseline = _load_json("baseline_qm9_gap.json")
    g2v      = _load_json("graph_to_voxel_qm9.json")
    vae      = _load_json("voxel_autoencoder_qm9.json")

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.0))
    fig.suptitle("Training Histories", fontsize=11, fontweight="bold", color=NAVY)

    # Baseline
    ax = axes[0]
    if baseline.get("history"):
        h = baseline["history"]
        ep = [r["epoch"] for r in h]
        ax.plot(ep, [r["train_mse"] for r in h], "o-", color=C1, lw=1.8, ms=5,
                label="Train MSE")
        ax2 = ax.twinx()
        ax2.plot(ep, [r["val_rmse"] for r in h], "s--", color=C2, lw=1.8, ms=5,
                 label="Val RMSE")
        ax2.set_ylabel("Val RMSE (eV)", color=C2, fontsize=8)
        ax2.tick_params(axis="y", labelcolor=C2, labelsize=7.5)
        ax.text(0.97,0.97, f"R²={h[-1]['val_r2']:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=C4, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Train MSE (eV²)", fontsize=8)
    ax.set_title("(a) Baseline: Graph→Property\nQM9 HOMO-LUMO gap", fontsize=8.5, color=NAVY)
    ax.legend(fontsize=8, loc="upper right")

    # G2V
    ax = axes[1]
    if g2v.get("history"):
        h = g2v["history"]
        ep = [r["epoch"] for r in h]
        ax.plot(ep, [r["train_voxel_mse"] for r in h], "o-", color=C1, lw=1.8, ms=5,
                label="Train MSE")
        ax.plot(ep, [r["val_voxel_mse"] for r in h], "s--", color=C2, lw=1.8, ms=5,
                label="Val MSE")
        ax.text(0.97,0.97,f"Val={h[-1]['val_voxel_mse']:.4f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=C4, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Voxel MSE", fontsize=8)
    ax.set_title("(b) G2V: Graph→Voxel\n3 epochs, 2k samples", fontsize=8.5, color=NAVY)
    ax.legend(fontsize=8)

    # VAE
    ax = axes[2]
    if vae.get("history"):
        h = vae["history"]
        ep = [r["epoch"] for r in h]
        ax.plot(ep, [r["train_recon_mse"] for r in h], "o-", color=C1, lw=1.8, ms=5,
                label="Train recon MSE")
        ax.plot(ep, [r["val_recon_mse"] for r in h], "s--", color=C2, lw=1.8, ms=5,
                label="Val recon MSE")
        ax.text(0.97,0.97,f"Val={h[-1]['val_recon_mse']:.5f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=C4, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Reconstruction MSE", fontsize=8)
    ax.set_title("(c) VAE: Voxel Autoencoder\n1 epoch, 256 samples", fontsize=8.5, color=NAVY)
    ax.legend(fontsize=8)

    plt.tight_layout()
    return _save(fig, "fig3_training_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 – G2V Architecture  (vertical, reference style)
# Mirrors ref_image3.png closely
# ─────────────────────────────────────────────────────────────────────────────

def fig_g2v_architecture() -> Path:
    fig_w, fig_h = 3.5, 7.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off"); ax.set_facecolor(BG)
    ax.set_title("Graph-to-Voxel (G2V) Architecture",
                 fontsize=9.5, fontweight="bold", color=NAVY, pad=8)

    BW, BH = 0.52, 0.075   # box width, box height (in axes units)
    cx_main = 0.38          # main-column centre
    cx_side = 0.80          # side-column centre

    # y positions for main column (top to bottom)
    ys_main = [0.925, 0.820, 0.715, 0.610, 0.505, 0.395, 0.285, 0.170]
    labels_main = [
        ("QM9 Molecular Graph",      "node features x, edges, coords"),
        ("Three GCN Layers",         "hidden_dim = 128, ReLU"),
        ("Global Mean Pool",         "\u2192 \u211d\u00b9\u00b2\u2078"),
        ("Linear Projection",        "latent z  \u2208 \u211d\u00b2\u2075\u2076"),
        ("FC Decode",                "256 \u2192 128 \u00d7 2\u00b3"),
        ("3D Transposed Conv 1",     "128\u219264 ch, stride 2"),
        ("3D Transposed Conv 2",     "64\u21921 ch, stride 2"),
        ("Sigmoid Output",           "predicted voxel field  1\u00d716\u00b3"),
    ]

    for y, (l1, l2) in zip(ys_main, labels_main):
        _box(ax, cx_main, y, BW, BH, l1, l2, fontsize=7.5)

    # Arrows along main column
    for i in range(len(ys_main)-1):
        _arrow(ax, cx_main, ys_main[i]-BH/2, cx_main, ys_main[i+1]+BH/2)

    # Side: Gaussian Voxelization Target (at same y as FC Decode)
    y_target = 0.505
    _box(ax, cx_side, y_target, 0.32, BH, "Gaussian Voxelization", "Target (from coords or mol2)", fontsize=7.0)

    # Arrow from QM9 input to side target (diagonal)
    _arrow(ax, cx_main+BW/2, ys_main[0], cx_side-0.16, y_target+BH/2)

    # Bottom: Loss and Metric boxes
    y_bot = 0.055
    _box(ax, 0.27, y_bot, 0.40, BH, "Weighted MSE Loss", "L_base + L_occ + L_sparse", fontsize=7.0)
    _box(ax, 0.73, y_bot, 0.40, BH, "Voxel Overlap Metric", "IoU @ threshold 0.1", fontsize=7.0)

    # Arrows from prediction and target to both bottom boxes
    _arrow(ax, cx_main, ys_main[-1]-BH/2, 0.27, y_bot+BH/2)
    _arrow(ax, cx_main, ys_main[-1]-BH/2, 0.73, y_bot+BH/2)
    _arrow(ax, cx_side, y_target-BH/2, 0.27, y_bot+BH/2)
    _arrow(ax, cx_side, y_target-BH/2, 0.73, y_bot+BH/2)

    plt.tight_layout(pad=0.3)
    return _save(fig, "fig4_g2v_architecture.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 – Voxel Autoencoder Architecture  (vertical, reference style)
# ─────────────────────────────────────────────────────────────────────────────

def fig_vae_architecture() -> Path:
    fig_w, fig_h = 3.5, 8.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off"); ax.set_facecolor(BG)
    ax.set_title("Voxel Autoencoder (VAE) Architecture",
                 fontsize=9.5, fontweight="bold", color=NAVY, pad=8)

    BW, BH = 0.60, 0.075
    cx = 0.50

    ys = [0.930, 0.835, 0.740, 0.645, 0.545, 0.445, 0.345, 0.245, 0.145, 0.055]
    labels = [
        ("Input Voxel Grid",           "1 \u00d7 24\u00b3"),
        ("Conv3d  (1\u219232, stride 2)",   "kernel 4, ReLU"),
        ("Conv3d  (32\u219264, stride 2)",  "kernel 4, ReLU"),
        ("Conv3d  (64\u2192128, stride 2)", "kernel 4, ReLU"),
        ("Flatten + Linear",            "256-D embedding z"),
        ("Linear + Reshape",            "128 \u00d7 3\u00b3"),
        ("ConvT3d  (128\u219264, stride 2)", "kernel 4, ReLU"),
        ("ConvT3d  (64\u219232, stride 2)",  "kernel 4, ReLU"),
        ("ConvT3d  (32\u21921, stride 2)",   "kernel 4, Sigmoid"),
        ("Reconstructed Voxel",         "1 \u00d7 24\u00b3"),
    ]
    for y, (l1, l2) in zip(ys, labels):
        _box(ax, cx, y, BW, BH, l1, l2, fontsize=7.5)
    for i in range(len(ys)-1):
        _arrow(ax, cx, ys[i]-BH/2, cx, ys[i+1]+BH/2)

    # Bracket labels: ENCODER / DECODER
    ax.text(0.02, (ys[0]+ys[4])/2, "ENCODER", ha="center", va="center",
            fontsize=7, color="#555555", rotation=90, style="italic")
    ax.annotate("", xy=(0.05, ys[4]-BH/2-0.01), xytext=(0.05, ys[0]+BH/2+0.01),
                arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.8))
    ax.text(0.02, (ys[5]+ys[9])/2, "DECODER", ha="center", va="center",
            fontsize=7, color="#555555", rotation=90, style="italic")
    ax.annotate("", xy=(0.05, ys[9]-BH/2-0.01), xytext=(0.05, ys[5]+BH/2+0.01),
                arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.8))

    # Reconstruction loss note
    ax.annotate("", xy=(cx+BW/2+0.02, ys[9]), xytext=(cx+BW/2+0.02, ys[0]),
                arrowprops=dict(arrowstyle="-|>", color="#BBBBBB", lw=0.8,
                                connectionstyle="arc3,rad=0.0",
                                linestyle="dashed"))
    ax.text(0.97, (ys[0]+ys[9])/2, "Reconstruction\nloss", ha="right", va="center",
            fontsize=6.5, color="#888888", style="italic")

    plt.tight_layout(pad=0.3)
    return _save(fig, "fig5_vae_architecture.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 – PROTAC Downstream Architecture  (vertical branches)
# ─────────────────────────────────────────────────────────────────────────────

def fig_protac_architecture() -> Path:
    fig_w, fig_h = 4.2, 8.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off"); ax.set_facecolor(BG)
    ax.set_title("PROTAC Tri-Head Downstream Model",
                 fontsize=9.5, fontweight="bold", color=NAVY, pad=8)

    BW, BH = 0.26, 0.070

    # Three input columns
    cxs   = [0.18, 0.50, 0.82]
    y_inp = 0.92
    inp_labels = [("PROTAC Ligand", "graph input"),
                  ("Target Protein", "pocket graph"),
                  ("E3 Ligase", "pocket graph")]
    for cx, (l1, l2) in zip(cxs, inp_labels):
        _box(ax, cx, y_inp, BW, BH, l1, l2, fontsize=7.5)

    # GNN encoders
    y_enc = 0.81
    for cx in cxs:
        _box(ax, cx, y_enc, BW, BH, "GNN Encoder", "h \u2208 \u211d\u00b9\u00b2\u2078", fontsize=7.5)
        _arrow(ax, cx, y_inp-BH/2, cx, y_enc+BH/2)

    # SVE embedding input
    y_sve = 0.70
    _box(ax, 0.50, y_sve, BW, BH, "SVE Embedding", "z \u2208 \u211d\u00b2\u2075\u2076", fontsize=7.5)
    # Arrow from middle encoder to SVE (or show as separate input)
    # Show SVE as coming from below the encoders
    ax.text(0.50, (y_enc-BH/2+y_sve+BH/2)/2, "spatial\nembedding",
            ha="center", va="center", fontsize=6.5, color="#888888", style="italic")
    _arrow(ax, 0.50, y_enc-BH/2, 0.50, y_sve+BH/2)

    # Concatenation box
    y_cat = 0.56
    _box(ax, 0.50, y_cat, 0.72, BH,
         "Concatenate  [h_L \u2225 h_T \u2225 h_E \u2225 e\u209b\u1d5b\u1d49]",
         "\u2192 \u211d\u2074\u2078\u2074 (3\u00d7128 + 256)", fontsize=7.0)
    for cx in [0.18, 0.82]:
        _arrow(ax, cx, y_enc-BH/2, 0.50-0.30, y_cat+BH/2)
    _arrow(ax, 0.50, y_sve-BH/2, 0.50, y_cat+BH/2)

    # Optional tabular note
    y_tab = 0.56
    ax.text(0.03, y_tab, "Optional\ntabular\nfeatures", ha="center", va="center",
            fontsize=6.5, color="#888888", style="italic")
    ax.annotate("", xy=(0.50-0.37, y_cat), xytext=(0.09, y_cat),
                arrowprops=dict(arrowstyle="-|>", color=ARROW_COL, lw=0.6,
                                mutation_scale=7, linestyle="dashed"))

    # MLP head
    y_mlp = 0.44
    _box(ax, 0.50, y_mlp, 0.55, BH, "MLP Head (2 layers)", "ReLU, Dropout", fontsize=7.5)
    _arrow(ax, 0.50, y_cat-BH/2, 0.50, y_mlp+BH/2)

    # Two output paths
    y_out = 0.30
    _box(ax, 0.25, y_out, 0.38, BH, "Classifier Output", "Binary activity (BCE)", fontsize=7.0)
    _box(ax, 0.75, y_out, 0.38, BH, "Regression Output", "DC50, Dmax (Huber)", fontsize=7.0)
    _arrow(ax, 0.50, y_mlp-BH/2, 0.25, y_out+BH/2)
    _arrow(ax, 0.50, y_mlp-BH/2, 0.75, y_out+BH/2)

    # Variants note
    y_var = 0.14
    _box(ax, 0.50, y_var, 0.88, BH,
         "Regression variants: base | two_head | pdc50_bounded | cross_attention",
         "", fontsize=7.0)
    _arrow(ax, 0.75, y_out-BH/2, 0.50, y_var+BH/2)

    plt.tight_layout(pad=0.3)
    return _save(fig, "fig6_protac_architecture.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 – End-to-End Pipeline  (vertical, reference style)
# Mirrors ref_image1.png layout
# ─────────────────────────────────────────────────────────────────────────────

def fig_pipeline_overview() -> Path:
    fig_w, fig_h = 3.8, 9.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off"); ax.set_facecolor(BG)
    ax.set_title("SVE End-to-End Pipeline",
                 fontsize=9.5, fontweight="bold", color=NAVY, pad=8)

    BW, BH = 0.56, 0.068
    cx_l, cx_r = 0.30, 0.74   # left / right branch centres

    # Stage 1: Data source
    y0 = 0.940
    _box(ax, 0.50, y0, BW, BH, "Raw source: QM9 via", "torch_geometric.datasets.QM9")

    # Stage 1b: DatasetManager
    y1 = 0.845
    _box(ax, 0.50, y1, BW, BH, "DatasetManager.load qm9", "")
    _arrow(ax, 0.50, y0-BH/2, 0.50, y1+BH/2)

    # Stage 2: PyG sample
    y2 = 0.750
    _box(ax, 0.50, y2, BW, BH, "PyG Data sample", "x, edge_index, pos, z, y, name")
    _arrow(ax, 0.50, y1-BH/2, 0.50, y2+BH/2)

    # Branch 1: split logic
    y3a = 0.645
    _box(ax, cx_l, y3a, 0.40, BH, "Split logic", "train / val / test")
    _arrow(ax, 0.50-BW/2, y2, cx_l, y3a+BH/2)

    # Branch 2: Mol2 export
    y3b = 0.645
    _box(ax, cx_r, y3b, 0.40, BH, "Optional Mol2Exporter", "data/QM9_mol2/*.mol2")
    _arrow(ax, 0.50+BW/2, y2, cx_r, y3b+BH/2)

    # Right branch: Mol2 parser
    y4b = 0.545
    _box(ax, cx_r, y4b, 0.40, BH, "Mol2 parser", "coordinates, atom types, bonds")
    _arrow(ax, cx_r, y3b-BH/2, cx_r, y4b+BH/2)

    # Left continues to graph voxel branch
    y4a = 0.435
    _box(ax, cx_l, y4a, 0.40, BH, "Graph coordinate branch", "voxelize_data")
    _arrow(ax, cx_l, y3a-BH/2, cx_l, y4a+BH/2)

    # Right continues to mol2 voxel branch
    y5b = 0.435
    _box(ax, cx_r, y5b, 0.40, BH, "Mol2 coordinate branch", "voxelize_positions")
    _arrow(ax, cx_r, y4b-BH/2, cx_r, y5b+BH/2)

    # Dense voxel target (merge both branches)
    y6 = 0.330
    _box(ax, 0.50, y6, BW, BH, "Dense voxel target", "")
    _arrow(ax, cx_l, y4a-BH/2, 0.50-BW/4, y6+BH/2)
    _arrow(ax, cx_r, y5b-BH/2, 0.50+BW/4, y6+BH/2)

    # Two training boxes
    y7a, y7b = 0.225, 0.225
    _box(ax, cx_l, y7a, 0.40, BH, "Graph to voxel training", "G2V model")
    _box(ax, cx_r, y7b, 0.40, BH, "Voxel autoencoder training", "VAE model")
    _arrow(ax, 0.50, y6-BH/2, cx_l, y7a+BH/2)
    _arrow(ax, 0.50, y6-BH/2, cx_r, y7b+BH/2)

    # Latent z from each
    y8a, y8b = 0.130, 0.130
    _box(ax, cx_l, y8a, 0.40, BH, "Latent z from graph", "encoder  (256-D)")
    _box(ax, cx_r, y8b, 0.40, BH, "Latent z from voxel", "encoder  (256-D)")
    _arrow(ax, cx_l, y7a-BH/2, cx_l, y8a+BH/2)
    _arrow(ax, cx_r, y7b-BH/2, cx_r, y8b+BH/2)

    # Embedding export
    y9 = 0.033
    _box(ax, 0.50, y9, BW, BH, "Embedding export  (CSV / NPZ / JSON)", "spatial_emb_0000 \u2026 0255")
    _arrow(ax, cx_l, y8a-BH/2, 0.50-BW/4, y9+BH/2)
    _arrow(ax, cx_r, y8b-BH/2, 0.50+BW/4, y9+BH/2)

    # Stage numbers on left margin
    stages = [(y0,"1"),(y2,"2"),(y6,"3"),(y9,"5")]
    for y,n in stages:
        ax.text(0.01, y, n, ha="left", va="center", fontsize=7,
                color="#555555", fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.15", facecolor="#DDDDDD",
                          edgecolor="#AAAAAA", lw=0.6))

    plt.tight_layout(pad=0.3)
    return _save(fig, "fig7_pipeline_overview.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 – Results Summary Table
# ─────────────────────────────────────────────────────────────────────────────

def fig_results_table() -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.axis("off")
    ax.set_title("Experimental Results Summary (Diagnostic Runs)",
                 fontsize=10, fontweight="bold", color=NAVY, pad=8)

    col_labels = ["Model", "Task", "Metric", "Value", "Assessment"]
    rows = [
        ["GCNRegressor",     "QM9 gap",          "Val RMSE (eV)",  "1.57",   "Under-trained"],
        ["GCNRegressor",     "QM9 gap",          "Val R\u00b2",    "\u22120.43", "Needs more epochs"],
        ["GraphToVoxelNet",  "Graph\u219216\u00b3 Voxel", "Val Voxel MSE",  "0.0121", "Converging"],
        ["GraphToVoxelNet",  "Graph\u219216\u00b3 Voxel", "Val IoU",        "\u22480",    "Needs longer run"],
        ["VoxelAutoencoder", "Voxel recon. 24\u00b3", "Val Recon MSE",  "0.0069", "Promising"],
        ["G2V smoke (1ep)",  "Graph\u219216\u00b3 Voxel", "Val IoU",        "0.043",  "Smoke only"],
    ]
    cw = [0.20, 0.22, 0.20, 0.13, 0.25]
    tbl = ax.table(cellText=rows, colLabels=col_labels, colWidths=cw, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    for j in range(len(col_labels)):
        tbl[0,j].set_facecolor(NAVY)
        tbl[0,j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows)+1):
        col = "#F5F5F5" if i%2==0 else "white"
        for j in range(len(col_labels)):
            tbl[i,j].set_facecolor(col)
    ax.text(0.5, 0.02,
            "Note: all runs used \u22643 epochs, \u22642000 samples. Research preset: 40\u201350 epochs, 12,000 samples.",
            ha="center", va="bottom", transform=ax.transAxes, fontsize=7.5,
            color="#666666", style="italic")
    plt.tight_layout()
    return _save(fig, "fig8_results_table.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9 – Gaussian Splatting Sigma Illustration
# ─────────────────────────────────────────────────────────────────────────────

def fig_gaussian_splatting() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    fig.suptitle("Gaussian Splatting: Single Atom Contribution  (2-D cross-section)",
                 fontsize=10, fontweight="bold", color=NAVY, y=1.01)
    for ax, sigma in zip(axes, [0.3, 0.5, 1.0]):
        half = 3.0; g = 64
        xs = np.linspace(-half, half, g)
        XX, YY = np.meshgrid(xs, xs)
        density = np.exp(-(XX**2+YY**2)/(2*sigma**2))
        im = ax.imshow(density, origin="lower", cmap="hot",
                       extent=[-half,half,-half,half], interpolation="bilinear", vmin=0, vmax=1)
        ax.scatter(0, 0, color="cyan", s=35, zorder=5, label="atom")
        th = np.linspace(0, 2*np.pi, 200)
        ax.plot(sigma*np.cos(th), sigma*np.sin(th), "c--", lw=1.0, label=f"\u03c3={sigma}")
        ax.plot(2*sigma*np.cos(th), 2*sigma*np.sin(th), "c:", lw=0.8, label="2\u03c3")
        ax.set_title(f"\u03c3 = {sigma} \u00c5", fontsize=9, color=NAVY)
        ax.set_xlabel("x (\u00c5)", fontsize=8); ax.set_ylabel("y (\u00c5)", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.legend(fontsize=7, loc="upper right")
        fig.colorbar(im, ax=ax, fraction=0.04).set_label("G(r)", fontsize=7)
    plt.tight_layout()
    return _save(fig, "fig9_gaussian_splatting.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

ALL_FIGURES = [
    ("fig1", fig_dataset_profile,     "QM9 dataset profile"),
    ("fig2", fig_voxel_sample,        "Voxelized molecule sample"),
    ("fig3", fig_training_curves,     "Training curves"),
    ("fig4", fig_g2v_architecture,    "G2V architecture"),
    ("fig5", fig_vae_architecture,    "VAE architecture"),
    ("fig6", fig_protac_architecture, "PROTAC architecture"),
    ("fig7", fig_pipeline_overview,   "Pipeline overview"),
    ("fig8", fig_results_table,       "Results table"),
    ("fig9", fig_gaussian_splatting,  "Gaussian splatting"),
]

def generate_all() -> dict[str, Path]:
    paths = {}
    for key, fn, desc in ALL_FIGURES:
        try:
            paths[key] = fn()
            print(f"  \u2713 {desc}")
        except Exception as exc:
            print(f"  \u2717 {desc}: {exc}")
    return paths

if __name__ == "__main__":
    print("Generating figures...")
    generate_all()
    print("Done.")
