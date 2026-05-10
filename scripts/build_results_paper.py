#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
SOURCE_MD = RESULTS_DIR / "paper_spatial_molecular_embeddings.md"
OUTPUT_DOCX = RESULTS_DIR / "paper_spatial_molecular_embeddings.docx"
DIAGNOSTICS_JSON = RESULTS_DIR / "model_diagnostics.json"


def load_diagnostics() -> dict:
    if not DIAGNOSTICS_JSON.exists():
        raise FileNotFoundError(f"Missing diagnostics JSON: {DIAGNOSTICS_JSON}")
    return json.loads(DIAGNOSTICS_JSON.read_text(encoding="utf-8"))


def build_markdown(payload: dict) -> str:
    models = payload.get("models", {})
    baseline = models.get("baseline_qm9_gap", {})
    graph = models.get("graph_to_voxel_qm9", {})
    auto = models.get("voxel_autoencoder_qm9", {})

    baseline_metrics = baseline.get("test_metrics", {})
    graph_metrics = graph.get("test_metrics", {})
    auto_metrics = auto.get("test_metrics", {})

    return f"""# Spatial Molecular Embeddings Results Note

## Purpose

This note records the current state of the spatial embedding project and the architectural changes made after inspecting the numerical and visual reconstruction failures of the voxel models.

The project goal is not only to train a spatial representation on QM9, but to determine whether a usable spatial vector embedding (SVE) can later improve downstream DegradeMaster tasks when generated on the DegradeMaster PROTAC molecules themselves.

## Current Numerical State

### Baseline QM9 regressor

- Epochs: {baseline.get('epochs')}
- Max samples: {baseline.get('max_samples')}
- Test RMSE: {baseline_metrics.get('val_rmse')}
- Test MAE: {baseline_metrics.get('val_mae')}
- Test R2: {baseline_metrics.get('val_r2')}

### Graph-to-voxel model

- Epochs: {graph.get('epochs')}
- Max samples: {graph.get('max_samples')}
- Test voxel MSE: {graph_metrics.get('val_voxel_mse')}
- Test voxel overlap: {graph_metrics.get('val_voxel_overlap')}

### Mol2 voxel autoencoder

- Epochs: {auto.get('epochs')}
- Max samples: {auto.get('max_samples')}
- Test reconstruction MSE: {auto_metrics.get('val_recon_mse')}

## Diagnosis

The baseline regressor now shows positive test $R^2$, so the scalar prediction path is no longer failing catastrophically.

The graph-to-voxel model improved substantially over the old smoke and 3-epoch runs, but visual inspection still shows a diffuse volumetric prediction rather than a sharply localized molecular structure. This means the latent representation may be learning coarse occupancy trends while still underperforming as a faithful structural embedding.

The mol2 voxel autoencoder is currently the more plausible embedding path for downstream use, but its reconstruction quality still needs architectural strengthening before it should be trusted for deployment-level downstream augmentation.

## Architecture Changes Applied

The original voxel autoencoder used a simple strided encoder and transposed-convolution decoder with a positively biased output regime. To improve reconstruction fidelity, the following changes were applied:

1. **Per-sample voxel max normalization** of the training input.
2. **3D U-Net style skip-connected encoder-decoder** structure.
3. **Instance normalization and LeakyReLU blocks** inside the 3D convolution stack.
4. **Sigmoid-bounded reconstruction output** for normalized voxel targets.
5. **Hybrid reconstruction loss** combining MSE, L1, occupied-region loss, soft Dice-style overlap loss, and a sparsity regularizer.

These changes were chosen because the earlier model often produced diffuse positive fields instead of localized reconstructions.

## Literature References Used For The Fix

The revised voxel autoencoder design was guided by established volumetric reconstruction and segmentation literature:

1. **3D ShapeNets** (Wu et al., 2015) introduced voxel-grid-based 3D shape modeling and treated 3D structure as occupancy on a voxel lattice. This is directly relevant because the current project also learns from voxelized structure rather than only graph topology.

2. **3D U-Net** (Cicek et al., 2016) showed that encoder-decoder skip connections are highly effective for dense volumetric prediction. This motivated the move from a plain bottlenecked decoder to a skip-connected 3D architecture.

3. **V-Net** (Milletari et al., 2016) highlighted the importance of overlap-sensitive objectives such as Dice-style losses for highly imbalanced volumetric foreground/background settings. This motivated the addition of a Dice-informed occupancy term rather than relying on plain MSE alone.

## References

Wu, Z., Song, S., Khosla, A., Yu, F., Zhang, L., Tang, X., & Xiao, J. (2015). *3D ShapeNets: A Deep Representation for Volumetric Shapes*. CVPR. arXiv:1406.5670.

Cicek, O., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016). *3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation*. MICCAI. arXiv:1606.06650.

Milletari, F., Navab, N., & Ahmadi, S.-A. (2016). *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation*. arXiv:1606.04797.

## Interpretation For Next Phase

The immediate next question is not whether QM9 embeddings can be fed directly into DegradeMaster. They cannot. Instead, the current task is to identify which embedding architecture is strong enough to be run on the **DegradeMaster PROTAC mol2 dataset**.

If the revised mol2 voxel autoencoder yields visibly sharper reconstructions and better overlap-style behavior after retraining, it should be the first embedding generator used to build `PROTAC_sve` and `PROTAC_regression_sve` for the downstream baseline-vs-SVE comparison.
"""


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = load_diagnostics()
    markdown = build_markdown(payload)
    SOURCE_MD.write_text(markdown, encoding="utf-8")

    subprocess.run([
        "pandoc",
        str(SOURCE_MD),
        "--from",
        "markdown",
        "--to",
        "docx",
        "--output",
        str(OUTPUT_DOCX),
    ], check=True)
    print(f"Wrote markdown: {SOURCE_MD}")
    print(f"Wrote docx: {OUTPUT_DOCX}")


if __name__ == "__main__":
    main()
