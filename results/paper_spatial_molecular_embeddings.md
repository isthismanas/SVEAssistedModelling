# Spatial Molecular Embeddings Results Note

## Purpose

This note records the current state of the spatial embedding project and the architectural changes made after inspecting the numerical and visual reconstruction failures of the voxel models.

The project goal is not only to train a spatial representation on QM9, but to determine whether a usable spatial vector embedding (SVE) can later improve downstream DegradeMaster tasks when generated on the DegradeMaster PROTAC molecules themselves.

## Current Numerical State

### Baseline QM9 regressor

- Epochs: 50
- Max samples: 5000
- Test RMSE: 0.9365558965227374
- Test MAE: 0.7276141533851623
- Test R2: 0.5176862690564347

### Graph-to-voxel model

- Epochs: 40
- Max samples: 12000
- Test voxel MSE: 0.014614307678923352
- Test voxel overlap: 0.2551634079562932

### Mol2 voxel autoencoder

- Epochs: 40
- Max samples: 12000
- Test reconstruction MSE: 0.012593637961670415

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
