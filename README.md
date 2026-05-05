# Molecular Interaction Simulation (Codebase Version)

This repository trains molecular graph models, including a graph-to-voxel model for spatial learning.

Current focus:
- Train a scalar QM9 baseline (`graph -> property`).
- Train a spatial model (`graph -> voxel grid`).
- Keep code lean around core research loops.

## Quick start

```bash
python3 scripts/prepare_datasets.py --dataset-ids qm9 --export-mol2 --max-mol2 0
python3 scripts/train_qm9_baseline.py --target gap --epochs 3 --max-samples 2000
python3 scripts/train_graph_to_voxel.py --epochs 3 --max-samples 2000 --grid-size 16
python3 scripts/visualize_voxel_vs_mol2.py --dataset-id qm9 --sample-index 0 --auto-export-mol2
python3 scripts/visualize_training_metrics.py
python3 scripts/result_showcase.py --dataset-id qm9 --sample-index 0 --mol2-dir data/QM9_mol2
python3 scripts/result_showcase.py --dataset-id qm9 --sample-indices 0,100,1000 --mol2-dir data/QM9_mol2
python3 scripts/build_visualization_index.py
python3 scripts/train_mol2_spatial_encoder.py --mol2-dir data/QM9_mol2 --epochs 8 --max-samples 12000 --device cpu
python3 scripts/export_degrademaster_embeddings.py --mol2-dir data/QM9_mol2 --checkpoint-path artifacts/voxel_autoencoder_qm9.pt --output-npz artifacts/degrademaster_mol2_embeddings.npz
python3 scripts/train_graph_to_voxel.py --epochs 8 --batch-size 32 --max-samples 12000 --grid-size 16 --latent-dim 256 --device cpu --mol2-dir data/QM9_mol2 --checkpoint-path artifacts/graph_to_voxel_qm9.pt
python3 scripts/export_graph_spatial_embeddings.py --dataset-id qm9 --checkpoint-path artifacts/graph_to_voxel_qm9.pt --split all --output-csv artifacts/degrademaster_graph_spatial_embeddings.csv
python3 scripts/generate_report_examples.py
python3 scripts/build_academic_report.py
python3 scripts/build_assignment2_report.py
```

## Structure

- `molsim/metrics.py`: reusable metric evaluators.
- `molsim/data/`: dataset loading and QM9 target adaptation.
- `molsim/models/`: baseline neural architectures.
- `molsim/spatial/`: voxelization and mol2 parsing utilities.
- `molsim/training/`: train/eval loop utilities.
- `scripts/prepare_datasets.py`: dataset download and mol2 export utility.
- `scripts/train_qm9_baseline.py`: scalar baseline training entrypoint.
- `scripts/train_graph_to_voxel.py`: graph-to-voxel training entrypoint.
- `scripts/visualize_voxel_vs_mol2.py`: Plotly 3D voxel vs mol2 comparison.
- `scripts/visualize_training_metrics.py`: Plotly dashboard for training artifacts.
- `scripts/result_showcase.py`: one-page consolidated Plotly showcase.
- `scripts/build_visualization_index.py`: clickable HTML index for all generated visualizations.
- `scripts/train_mol2_spatial_encoder.py`: train mol2 voxel autoencoder encoder.
- `scripts/export_degrademaster_embeddings.py`: export mol2 embeddings for DegradeMaster input.
- `scripts/export_graph_spatial_embeddings.py`: export graph-derived spatial embeddings for DegradeMaster input.
- `scripts/generate_report_examples.py`: generate report example assets for one QM9 sample.
- `scripts/build_academic_report.py`: build the academic Word report with Mermaid architecture figures.
- `scripts/build_assignment2_report.py`: build the assignment-specific data and method report.
- `docs/stage_1_detailed.md`: detailed Stage 1 methodology with equations.
- `docs/SPATIAL_FUSION_ACADEMIC_REPORT.md`: manuscript source for the academic report.
- `docs/SPATIAL_FUSION_ACADEMIC_REPORT.docx`: generated Microsoft Word report.
- `docs/ASSIGNMENT2_DATA_METHOD_REPORT.md`: assignment-specific manuscript source.
- `docs/ASSIGNMENT2_DATA_METHOD_REPORT.docx`: generated assignment report in Microsoft Word format.
- `docs/RUN_PROJECT_END_TO_END.md`: full terminal command runbook.
- `ROADMAP.md`: staged implementation roadmap.
- `AGENTS.md`: local repo operating instructions.
