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

## Fresh Machine Setup

Use this sequence on a new machine such as the Zephyrus.

1. Clone the repo.

```bash
git clone https://github.com/isthismanas/SVEAssistedModelling.git
cd SVEAssistedModelling
```

2. Create a Python 3.11 environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

3. Install PyTorch for your machine first.

- For NVIDIA/CUDA, use the official PyTorch install command from pytorch.org.
- For CPU-only, install the CPU build.

4. Install PyG and scatter packages that match your PyTorch build.

- `torch-geometric`
- `torch-scatter`

Use the matching wheel instructions from the PyG install guide.

5. Install this repo and the extra runtime packages used by downstream and notebooks.

```bash
pip install -e .
pip install pandas pyyaml biopython jupyter matplotlib nbformat
```

6. Install chemistry/runtime dependencies needed by the mirrored downstream project.

- `rdkit`
- `openbabel`

On Linux, these are often easiest via `conda`/`mamba`, or via your system package manager plus pip depending on your setup.

7. Verify the core imports.

```bash
python -c "import torch, torch_geometric, yaml, pandas, Bio"
python -c "import torch_scatter"
python -c "from rdkit import Chem"
```

8. Prepare data and regenerate local artifacts.

- QM9 prep:

```bash
python3 scripts/prepare_datasets.py --dataset-ids qm9
python3 scripts/prepare_datasets.py --dataset-ids qm9 --export-mol2 --max-mol2 1000
```

- For downstream PROTAC work, regenerate the local `data/` layout before training. Large datasets and checkpoints are intentionally not stored in git.

9. Run smoke tests.

```bash
python3 scripts/train_qm9_baseline.py --target gap --epochs 3 --max-samples 2000 --device cpu
python3 scripts/train_graph_to_voxel.py --epochs 3 --max-samples 2000 --grid-size 16 --device cpu
python3 scripts/train_mol2_spatial_encoder.py --mol2-dir data/QM9_mol2 --epochs 8 --max-samples 12000 --device cpu
```

10. Open the main notebook after the environment, data, and artifacts exist.

- Main demo notebook: `notebooks/experiment_runner_and_analysis.ipynb`

## Reproducibility Notes

- This repo intentionally does not include large `data/` folders, trained checkpoints, or generated run directories.
- `pyproject.toml` covers the core `molsim` dependencies, but the mirrored downstream project needs extra runtime packages such as `pandas`, `pyyaml`, `biopython`, `rdkit`, `openbabel`, and `torch-scatter`.
- On this Mac checkout, `molsim` was run on `mps` and downstream DegradeMaster was effectively CPU-only. On the Zephyrus, use CUDA if your PyTorch and PyG stack are installed consistently.
- Report builders also require external tools not declared in `pyproject.toml`: `pandoc`, Mermaid CLI (`mmdc`), and Chrome/Chromium.

## Notebook Export

- Preferred export command:

```bash
python3 scripts/export_notebook.py notebooks/experiment_runner_and_analysis.ipynb --to html
```

- PDF export requires a TeX distribution with `xelatex` on `PATH`:

```bash
python3 scripts/export_notebook.py notebooks/experiment_runner_and_analysis.ipynb --to pdf
```

- On this Mac, the VS Code export button may fail even when the selected interpreter is correct because `jupyter nbconvert` can resolve to a global `jupyter-nbconvert` instead of the environment-local one. The helper script above avoids that by calling `python -m nbconvert` directly from the active environment.
- If you export to LaTeX/PDF, Plotly cells are not preserved as interactive figures. HTML export is the better fit for this notebook.

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
