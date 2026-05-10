# AGENTS.md

## Scope
- Repository root is the working project. The old path `Molecule interaction simulation proof of concept/` referenced in prior docs does not exist in this checkout.

## Layout That Matters
- `scripts/` is the real entrypoint surface. Run workflows as `python3 scripts/<name>.py` from the repo root.
- `molsim/data/` owns dataset loading, QM9 target adaptation, mol2 export, and the reusable seeded splitter (`DatasetManager.split_indices`).
- `molsim/models/` owns model definitions.
- `molsim/training/` owns training loops.
- `molsim/spatial/` owns voxelization and mol2 parsing.
- `molsim/metrics.py` is the central metrics file.
- `downstream/` is a mirrored DegradeMaster subtree with its own runtime assumptions.

## Verified Commands
- Prepare QM9 data: `python3 scripts/prepare_datasets.py --dataset-ids qm9`
- Export QM9 mol2 files: `python3 scripts/prepare_datasets.py --dataset-ids qm9 --export-mol2 --max-mol2 1000`
- Train scalar baseline: `python3 scripts/train_qm9_baseline.py --target gap --epochs 3 --max-samples 2000 --device cpu`
- Train graph-to-voxel model: `python3 scripts/train_graph_to_voxel.py --epochs 3 --max-samples 2000 --grid-size 16 --device cpu`
- Train mol2 autoencoder: `python3 scripts/train_mol2_spatial_encoder.py --mol2-dir data/QM9_mol2 --epochs 8 --max-samples 12000 --device cpu`
- Export graph spatial embeddings: `python3 scripts/export_graph_spatial_embeddings.py --dataset-id qm9 --checkpoint-path artifacts/graph_to_voxel_qm9.pt --split all`
- Export mol2 embeddings: `python3 scripts/export_degrademaster_embeddings.py --mol2-dir data/QM9_mol2 --checkpoint-path artifacts/voxel_autoencoder_qm9.pt`

## Workflow Quirks
- Most root scripts insert `PROJECT_ROOT` into `sys.path` themselves. Prefer running the scripts directly, not `python -m ...`.
- Quick verification is done by shrinking `--max-samples` and `--epochs`; there is no test or CI config in this repo.
- Artifacts are written under `artifacts/`. Dataset roots are under `data/`.

## Split And Metrics Caveat
- Do not assume every training script uses `DatasetManager.split_indices` yet.
- `scripts/export_graph_spatial_embeddings.py` uses `DatasetManager.split_indices` with `SplitConfig(seed=42)`.
- `scripts/train_qm9_baseline.py`, `scripts/train_graph_to_voxel.py`, and `scripts/train_mol2_spatial_encoder.py` currently do their own local shuffle-then-split logic.
- If you change metric behavior, keep it in `molsim/metrics.py` instead of duplicating logic in scripts.

## Downstream Gotchas
- Root wrappers for the mirrored downstream project are `scripts/run_degrademaster_classification.py`, `scripts/run_degrademaster_regression_preprocess.py`, and `scripts/run_degrademaster_regression.py`.
- Those wrappers run with `downstream/` as the working directory because the mirrored code depends on relative paths.
- `downstream/config/` has been restored locally, along with missing `regression_models/`, `tokenizer/`, and `utils/` code copied from the local DegradeMaster source tree. This made the downstream wrappers runnable in this checkout.
- Downstream training still depends on ignored local data/config state, so treat it as reproducible only on this machine unless those inputs are exported cleanly.
- DegradeMaster runs should currently be treated as CPU-only on this Mac. `molsim` can use `mps`, but downstream `torch_scatter` and related runtime assumptions were not stable on Apple GPU.

## Report Builders
- `scripts/build_academic_report.py` and `scripts/build_assignment2_report.py` require external tools that are not declared in `pyproject.toml`: `pandoc`, Mermaid CLI (`mmdc`), and Chrome/Chromium for Mermaid rendering.

## State File
- `project_state.json` still carries stale absolute artifact paths from an older machine path. Do not trust its `artifacts` values as current filesystem locations without re-verifying them.

## Current Status
- The main stakeholder/demo notebook is `notebooks/experiment_runner_and_analysis.ipynb`.
- Current best SVE generator is the mol2 voxel autoencoder in `molsim/models/voxel_autoencoder.py`; the graph-to-voxel path is still weaker visually.
- PROTAC SVE export and merged dataset prep have already been completed locally. Canonical data roots in this repo are `data/PROTAC`, `data/PROTAC_sve`, `data/PROTAC_regression`, and `data/PROTAC_regression_sve`.
- Completed downstream classification runs exist for baseline and with-SVE. Baseline currently outperforms with-SVE.
- Completed downstream regression runs exist for `base`, `two_head`, `cross_attention`, `pdc50_bounded`, and `tabular`, each with baseline and with-SVE variants.
- Current best downstream regression result is `regression_cross_attention_with_sve`, with mean RMSE around `223`.
- Current interpretation is architecture-dependent SVE value: it hurts the base regression model and classification, helps `two_head`, helps `cross_attention` the most, helps `pdc50_bounded` but not enough overall, and has no effect on `tabular`.
- Results registries used by the notebook live under `results/`, especially `results/degrademaster_results.json` and `results/model_diagnostics.json`.
- The latest notebook issue already fixed in this checkout was a final-takeaway summary bug caused by subtracting string-valued columns in `downstream_sve_comparison_frame()`. The helper now filters to numeric metrics only.
