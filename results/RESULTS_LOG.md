# Results Log

This file is the human-readable summary for experiments run from this repository.

Machine-readable run records live in `results/experiment_log.jsonl`.
Raw command logs written by the notebook live under `results/generated/command_logs/` and are gitignored.

## Run Summary

| Date | Family | Run | Source Artifact | Notes |
| --- | --- | --- | --- | --- |
| _fill after runs_ | `molsim` / `degrademaster` | _name_ | _path_ | _short summary_ |

## Comparison Checklist

- `molsim` baseline QM9 regression metrics captured
- `molsim` graph-to-voxel training history captured
- `molsim` mol2 autoencoder history captured
- DegradeMaster classification baseline captured
- DegradeMaster regression variant metrics captured for `base`
- DegradeMaster regression variant metrics captured for `two_head`
- DegradeMaster regression variant metrics captured for `cross_attention`
- DegradeMaster regression variant metrics captured for `pdc50_bounded`
- DegradeMaster regression variant metrics captured for `tabular`
- SVE-augmented downstream comparison captured after feature merge

## Notes

- `molsim` writes structured histories directly into artifact JSON files under `artifacts/`.
- DegradeMaster regression writes final metrics to `final_metrics.json` and predictions to `test_predictions.npz` inside each run directory.
- The notebook `notebooks/experiment_runner_and_analysis.ipynb` can run commands, append JSONL entries, and build 2D/3D visualizations from those artifacts.
