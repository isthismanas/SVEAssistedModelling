#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_PATH = RESULTS_DIR / "model_diagnostics.json"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _assess_baseline(payload: dict) -> dict:
    test = payload.get("test_metrics", {})
    r2 = float(test.get("val_r2", float("nan")))
    return {
        "epochs": payload.get("epochs"),
        "max_samples": payload.get("max_samples"),
        "test_metrics": test,
        "assessment": "poor" if r2 < 0.0 else "needs_review",
        "notes": [
            "Negative test R2 means the current baseline is not yet reliable for generalization claims."
        ],
    }


def _assess_graph_voxel(payload: dict) -> dict:
    test = payload.get("test_metrics", {})
    overlap = float(test.get("val_voxel_overlap", 0.0))
    mse = float(test.get("val_voxel_mse", float("nan")))
    notes = []
    if overlap < 0.05:
        notes.append("Voxel overlap is near zero, indicating the model is not reconstructing localized structure.")
    if payload.get("epochs", 0) < 10:
        notes.append("Training run is far too short for a deployment-quality spatial decoder.")
    return {
        "epochs": payload.get("epochs"),
        "max_samples": payload.get("max_samples"),
        "test_metrics": test,
        "assessment": "poor" if overlap < 0.05 else "needs_review",
        "notes": notes,
        "failure_mode_hypothesis": [
            "Sparse voxel targets plus plain MSE favored diffuse low-information predictions.",
            "Old decoder output activation biased predictions toward globally positive fields.",
        ],
        "recommended_changes_applied": [
            "Shifted softplus output so zero logits map to zero output.",
            "Added occupied-region weighting in reconstruction loss.",
            "Added sparsity penalty to discourage filled-cube predictions.",
            "Raised graph-to-voxel training defaults to larger sample count and longer runs.",
        ],
        "reference_mse": mse,
    }


def _assess_autoencoder(payload: dict) -> dict:
    test = payload.get("test_metrics", {})
    notes = []
    if payload.get("epochs", 0) < 10:
        notes.append("Autoencoder run is too short to judge deployment readiness.")
    if payload.get("max_samples", 0) < 1000:
        notes.append("Autoencoder was trained on a very small subset of QM9 mol2 files.")
    return {
        "epochs": payload.get("epochs"),
        "max_samples": payload.get("max_samples"),
        "test_metrics": test,
        "assessment": "weak_but_promising",
        "notes": notes,
        "recommended_changes_applied": [
            "Shifted softplus output so zero logits map to zero output.",
            "Added occupied-region weighting in reconstruction loss.",
            "Added sparsity penalty to discourage diffuse reconstruction fields.",
            "Raised default autoencoder training duration.",
        ],
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    baseline = _load_json(ARTIFACTS_DIR / "baseline_qm9_gap.json")
    graph = _load_json(ARTIFACTS_DIR / "graph_to_voxel_qm9.json")
    graph_smoke = _load_json(ARTIFACTS_DIR / "graph_to_voxel_qm9_smoke.json")
    autoencoder = _load_json(ARTIFACTS_DIR / "voxel_autoencoder_qm9.json")

    payload = {
        "project_root": str(PROJECT_ROOT),
        "diagnostics_generated_from": "existing artifact json files",
        "models": {},
    }

    if baseline is not None:
        payload["models"]["baseline_qm9_gap"] = _assess_baseline(baseline)
    if graph is not None:
        payload["models"]["graph_to_voxel_qm9"] = _assess_graph_voxel(graph)
    if graph_smoke is not None:
        payload["models"]["graph_to_voxel_qm9_smoke"] = _assess_graph_voxel(graph_smoke)
    if autoencoder is not None:
        payload["models"]["voxel_autoencoder_qm9"] = _assess_autoencoder(autoencoder)

    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote diagnostics: {OUTPUT_PATH}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
