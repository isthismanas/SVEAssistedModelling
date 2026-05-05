#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Plotly dashboard from baseline/voxel training artifacts.")
    parser.add_argument(
        "--baseline-artifact",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "baseline_qm9_gap.json"),
        help="Path to baseline artifact JSON.",
    )
    parser.add_argument(
        "--voxel-artifact",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "graph_to_voxel_qm9.json"),
        help="Path to voxel artifact JSON.",
    )
    parser.add_argument(
        "--output-html",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "visualizations" / "training_metrics_dashboard.html"),
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError("plotly is required: pip install plotly") from exc

    args = parse_args()

    baseline = _load_json(Path(args.baseline_artifact).resolve())
    voxel = _load_json(Path(args.voxel_artifact).resolve())

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Baseline Epoch Curves",
            "Voxel Epoch Curves",
            "Baseline Test Metrics",
            "Voxel Test Metrics",
        ),
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "bar"}, {"type": "bar"}]],
    )

    if baseline and baseline.get("history"):
        history = baseline["history"]
        epochs = [h.get("epoch") for h in history]

        fig.add_trace(
            go.Scatter(x=epochs, y=[h.get("train_mse") for h in history], mode="lines+markers", name="baseline train_mse"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=[h.get("val_rmse") for h in history], mode="lines+markers", name="baseline val_rmse"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=[h.get("val_mae") for h in history], mode="lines+markers", name="baseline val_mae"),
            row=1,
            col=1,
        )

        test = baseline.get("test_metrics", {})
        fig.add_trace(
            go.Bar(x=list(test.keys()), y=list(test.values()), name="baseline test"),
            row=2,
            col=1,
        )

    if voxel and voxel.get("history"):
        history = voxel["history"]
        epochs = [h.get("epoch") for h in history]

        fig.add_trace(
            go.Scatter(x=epochs, y=[h.get("train_voxel_mse") for h in history], mode="lines+markers", name="voxel train_mse"),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=[h.get("val_voxel_mse") for h in history], mode="lines+markers", name="voxel val_mse"),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=[h.get("val_voxel_overlap") for h in history],
                mode="lines+markers",
                name="voxel overlap",
            ),
            row=1,
            col=2,
        )

        test = voxel.get("test_metrics", {})
        fig.add_trace(
            go.Bar(x=list(test.keys()), y=list(test.values()), name="voxel test"),
            row=2,
            col=2,
        )

    fig.update_layout(
        title="Training Result Dashboard",
        width=1400,
        height=900,
        barmode="group",
    )

    out_path = Path(args.output_html).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Wrote dashboard: {out_path}")


if __name__ == "__main__":
    main()
