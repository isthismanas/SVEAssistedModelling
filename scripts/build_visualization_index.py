#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build index.html for visualization artifacts.")
    parser.add_argument(
        "--visualizations-dir",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "visualizations"),
        help="Directory containing visualization HTML files.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="",
        help="Optional explicit output HTML path. Default: <visualizations-dir>/index.html",
    )
    return parser.parse_args()


def _classify(name: str) -> str:
    if name.startswith("result_showcase"):
        return "Result Showcase"
    if name.startswith("voxel_vs_mol2"):
        return "Voxel vs mol2"
    if name.startswith("training_metrics"):
        return "Training Metrics"
    return "Other"


def _relative_link(base_dir: Path, file_path: Path) -> str:
    return file_path.relative_to(base_dir).as_posix()


def _render_group(group_name: str, files: list[Path], base_dir: Path) -> str:
    if not files:
        return ""

    items = []
    for f in files:
        href = html.escape(_relative_link(base_dir, f))
        label = html.escape(f.name)
        items.append(
            f"""
            <li>
              <a class=\"file-link\" href=\"{href}\" target=\"_blank\" rel=\"noopener noreferrer\">{label}</a>
            </li>
            """.strip()
        )

    items_html = "\n".join(items)

    return f"""
    <section class=\"group\">
      <h2>{html.escape(group_name)}</h2>
      <ul>
        {items_html}
      </ul>
    </section>
    """.strip()


def build_index(visualizations_dir: Path, output_file: Path) -> Path:
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    html_files = sorted(
        [
            p
            for p in visualizations_dir.glob("*.html")
            if p.name != output_file.name
        ]
    )

    grouped: dict[str, list[Path]] = {
        "Result Showcase": [],
        "Voxel vs mol2": [],
        "Training Metrics": [],
        "Other": [],
    }

    for file_path in html_files:
        grouped[_classify(file_path.name)].append(file_path)

    sections = [
        _render_group("Result Showcase", grouped["Result Showcase"], visualizations_dir),
        _render_group("Voxel vs mol2", grouped["Voxel vs mol2"], visualizations_dir),
        _render_group("Training Metrics", grouped["Training Metrics"], visualizations_dir),
        _render_group("Other", grouped["Other"], visualizations_dir),
    ]
    sections_html = "\n\n".join([s for s in sections if s])

    page = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Visualization Index</title>
  <style>
    :root {{
      --bg: #0b132b;
      --panel: #1c2541;
      --text: #f8f9fa;
      --muted: #c9d6ea;
      --accent: #5bc0be;
      --accent-2: #6fffe9;
      --border: #33415c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 28px;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      color: var(--text);
      background: radial-gradient(circle at top right, #1d2d50 0%, var(--bg) 55%);
    }}
    .container {{
      max-width: 980px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 10px 0;
      font-size: 2rem;
      letter-spacing: 0.2px;
    }}
    .meta {{
      color: var(--muted);
      margin-bottom: 20px;
      font-size: 0.95rem;
    }}
    .group {{
      background: linear-gradient(180deg, rgba(28,37,65,0.92), rgba(20,29,51,0.92));
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px;
      margin-bottom: 14px;
    }}
    h2 {{
      margin: 0 0 10px 0;
      font-size: 1.1rem;
      color: var(--accent-2);
    }}
    ul {{
      list-style: none;
      margin: 0;
      padding: 0;
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
    }}
    .file-link {{
      display: block;
      text-decoration: none;
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 10px;
      background: rgba(255,255,255,0.02);
      padding: 10px 12px;
      transition: transform 120ms ease, border-color 120ms ease, background 120ms ease;
      word-break: break-word;
    }}
    .file-link:hover {{
      border-color: var(--accent);
      background: rgba(91,192,190,0.12);
      transform: translateY(-1px);
    }}
  </style>
</head>
<body>
  <div class=\"container\">
    <h1>Visualization Index</h1>
    <div class=\"meta\">Directory: {html.escape(str(visualizations_dir))} | Files: {len(html_files)}</div>
    {sections_html if sections_html else '<p>No visualization HTML files found yet.</p>'}
  </div>
</body>
</html>
"""

    output_file.write_text(page)
    return output_file


def main() -> None:
    args = parse_args()
    visualizations_dir = Path(args.visualizations_dir).resolve()
    output_file = Path(args.output_file).resolve() if args.output_file else (visualizations_dir / "index.html")

    out = build_index(visualizations_dir, output_file)
    print(f"Wrote visualization index: {out}")


if __name__ == "__main__":
    main()
