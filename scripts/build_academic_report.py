#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
ASSETS_DIR = DOCS_DIR / "report_assets"
REPORT_MD = DOCS_DIR / "SPATIAL_FUSION_ACADEMIC_REPORT.md"
REPORT_DOCX = DOCS_DIR / "SPATIAL_FUSION_ACADEMIC_REPORT.docx"


def _resolve_mmdc() -> str:
    candidates = [
        shutil.which("mmdc"),
        str(Path.home() / ".npm" / "_npx" / "668c188756b835f3" / "node_modules" / ".bin" / "mmdc"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("Mermaid CLI not found. Install mermaid-cli or ensure the cached binary exists.")


def _resolve_chrome() -> str:
    candidates = [
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise FileNotFoundError("Chrome/Chromium executable not found for Mermaid rendering.")


def _render_mermaid(mmdc: str, chrome_path: str, source: Path, output: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(
            {
                "executablePath": chrome_path,
                "args": ["--no-sandbox", "--disable-setuid-sandbox"],
            },
            handle,
            indent=2,
        )
        handle.write("\n")
        puppeteer_cfg = Path(handle.name)

    cmd = [
        mmdc,
        "-i",
        str(source),
        "-o",
        str(output),
        "-t",
        "neutral",
        "-w",
        "1600",
        "-H",
        "900",
        "-b",
        "white",
        "-p",
        str(puppeteer_cfg),
        "-q",
    ]
    try:
        subprocess.run(cmd, check=True)
    finally:
        puppeteer_cfg.unlink(missing_ok=True)


def _build_docx() -> None:
    pandoc = shutil.which("pandoc")
    if not pandoc:
        raise FileNotFoundError("pandoc is required to build the .docx report.")

    cmd = [
        pandoc,
        str(REPORT_MD),
        "--from",
        "markdown+tex_math_dollars",
        "--resource-path",
        str(DOCS_DIR),
        "--to",
        "docx",
        "--output",
        str(REPORT_DOCX),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    if not REPORT_MD.exists():
        raise FileNotFoundError(f"Missing report source: {REPORT_MD}")

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    example_builder = PROJECT_ROOT / "scripts" / "generate_report_examples.py"
    subprocess.run([sys.executable, str(example_builder)], check=True)

    mmdc = _resolve_mmdc()
    chrome_path = _resolve_chrome()

    diagrams = [
        ("data_pipeline_overview.mmd", "data_pipeline_overview.png"),
        ("graph_to_voxel_architecture.mmd", "graph_to_voxel_architecture.png"),
        ("voxel_autoencoder_architecture.mmd", "voxel_autoencoder_architecture.png"),
    ]
    for source_name, output_name in diagrams:
        _render_mermaid(
            mmdc=mmdc,
            chrome_path=chrome_path,
            source=ASSETS_DIR / source_name,
            output=ASSETS_DIR / output_name,
        )

    _build_docx()
    print(f"Built academic report: {REPORT_DOCX}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}: {exc.cmd}", file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
