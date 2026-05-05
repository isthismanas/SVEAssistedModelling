#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
ASSETS_DIR = DOCS_DIR / "report_assets"
REPORT_MD = DOCS_DIR / "ASSIGNMENT2_DATA_METHOD_REPORT.md"
REPORT_DOCX = DOCS_DIR / "ASSIGNMENT2_DATA_METHOD_REPORT.docx"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}


def _w(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


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


def _ensure_child(parent: ET.Element, tag: str) -> ET.Element:
    child = parent.find(tag, NS)
    if child is None:
        child = ET.SubElement(parent, _w(tag.split(":")[1]))
    return child


def _set_style_size(root: ET.Element, style_id: str, size_half_points: int) -> None:
    for style in root.findall("w:style", NS):
        if style.get(_w("styleId")) != style_id:
            continue
        rpr = style.find("w:rPr", NS)
        if rpr is None:
            rpr = ET.SubElement(style, _w("rPr"))
        sz = rpr.find("w:sz", NS)
        if sz is None:
            sz = ET.SubElement(rpr, _w("sz"))
        sz.set(_w("val"), str(size_half_points))
        szcs = rpr.find("w:szCs", NS)
        if szcs is None:
            szcs = ET.SubElement(rpr, _w("szCs"))
        szcs.set(_w("val"), str(size_half_points))
        break


def _make_sectpr(columns: int, next_page: bool = False) -> ET.Element:
    sect = ET.Element(_w("sectPr"))

    if next_page:
        sect_type = ET.SubElement(sect, _w("type"))
        sect_type.set(_w("val"), "nextPage")

    pg_sz = ET.SubElement(sect, _w("pgSz"))
    pg_sz.set(_w("w"), "11906")
    pg_sz.set(_w("h"), "16838")

    pg_mar = ET.SubElement(sect, _w("pgMar"))
    for key, value in {
        "top": "720",
        "bottom": "720",
        "left": "720",
        "right": "720",
        "header": "360",
        "footer": "360",
        "gutter": "0",
    }.items():
        pg_mar.set(_w(key), value)

    cols = ET.SubElement(sect, _w("cols"))
    cols.set(_w("num"), str(columns))
    cols.set(_w("space"), "360")

    return sect


def _compact_assignment_docx(path: Path) -> None:
    with ZipFile(path, "r") as zin:
        files = {name: zin.read(name) for name in zin.namelist()}

    styles_root = ET.fromstring(files["word/styles.xml"])
    doc_defaults = styles_root.find("w:docDefaults", NS)
    if doc_defaults is not None:
        rpr_default = _ensure_child(doc_defaults, "w:rPrDefault")
        rpr = _ensure_child(rpr_default, "w:rPr")
        for tag in ("w:sz", "w:szCs"):
            node = rpr.find(tag, NS)
            if node is None:
                node = ET.SubElement(rpr, _w(tag.split(":")[1]))
            node.set(_w("val"), "19")

        ppr_default = _ensure_child(doc_defaults, "w:pPrDefault")
        ppr = _ensure_child(ppr_default, "w:pPr")
        spacing = ppr.find("w:spacing", NS)
        if spacing is None:
            spacing = ET.SubElement(ppr, _w("spacing"))
        spacing.set(_w("after"), "60")
        spacing.set(_w("line"), "220")
        spacing.set(_w("lineRule"), "auto")

    _set_style_size(styles_root, "Heading1", 28)
    _set_style_size(styles_root, "Heading2", 22)
    _set_style_size(styles_root, "Heading3", 20)
    _set_style_size(styles_root, "Title", 32)
    files["word/styles.xml"] = ET.tostring(styles_root, encoding="utf-8", xml_declaration=True)

    document_root = ET.fromstring(files["word/document.xml"])
    body = document_root.find("w:body", NS)
    if body is None:
        raise RuntimeError("Word document body not found")

    paragraphs = body.findall("w:p", NS)
    refs_index = None
    for idx, para in enumerate(paragraphs):
        texts = [t.text or "" for t in para.findall(".//w:t", NS)]
        if "".join(texts).strip() == "References":
            refs_index = idx
            break

    if refs_index is not None:
        break_para = ET.Element(_w("p"))
        ppr = ET.SubElement(break_para, _w("pPr"))
        ppr.append(_make_sectpr(columns=2, next_page=True))
        body.insert(refs_index, break_para)

    final_sect = body.find("w:sectPr", NS)
    if final_sect is not None:
        body.remove(final_sect)
    body.append(_make_sectpr(columns=1, next_page=False))

    files["word/document.xml"] = ET.tostring(document_root, encoding="utf-8", xml_declaration=True)

    with ZipFile(path, "w", compression=ZIP_DEFLATED) as zout:
        for name, data in files.items():
            zout.writestr(name, data)


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
    ]
    for source_name, output_name in diagrams:
        _render_mermaid(
            mmdc=mmdc,
            chrome_path=chrome_path,
            source=ASSETS_DIR / source_name,
            output=ASSETS_DIR / output_name,
        )

    _build_docx()
    _compact_assignment_docx(REPORT_DOCX)
    print(f"Built assignment report: {REPORT_DOCX}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}: {exc.cmd}", file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
