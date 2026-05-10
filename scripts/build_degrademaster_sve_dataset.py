#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append SVE embeddings to DegradeMaster PROTAC features.")
    parser.add_argument("--source-data-root", type=str, default="data/PROTAC")
    parser.add_argument("--output-data-root", type=str, default="data/PROTAC_sve")
    parser.add_argument("--name-json", type=str, default="data/PROTAC/name.json")
    parser.add_argument(
        "--embeddings-npz",
        type=str,
        default="artifacts/degrademaster_protac_embeddings.npz",
        help="NPZ produced from downstream PROTAC mol2 files.",
    )
    parser.add_argument(
        "--protac-feature-path",
        type=str,
        default="data/PROTAC/features/protac_feature.npy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = (PROJECT_ROOT / args.source_data_root).resolve()
    output_root = (PROJECT_ROOT / args.output_data_root).resolve()
    name_json = (PROJECT_ROOT / args.name_json).resolve()
    embeddings_npz = (PROJECT_ROOT / args.embeddings_npz).resolve()
    protac_feature_path = (PROJECT_ROOT / args.protac_feature_path).resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"Source data root not found: {source_root}")
    if not name_json.exists():
        raise FileNotFoundError(f"name.json not found: {name_json}")
    if not embeddings_npz.exists():
        raise FileNotFoundError(f"Embeddings NPZ not found: {embeddings_npz}")
    if not protac_feature_path.exists():
        raise FileNotFoundError(f"protac_feature.npy not found: {protac_feature_path}")

    shutil.copytree(source_root, output_root, dirs_exist_ok=True)

    with open(name_json, "r", encoding="utf-8") as f:
        name_dict = json.load(f)

    arr = np.load(embeddings_npz, allow_pickle=True)
    emb_names = [str(x) for x in arr["names"].tolist()]
    embeddings = np.asarray(arr["embeddings"], dtype=np.float32)
    emb_by_name = {name: embeddings[i] for i, name in enumerate(emb_names)}

    protac_feature = np.load(protac_feature_path, allow_pickle=True).astype(np.float32)
    emb_dim = int(embeddings.shape[1])
    sve_block = np.zeros((protac_feature.shape[0], emb_dim), dtype=np.float32)

    matched = 0
    missing: list[str] = []
    for key, meta in name_dict.items():
        idx = int(key)
        stem = Path(meta["protac_path"]).stem
        if stem in emb_by_name and 0 <= idx < sve_block.shape[0]:
            sve_block[idx] = emb_by_name[stem]
            matched += 1
        else:
            missing.append(stem)

    out_features = np.concatenate((protac_feature, sve_block), axis=1)
    out_feature_path = output_root / "features" / "protac_feature.npy"
    np.save(out_feature_path, out_features)

    meta = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "embeddings_npz": str(embeddings_npz),
        "original_protac_dim": int(protac_feature.shape[1]),
        "embedding_dim": emb_dim,
        "new_protac_dim": int(out_features.shape[1]),
        "matched_rows": matched,
        "total_name_rows": int(len(name_dict)),
        "missing_embedding_names": missing[:50],
    }
    meta_path = output_root / "sve_merge_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
