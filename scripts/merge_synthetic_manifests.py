#!/usr/bin/env python3
"""
Merge NeMo-style JSONL manifests (one JSON object per line).
Deduplicates by (audio_filepath, offset, duration) — first occurrence wins.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple


def _key(obj: dict) -> Tuple:
    return (
        str(obj.get("audio_filepath", "")),
        float(obj.get("offset", 0.0)),
        float(obj.get("duration", -1.0)),
    )


def merge_manifests(
    input_paths: List[Path],
    out_path: Path,
) -> tuple[int, int]:
    seen = set()
    written = 0
    skipped_dup = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        for p in input_paths:
            if not p.exists():
                raise FileNotFoundError(p)
            with open(p, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    k = _key(obj)
                    if k in seen:
                        skipped_dup += 1
                        continue
                    seen.add(k)
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    written += 1
    return written, skipped_dup


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge synthetic_data JSONL manifests")
    parser.add_argument(
        "--synthetic-dir",
        type=Path,
        default=None,
        help="Directory containing manifests (default: <workspace>/synthetic_data or env SYNTHETIC_DATA_DIR)",
    )
    parser.add_argument(
        "--out-train",
        type=Path,
        default=None,
        help="Output train manifest path",
    )
    parser.add_argument(
        "--out-val",
        type=Path,
        default=None,
        help="Output val manifest path",
    )
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent.parent
    syn = args.synthetic_dir
    if syn is None:
        syn = Path(__import__("os").environ.get("SYNTHETIC_DATA_DIR", ""))
        if not syn:
            for cand in (
                workspace / "synthetic_data",
                Path("/mnt/data/workspace/synthetic_data"),
            ):
                if cand.is_dir():
                    syn = cand
                    break
        if not syn or not syn.is_dir():
            raise SystemExit("Could not find synthetic_data; pass --synthetic-dir")

    train_names = [
        "train_2to8spk_new.json",
        "train_2to8spk_new2_fixed.json",
        "train_2to8spk_v3.json",
        "train_2to8spk_v4.json",
    ]
    val_names = [
        "val_2to8spk_new.json",
        "val_2to8spk_new2_fixed.json",
        "val_2to8spk_v3.json",
        "val_2to8spk_v4.json",
    ]

    out_train = args.out_train or (syn / "train_2to8spk_combined_new_new2fixed_v3_v4.json")
    out_val = args.out_val or (syn / "val_2to8spk_combined_new_new2fixed_v3_v4.json")

    train_paths = [syn / n for n in train_names]
    val_paths = [syn / n for n in val_names]

    print(f"synthetic_dir: {syn}")
    print("Train inputs:", *[p.name for p in train_paths], sep="\n  ")
    n_t, d_t = merge_manifests(train_paths, out_train)
    print(f"  -> {out_train}  lines={n_t}  duplicates_skipped={d_t}")

    print("Val inputs:", *[p.name for p in val_paths], sep="\n  ")
    n_v, d_v = merge_manifests(val_paths, out_val)
    print(f"  -> {out_val}  lines={n_v}  duplicates_skipped={d_v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
