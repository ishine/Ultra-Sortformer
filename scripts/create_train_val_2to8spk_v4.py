#!/usr/bin/env python3
"""
train_2to8spk_v4.json, val_2to8spk_v4.json 생성

요구사항:
- 합성: spk 2~5 각 50개, spk 6~7 각 100개, spk 8 은 800개 (ov0.05 / ov0.15 각 절반)
- 합성은 training_* 폴더에서만 (validation_* 는 test용으로 사용 안 함)
- 기존 new / new2 / v3 등 이미 쓴 manifest 항목은 제외
- 리얼: alimeeting, ami_ihm, ami_sdm 각 200개 (train)
- val 리얼: 소스당 일정 개수, 후보가 부족하면 중복 허용

사용법:
  python scripts/create_train_val_2to8spk_v4.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
SYNTH_DIR = WORKSPACE / "synthetic_data"

# 여러 작업 루트 (데이터가 /mnt/data/workspace 에만 있을 수 있음)
WORKSPACE_ROOTS = [WORKSPACE, Path("/mnt/data/workspace")]


def _resolve_path(rel: str) -> Path:
    for root in WORKSPACE_ROOTS:
        p = root / rel
        if p.exists():
            return p
    return WORKSPACE / rel


# 제외: 이전 학습/검증에 쓴 manifest의 모든 항목
EXCLUDE_MANIFESTS = [
    SYNTH_DIR / "train_2to8spk_new.json",
    SYNTH_DIR / "train_2to8spk_new2.json",
    SYNTH_DIR / "train_2to8spk_new2_fixed.json",
    SYNTH_DIR / "val_2to8spk_new.json",
    SYNTH_DIR / "val_2to8spk_new2.json",
    SYNTH_DIR / "val_2to8spk_new2_fixed.json",
    SYNTH_DIR / "train_2to8spk_v3.json",
    SYNTH_DIR / "val_2to8spk_v3.json",
]

# 합성 (train): (spk, ov) -> 개수
SYNTH_COUNTS_TRAIN = {
    (2, "ov0.05"): 25,
    (2, "ov0.15"): 25,
    (3, "ov0.05"): 25,
    (3, "ov0.15"): 25,
    (4, "ov0.05"): 25,
    (4, "ov0.15"): 25,
    (5, "ov0.05"): 25,
    (5, "ov0.15"): 25,
    (6, "ov0.05"): 50,
    (6, "ov0.15"): 50,
    (7, "ov0.05"): 50,
    (7, "ov0.15"): 50,
    (8, "ov0.05"): 400,
    (8, "ov0.15"): 400,
}

# val 합성: train 대비 약 1/5 (v3와 동일 비율)
VAL_SYNTH_FRACTION = 5

# 리얼 train: 소스당 개수
REAL_PER_SOURCE_TRAIN = 200

# 리얼 val: 소스당 개수 (부족 시 중복)
REAL_PER_SOURCE_VAL = 50

REAL_SOURCE_RELS = [
    ("data/alimeeting_prepared/train/manifest.json", "alimeeting"),
    ("data/ami_prepared/ihm/train/manifest.json", "ami_ihm"),
    ("data/ami_prepared/sdm/train/manifest.json", "ami_sdm"),
]


def norm_key(entry: dict) -> str:
    ap = entry.get("audio_filepath", "")
    offset = entry.get("offset", 0)
    for prefix in ["/mnt/data/workspace/", "/home/devsy/workspace/"]:
        if ap.startswith(prefix):
            ap = ap[len(prefix) :]
            break
    return f"{ap}|{offset}"


def load_manifest(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def load_excluded_keys() -> set[str]:
    keys: set[str] = set()
    for p in EXCLUDE_MANIFESTS:
        if not p.exists():
            print(f"[WARN] Exclude manifest not found: {p}", file=sys.stderr)
            continue
        for e in load_manifest(p):
            keys.add(norm_key(e))
    return keys


def _entries_from_folder(folder: Path, spk: int) -> list[dict]:
    entries = []
    for wav in sorted(folder.glob("*.wav")):
        rttm = folder / f"{wav.stem}.rttm"
        if not rttm.exists():
            continue
        ap = str(wav.resolve())
        rp = str(rttm.resolve())
        entries.append(
            {
                "audio_filepath": ap,
                "offset": 0.0,
                "duration": 180.0,
                "label": "infer",
                "text": "-",
                "num_speakers": spk,
                "rttm_filepath": rp,
                "uem_filepath": "",
                "ctm_filepath": "",
            }
        )
    return entries


def load_synthetic_pool() -> dict[tuple[int, str], list[dict]]:
    pool: dict[tuple[int, str], list[dict]] = {}
    for spk in range(2, 9):
        for ov in ["ov0.05", "ov0.15"]:
            folder = SYNTH_DIR / f"training_1000sess_{spk}spk_180s_sil0.1_{ov}"
            manifest = folder / "manifest.json"
            if manifest.exists():
                entries = load_manifest(manifest)
            else:
                entries = _entries_from_folder(folder, spk)
                if not entries:
                    print(f"[WARN] No entries from {folder}", file=sys.stderr)
            for e in entries:
                if e.get("duration") == -1 or e.get("duration") is None:
                    e["duration"] = 180.0
                if "offset" not in e or e.get("offset") is None:
                    e["offset"] = 0.0
            pool[(spk, ov)] = entries
    return pool


def _sample_with_optional_replace(candidates: list[dict], n: int) -> list[dict]:
    """고유 샘플이 n개 미만이면 replacement로 채움."""
    if not candidates or n <= 0:
        return []
    if len(candidates) >= n:
        return random.sample(candidates, n)
    print(
        f"[WARN] synthetic: need {n}, unique pool {len(candidates)} — "
        f"filling {n - len(candidates)} with replacement",
        file=sys.stderr,
    )
    return random.choices(candidates, k=n)


def sample_train_val_synthetic(
    pool: dict[tuple[int, str], list[dict]],
    excluded: set[str],
    train_counts: dict[tuple[int, str], int],
    val_counts: dict[tuple[int, str], int],
) -> tuple[list[dict], list[dict]]:
    """train/val 합성: 가능하면 서로 겹치지 않게 뽑고, 부족하면 replacement."""
    train_out: list[dict] = []
    val_out: list[dict] = []
    for (spk, ov), n_t in train_counts.items():
        n_v = val_counts.get((spk, ov), 0)
        candidates = [e for e in pool[(spk, ov)] if norm_key(e) not in excluded]
        if not candidates:
            print(f"[WARN] {(spk, ov)}: empty pool after exclude", file=sys.stderr)
            continue
        total = n_t + n_v
        if len(candidates) >= total:
            picked = random.sample(candidates, total)
            train_out.extend(picked[:n_t])
            val_out.extend(picked[n_t:])
            continue
        # 고유 개수 부족: train 우선, val은 train과 키 겹침 최소화 후 replacement
        train_pick = _sample_with_optional_replace(candidates, n_t)
        tk = {norm_key(e) for e in train_pick}
        val_cand = [e for e in candidates if norm_key(e) not in tk]
        if len(val_cand) >= n_v:
            val_pick = random.sample(val_cand, n_v)
        elif val_cand:
            val_pick = _sample_with_optional_replace(val_cand, n_v)
        else:
            val_pick = _sample_with_optional_replace(candidates, n_v)
        train_out.extend(train_pick)
        val_out.extend(val_pick)
    return train_out, val_out


def sample_real_train(
    sources: list[tuple[Path, str]],
    excluded: set[str],
    per_source: int,
) -> list[dict]:
    out: list[dict] = []
    for manifest_path, _name in sources:
        if not manifest_path.exists():
            print(f"[WARN] {manifest_path} not found", file=sys.stderr)
            continue
        entries = load_manifest(manifest_path)
        candidates = [e for e in entries if norm_key(e) not in excluded]
        if len(candidates) < per_source:
            print(
                f"[WARN] {manifest_path}: need {per_source} train real, "
                f"only {len(candidates)} after exclude — using all (no dup)",
                file=sys.stderr,
            )
        take = min(per_source, len(candidates))
        out.extend(random.sample(candidates, take))
    return out


def sample_real_val(
    sources: list[tuple[Path, str]],
    excluded: set[str],
    per_source: int,
) -> list[dict]:
    """후보가 per_source보다 적으면 중복 허용."""
    out: list[dict] = []
    for manifest_path, _name in sources:
        if not manifest_path.exists():
            print(f"[WARN] {manifest_path} not found", file=sys.stderr)
            continue
        entries = load_manifest(manifest_path)
        candidates = [e for e in entries if norm_key(e) not in excluded]
        if not candidates:
            print(f"[WARN] {manifest_path}: no candidates for val", file=sys.stderr)
            continue
        if len(candidates) >= per_source:
            out.extend(random.sample(candidates, per_source))
        else:
            print(
                f"[INFO] {manifest_path}: val real {per_source} requested, "
                f"{len(candidates)} unique — sampling with replacement",
                file=sys.stderr,
            )
            out.extend(random.choices(candidates, k=per_source))
    return out


def build_val_synth_counts(train_counts: dict[tuple[int, str], int]) -> dict[tuple[int, str], int]:
    return {k: max(1, v // VAL_SYNTH_FRACTION) for k, v in train_counts.items()}


def resolve_real_sources() -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for rel, name in REAL_SOURCE_RELS:
        p = _resolve_path(rel)
        out.append((p, name))
    return out


def summarize(entries: list[dict]) -> dict[str, int]:
    from collections import defaultdict

    c: defaultdict[str, int] = defaultdict(int)
    for e in entries:
        ap = e.get("audio_filepath", "")
        if "training_1000sess" in ap:
            for spk in range(2, 9):
                if f"{spk}spk" in ap:
                    ov = "ov0.15" if "ov0.15" in ap else "ov0.05"
                    c[f"{spk}spk_{ov}"] += 1
                    break
        elif "alimeeting" in ap:
            c["alimeeting"] += 1
        elif "ami_prepared/ihm" in ap:
            c["ami_ihm"] += 1
        elif "ami_prepared/sdm" in ap:
            c["ami_sdm"] += 1
    return dict(c)


def main() -> None:
    random.seed(42)
    excluded = load_excluded_keys()
    print(f"Excluded keys: {len(excluded)}", file=sys.stderr)

    real_sources = resolve_real_sources()
    for p, name in real_sources:
        print(f"Real source [{name}]: {p} exists={p.exists()}", file=sys.stderr)

    pool = load_synthetic_pool()
    val_synth_counts = build_val_synth_counts(SYNTH_COUNTS_TRAIN)
    train_synth, val_synth = sample_train_val_synthetic(
        pool, excluded, SYNTH_COUNTS_TRAIN, val_synth_counts
    )

    train_real = sample_real_train(real_sources, excluded, REAL_PER_SOURCE_TRAIN)
    train_keys = {norm_key(e) for e in train_synth}
    train_real_keys = {norm_key(e) for e in train_real}

    val_real = sample_real_val(
        real_sources,
        excluded | train_keys | train_real_keys,
        REAL_PER_SOURCE_VAL,
    )

    train_all = train_synth + train_real
    random.shuffle(train_all)
    val_all = val_synth + val_real
    random.shuffle(val_all)

    train_out = SYNTH_DIR / "train_2to8spk_v4.json"
    val_out = SYNTH_DIR / "val_2to8spk_v4.json"

    with open(train_out, "w", encoding="utf-8") as f:
        for e in train_all:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    with open(val_out, "w", encoding="utf-8") as f:
        for e in val_all:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"Written: {train_out} ({len(train_all)} entries)")
    print(f"Written: {val_out} ({len(val_all)} entries)")
    print("\nTrain summary:", summarize(train_all))
    print("Val summary:", summarize(val_all))


if __name__ == "__main__":
    main()
