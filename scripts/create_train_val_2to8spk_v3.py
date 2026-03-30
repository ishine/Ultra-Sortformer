#!/usr/bin/env python3
"""
train_2to8spk_v3.json, val_2to8spk_v3.json 생성

요구사항:
- 합성: spk2~4 각 50개, spk5~6 각 100개, spk7 200개, spk8 500개
- ov0.05 / ov0.15 절반씩
- 합성은 training_* 에서만, 기존 train_2to8spk_new*.json, val_2to8spk_new*.json 사용 데이터 제외
- 리얼: alimeeting, ami_ihm, ami_sdm 각 100개 (기존 사용 데이터 제외)

사용법:
  python scripts/create_train_val_2to8spk_v3.py
"""
import json
import random
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
SYNTH_DIR = WORKSPACE / "synthetic_data"

# 제외할 manifest (train/val 모두)
EXCLUDE_MANIFESTS = [
    SYNTH_DIR / "train_2to8spk_new.json",
    SYNTH_DIR / "train_2to8spk_new2_fixed.json",
    SYNTH_DIR / "val_2to8spk_new.json",
    SYNTH_DIR / "val_2to8spk_new2_fixed.json",
]

# 합성 데이터 구성: (spk, ov) -> 개수
# spk 2~4: 50 each → ov0.05: 25, ov0.15: 25
# spk 5~6: 100 each → ov0.05: 50, ov0.15: 50
# spk 7: 200 → ov0.05: 100, ov0.15: 100
# spk 8: 500 → ov0.05: 250, ov0.15: 250
SYNTH_COUNTS = {
    (2, "ov0.05"): 25, (2, "ov0.15"): 25,
    (3, "ov0.05"): 25, (3, "ov0.15"): 25,
    (4, "ov0.05"): 25, (4, "ov0.15"): 25,
    (5, "ov0.05"): 50, (5, "ov0.15"): 50,
    (6, "ov0.05"): 50, (6, "ov0.15"): 50,
    (7, "ov0.05"): 100, (7, "ov0.15"): 100,
    (8, "ov0.05"): 250, (8, "ov0.15"): 250,
}

# 리얼 데이터 소스 (manifest 경로, 최대 100개)
REAL_SOURCES = [
    (WORKSPACE / "data/alimeeting_prepared/train/manifest.json", "alimeeting"),
    (WORKSPACE / "data/ami_prepared/ihm/train/manifest.json", "ami_ihm"),
    (WORKSPACE / "data/ami_prepared/sdm/train/manifest.json", "ami_sdm"),
]


def norm_key(entry: dict) -> str:
    """중복 제거용 키: audio path + offset (경로 정규화)"""
    ap = entry.get("audio_filepath", "")
    offset = entry.get("offset", 0)
    # /mnt/data/workspace, /home/devsy/workspace 통일
    for prefix in ["/mnt/data/workspace/", "/home/devsy/workspace/"]:
        if ap.startswith(prefix):
            ap = ap[len(prefix):]
            break
    return f"{ap}|{offset}"


def load_manifest(path: Path) -> list[dict]:
    out = []
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
    keys = set()
    for p in EXCLUDE_MANIFESTS:
        if not p.exists():
            print(f"[WARN] Exclude manifest not found: {p}", file=sys.stderr)
            continue
        for e in load_manifest(p):
            keys.add(norm_key(e))
    return keys


def _entries_from_folder(folder: Path, spk: int) -> list[dict]:
    """manifest 없을 때 폴더에서 wav/rttm 스캔하여 entry 리스트 생성"""
    entries = []
    for wav in sorted(folder.glob("*.wav")):
        rttm = folder / f"{wav.stem}.rttm"
        if not rttm.exists():
            continue
        ap = str(wav.resolve())
        rp = str(rttm.resolve())
        entries.append({
            "audio_filepath": ap,
            "offset": 0.0,
            "duration": 180.0,
            "label": "infer",
            "text": "-",
            "num_speakers": spk,
            "rttm_filepath": rp,
            "uem_filepath": "",
            "ctm_filepath": "",
        })
    return entries


def load_synthetic_pool() -> dict[tuple[int, str], list[dict]]:
    """(spk, ov) -> entries from training manifest (또는 폴더 스캔)"""
    pool = {}
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
            # duration=180으로 통일 (기존 -1 방지)
            for e in entries:
                if e.get("duration") == -1 or e.get("duration") is None:
                    e["duration"] = 180.0
                if "offset" not in e or e.get("offset") is None:
                    e["offset"] = 0.0
            pool[(spk, ov)] = entries
    return pool


def sample_synthetic(pool: dict, excluded: set[str], counts: dict) -> list[dict]:
    out = []
    for (spk, ov), n in counts.items():
        candidates = [e for e in pool[(spk, ov)] if norm_key(e) not in excluded]
        if len(candidates) < n:
            print(f"[WARN] {(spk, ov)}: need {n}, available {len(candidates)}", file=sys.stderr)
        chosen = random.sample(candidates, min(n, len(candidates)))
        out.extend(chosen)
    return out


def sample_real(sources: list, excluded: set[str], per_source: int) -> list[dict]:
    out = []
    for manifest_path, name in sources:
        if not manifest_path.exists():
            print(f"[WARN] {manifest_path} not found", file=sys.stderr)
            continue
        entries = load_manifest(manifest_path)
        candidates = [e for e in entries if norm_key(e) not in excluded]
        chosen = random.sample(candidates, min(per_source, len(candidates)))
        out.extend(chosen)
    return out


def main():
    random.seed(42)
    excluded = load_excluded_keys()
    print(f"Excluded keys: {len(excluded)}", file=sys.stderr)

    # 합성
    pool = load_synthetic_pool()
    train_synth = sample_synthetic(pool, excluded, SYNTH_COUNTS)
    # val용 합성: train에서 사용한 것 제외 후 별도 샘플 (동일 비율, 적은 수)
    val_synth_counts = {k: max(1, v // 5) for k, v in SYNTH_COUNTS.items()}  # 약 20%
    train_keys = {norm_key(e) for e in train_synth}
    val_synth = sample_synthetic(pool, excluded | train_keys, val_synth_counts)

    # 리얼
    train_real = sample_real(REAL_SOURCES, excluded, 100)
    train_real_keys = {norm_key(e) for e in train_real}
    val_real = sample_real(REAL_SOURCES, excluded | train_keys | train_real_keys, 30)  # val용 30개씩

    # 셔플 후 저장
    train_all = train_synth + train_real
    random.shuffle(train_all)

    val_all = val_synth + val_real
    random.shuffle(val_all)

    train_out = SYNTH_DIR / "train_2to8spk_v3.json"
    val_out = SYNTH_DIR / "val_2to8spk_v3.json"

    with open(train_out, "w", encoding="utf-8") as f:
        for e in train_all:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    with open(val_out, "w", encoding="utf-8") as f:
        for e in val_all:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"Written: {train_out} ({len(train_all)} entries)")
    print(f"Written: {val_out} ({len(val_all)} entries)")

    # 요약
    from collections import defaultdict
    def summarize(entries):
        c = defaultdict(int)
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

    print("\nTrain summary:", summarize(train_all))
    print("Val summary:", summarize(val_all))


if __name__ == "__main__":
    main()
