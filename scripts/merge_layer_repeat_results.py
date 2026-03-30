#!/usr/bin/env python3
"""
두 개의 layer_repeat_experiment 결과를 병합합니다.
각 config의 메트릭을 파일 수 기준 가중 평균으로 합칩니다.

사용법:
    python scripts/merge_layer_repeat_results.py \
        --inputs results/layer_repeat_half0.json results/layer_repeat_half1.json \
        --output results/layer_repeat_merged.json
"""
import argparse
import json
from pathlib import Path


def weighted_merge(r0: dict, n0: int, r1: dict, n1: int) -> dict:
    total = n0 + n1
    metrics = ["DER", "CER", "FA", "MISS", "Spk_Count_Acc"]
    merged = {k: v for k, v in r0.items()}
    for m in metrics:
        if m in r0 and m in r1:
            merged[m] = (r0[m] * n0 + r1[m] * n1) / total
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs=2, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = []
    for p in args.inputs:
        with open(p) as f:
            data.append(json.load(f))

    d0, d1 = data
    n0 = sum(1 for _ in open(d0.get("manifest", "")) if _.strip()) if Path(d0.get("manifest","")).exists() else 406
    n1 = sum(1 for _ in open(d1.get("manifest", "")) if _.strip()) if Path(d1.get("manifest","")).exists() else 406

    # manifest 크기 추정 (파일명에서)
    try:
        n0 = sum(1 for _ in open(d0["manifest"]) if _.strip())
        n1 = sum(1 for _ in open(d1["manifest"]) if _.strip())
    except Exception:
        n0, n1 = 406, 406

    print(f"half0: {n0}개, half1: {n1}개, 합계: {n0+n1}개")

    # baseline 병합
    merged_baseline = weighted_merge(d0["baseline"], n0, d1["baseline"], n1)

    # results 병합 (start_layer, end_layer, block_size 키로 매칭)
    def key(r):
        return (r["start_layer"], r["end_layer"], r["block_size"])

    map0 = {key(r): r for r in d0["results"]}
    map1 = {key(r): r for r in d1["results"]}

    all_keys = set(map0.keys()) | set(map1.keys())
    merged_results = []
    for k in sorted(all_keys):
        if k in map0 and k in map1:
            merged_results.append(weighted_merge(map0[k], n0, map1[k], n1))
        elif k in map0:
            merged_results.append(map0[k])
            print(f"  [WARN] half1에 없음: {k}")
        else:
            merged_results.append(map1[k])
            print(f"  [WARN] half0에 없음: {k}")

    output = {
        "model": d0["model"],
        "manifest": f"{d0['manifest']} + {d1['manifest']}",
        "dataset": d0["dataset"].replace("_half0", "").replace("_half1", ""),
        "encoder": d0["encoder"],
        "num_layers": d0["num_layers"],
        "n_files": n0 + n1,
        "baseline": merged_baseline,
        "results": merged_results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"저장: {args.output} ({len(merged_results)}개 configs)")


if __name__ == "__main__":
    main()
