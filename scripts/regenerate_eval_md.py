#!/usr/bin/env python3
"""
eval_results.json을 읽어 total_der, total_spk_count_acc를 계산하고 eval_results.md를 재생성.
각 모델 섹션에 total 행 추가.
"""
import json
import sys
from pathlib import Path

HEADER = """# Sortformer Evaluation Results

"""


# 합성(val_2spk~val_8spk) 제외한 리얼 데이터셋
REAL_DATASETS = {
    "alimeeting", "ami_ihm_test", "ami_sdm_test", "ami_test",
    "callhome_eng", "callhome_deu", "callhome_jpn", "callhome_spa", "callhome_zho",
    "kdomainconf_test30", "kdomainconf_val_3_4spk_test30", "kaddress_test30", "kemergency_test30",
}


def compute_totals(results: dict, keys: list | None = None) -> tuple[float | None, float | None]:
    """keys가 있으면 해당 키만, 없으면 전체."""
    if keys is not None:
        vals = [results[k] for k in keys if k in results]
    else:
        vals = list(results.values())
    if not vals:
        return None, None
    ders = [r["DER"] for r in vals]
    spks = [r["Spk_Count_Acc"] for r in vals if r.get("Spk_Count_Acc") is not None]
    total_der = sum(ders) / len(ders) if ders else None
    total_spk = sum(spks) / len(spks) if spks else None
    return total_der, total_spk


def build_model_section(mn: str, mdata: dict) -> str:
    dorder = mdata.get("dataset_order", [])
    mresults = mdata.get("results", {})
    mrows = [mresults[n] for n in dorder if n in mresults]
    if not mrows:
        return ""
    sec = [f"## {mn}\n\n", f"Model: `{mdata.get('model', '')}`\n\n"]
    train_info = mdata.get("training_info")
    if train_info and isinstance(train_info, dict) and (train_info.get("train") or train_info.get("val")):
        sec.append("학습 데이터:\n\n")
        sec.append("| 데이터셋 | Train | Val |\n")
        sec.append("|----------|-------|-----|\n")
        for src in sorted(set(train_info.get("train", {})) | set(train_info.get("val", {}))):
            t = train_info.get("train", {}).get(src, "-")
            v = train_info.get("val", {}).get(src, "-")
            sec.append(f"| {src} | {t} | {v} |\n")
        t_m = train_info.get("train_manifest", "")
        v_m = train_info.get("val_manifest", "")
        if t_m and v_m:
            sec.append(f"\ntrain: `{t_m}` | val: `{v_m}`\n\n")
        elif t_m or v_m:
            sec.append(f"\nmanifest: `{t_m or v_m}`\n\n")
    elif train_info and isinstance(train_info, dict):
        t_m = train_info.get("train_manifest", "") or train_info.get("manifest", "")
        v_m = train_info.get("val_manifest", "")
        if t_m or v_m:
            sec.append(f"학습 데이터: train `{t_m}`" + (f", val `{v_m}`" if v_m else "") + "\n\n")
    sec.append("| dataset | FA | MISS | CER | DER | Spk_Count_Acc |\n")
    sec.append("|---------|-----|------|-----|-----|---------------|\n")
    for r in mrows:
        spk = r.get("Spk_Count_Acc")
        spk_str = f"{spk*100:.2f}%" if spk is not None else "-"
        sec.append(
            f"| {r['dataset']} | {r['FA']*100:.2f}% | {r['MISS']*100:.2f}% "
            f"| {r['CER']*100:.2f}% | {r['DER']*100:.2f}% | {spk_str} |\n"
        )
    t_der = mdata.get("total_der")
    t_spk = mdata.get("total_spk_count_acc")
    if t_der is None or t_spk is None:
        t_der, t_spk = compute_totals(mresults)
    if t_der is not None or t_spk is not None:
        if t_der is None:
            t_der = sum(r["DER"] for r in mrows) / len(mrows)
        t_spk_str = f"{t_spk*100:.2f}%" if t_spk is not None else "-"
        sec.append(f"| **total** | - | - | - | **{t_der*100:.2f}%** | **{t_spk_str}** |\n")
    real_keys = [n for n in dorder if n in mresults and n in REAL_DATASETS]
    if real_keys:
        r_der, r_spk = compute_totals(mresults, real_keys)
        if r_der is not None or r_spk is not None:
            r_spk_str = f"{r_spk*100:.2f}%" if r_spk is not None else "-"
            sec.append(f"| **total (real)** | - | - | - | **{r_der*100:.2f}%** | **{r_spk_str}** |\n")
    sec.append("\n")
    return "".join(sec)


def main():
    workspace = Path(__file__).parent.parent
    results_dir = workspace / "results"
    json_path = results_dir / "eval_results.json"
    md_path = results_dir / "eval_results.md"
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        md_path = Path(sys.argv[2])
    if not json_path.exists():
        print(f"Not found: {json_path}", file=sys.stderr)
        sys.exit(1)
    with open(json_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    for mn, mdata in all_data.items():
        mresults = mdata.get("results", {})
        dorder = mdata.get("dataset_order", [])
        if "total_der" not in mdata or "total_spk_count_acc" not in mdata:
            t_der, t_spk = compute_totals(mresults)
            if t_der is not None:
                mdata["total_der"] = t_der
            if t_spk is not None:
                mdata["total_spk_count_acc"] = t_spk
        real_keys = [n for n in dorder if n in mresults and n in REAL_DATASETS]
        if real_keys:
            r_der, r_spk = compute_totals(mresults, real_keys)
            if r_der is not None:
                mdata["total_real_der"] = r_der
            if r_spk is not None:
                mdata["total_real_spk_count_acc"] = r_spk
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"Updated JSON with totals: {json_path}", file=sys.stderr)

    # 상단 요약: total, total (real)만, 성능순 정렬
    rows = []
    for mn, mdata in all_data.items():
        t_der = mdata.get("total_der")
        t_spk = mdata.get("total_spk_count_acc")
        r_der = mdata.get("total_real_der")
        r_spk = mdata.get("total_real_spk_count_acc")
        if t_der is None and t_spk is None and r_der is None and r_spk is None:
            continue
        rows.append({
            "model": mn,
            "total_der": t_der,
            "total_spk": t_spk,
            "total_real_der": r_der,
            "total_real_spk": r_spk,
        })
    by_der = sorted(rows, key=lambda x: (x["total_real_der"] or 999, x["total_der"] or 999))
    summary = ["## 요약 (total / total real)\n\n"]
    summary.append("**DER 낮을수록 좋음** (total real 기준 정렬)\n\n")
    summary.append("| 순위 | model | total DER | total Spk_Count_Acc | total (real) DER | total (real) Spk_Count_Acc |\n")
    summary.append("|------|-------|-----------|---------------------|------------------|---------------------------|\n")
    for i, r in enumerate(by_der, 1):
        t_der_s = f"{r['total_der']*100:.2f}%" if r["total_der"] is not None else "-"
        t_spk_s = f"{r['total_spk']*100:.2f}%" if r["total_spk"] is not None else "-"
        r_der_s = f"{r['total_real_der']*100:.2f}%" if r["total_real_der"] is not None else "-"
        r_spk_s = f"{r['total_real_spk']*100:.2f}%" if r["total_real_spk"] is not None else "-"
        summary.append(f"| {i} | {r['model']} | {t_der_s} | {t_spk_s} | {r_der_s} | {r_spk_s} |\n")
    summary.append("\n---\n\n")

    dataset_desc = """## 데이터셋 설명

| dataset | 설명 | 언어 |
|---------|------|------|
| val_2spk ~ val_8spk | 합성 검증 데이터 (2~8화자, 90초, silence/overlap 포함) | 한국어 |
| alimeeting | AliMeeting 회의 음성 | 중국어 |
| ami_ihm_test | AMI 코퍼스 IHM (개별 헤드셋 마이크) test | 영어 |
| ami_sdm_test | AMI 코퍼스 SDM (단일 원거리 마이크) test | 영어 |
| callhome_eng | CallHome 영어 | 영어 |
| callhome_deu | CallHome 독일어 | 독일어 |
| callhome_jpn | CallHome 일본어 | 일본어 |
| callhome_spa | CallHome 스페인어 | 스페인어 |
| callhome_zho | CallHome 중국어 | 중국어 |
| kdomainconf_test30 | kdomainconf 5화자 30개 샘플 | 한국어 |
| kdomainconf_val_3_4spk_test30 | kdomainconf validation 3~4화자 30개 샘플 | 한국어 |
| kaddress_test30 | kaddress 주소 음성 30개 샘플 | 한국어 |
| kemergency_test30 | kemergency 긴급 음성 30개 샘플 | 한국어 |

"""
    sections = [build_model_section(mn, mdata) for mn, mdata in all_data.items()]
    md_text = HEADER + "".join(summary) + dataset_desc + "\n".join(s for s in sections if s)
    md_path.write_text(md_text, encoding="utf-8")
    print(f"Written: {md_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
