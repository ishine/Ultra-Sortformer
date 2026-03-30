#!/usr/bin/env python3
"""
Sortformer 8spk 모델 평가 (DER 계산).
결과를 단일 JSON + MD로 저장 (모델명으로 구분).

사용법:
    # 전체 데이터셋 평가 → results/eval_results.json, .md에 추가/갱신
    python scripts/eval.py --model-path /path/to/model.nemo
    # Lightning 체크포인트: 확장자 .ckpt면 자동으로 load_from_checkpoint 사용
    python scripts/eval.py --model-path /path/to/last.ckpt

    # 단일 manifest만 평가 (기존 결과에 해당 데이터셋만 갱신/추가)
    python scripts/eval.py \\
        --manifest /path/to/manifest.json \\
        --dataset-name kdomainconf_test30 \\
        --model-path /path/to/model.nemo
"""
import json
import os
import re
import sys
import tempfile
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "NeMo"))

import torch
from nemo.collections.asr.metrics.der import score_labels
from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.parts.utils.speaker_utils import (
    audio_rttm_map,
    get_uniqname_from_filepath,
    labels_to_pyannote_object,
    rttm_to_labels,
)

# 전체 평가 대상 데이터셋 (manifest 경로, 출력 서브디렉터리명). 순서: val 2~8, alimeeting, ami, callhome, k시리즈
DEFAULT_DATASETS = [
    ("synthetic_data/validation_100sess_2spk_90s_sil0.1_ov0/manifest.json", "val_2spk"),
    ("synthetic_data/validation_100sess_3spk_90s_sil0.1_ov0/manifest.json", "val_3spk"),
    ("synthetic_data/validation_100sess_4spk_90s_sil0.1_ov0/manifest.json", "val_4spk"),
    ("synthetic_data/validation_100sess_5spk_90s_sil0.1_ov0/manifest.json", "val_5spk"),
    ("synthetic_data/validation_100sess_6spk_90s_sil0.1_ov0/manifest.json", "val_6spk"),
    ("synthetic_data/validation_100sess_7spk_90s_sil0.1_ov0/manifest.json", "val_7spk"),
    ("synthetic_data/validation_100sess_8spk_90s_sil0.1_ov0/manifest.json", "val_8spk"),
    ("data/alimeeting_prepared/test/manifest.json", "alimeeting"),
    ("data/ami_prepared/ihm/test/manifest.json", "ami_ihm_test"),
    ("data/ami_prepared/sdm/test/manifest.json", "ami_sdm_test"),
    ("data/callhome_prepared/eng/manifest.json", "callhome_eng"),
    ("data/callhome_prepared/deu/manifest.json", "callhome_deu"),
    ("data/callhome_prepared/jpn/manifest.json", "callhome_jpn"),
    ("data/callhome_prepared/spa/manifest.json", "callhome_spa"),
    ("data/callhome_prepared/zho/manifest.json", "callhome_zho"),
    ("data/kdomainconf_speech_5speakers/manifest_test_30.json", "kdomainconf_test30"),
    ("data/kdomainconf_speech_validation_3_4speakers/manifest_test_30.json", "kdomainconf_val_3_4spk_test30"),
    ("data/kaddress_speech/manifest_test_30.json", "kaddress_test30"),
    ("data/kemergency_speech/manifest_test_30.json", "kemergency_test30"),
]


def summarize_manifest_by_source(manifest_path: str) -> dict[str, int]:
    """manifest에서 데이터 소스별 개수 반환."""
    from collections import Counter
    path = Path(manifest_path)
    if not path.exists():
        return {}
    by_source = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = e.get("audio_filepath", "")
            if "ami_prepared" in p:
                by_source["AMI"] += 1
            elif "training_" in p or "synthetic" in p.lower():
                for spk in ["2spk", "3spk", "4spk", "5spk", "6spk", "7spk", "8spk"]:
                    if spk in p:
                        by_source[f"합성_{spk}"] += 1
                        break
                else:
                    by_source["합성"] += 1
            else:
                by_source["기타"] += 1
    return dict(by_source)


def format_training_info(train_path: str | None, val_path: str | None) -> dict:
    """train/val manifest 경로로 학습 데이터 설명 dict 생성 (표 출력용)."""
    train_counts = summarize_manifest_by_source(train_path) if train_path else {}
    val_counts = summarize_manifest_by_source(val_path) if val_path else {}
    if not train_counts and not val_counts:
        return {"train_manifest": train_path or "", "val_manifest": val_path or ""}
    return {
        "train_manifest": Path(train_path).name if train_path else "",
        "val_manifest": Path(val_path).name if val_path else "",
        "train": train_counts,
        "val": val_counts,
    }


def load_manifest(manifest_path: str) -> tuple[list[str], list[str]]:
    """JSONL manifest에서 audio_filepath, rttm_filepath 리스트 반환."""
    paths, rttm_paths = [], []
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return [], []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            paths.append(entry["audio_filepath"])
            rttm_paths.append(entry.get("rttm_filepath") or entry["audio_filepath"].rsplit(".", 1)[0] + ".rttm")
    return paths, rttm_paths


def label_line_to_rttm_line(line: str, file_id: str) -> str:
    """'start end speaker_id' 형식을 표준 RTTM 한 줄로 변환."""
    line = line.strip()
    if not line:
        return ""
    parts = line.split()
    if len(parts) < 3:
        return ""
    start, end, speaker = float(parts[0]), float(parts[1]), parts[2]
    duration = end - start
    return f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>"


def run_eval(
    diar_model,
    manifest_path: str,
    dataset_name: str,
    batch_size: int = 1,
    collar: float = 0.25,
    ignore_overlap: bool = False,
) -> tuple[str, dict | None]:
    """단일 manifest에 대해 인퍼런스 및 DER 계산. (텍스트, 구조화 결과) 반환."""
    audio_paths, _ = load_manifest(manifest_path)
    if not audio_paths:
        return f"[SKIP] {dataset_name}: no entries or file not found\n", None

    predicted_segments = diar_model.diarize(
        audio=audio_paths,
        batch_size=batch_size,
        verbose=False,
        postprocessing_yaml=None,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for audio_path, segments in zip(audio_paths, predicted_segments):
            uniq_id = get_uniqname_from_filepath(audio_path)
            out_path = tmpdir / f"{uniq_id}.rttm"
            with open(out_path, "w", encoding="utf-8") as f:
                for line in segments:
                    line = line.strip()
                    if not line:
                        continue
                    rttm_line = label_line_to_rttm_line(line, uniq_id)
                    if rttm_line:
                        f.write(rttm_line + "\n")

        AUDIO_RTTM_MAP = audio_rttm_map(manifest_path)
        all_reference, all_hypothesis = [], []
        for uniq_id, meta in AUDIO_RTTM_MAP.items():
            ref_path = meta.get("rttm_filepath")
            hyp_path = tmpdir / f"{uniq_id}.rttm"
            if ref_path is None or not os.path.exists(ref_path) or not hyp_path.exists():
                continue
            ref_labels = rttm_to_labels(ref_path)
            ref_ann = labels_to_pyannote_object(ref_labels, uniq_name=uniq_id)
            all_reference.append([uniq_id, ref_ann])
            hyp_labels = rttm_to_labels(str(hyp_path))
            hyp_ann = labels_to_pyannote_object(hyp_labels, uniq_name=uniq_id)
            all_hypothesis.append([uniq_id, hyp_ann])

    if len(all_reference) == 0:
        return f"[SKIP] {dataset_name}: no reference RTTM found\n", None

    result = score_labels(
        AUDIO_RTTM_MAP,
        all_reference,
        all_hypothesis,
        collar=collar,
        ignore_overlap=ignore_overlap,
        verbose=False,
    )
    if result is None:
        return f"[SKIP] {dataset_name}: score_labels returned None\n", None

    _, _, (DER, CER, FA, MISS) = result
    spk_count_acc = sum(
        1 for (_, ref_ann), (_, hyp_ann) in zip(all_reference, all_hypothesis)
        if len(ref_ann.labels()) == len(hyp_ann.labels())
    ) / len(all_reference)

    row = {
        "dataset": dataset_name,
        "FA": FA,
        "MISS": MISS,
        "CER": CER,
        "DER": DER,
        "Spk_Count_Acc": spk_count_acc,
    }
    txt = (
        f"Cumulative Results ({dataset_name}) for collar {collar} sec and ignore_overlap {ignore_overlap}:\n"
        f"    | FA: {FA:.4f} | MISS: {MISS:.4f} | CER: {CER:.4f} | DER: {DER:.4f} | Spk. Count Acc. {spk_count_acc:.4f}\n"
    )
    return txt, row


def main():
    import argparse
    workspace = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description="Sortformer 8spk evaluation")
    parser.add_argument("--manifest", type=str, default=None, help="단일 manifest (미지정 시 전체 데이터셋 평가)")
    parser.add_argument("--dataset-name", type=str, default=None,
        help="단일 manifest 시 사용할 데이터셋명 (미지정 시 manifest 경로에서 추출)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(workspace / "results"),
        help="결과 저장 디렉터리. 기본: workspace/results",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="NeMo .nemo 경로, 또는 Lightning .ckpt (확장자로 자동 구분)",
    )
    parser.add_argument("--ckpt-path", type=str, default=None, help="Lightning .ckpt (명시 시 --model-path보다 우선)")
    parser.add_argument("--train-manifest", type=str, default=None,
        help="학습에 사용한 train manifest 경로 (MD에 표시, --val-manifest와 함께 사용 시 소스별 개수 자동 집계)",
    )
    parser.add_argument("--val-manifest", type=str, default=None,
        help="학습에 사용한 val manifest 경로 (--train-manifest와 함께 사용 시 소스별 개수 자동 집계)",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--collar", type=float, default=0.25)
    parser.add_argument("--ignore-overlap", action="store_true")
    parser.add_argument("--format", choices=["md", "json", "all"], default="json",
        help="출력 형식: md, json(기본), all",
    )
    args = parser.parse_args()

    if not args.model_path and not args.ckpt_path:
        parser.error("--model-path 또는 --ckpt-path 필요")

    ckpt_file = args.ckpt_path
    nemo_file = args.model_path
    if ckpt_file is None and nemo_file and str(nemo_file).lower().endswith(".ckpt"):
        ckpt_file = nemo_file
        nemo_file = None

    model_src = ckpt_file or nemo_file
    model_name = Path(model_src).stem  # e.g. sortformer_8spk_2300sess_180s_5to8spk500

    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    map_loc = "cuda" if torch.cuda.is_available() else "cpu"
    if ckpt_file:
        print(f"Loading Lightning checkpoint: {ckpt_file}", file=sys.stderr)
        diar_model = SortformerEncLabelModel.load_from_checkpoint(
            checkpoint_path=ckpt_file,
            map_location=map_loc,
            strict=False,
            weights_only=False,
        )
    else:
        print(f"Loading NeMo model: {nemo_file}", file=sys.stderr)
        diar_model = SortformerEncLabelModel.restore_from(
            restore_path=nemo_file,
            map_location=map_loc,
        )
    diar_model.eval()
    if torch.cuda.is_available():
        diar_model = diar_model.cuda()

    diar_model.sortformer_modules.chunk_len = 340
    diar_model.sortformer_modules.chunk_right_context = 40
    diar_model.sortformer_modules.fifo_len = 40
    diar_model.sortformer_modules.spkcache_update_period = 300

    results_lines = []
    rows = []
    if args.manifest:
        ds_name = args.dataset_name or Path(args.manifest).stem
        datasets = [(args.manifest, ds_name)]
    else:
        datasets = DEFAULT_DATASETS

    for manifest_path, name in datasets:
        manifest_path = Path(manifest_path)
        if not manifest_path.is_absolute():
            manifest_path = Path.cwd() / manifest_path
        print(f"Running: {name} ({manifest_path})", file=sys.stderr)
        txt, row = run_eval(
            diar_model,
            str(manifest_path),
            dataset_name=name,
            batch_size=args.batch_size,
            collar=args.collar,
            ignore_overlap=args.ignore_overlap,
        )
        results_lines.append(txt)
        print(txt, file=sys.stderr)
        if row:
            rows.append(row)

    # 단일 파일: results/eval_results.json, eval_results.md
    results_json_path = out_base / "eval_results.json"
    out_md_path = out_base / "eval_results.md"

    def _load_all_models(json_path: Path) -> dict:
        if not json_path.exists():
            return {}
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _get_model_data(all_data: dict, mn: str) -> tuple[dict, list]:
        m = all_data.get(mn, {})
        return dict(m.get("results", {})), list(m.get("dataset_order", []))

    all_models_data = _load_all_models(results_json_path)
    all_results, dataset_order = _get_model_data(all_models_data, model_name)

    # 기존 모델별 json에서 마이그레이션 (단일 파일 전환 시 일회성)
    legacy_path = out_base / f"{model_name}.json"
    if not all_results and legacy_path.exists():
        try:
            old = json.loads(legacy_path.read_text(encoding="utf-8"))
            all_results, dataset_order = _get_model_data({model_name: old}, model_name)
        except Exception:
            pass

    # 새 결과 반영 (단일: 해당 데이터셋만 갱신/추가, 전체: 교체)
    if args.manifest:
        for r in rows:
            name = r["dataset"]
            all_results[name] = r
            if name not in dataset_order:
                dataset_order.append(name)
    else:
        all_results = {r["dataset"]: r for r in rows}
        dataset_order = [d[1] for d in DEFAULT_DATASETS]

    REAL_DATASETS = {
        "alimeeting", "ami_ihm_test", "ami_sdm_test", "ami_test",
        "callhome_eng", "callhome_deu", "callhome_jpn", "callhome_spa", "callhome_zho",
        "kdomainconf_test30", "kdomainconf_val_3_4spk_test30", "kaddress_test30", "kemergency_test30",
    }

    def _compute_totals(res: dict, keys: list | None = None) -> tuple[float | None, float | None]:
        """results dict에서 total_der, total_spk_count_acc 계산 (데이터셋별 평균)."""
        vals = [res[k] for k in keys if k in res] if keys else list(res.values())
        if not vals:
            return None, None
        ders = [r["DER"] for r in vals]
        spks = [r["Spk_Count_Acc"] for r in vals if r.get("Spk_Count_Acc") is not None]
        total_der = sum(ders) / len(ders) if ders else None
        total_spk = sum(spks) / len(spks) if spks else None
        return total_der, total_spk

    model_entry = {"model": model_src, "dataset_order": dataset_order, "results": all_results}
    t_der, t_spk = _compute_totals(all_results)
    if t_der is not None:
        model_entry["total_der"] = t_der
    if t_spk is not None:
        model_entry["total_spk_count_acc"] = t_spk
    real_keys = [n for n in dataset_order if n in all_results and n in REAL_DATASETS]
    if real_keys:
        r_der, r_spk = _compute_totals(all_results, real_keys)
        if r_der is not None:
            model_entry["total_real_der"] = r_der
        if r_spk is not None:
            model_entry["total_real_spk_count_acc"] = r_spk
    if args.train_manifest or args.val_manifest:
        model_entry["training_info"] = format_training_info(args.train_manifest, args.val_manifest)
    elif all_models_data.get(model_name, {}).get("training_info"):
        model_entry["training_info"] = all_models_data[model_name]["training_info"]
    all_models_data[model_name] = model_entry

    fmt = args.format
    if rows and fmt in ("md", "all"):
        def _build_model_section(mn: str, mdata: dict) -> str:
            """단일 모델의 md 섹션 문자열 생성."""
            dorder = mdata.get("dataset_order", [])
            mresults = mdata.get("results", {})
            mrows = [mresults[n] for n in dorder if n in mresults]
            if not mrows:
                return ""
            sec = [f"## {mn}\n\n", f"Model: `{mdata.get('model', '')}`\n\n"]
            train_info = mdata.get("training_info")
            if train_info:
                if isinstance(train_info, dict) and (train_info.get("train") or train_info.get("val")):
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
                elif isinstance(train_info, dict):
                    t_m = train_info.get("train_manifest", "") or train_info.get("manifest", "")
                    v_m = train_info.get("val_manifest", "")
                    sec.append(f"학습 데이터: train `{t_m}`" + (f", val `{v_m}`" if v_m else "") + "\n\n")
                else:
                    sec.append(f"학습 데이터: {train_info}\n\n")
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
                t_der, t_spk = _compute_totals(mresults)
            if t_der is not None or t_spk is not None:
                if t_der is None:
                    t_der = sum(r["DER"] for r in mrows) / len(mrows)
                if t_spk is None:
                    spk_vals = [r["Spk_Count_Acc"] for r in mrows if r.get("Spk_Count_Acc") is not None]
                    t_spk = sum(spk_vals) / len(spk_vals) if spk_vals else None
                t_spk_str = f"{t_spk*100:.2f}%" if t_spk is not None else "-"
                sec.append(f"| **total** | - | - | - | **{t_der*100:.2f}%** | **{t_spk_str}** |\n")
            real_keys = [n for n in dorder if n in mresults and n in REAL_DATASETS]
            if real_keys:
                r_der, r_spk = _compute_totals(mresults, real_keys)
                if r_der is not None or r_spk is not None:
                    r_spk_str = f"{r_spk*100:.2f}%" if r_spk is not None else "-"
                    sec.append(f"| **total (real)** | - | - | - | **{r_der*100:.2f}%** | **{r_spk_str}** |\n")
            sec.append("\n")
            return "".join(sec)

        HEADER = (
            "# Sortformer Evaluation Results\n\n"
            "## 데이터셋 설명\n\n"
            "| dataset | 설명 | 언어 |\n"
            "|---------|------|------|\n"
            "| val_2spk ~ val_8spk | 합성 검증 데이터 (2~8화자, 90초, silence/overlap 포함) | 한국어 |\n"
            "| alimeeting | AliMeeting 회의 음성 | 중국어 |\n"
            "| ami_ihm_test | AMI 코퍼스 IHM (개별 헤드셋 마이크) test | 영어 |\n"
            "| ami_sdm_test | AMI 코퍼스 SDM (단일 원거리 마이크) test | 영어 |\n"
            "| callhome_eng | CallHome 영어 | 영어 |\n"
            "| callhome_deu | CallHome 독일어 | 독일어 |\n"
            "| callhome_jpn | CallHome 일본어 | 일본어 |\n"
            "| callhome_spa | CallHome 스페인어 | 스페인어 |\n"
            "| callhome_zho | CallHome 중국어 | 중국어 |\n"
            "| kdomainconf_test30 | kdomainconf 5화자 30개 샘플 | 한국어 |\n"
            "| kdomainconf_val_3_4spk_test30 | kdomainconf validation 3~4화자 30개 샘플 | 한국어 |\n"
            "| kaddress_test30 | kaddress 주소 음성 30개 샘플 | 한국어 |\n"
            "| kemergency_test30 | kemergency 긴급 음성 30개 샘플 | 한국어 |\n\n"
        )

        new_section = _build_model_section(model_name, model_entry)

        if out_md_path.exists():
            md_text = out_md_path.read_text(encoding="utf-8")
            # 기존 섹션이 있으면 교체, 없으면 맨 끝에 추가
            section_marker = f"## {model_name}\n"
            if section_marker in md_text:
                # 해당 섹션의 시작~다음 ## 사이를 교체
                import re
                pattern = rf"(## {re.escape(model_name)}\n).*?(?=\n## |\Z)"
                md_text = re.sub(pattern, new_section.rstrip("\n"), md_text, flags=re.DOTALL)
            else:
                md_text = md_text.rstrip("\n") + "\n\n" + new_section
        else:
            md_text = HEADER + new_section

        out_md_path.write_text(md_text, encoding="utf-8")
        print(f"Results (md): {out_md_path}", file=sys.stderr)

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(all_models_data, f, ensure_ascii=False, indent=2)
    print(f"Results (json): {results_json_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
