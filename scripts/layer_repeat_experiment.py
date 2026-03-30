#!/usr/bin/env python3
"""
Sortformer 레이어 복제 실험 (LLM Neuroanatomy 스타일)

Transformer 또는 Conformer 인코더의 특정 레이어 블록 [start, end]를
두 번 통과시키고 DER 변화를 측정합니다. 학습 없이 가중치 수정 없음.

사용법:
    # Transformer 인코더 (기본)
    python scripts/layer_repeat_experiment.py \\
        --model-path /path/to/model.nemo \\
        --manifest data/alimeeting_prepared/test/manifest.json \\
        --dataset-name alimeeting \\
        --encoder transformer \\
        --output results/layer_repeat_transformer.json

    # Conformer 인코더
    python scripts/layer_repeat_experiment.py \\
        --model-path /path/to/model.nemo \\
        --manifest data/alimeeting_prepared/test/manifest.json \\
        --dataset-name alimeeting \\
        --encoder conformer \\
        --output results/layer_repeat_conformer.json
"""
import json
import os
import sys
import tempfile
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
from nemo.collections.common.parts import form_attention_mask


def label_line_to_rttm_line(line: str, file_id: str) -> str:
    line = line.strip()
    if not line:
        return ""
    parts = line.split()
    if len(parts) < 3:
        return ""
    start, end, speaker = float(parts[0]), float(parts[1]), parts[2]
    duration = end - start
    return f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>"


def load_manifest(manifest_path: str) -> tuple[list[str], list[str]]:
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
            rttm_paths.append(
                entry.get("rttm_filepath") or entry["audio_filepath"].rsplit(".", 1)[0] + ".rttm"
            )
    return paths, rttm_paths


def patch_conformer_repeat_block(encoder, start_layer: int, end_layer: int):
    """
    ConformerEncoder의 layers를 임시로 교체하여 [start_layer, end_layer] 블록을
    두 번 통과시킵니다.

    반환값: restore 함수 (finally 블록에서 반드시 호출)
    """
    import torch.nn as nn

    original_layers = encoder.layers
    original_reduction_pos = getattr(encoder, "reduction_position", None)
    block_size = end_layer - start_layer + 1

    # 레이어 참조를 반복 삽입 (가중치 복사 없음 - 같은 파라미터 공유)
    new_layers = (
        list(original_layers[: end_layer + 1])           # 0 ~ end
        + list(original_layers[start_layer : end_layer + 1])  # start ~ end (반복)
        + list(original_layers[end_layer + 1 :])          # end+1 ~ N-1
    )
    encoder.layers = nn.ModuleList(new_layers)

    # reduction_position이 반복 블록 뒤에 있으면 인덱스 조정
    if original_reduction_pos is not None and original_reduction_pos > end_layer:
        encoder.reduction_position = original_reduction_pos + block_size

    def restore():
        encoder.layers = original_layers
        encoder.reduction_position = original_reduction_pos

    return restore


def make_forward_with_repeat_block(transformer_encoder, start_layer: int, end_layer: int):
    """TransformerEncoder.forward를 [start_layer, end_layer] 블록을 두 번 통과하도록 래핑."""

    def _forward(encoder_states, encoder_mask, encoder_mems_list=None, return_mems=False):
        encoder_attn_mask = form_attention_mask(encoder_mask, transformer_encoder.diag)
        memory_states = (
            torch.cat((encoder_mems_list[0], encoder_states), dim=1)
            if encoder_mems_list is not None
            else encoder_states
        )
        num_layers = len(transformer_encoder.layers)

        def run_layers(lo: int, hi: int):
            nonlocal encoder_states, memory_states
            for i in range(lo, hi):
                layer = transformer_encoder.layers[i]
                encoder_states = layer(encoder_states, encoder_attn_mask, memory_states)
                memory_states = (
                    torch.cat((encoder_mems_list[i + 1], encoder_states), dim=1)
                    if encoder_mems_list is not None
                    else encoder_states
                )

        # 0 .. start-1
        run_layers(0, start_layer)
        # start .. end (첫 번째 통과)
        run_layers(start_layer, end_layer + 1)
        # start .. end (두 번째 통과 = 복제)
        run_layers(start_layer, end_layer + 1)
        # end+1 .. last
        run_layers(end_layer + 1, num_layers)

        if transformer_encoder.final_layer_norm is not None:
            encoder_states = transformer_encoder.final_layer_norm(encoder_states)

        return encoder_states

    return _forward


def run_baseline_eval(diar_model, manifest_path: str, dataset_name: str, collar: float, ignore_overlap: bool) -> dict | None:
    """원본 모델(패치 없음)로 평가."""
    audio_paths, _ = load_manifest(manifest_path)
    if not audio_paths:
        return None
    predicted_segments = diar_model.diarize(
        audio=audio_paths,
        batch_size=1,
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
        return None
    result = score_labels(
        AUDIO_RTTM_MAP,
        all_reference,
        all_hypothesis,
        collar=collar,
        ignore_overlap=ignore_overlap,
        verbose=False,
    )
    if result is None:
        return None
    _, _, (DER, CER, FA, MISS) = result
    spk_count_acc = sum(
        1 for (_, ref_ann), (_, hyp_ann) in zip(all_reference, all_hypothesis)
        if len(ref_ann.labels()) == len(hyp_ann.labels())
    ) / len(all_reference)
    return {
        "start_layer": -1,
        "end_layer": -1,
        "block_size": 0,
        "DER": DER,
        "CER": CER,
        "FA": FA,
        "MISS": MISS,
        "Spk_Count_Acc": spk_count_acc,
    }


def run_eval_with_repeat_block(
    diar_model,
    manifest_path: str,
    start_layer: int,
    end_layer: int,
    encoder_type: str = "transformer",
    batch_size: int = 1,
    collar: float = 0.25,
    ignore_overlap: bool = False,
) -> dict | None:
    """레이어 블록 [start, end] 복제 모드로 평가. DER, CER, FA, MISS, Spk_Count_Acc 반환."""
    audio_paths, _ = load_manifest(manifest_path)
    if not audio_paths:
        return None

    if encoder_type == "conformer":
        restore_fn = patch_conformer_repeat_block(diar_model.encoder, start_layer, end_layer)
        try:
            predicted_segments = diar_model.diarize(
                audio=audio_paths,
                batch_size=batch_size,
                verbose=False,
                postprocessing_yaml=None,
            )
        finally:
            restore_fn()
    else:
        te = diar_model.transformer_encoder
        original_forward = te.forward
        try:
            te.forward = make_forward_with_repeat_block(te, start_layer, end_layer)
            predicted_segments = diar_model.diarize(
                audio=audio_paths,
                batch_size=batch_size,
                verbose=False,
                postprocessing_yaml=None,
            )
        finally:
            te.forward = original_forward

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
        return None

    result = score_labels(
        AUDIO_RTTM_MAP,
        all_reference,
        all_hypothesis,
        collar=collar,
        ignore_overlap=ignore_overlap,
        verbose=False,
    )
    if result is None:
        return None

    _, _, (DER, CER, FA, MISS) = result
    spk_count_acc = sum(
        1 for (_, ref_ann), (_, hyp_ann) in zip(all_reference, all_hypothesis)
        if len(ref_ann.labels()) == len(hyp_ann.labels())
    ) / len(all_reference)

    return {
        "start_layer": start_layer,
        "end_layer": end_layer,
        "block_size": end_layer - start_layer + 1,
        "DER": DER,
        "CER": CER,
        "FA": FA,
        "MISS": MISS,
        "Spk_Count_Acc": spk_count_acc,
    }


def main():
    import argparse

    workspace = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description="Sortformer layer repeat experiment")
    parser.add_argument("--model-path", type=str, required=True, help=".nemo 모델 경로")
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(workspace / "synthetic_data/validation_100sess_5spk_90s_sil0.1_ov0/manifest.json"),
        help="평가 manifest",
    )
    parser.add_argument("--dataset-name", type=str, default="val_5spk", help="데이터셋명")
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="고정 블록 크기 (지정 시 해당 크기만 시도)",
    )
    parser.add_argument("--block-size-min", type=int, default=2, help="블록 크기 최솟값 (--block-size 미지정 시)")
    parser.add_argument("--block-size-max", type=int, default=7, help="블록 크기 최댓값 (--block-size 미지정 시)")
    parser.add_argument(
        "--output",
        type=str,
        default=str(workspace / "results/layer_repeat_experiment.json"),
        help="결과 JSON 저장 경로",
    )
    parser.add_argument("--collar", type=float, default=0.25)
    parser.add_argument("--ignore-overlap", action="store_true")
    parser.add_argument(
        "--encoder",
        choices=["transformer", "conformer"],
        default="transformer",
        help="반복 실험 대상 인코더 (기본: transformer)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = workspace / manifest_path
    if not manifest_path.exists():
        print(f"Error: manifest not found: {manifest_path}")
        return 1

    print(f"Loading model: {args.model_path}")
    diar_model = SortformerEncLabelModel.restore_from(
        restore_path=args.model_path,
        map_location="cuda" if torch.cuda.is_available() else "cpu",
    )
    diar_model.eval()
    if torch.cuda.is_available():
        diar_model = diar_model.cuda()

    # streaming params
    diar_model.sortformer_modules.chunk_len = 340
    diar_model.sortformer_modules.chunk_right_context = 40
    diar_model.sortformer_modules.fifo_len = 40
    diar_model.sortformer_modules.spkcache_update_period = 300

    num_transformer_layers = len(diar_model.transformer_encoder.layers)
    num_conformer_layers = len(diar_model.encoder.layers)
    print(f"Transformer layers: {num_transformer_layers}")
    print(f"Conformer layers:   {num_conformer_layers}")

    if args.encoder == "transformer":
        num_layers = num_transformer_layers
        print(f"실험 대상: Transformer encoder ({num_layers}층)")
    else:
        num_layers = num_conformer_layers
        reduction_pos = getattr(diar_model.encoder, "reduction_position", None)
        print(f"실험 대상: Conformer encoder ({num_layers}층), reduction_position={reduction_pos}")

    # 블록 조합 생성
    if args.block_size is not None:
        block_sizes = [args.block_size]
    else:
        block_sizes = list(range(args.block_size_min, args.block_size_max + 1))

    configs = []
    for bs in block_sizes:
        for start in range(0, num_layers - bs + 1):
            configs.append((start, start + bs - 1))

    # baseline (복제 없음)
    print("Running baseline (no repeat)...")
    baseline = run_baseline_eval(
        diar_model,
        str(manifest_path),
        args.dataset_name,
        collar=args.collar,
        ignore_overlap=args.ignore_overlap,
    )
    if baseline is None:
        print("Error: baseline eval failed")
        return 1
    results = [baseline]
    print(f"Baseline DER: {baseline['DER']:.4f}")

    for i, (start, end) in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] Repeat layers [{start}, {end}] (size={end-start+1})...")
        row = run_eval_with_repeat_block(
            diar_model,
            str(manifest_path),
            start_layer=start,
            end_layer=end,
            encoder_type=args.encoder,
            collar=args.collar,
            ignore_overlap=args.ignore_overlap,
        )
        if row:
            results.append(row)
            delta = row["DER"] - baseline["DER"]
            print(f"    DER: {row['DER']:.4f} (Δ {delta:+.4f})")

    # 저장
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model_path,
                "manifest": str(manifest_path),
                "dataset": args.dataset_name,
                "encoder": args.encoder,
                "num_layers": num_layers,
                "baseline": baseline,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Results saved to {out_path}")

    # 최적 구간 출력
    repeat_results = [r for r in results if r["start_layer"] >= 0]
    if repeat_results:
        best = min(repeat_results, key=lambda x: x["DER"])
        print(f"\nBest repeat block: [{best['start_layer']}, {best['end_layer']}] DER={best['DER']:.4f}")
        if best["DER"] < baseline["DER"]:
            print(f"  → Improvement: {baseline['DER'] - best['DER']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
