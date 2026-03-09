#!/usr/bin/env python3
import itertools
import json
import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import open_dict

# Add NeMo to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent / "NeMo"))

from nemo.collections.asr.metrics.der import score_labels
from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.parts.utils.speaker_utils import (
    audio_rttm_map,
    get_uniqname_from_filepath,
    labels_to_pyannote_object,
    rttm_to_labels,
)


def safe_rttm_to_labels(rttm_path: str):
    """RTTM 파일에서 유효한 라인만 읽어 'start end speaker' 리스트 반환. 빈/잘못된 줄은 건너뜀."""
    labels = []
    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # RTTM: SPEAKER file_id 1 start duration ... speaker_id (최소 8칸)
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                try:
                    start = float(parts[3])
                    duration = float(parts[4])
                    end = start + duration
                    speaker = parts[7]
                    labels.append(f"{start} {end} {speaker}")
                except (ValueError, IndexError):
                    continue
    return labels


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


def load_manifest(manifest_path: str):
    """JSONL manifest에서 audio_filepath, rttm_filepath 리스트 반환."""
    paths = []
    rttm_paths = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            paths.append(entry["audio_filepath"])
            rttm_paths.append(entry.get("rttm_filepath") or entry["audio_filepath"].rsplit(".", 1)[0] + ".rttm")
    return paths, rttm_paths


def extend_sortformer_to_n_speakers(model, new_num_spk=5, init_mode="conservative"):
    """
    Sortformer 모델의 마지막 출력층을 4화자에서 new_num_spk화자로 확장합니다.

    init_mode:
      - "conservative" (기본): 추가 출력을 gain=0.01로 작게 초기화.
        → Fine-tuning 전 5번째 채널은 거의 0에 가깝게 나옴. 학습 시 기존 4화자 안정적.
      - "match_scale": 추가 출력을 기존 4화자 가중치 분포와 비슷하게 초기화(평균 복사 + 소량 노이즈).
        → 학습 없이 Inference만 할 때 5번째 화자가 어느 정도 감지될 수 있음. 실험용.
    """
    mod = model.sortformer_modules
    old_n = mod.n_spk
    if old_n >= new_num_spk:
        return

    device = next(mod.parameters()).device
    dtype = next(mod.parameters()).dtype

    def init_new_rows(weight, bias, old_n, new_n):
        with torch.no_grad():
            if init_mode == "match_scale":
                # 기존 화자 가중치 평균을 복사한 뒤 소량 노이즈로 차이 둠 (Inference만 할 때 5번째 채널이 어느 정도 반응하도록)
                num_new = new_n - old_n
                mean_row = weight[:old_n].mean(dim=0, keepdim=True)  # (1, in_features)
                weight[old_n:].copy_(mean_row.expand(num_new, -1))
                weight[old_n:].add_(torch.randn_like(weight[old_n:], device=weight.device, dtype=weight.dtype) * 0.02)
                bias[old_n:].fill_(bias[:old_n].mean().item())
            else:
                # conservative: 작게 시작 (Fine-tuning용, 기존 4화자 예측 유지)
                nn.init.xavier_uniform_(weight[old_n:], gain=0.01)
                bias[old_n:].zero_()

    # 1. hidden_to_spks: (2*hidden_size -> n_spk) 확장
    old_lin = mod.hidden_to_spks
    new_lin = nn.Linear(old_lin.in_features, new_num_spk, device=device, dtype=dtype)
    with torch.no_grad():
        new_lin.weight[:old_n].copy_(old_lin.weight)
        new_lin.bias[:old_n].copy_(old_lin.bias)
    init_new_rows(new_lin.weight, new_lin.bias, old_n, new_num_spk)
    mod.hidden_to_spks = new_lin.to(device)

    # 2. single_hidden_to_spks: (hidden_size -> n_spk) 확장
    old_lin2 = mod.single_hidden_to_spks
    new_lin2 = nn.Linear(old_lin2.in_features, new_num_spk, device=device, dtype=dtype)
    with torch.no_grad():
        new_lin2.weight[:old_n].copy_(old_lin2.weight)
        new_lin2.bias[:old_n].copy_(old_lin2.bias)
    init_new_rows(new_lin2.weight, new_lin2.bias, old_n, new_num_spk)
    mod.single_hidden_to_spks = new_lin2.to(device)

    # 3. 화자 수 속성 갱신
    mod.n_spk = new_num_spk

    # 4. 모델 config 및 speaker_permutations 갱신 (추론 시 일관성)
    if hasattr(model, "_cfg") and model._cfg is not None:
        with open_dict(model._cfg):
            model._cfg.max_num_of_spks = new_num_spk
    model.speaker_permutations = torch.tensor(
        list(itertools.permutations(range(new_num_spk))), device=device
    )

    # 5. spkcache_len 제약: (1 + spkcache_sil_frames_per_spk) * n_spk 이상이어야 함
    min_spkcache = (1 + mod.spkcache_sil_frames_per_spk) * new_num_spk
    if mod.spkcache_len < min_spkcache:
        mod.spkcache_len = min_spkcache


def main():
    # 공식 NeMo e2e_diarize_speech.py와 동일한 파라미터 이름 지원
    # (https://github.com/NVIDIA/NeMo/blob/main/examples/speaker_tasks/diarization/neural_diarizer/e2e_diarize_speech.py)
    parser = argparse.ArgumentParser(
        description="Speaker Diarization Inference (compatible with NeMo e2e_diarize_speech.py params)"
    )
    parser.add_argument("--audio", type=str, default=None, help="Path to a single audio file")
    parser.add_argument(
        "--manifest",
        "--dataset-manifest",
        dest="manifest",
        type=str,
        default=None,
        help="Path to JSONL manifest (audio_filepath per line); alias: --dataset-manifest",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to .nemo model (default: workspace/diar_streaming_sortformer_4spk-v2.1/...)",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Path to PyTorch Lightning .ckpt (if set, will load weights into the restored model and run inference)",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=5,
        help="Number of speaker output heads (default: 5, extend from 4)",
    )
    parser.add_argument(
        "--extend-init-mode",
        type=str,
        choices=("conservative", "match_scale"),
        default="conservative",
        help="5번째 화자층 초기화: conservative=작게 시작(Fine-tuning용), match_scale=기존 화자와 비슷한 스케일(Inference만 할 때 실험용)",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1, use 1 for best accuracy)")
    # 스트리밍 파라미터 (공식 스크립트와 동일 이름)
    parser.add_argument("--spkcache-len", type=int, default=188, help="Speaker cache length in frames (default: 188)")
    parser.add_argument(
        "--spkcache-update-period",
        type=int,
        default=300,
        help="Speaker cache update period in frames (default: 300)",
    )
    parser.add_argument("--fifo-len", type=int, default=40, help="FIFO buffer length in frames (default: 40)")
    parser.add_argument("--chunk-len", type=int, default=340, help="Chunk length in frames (default: 340)")
    parser.add_argument(
        "--chunk-right-context",
        type=int,
        default=40,
        help="Chunk right context in frames (default: 40)",
    )
    parser.add_argument(
        "--chunk-left-context",
        type=int,
        default=1,
        help="Chunk left context in frames (default: 1)",
    )
    # 출력 및 평가
    parser.add_argument(
        "--out-rttm-dir",
        type=str,
        default=None,
        help="Hypothesis RTTM 저장 디렉터리 (지정 시 참조 RTTM을 덮어쓰지 않음; DER 계산 시 필요)",
    )
    parser.add_argument(
        "--no-der",
        action="store_true",
        help="DER(Diarization Error Rate) 계산 생략 (manifest에 rttm_filepath 있고 --out-rttm-dir 있을 때만 DER 계산)",
    )
    parser.add_argument(
        "--postprocessing-yaml",
        type=str,
        default=None,
        help="후처리 파라미터 YAML 경로 (onset, offset, pad_onset, pad_offset, min_duration_on, min_duration_off 등)",
    )
    parser.add_argument(
        "--collar",
        type=float,
        default=0.25,
        help="DER 계산 시 collar(초) (default: 0.25)",
    )
    parser.add_argument(
        "--ignore-overlap",
        action="store_true",
        help="DER 계산 시 겹치는 구간 무시",
    )
    args = parser.parse_args()

    if not args.audio and not args.manifest:
        parser.error("Provide either --audio or --manifest")
    if args.audio and args.manifest:
        parser.error("Provide only one of --audio or --manifest")

    workspace_root = Path(__file__).resolve().parent.parent
    model_path = args.model_path or str(
        workspace_root / "diar_streaming_sortformer_4spk-v2.1/diar_streaming_sortformer_4spk-v2.1.nemo"
    )
    diar_model = SortformerEncLabelModel.restore_from(restore_path=model_path)
    diar_model.eval()
    if torch.cuda.is_available():
        diar_model = diar_model.cuda()
        print("Using GPU for inference (device: %s)" % next(diar_model.parameters()).device, file=sys.stderr)
    else:
        print("CUDA not available, using CPU for inference", file=sys.stderr)

    # 4화자 모델을 N화자 출력으로 확장 (기본 5)
    if args.num_speakers > 4:
        extend_sortformer_to_n_speakers(
            diar_model,
            new_num_spk=args.num_speakers,
            init_mode=args.extend_init_mode,
        )

    # (Optional) Load .ckpt weights into the model
    # Note: ckpt must be compatible with the current model architecture (including num_speakers).
    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        missing, unexpected = diar_model.load_state_dict(state_dict, strict=False)
        print(
            f"Loaded ckpt weights from {args.ckpt_path}. Missing={len(missing)}, Unexpected={len(unexpected)}",
            file=sys.stderr,
        )
        diar_model.eval()
        if torch.cuda.is_available():
            diar_model = diar_model.cuda()

    # 스트리밍 파라미터 (공식 e2e_diarize_speech.py와 동일)
    if diar_model.streaming_mode:
        diar_model.sortformer_modules.spkcache_len = args.spkcache_len
        diar_model.sortformer_modules.spkcache_update_period = args.spkcache_update_period
        diar_model.sortformer_modules.fifo_len = args.fifo_len
        diar_model.sortformer_modules.chunk_len = args.chunk_len
        diar_model.sortformer_modules.chunk_right_context = args.chunk_right_context
        diar_model.sortformer_modules.chunk_left_context = args.chunk_left_context

    if args.manifest:
        # Manifest 기반 일괄 인퍼런스
        audio_paths, rttm_paths = load_manifest(args.manifest)
        if not audio_paths:
            print("No entries in manifest.", file=sys.stderr)
            return
        out_rttm_dir = Path(args.out_rttm_dir) if args.out_rttm_dir else None
        if out_rttm_dir:
            out_rttm_dir.mkdir(parents=True, exist_ok=True)

        print(f"Running diarization on {len(audio_paths)} files from manifest...", file=sys.stderr)
        predicted_segments = diar_model.diarize(
            audio=audio_paths,
            batch_size=args.batch_size,
            verbose=True,
            postprocessing_yaml=args.postprocessing_yaml,
        )

        for i, (audio_path, rttm_path, segments) in enumerate(
            zip(audio_paths, rttm_paths, predicted_segments)
        ):
            if out_rttm_dir:
                uniq_id = get_uniqname_from_filepath(audio_path)
                out_path = out_rttm_dir / f"{uniq_id}.rttm"
            else:
                uniq_id = get_uniqname_from_filepath(audio_path)
                out_path = Path(rttm_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                for line in segments:
                    line = line.strip()
                    if not line:
                        continue
                    rttm_line = label_line_to_rttm_line(line, uniq_id)
                    if rttm_line:
                        f.write(rttm_line + "\n")
            if (i + 1) % 50 == 0 or (i + 1) == len(audio_paths):
                print(f"  Wrote {i + 1}/{len(audio_paths)} RTTM files.", file=sys.stderr)
        print(f"Done. Wrote {len(predicted_segments)} RTTM files.", file=sys.stderr)

        # DER 계산 (--out-rttm-dir 사용 시 참조 RTTM과 비교)
        if not args.no_der and out_rttm_dir:
            AUDIO_RTTM_MAP = audio_rttm_map(args.manifest)
            all_reference, all_hypothesis = [], []
            for uniq_id, meta in AUDIO_RTTM_MAP.items():
                ref_path = meta.get("rttm_filepath")
                hyp_path = out_rttm_dir / f"{uniq_id}.rttm"
                if ref_path is None or not os.path.exists(ref_path) or not hyp_path.exists():
                    continue
                ref_labels = rttm_to_labels(ref_path)
                ref_ann = labels_to_pyannote_object(ref_labels, uniq_name=uniq_id)
                all_reference.append([uniq_id, ref_ann])
                hyp_labels = rttm_to_labels(str(hyp_path))
                hyp_ann = labels_to_pyannote_object(hyp_labels, uniq_name=uniq_id)
                all_hypothesis.append([uniq_id, hyp_ann])
            if len(all_reference) == 0:
                print(
                    "DER skipped: need --out-rttm-dir and manifest with rttm_filepath (reference RTTM).",
                    file=sys.stderr,
                )
            else:
                print(f"Computing DER on {len(all_reference)} files...", file=sys.stderr)
                score_labels(
                    AUDIO_RTTM_MAP,
                    all_reference,
                    all_hypothesis,
                    collar=args.collar,
                    ignore_overlap=args.ignore_overlap,
                    verbose=True,
                )
    else:
        # 단일 파일 인퍼런스 (stdout 출력)
        predicted_segments = diar_model.diarize(
            audio=[args.audio],
            batch_size=1,
            postprocessing_yaml=args.postprocessing_yaml,
        )
        for segment in predicted_segments[0]:
            print(segment)


if __name__ == "__main__":
    main()
