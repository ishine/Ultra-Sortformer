#!/usr/bin/env python3
"""
4spk Sortformer .nemo를 로드한 뒤, 출력층만 Nspk로 확장한 state_dict를 만들어 Nspk 모델에 넣습니다.
- hidden_to_spks, single_hidden_to_spks: 4spk의 첫 4개 행은 그대로 복사,
  5~N번째 화자는 기존 화자들의 통계에 맞춰 직교(Orthogonal) 초기화 (SVD 활용).
- 나머지 파라미터는 shape 일치 시 그대로 복사.
저장된 Nspk .nemo를 학습 시 init_from_nemo_model로 사용하면 됩니다.
"""
import argparse
import sys
from pathlib import Path

import torch


def extend_state_dict_Nspk_to_Mspk(state_src, state_dst, src_spk: int, dst_spk: int):
    """
    Nspk 모델의 지식을 보존하며 Mspk로 확장합니다.
    (N+1)~M번째 화자는 기존 화자들의 통계치에 맞춰 직교(Orthogonal) 초기화됩니다.
    """
    out_keys = [
        "sortformer_modules.hidden_to_spks.weight",
        "sortformer_modules.hidden_to_spks.bias",
        "sortformer_modules.single_hidden_to_spks.weight",
        "sortformer_modules.single_hidden_to_spks.bias",
    ]
    new_state = {}

    for k, v_dst in state_dst.items():
        if k not in state_src:
            continue
        v_src = state_src[k]

        # 1. 크기가 같으면 (인코더 등) 그대로 복사
        if v_src.shape == v_dst.shape:
            new_state[k] = v_src.clone()
            continue

        # 2. 출력층 확장 (src_spk -> dst_spk)
        if k in out_keys:
            new = v_dst.clone()
            with torch.no_grad():
                # 기존 1~src_spk 화자 가중치 복사
                new[:src_spk] = v_src

                # (src_spk+1)~dst_spk 화자 '직교' 초기화
                if v_src.dim() == 2:  # Weight matrix
                    existing = v_src.float()
                    avg_norm = torch.norm(existing, p=2, dim=1).mean()
                    avg_mean = existing.mean()
                    avg_std = existing.std().clamp(min=1e-6)

                    _, _, Vh = torch.linalg.svd(existing, full_matrices=True)
                    for i in range(src_spk, dst_spk):
                        if Vh.shape[0] <= i:
                            new_vec = existing.mean(dim=0, keepdim=True) + torch.randn_like(existing[:1]) * avg_std
                        else:
                            new_vec = Vh[i : i + 1, :].clone()
                        new_vec = (new_vec - new_vec.mean()) / (new_vec.std() + 1e-6)
                        new_vec = new_vec * avg_std + avg_mean
                        current_norm = torch.norm(new_vec, p=2)
                        if current_norm > 1e-6:
                            new_vec = new_vec * (avg_norm / current_norm)
                        new[i : i + 1] = new_vec.to(v_dst.dtype)

                else:  # Bias vector
                    for i in range(src_spk, dst_spk):
                        new[i] = v_src.mean() + torch.randn(1, device=v_src.device, dtype=v_src.dtype).squeeze() * v_src.std()

            new_state[k] = new

    return new_state


def main():
    parser = argparse.ArgumentParser(description="Extend 4spk Sortformer to Nspk and save .nemo")
    parser.add_argument("--nemo_4spk", required=True, help="Path to 4spk .nemo file")
    parser.add_argument("--out_nspk", required=True, help="Path to save Nspk .nemo file")
    parser.add_argument(
        "--num_spks",
        type=int,
        default=8,
        help="목표 화자 수 (기본 8)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Nspk용 config yaml (없으면 4spk config에서 max_num_of_spks로 오버라이드)",
    )
    args = parser.parse_args()

    src_spk = 4
    dst_spk = args.num_spks
    if dst_spk <= src_spk:
        print(f"num_spks({dst_spk})는 소스 화자 수({src_spk})보다 커야 합니다.")
        return 1

    # NeMo import (workspace root에서 실행 가정)
    workspace = Path(__file__).resolve().parent.parent
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))

    try:
        from nemo.collections.asr.models import SortformerEncLabelModel
        from omegaconf import OmegaConf
    except ImportError as e:
        print("NeMo import 실패. 프로젝트 루트에서 실행하세요:", e)
        return 1

    nemo_4 = Path(args.nemo_4spk)
    if not nemo_4.exists():
        print(f"파일 없음: {nemo_4}")
        return 1

    # 4spk 로드
    print(f"4spk 로드: {nemo_4}")
    model_src = SortformerEncLabelModel.restore_from(str(nemo_4), map_location="cpu")
    state_src = model_src.state_dict()

    # Nspk config
    if args.config and Path(args.config).exists():
        cfg = OmegaConf.load(args.config)
    else:
        config_path = workspace / "NeMo/examples/speaker_tasks/diarization/conf/neural_diarizer/streaming_sortformer_diarizer_4spk-v2.yaml"
        if not config_path.exists():
            print(f"Config 없음: {config_path}")
            return 1
        cfg = OmegaConf.load(config_path)
    cfg.model.max_num_of_spks = dst_spk

    # 모델 생성 시 train_ds/validation_ds.manifest_filepath 필수이므로 더미 설정
    dummy_manifest = workspace / "data/kdomainconf_speech_train_5speaker/train_manifest.json"
    if not dummy_manifest.exists():
        dummy_manifest = workspace / "data/kdomainconf_speech_train_5speaker/manifest.json"
    if not dummy_manifest.exists():
        import tempfile
        td = Path(tempfile.gettempdir()) / "sortformer_extend_dummy"
        td.mkdir(parents=True, exist_ok=True)
        dummy_manifest = td / "dummy_manifest.json"
        with open(dummy_manifest, "w") as f:
            f.write(f'{{"audio_filepath": "/tmp/dummy.wav", "offset": 0, "duration": 10.0, "text": "-", "num_speakers": {dst_spk}, "rttm_filepath": null}}\n')
    OmegaConf.update(cfg.model.train_ds, "manifest_filepath", str(dummy_manifest))
    OmegaConf.update(cfg.model.validation_ds, "manifest_filepath", str(dummy_manifest))

    # Nspk 모델 생성 (trainer 없음)
    print(f"{dst_spk}spk 모델 생성 (max_num_of_spks={dst_spk})")
    model_dst = SortformerEncLabelModel(cfg=cfg.model, trainer=None)
    state_dst = model_dst.state_dict()

    merged = extend_state_dict_Nspk_to_Mspk(state_src, state_dst, src_spk, dst_spk)
    missing, unexpected = model_dst.load_state_dict(merged, strict=False)
    print(f"로드된 파라미터: {len(merged)}개")
    print(f"Missing (랜덤 유지): {len(missing)}개")
    if missing:
        for m in missing[:10]:
            print("  -", m)
        if len(missing) > 10:
            print("  ...")

    out_path = Path(args.out_nspk)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_dst.save_to(str(out_path))
    print(f"{dst_spk}spk .nemo 저장: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
