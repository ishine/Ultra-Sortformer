#!/usr/bin/env python3
"""
출력층 분리 변환 스크립트

기존 n_spk 통합 출력층을 n_base_spks + n_new_spks 분리 구조로 변환합니다.
- single_hidden_to_spks (n_spk)  →  single_hidden_to_spks_base (n_base) + single_hidden_to_spks_new (n_new)
- 기존 가중치는 _base에 복사, _new는 Xavier 초기화

변환 후 모델은 config에 n_base_spks가 기록되므로,
optimizer param_groups에서 regex로 lr를 분리할 수 있습니다.

사용법:
    python scripts/split_output_layer.py \\
        --model-path /path/to/5spk_model.nemo \\
        --n-base-spks 4 \\
        --out /path/to/5spk_model_split.nemo
"""
import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "NeMo"))

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from nemo.collections.asr.models import SortformerEncLabelModel


def split_output_layer(model: SortformerEncLabelModel, n_base_spks: int) -> None:
    sm = model.sortformer_modules
    n_spk = sm.n_spk
    n_new = n_spk - n_base_spks

    if n_base_spks <= 0 or n_new <= 0:
        raise ValueError(f"n_base_spks={n_base_spks}는 1 이상 n_spk({n_spk}) 미만이어야 합니다.")

    if sm.n_base_spks > 0:
        print(f"이미 분리 모드입니다 (n_base_spks={sm.n_base_spks}). 스킵.")
        return

    src = sm.single_hidden_to_spks
    hidden_size = sm.hidden_size

    # 분리 레이어 생성
    base_layer = nn.Linear(hidden_size, n_base_spks)
    new_layer = nn.Linear(hidden_size, n_new)

    # 기존 가중치 분리 복사: 이미 학습된 가중치를 그대로 유지
    with torch.no_grad():
        base_layer.weight.copy_(src.weight[:n_base_spks])
        base_layer.bias.copy_(src.bias[:n_base_spks])
        new_layer.weight.copy_(src.weight[n_base_spks:])
        new_layer.bias.copy_(src.bias[n_base_spks:])

    sm.single_hidden_to_spks_base = base_layer
    sm.single_hidden_to_spks_new = new_layer
    sm.n_base_spks = n_base_spks

    # 기존 통합 레이어 제거
    del sm.single_hidden_to_spks

    print(f"출력층 분리 완료: single_hidden_to_spks({n_spk})")
    print(f"  → single_hidden_to_spks_base({n_base_spks})  [기존 가중치 복사, 뉴런 0~{n_base_spks-1}]")
    print(f"  → single_hidden_to_spks_new({n_new})          [기존 가중치 복사, 뉴런 {n_base_spks}~{n_spk-1}]")


def main():
    parser = argparse.ArgumentParser(description="출력층 분리 변환")
    parser.add_argument("--model-path", required=True, help="원본 .nemo 경로")
    parser.add_argument("--n-base-spks", type=int, required=True, help="기존(base) 화자 수")
    parser.add_argument("--out", required=True, help="저장할 .nemo 경로")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(f"모델 로드: {args.model_path}")
    model = SortformerEncLabelModel.restore_from(
        restore_path=args.model_path,
        map_location=torch.device(args.device),
    )
    model.eval()

    split_output_layer(model, args.n_base_spks)

    # config에 n_base_spks 반영
    try:
        OmegaConf.set_struct(model.cfg, False)
        model.cfg.sortformer_modules.n_base_spks = args.n_base_spks
        OmegaConf.set_struct(model.cfg, True)
        print(f"cfg.sortformer_modules.n_base_spks → {args.n_base_spks}")
    except Exception as e:
        print(f"[WARN] cfg 업데이트 실패: {e}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"저장: {out_path}")
    model.save_to(str(out_path))
    print("완료")


if __name__ == "__main__":
    sys.exit(main())
