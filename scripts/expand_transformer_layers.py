#!/usr/bin/env python3
"""
Sortformer Transformer 레이어 블록 복제 (영구 확장)

실험에서 [start, end] 블록을 두 번 통과할 때 성능이 올랐다면,
해당 블록의 가중치를 실제로 복사해 삽입하여 영구적인 모델로 저장합니다.

사용법:
    python scripts/expand_transformer_layers.py \
        --model-path /path/to/model.nemo \
        --start-layer 10 \
        --end-layer 16 \
        --out /path/to/expanded_model.nemo
"""
import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "NeMo"))

import torch
import torch.nn as nn
from nemo.collections.asr.models import SortformerEncLabelModel


def expand_transformer_layers(model: SortformerEncLabelModel, start: int, end: int) -> None:
    """
    transformer_encoder.layers 의 [start, end] 블록을 복제하여
    end 바로 뒤에 삽입합니다.

    원본: 0, 1, ..., start, ..., end, end+1, ..., N-1
    결과: 0, 1, ..., start, ..., end, start', ..., end', end+1, ..., N-1
    """
    layers = model.transformer_encoder.layers
    num_layers = len(layers)

    print(f"원본 레이어 수: {num_layers}")
    print(f"복제 대상 블록: [{start}, {end}] (크기 {end - start + 1})")

    if start < 0 or end >= num_layers or start > end:
        raise ValueError(f"레이어 범위 오류: [{start}, {end}] (총 {num_layers}층)")

    # 블록 복제 (깊은 복사 = 독립적인 가중치)
    cloned_block = nn.ModuleList([
        copy.deepcopy(layers[i]) for i in range(start, end + 1)
    ])

    # 새 레이어 목록 구성: 0..end + 복제 + end+1..N-1
    new_layers = (
        list(layers[:end + 1])       # 0 ~ end
        + list(cloned_block)          # start' ~ end' (복제본)
        + list(layers[end + 1:])      # end+1 ~ N-1
    )

    model.transformer_encoder.layers = nn.ModuleList(new_layers)
    new_num_layers = len(new_layers)
    print(f"확장 후 레이어 수: {new_num_layers}")

    # config의 num_layers를 실제 레이어 수와 동기화해야
    # save_to → restore_from 시 모델 구조가 올바르게 생성됨
    try:
        from omegaconf import OmegaConf
        OmegaConf.set_struct(model.cfg, False)
        model.cfg.transformer_encoder.num_layers = new_num_layers
        OmegaConf.set_struct(model.cfg, True)
        print(f"cfg.transformer_encoder.num_layers → {new_num_layers}")
    except Exception as e:
        print(f"[WARN] cfg 업데이트 실패: {e}")


def main():
    parser = argparse.ArgumentParser(description="Transformer 레이어 블록 복제 후 .nemo 저장")
    parser.add_argument("--model-path", required=True, help="원본 .nemo 경로")
    parser.add_argument("--start-layer", type=int, required=True, help="복제 시작 레이어 인덱스")
    parser.add_argument("--end-layer", type=int, required=True, help="복제 끝 레이어 인덱스 (포함)")
    parser.add_argument("--out", required=True, help="저장할 .nemo 경로")
    parser.add_argument("--device", default="cpu", help="로딩 디바이스 (기본: cpu)")
    args = parser.parse_args()

    print(f"모델 로드: {args.model_path}")
    model = SortformerEncLabelModel.restore_from(
        restore_path=args.model_path,
        map_location=torch.device(args.device),
    )
    model.eval()

    expand_transformer_layers(model, args.start_layer, args.end_layer)

    out_path = Path(args.out)
    if not out_path.suffix == ".nemo":
        out_path = out_path.with_suffix(".nemo")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"저장: {out_path}")
    model.save_to(str(out_path))
    print("완료")


if __name__ == "__main__":
    sys.exit(main())
