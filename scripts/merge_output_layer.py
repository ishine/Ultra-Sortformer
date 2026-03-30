#!/usr/bin/env python3
"""
출력층 병합 변환 (split_output_layer.py 의 역연산)

single_hidden_to_spks_base + single_hidden_to_spks_new  →  single_hidden_to_spks (n_spk)

학습 시 split LR·장식 상관 손실 등을 위해 층을 나눴어도, 추론 전에는 위 병합이
수학적으로 동일한 출력을 냅니다. NeMo 기본 SortformerModules(n_base_spks=0)와
가중치 키가 맞아 Hugging Face에서 monkey-patch 없이 from_pretrained 가능합니다.

사용법:
    python scripts/merge_output_layer.py \\
        --model-path /path/to/8spk_split.nemo \\
        --out /path/to/8spk_merged.nemo
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "NeMo"))

import torch
import torch.nn as nn
from omegaconf import OmegaConf, open_dict
from nemo.collections.asr.models import SortformerEncLabelModel


def merge_output_layer(model: SortformerEncLabelModel) -> None:
    sm = model.sortformer_modules
    if sm.n_base_spks <= 0:
        print("이미 통합 모드입니다 (n_base_spks=0). 스킵.")
        return

    n_base = sm.n_base_spks
    n_spk = sm.n_spk
    n_new = n_spk - n_base
    hidden_size = sm.hidden_size

    merged = nn.Linear(hidden_size, n_spk)
    with torch.no_grad():
        merged.weight.copy_(
            torch.cat(
                [sm.single_hidden_to_spks_base.weight, sm.single_hidden_to_spks_new.weight],
                dim=0,
            )
        )
        merged.bias.copy_(
            torch.cat(
                [sm.single_hidden_to_spks_base.bias, sm.single_hidden_to_spks_new.bias],
                dim=0,
            )
        )

    del sm.single_hidden_to_spks_base
    del sm.single_hidden_to_spks_new
    sm.single_hidden_to_spks = merged
    sm.n_base_spks = 0

    print(f"출력층 병합 완료: single_hidden_to_spks_base({n_base}) + new({n_new})")
    print(f"  → single_hidden_to_spks({n_spk})")


def main():
    parser = argparse.ArgumentParser(description="출력층 base/new → 단일 레이어 병합")
    parser.add_argument("--model-path", required=True, help="분리 출력층 .nemo 경로")
    parser.add_argument("--out", required=True, help="저장할 .nemo 경로")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(f"모델 로드: {args.model_path}")
    model = SortformerEncLabelModel.restore_from(
        restore_path=args.model_path,
        map_location=torch.device(args.device),
    )
    model.eval()

    merge_output_layer(model)

    try:
        OmegaConf.set_struct(model.cfg, False)
        with open_dict(model.cfg.sortformer_modules):
            model.cfg.sortformer_modules.pop("n_base_spks", None)
        OmegaConf.set_struct(model.cfg, True)
        print("cfg.sortformer_modules 에서 n_base_spks 키 제거 (upstream NeMo 호환)")
    except Exception as e:
        print(f"[WARN] cfg 업데이트 실패: {e}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"저장: {out_path}")
    model.save_to(str(out_path))
    print("완료")


if __name__ == "__main__":
    sys.exit(main())
