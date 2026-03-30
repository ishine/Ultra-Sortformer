#!/usr/bin/env python3
"""
출력층 화자 수 확장 스크립트 (extend_sortformer_4spk_to_5spk.py 방식 기반)

기존 N화자 모델(통합 or _base/_new 분리 모두 지원)을 M화자로 확장합니다.
- 기존 N개 출력 뉴런 가중치는 그대로 복사
- 새 (N+1)~M 뉴런은 SVD 기반 직교 초기화 (기존 화자와 겹침 최소화)
- 출력 모델은 n_base_spks=N 분리 구조로 저장 (학습률 분리 지원)

사용법:
    python scripts/extend_output_layer.py \\
        --src /path/to/5spk_model.nemo \\
        --dst-spk 6 \\
        --out /path/to/6spk_model.nemo
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "NeMo"))

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from nemo.collections.asr.models import SortformerEncLabelModel


def orthogonal_extend_weight(src_weight: torch.Tensor, n_src: int, n_dst: int) -> torch.Tensor:
    """
    src_weight (n_src, hidden) 를 (n_dst, hidden) 로 확장.
    새 행은 SVD 직교 벡터 기반 초기화.
    """
    avg_norm = torch.norm(src_weight.float(), p=2, dim=1).mean()
    avg_mean = src_weight.float().mean()
    avg_std  = src_weight.float().std().clamp(min=1e-6)

    _, _, Vh = torch.linalg.svd(src_weight.float(), full_matrices=True)

    new_weight = src_weight.clone()
    rows_to_add = n_dst - n_src
    extra_rows = []
    for i in range(rows_to_add):
        idx = n_src + i
        if idx < Vh.shape[0]:
            new_vec = Vh[idx : idx + 1].clone()
        else:
            new_vec = src_weight.float().mean(dim=0, keepdim=True) + \
                      torch.randn_like(src_weight[:1].float()) * avg_std
        new_vec = (new_vec - new_vec.mean()) / (new_vec.std() + 1e-6)
        new_vec = new_vec * avg_std + avg_mean
        norm = torch.norm(new_vec, p=2)
        if norm > 1e-6:
            new_vec = new_vec * (avg_norm / norm)
        extra_rows.append(new_vec.to(src_weight.dtype))

    return torch.cat([new_weight] + extra_rows, dim=0)  # (n_dst, hidden)


def orthogonal_extend_bias(src_bias: torch.Tensor, n_src: int, n_dst: int) -> torch.Tensor:
    new_bias = src_bias.clone()
    extra = []
    for _ in range(n_dst - n_src):
        val = src_bias.mean() + torch.randn(1, device=src_bias.device, dtype=src_bias.dtype).squeeze() * src_bias.std()
        extra.append(val.unsqueeze(0))
    return torch.cat([new_bias] + extra, dim=0)  # (n_dst,)


def get_unified_output_weights(state_dict: dict) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    state_dict에서 출력층 가중치를 통합해서 반환.
    통합 모드(single_hidden_to_spks) 또는 분리 모드(_base + _new) 모두 지원.
    반환: (weight, bias, n_spk)
    """
    base_w_key = "sortformer_modules.single_hidden_to_spks_base.weight"
    base_b_key = "sortformer_modules.single_hidden_to_spks_base.bias"
    new_w_key  = "sortformer_modules.single_hidden_to_spks_new.weight"
    new_b_key  = "sortformer_modules.single_hidden_to_spks_new.bias"
    uni_w_key  = "sortformer_modules.single_hidden_to_spks.weight"
    uni_b_key  = "sortformer_modules.single_hidden_to_spks.bias"

    if base_w_key in state_dict:
        weight = torch.cat([state_dict[base_w_key], state_dict[new_w_key]], dim=0)
        bias   = torch.cat([state_dict[base_b_key], state_dict[new_b_key]], dim=0)
    else:
        weight = state_dict[uni_w_key]
        bias   = state_dict[uni_b_key]

    return weight, bias, weight.shape[0]


def main():
    parser = argparse.ArgumentParser(description="출력층 화자 수 확장")
    parser.add_argument("--src", required=True, help="원본 .nemo 경로")
    parser.add_argument("--dst-spk", type=int, required=True, help="목표 화자 수")
    parser.add_argument("--out", required=True, help="저장할 .nemo 경로")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(f"소스 모델 로드: {args.src}")
    model_src = SortformerEncLabelModel.restore_from(
        restore_path=args.src,
        map_location=torch.device(args.device),
    )
    state_src = model_src.state_dict()

    # 기존 출력층 가중치 통합 추출
    src_weight, src_bias, n_src = get_unified_output_weights(state_src)
    n_dst = args.dst_spk

    if n_dst <= n_src:
        print(f"오류: dst_spk({n_dst}) > src_spk({n_src}) 이어야 합니다.")
        return 1

    print(f"출력층 확장: {n_src}spk → {n_dst}spk (SVD 직교 초기화)")

    # 확장된 가중치 생성
    new_weight = orthogonal_extend_weight(src_weight, n_src, n_dst)  # (n_dst, hidden)
    new_bias   = orthogonal_extend_bias(src_bias, n_src, n_dst)      # (n_dst,)

    # 목표 모델 config 구성 (소스 cfg 기반)
    cfg = model_src.cfg
    OmegaConf.set_struct(cfg, False)
    cfg.max_num_of_spks = n_dst
    cfg.sortformer_modules.num_spks = n_dst
    cfg.sortformer_modules.n_base_spks = n_src   # 분리 구조: base=기존, new=신규
    OmegaConf.set_struct(cfg, True)

    # 목표 모델 생성 (n_base_spks=n_src → _base/_new 분리 구조)
    print(f"목표 모델 생성 (max_num_of_spks={n_dst}, n_base_spks={n_src})")
    model_dst = SortformerEncLabelModel(cfg=cfg, trainer=None)
    state_dst = model_dst.state_dict()

    # state_dict 병합
    new_state = {}
    for k, v_dst in state_dst.items():
        if k not in state_src:
            new_state[k] = v_dst  # 새 키는 기본값 유지
            continue
        v_src = state_src[k]
        if v_src.shape == v_dst.shape:
            new_state[k] = v_src.clone()
        else:
            new_state[k] = v_dst  # shape 불일치는 기본값 유지

    # 출력층 분리 구조 key에 확장 가중치 삽입
    base_w = "sortformer_modules.single_hidden_to_spks_base.weight"
    base_b = "sortformer_modules.single_hidden_to_spks_base.bias"
    new_w  = "sortformer_modules.single_hidden_to_spks_new.weight"
    new_b  = "sortformer_modules.single_hidden_to_spks_new.bias"

    new_state[base_w] = new_weight[:n_src].clone()
    new_state[base_b] = new_bias[:n_src].clone()
    new_state[new_w]  = new_weight[n_src:].clone()
    new_state[new_b]  = new_bias[n_src:].clone()

    missing, unexpected = model_dst.load_state_dict(new_state, strict=False)
    print(f"로드 완료. Missing={len(missing)}, Unexpected={len(unexpected)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"저장: {out_path}")
    model_dst.save_to(str(out_path))
    print("완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
