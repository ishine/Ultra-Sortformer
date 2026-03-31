#!/usr/bin/env python3
"""
Extend Sortformer speaker head from N to M outputs (unified or split checkpoint).
Copies existing rows; new rows use SVD-based orthogonal init. Saves with
n_base_spks=N (split head) for differential LR during fine-tuning.

Usage:
    python scripts/extend_output_layer.py \\
        --src /path/to/Nspk_model.nemo \\
        --dst-spk 8 \\
        --out /path/to/Mspk_model.nemo
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "NeMo"))

import torch
from omegaconf import OmegaConf
from nemo.collections.asr.models import SortformerEncLabelModel

SK_BASE_W = "sortformer_modules.single_hidden_to_spks_base.weight"
SK_BASE_B = "sortformer_modules.single_hidden_to_spks_base.bias"
SK_NEW_W = "sortformer_modules.single_hidden_to_spks_new.weight"
SK_NEW_B = "sortformer_modules.single_hidden_to_spks_new.bias"
SK_UNI_W = "sortformer_modules.single_hidden_to_spks.weight"
SK_UNI_B = "sortformer_modules.single_hidden_to_spks.bias"


def orthogonal_extend_weight(src_weight: torch.Tensor, n_src: int, n_dst: int) -> torch.Tensor:
    """Stack (n_src, H) weights to (n_dst, H); new rows from SVD directions of existing rows."""
    w = src_weight.float()
    avg_norm = torch.norm(w, p=2, dim=1).mean()
    avg_mean = w.mean()
    avg_std = w.std().clamp(min=1e-6)
    _, _, Vh = torch.linalg.svd(w, full_matrices=True)

    base = src_weight.clone()
    extra_rows = []
    for i in range(n_dst - n_src):
        idx = n_src + i
        if idx < Vh.shape[0]:
            new_vec = Vh[idx : idx + 1].clone()
        else:
            new_vec = w.mean(dim=0, keepdim=True) + torch.randn_like(w[:1]) * avg_std
        new_vec = (new_vec - new_vec.mean()) / (new_vec.std() + 1e-6)
        new_vec = new_vec * avg_std + avg_mean
        norm = torch.norm(new_vec, p=2)
        if norm > 1e-6:
            new_vec = new_vec * (avg_norm / norm)
        extra_rows.append(new_vec.to(src_weight.dtype))

    return torch.cat([base] + extra_rows, dim=0)


def orthogonal_extend_bias(src_bias: torch.Tensor, n_src: int, n_dst: int) -> torch.Tensor:
    new_bias = src_bias.clone()
    extra = []
    for _ in range(n_dst - n_src):
        val = src_bias.mean() + torch.randn(1, device=src_bias.device, dtype=src_bias.dtype).squeeze() * src_bias.std()
        extra.append(val.unsqueeze(0))
    return torch.cat([new_bias] + extra, dim=0)


def get_unified_output_weights(state_dict: dict) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return concatenated (weight, bias, n_spk) from unified or split head keys."""
    if SK_BASE_W in state_dict:
        weight = torch.cat([state_dict[SK_BASE_W], state_dict[SK_NEW_W]], dim=0)
        bias = torch.cat([state_dict[SK_BASE_B], state_dict[SK_NEW_B]], dim=0)
    else:
        weight = state_dict[SK_UNI_W]
        bias = state_dict[SK_UNI_B]
    return weight, bias, weight.shape[0]


def main():
    parser = argparse.ArgumentParser(description="Extend Sortformer output layer to more speakers")
    parser.add_argument("--src", required=True, help="Path to source .nemo checkpoint")
    parser.add_argument("--dst-spk", type=int, required=True, help="Target number of speakers")
    parser.add_argument("--out", required=True, help="Path to write extended .nemo")
    args = parser.parse_args()

    print(f"Loading source model: {args.src}")
    model_src = SortformerEncLabelModel.restore_from(
        restore_path=args.src,
        map_location=torch.device("cpu"),
    )
    state_src = model_src.state_dict()

    src_weight, src_bias, n_src = get_unified_output_weights(state_src)
    n_dst = args.dst_spk

    if n_dst <= n_src:
        print(f"Error: dst-spk ({n_dst}) must be greater than source speakers ({n_src}).")
        return 1

    print(f"Extending head: {n_src} spk → {n_dst} spk (SVD orthogonal init for new rows)")

    new_weight = orthogonal_extend_weight(src_weight, n_src, n_dst)
    new_bias = orthogonal_extend_bias(src_bias, n_src, n_dst)

    cfg = model_src.cfg
    OmegaConf.set_struct(cfg, False)
    cfg.max_num_of_spks = n_dst
    cfg.sortformer_modules.num_spks = n_dst
    cfg.sortformer_modules.n_base_spks = n_src
    OmegaConf.set_struct(cfg, True)

    print(f"Building target model (max_num_of_spks={n_dst}, n_base_spks={n_src})")
    model_dst = SortformerEncLabelModel(cfg=cfg, trainer=None)
    state_dst = model_dst.state_dict()

    # Copy matching shapes from source; keep dst init where shapes differ (e.g. new modules).
    new_state = {}
    for k, v_dst in state_dst.items():
        if k not in state_src:
            new_state[k] = v_dst
            continue
        v_src = state_src[k]
        if v_src.shape == v_dst.shape:
            new_state[k] = v_src.clone()
        else:
            new_state[k] = v_dst

    new_state[SK_BASE_W] = new_weight[:n_src].clone()
    new_state[SK_BASE_B] = new_bias[:n_src].clone()
    new_state[SK_NEW_W] = new_weight[n_src:].clone()
    new_state[SK_NEW_B] = new_bias[n_src:].clone()

    missing, unexpected = model_dst.load_state_dict(new_state, strict=False)
    print(f"State dict loaded. missing={len(missing)}, unexpected={len(unexpected)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving: {out_path}")
    model_dst.save_to(str(out_path))
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
