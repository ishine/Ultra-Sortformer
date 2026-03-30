#!/usr/bin/env python3
"""
.nemo (tar) 안의 model_weights.ckpt 만 수정해 Sortformer single_hidden 출력층을 병합합니다.

NeMo 런타임 없이 torch + PyYAML 만 필요합니다.
- single_hidden_to_spks_base + single_hidden_to_spks_new
  → single_hidden_to_spks (가중치 행 방향 concat)
- model_config.yaml 의 sortformer_modules.n_base_spks 제거 (upstream cfg 호환)

사용법:
    python scripts/merge_sortformer_head_in_nemo_tar.py \\
        --in model.nemo --out model_merged.nemo
"""
from __future__ import annotations

import argparse
import io
import sys
import tarfile
from pathlib import Path

import torch


def _merge_flat_state_dict(sd: dict) -> bool:
    """Returns True if merge was applied."""
    bw = "sortformer_modules.single_hidden_to_spks_base.weight"
    bb = "sortformer_modules.single_hidden_to_spks_base.bias"
    nw = "sortformer_modules.single_hidden_to_spks_new.weight"
    nb = "sortformer_modules.single_hidden_to_spks_new.bias"
    ow = "sortformer_modules.single_hidden_to_spks.weight"
    ob = "sortformer_modules.single_hidden_to_spks.bias"

    if bw not in sd or ow in sd:
        return False

    sd[ow] = torch.cat([sd[bw], sd[nw]], dim=0)
    sd[ob] = torch.cat([sd[bb], sd[nb]], dim=0)
    del sd[bw], sd[bb], sd[nw], sd[nb]
    return True


def _patch_yaml_remove_n_base(text: str) -> str:
    import yaml

    cfg = yaml.safe_load(text)
    sm = cfg.get("sortformer_modules")
    if isinstance(sm, dict) and "n_base_spks" in sm:
        del sm["n_base_spks"]
    return yaml.dump(cfg, default_flow_style=False, sort_keys=False, allow_unicode=True)


def merge_nemo(inp: Path, out: Path) -> None:
    buf = io.BytesIO()
    merged_ckpt = False
    with tarfile.open(inp, "r:*") as tar_in:
        with tarfile.open(fileobj=buf, mode="w") as tar_out:
            for member in tar_in.getmembers():
                f = tar_in.extractfile(member)
                if f is None:
                    continue
                data = f.read()
                name = member.name
                if name.endswith("model_weights.ckpt"):
                    ckpt = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
                    if isinstance(ckpt, dict) and "state_dict" in ckpt:
                        inner = ckpt["state_dict"]
                        if _merge_flat_state_dict(inner):
                            merged_ckpt = True
                    else:
                        if _merge_flat_state_dict(ckpt):
                            merged_ckpt = True
                    bio = io.BytesIO()
                    torch.save(ckpt, bio)
                    data = bio.getvalue()
                elif name.endswith("model_config.yaml"):
                    data = _patch_yaml_remove_n_base(data.decode("utf-8")).encode("utf-8")
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tar_out.addfile(info, io.BytesIO(data))
    if not merged_ckpt:
        print(
            "[WARN] 출력층 병합 스킵: base/new 키가 없거나 이미 single_hidden_to_spks 가 있습니다.",
            file=sys.stderr,
        )
    buf.seek(0)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(buf.read())
    print(f"저장: {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=".nemo tar 내 Sortformer 출력층 병합")
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    merge_nemo(args.inp, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
