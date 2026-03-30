#!/usr/bin/env python3
"""
Convert NeMo/PyTorch Lightning checkpoint (.ckpt) to NeMo format (.nemo).
Usage:
    python scripts/ckpt_to_nemo.py --ckpt path/to/best.ckpt --out ultra_diar_streaming_sortformer_8spk_v1.nemo
"""
import argparse
import sys
from pathlib import Path

import torch

# Add NeMo to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "NeMo"))

from nemo.collections.asr.models import SortformerEncLabelModel


def main():
    parser = argparse.ArgumentParser(description="Convert .ckpt to .nemo")
    parser.add_argument("--ckpt", required=True, help="Path to .ckpt file")
    parser.add_argument("--out", required=True, help="Output .nemo path")
    parser.add_argument("--device", default="cpu", help="Device for loading (cpu or cuda)")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    out_path = Path(args.out)

    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}")
        return 1

    if not out_path.suffix == ".nemo":
        out_path = out_path.with_suffix(".nemo")

    map_location = torch.device(args.device)
    print(f"Loading checkpoint: {ckpt_path}")
    model = SortformerEncLabelModel.load_from_checkpoint(
        checkpoint_path=str(ckpt_path),
        map_location=map_location,
        strict=False,
        weights_only=False,
    )
    model.eval()

    print(f"Saving to: {out_path}")
    model.save_to(str(out_path))
    print("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
