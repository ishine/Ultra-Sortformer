#!/usr/bin/env python3
"""
.nemo 안의 model_config.yaml 에서 sortformer_modules.n_base_spks 항목을 제거합니다.

워크스페이스 NeMo(분리 헤드용 n_base_spks 지원)로 저장한 체크포인트를
pip 설치 upstream NeMo로 restore_from / from_pretrained 하려면,
cfg에 남은 n_base_spks 키만으로도 __init__ 에 예기치 않은 인자가 전달되어 실패합니다.
키 자체를 삭제해야 합니다.

의존성: PyYAML (NeMo와 함께 설치되는 경우가 많음)

사용법:
    python scripts/strip_nemo_sortformer_cfg_for_upstream.py \\
        --in /path/to/model.nemo \\
        --out /path/to/out.nemo

    제자리 덮어쓰기:
    python scripts/strip_nemo_sortformer_cfg_for_upstream.py \\
        --in /path/to/model.nemo --in-place
"""
from __future__ import annotations

import argparse
import io
import sys
import tarfile
from pathlib import Path


def _patch_yaml_text(text: str) -> str:
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("PyYAML 이 필요합니다: pip install pyyaml") from e

    cfg = yaml.safe_load(text)
    sm = cfg.get("sortformer_modules")
    if isinstance(sm, dict) and "n_base_spks" in sm:
        del sm["n_base_spks"]
        print("제거됨: sortformer_modules.n_base_spks")
    else:
        print("변경 없음: sortformer_modules.n_base_spks 가 없거나 dict 가 아님")

    return yaml.dump(cfg, default_flow_style=False, sort_keys=False, allow_unicode=True)


def patch_nemo_archive(inp: Path, out: Path) -> None:
    buf = io.BytesIO()
    with tarfile.open(inp, "r:*") as tar_in:
        with tarfile.open(fileobj=buf, mode="w") as tar_out:
            for member in tar_in.getmembers():
                f = tar_in.extractfile(member)
                if f is None:
                    continue
                data = f.read()
                name = member.name
                if name == "model_config.yaml" or name.endswith("/model_config.yaml"):
                    data = _patch_yaml_text(data.decode("utf-8")).encode("utf-8")
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tar_out.addfile(info, io.BytesIO(data))
    buf.seek(0)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(buf.read())
    print(f"저장: {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description="upstream NeMo용 .nemo cfg 에서 n_base_spks 제거")
    ap.add_argument("--in", dest="inp", required=True, type=Path, help="입력 .nemo")
    ap.add_argument("--out", dest="out", type=Path, default=None, help="출력 .nemo (--in-place 와 배타)")
    ap.add_argument(
        "--in-place",
        action="store_true",
        help="입력 파일을 임시 파일 경유로 제자리 덮어쓰기 (별도 출력 파일 없음)",
    )
    args = ap.parse_args()
    if args.in_place:
        inp = args.inp.resolve()
        tmp = inp.with_name(inp.name + ".tmp")
        try:
            patch_nemo_archive(inp, tmp)
            tmp.replace(inp)
            print(f"제자리 덮어씀: {inp}")
        except BaseException:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise
        return 0
    if args.out is None:
        ap.error("--out 또는 --in-place 가 필요합니다.")
    patch_nemo_archive(args.inp, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
