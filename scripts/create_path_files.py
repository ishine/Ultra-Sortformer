#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WAV/RTTM 경로 리스트 파일 생성. 선택적으로 NeMo 스타일 manifest.json 생성.

- data_dir 내 *.wav 기준으로 audio_file_path_list.txt, rttm_file_path_list.txt 생성.
- --manifest 시 manifest.json 추가 생성. JSON 구조는 --dataset_type에 따라 다름.
"""
from pathlib import Path
import argparse
import json
import random


def _duration_from_json(json_path: Path, dataset_type: str) -> float | None:
    """dataset_type에 따라 JSON에서 duration(초) 반환."""
    try:
        if dataset_type == "synthetic":
            # JSONL: 각 줄이 {"offset", "duration", "num_speakers", ...}
            max_end = 0.0
            with open(json_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    o = data.get("offset", 0) or 0
                    d = data.get("duration", 0) or 0
                    max_end = max(max_end, float(o) + float(d))
            return max_end if max_end > 0 else None

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        if dataset_type == "multispeaker_single":
            flen = data.get("파일정보", {}).get("FileLength")
            if flen is not None:
                try:
                    return float(flen)
                except (TypeError, ValueError):
                    pass
            return None
        if dataset_type == "kdomainconf":
            utterances = data.get("utterance") or []
            if not utterances:
                return None
            max_end = 0.0
            for u in utterances:
                end = u.get("end")
                if end is None:
                    continue
                if isinstance(end, str):
                    end = float(end)
                max_end = max(max_end, end)
            return max_end if max_end > 0 else None
        return None
    except Exception:
        return None


def _duration_from_wav(wav_path: Path) -> float | None:
    """WAV 파일 길이(초) 반환. soundfile 또는 librosa 사용."""
    try:
        import soundfile as sf
        info = sf.info(str(wav_path))
        return float(info.duration)
    except ImportError:
        try:
            import librosa
            y, sr = librosa.load(str(wav_path), sr=None, duration=0)
            return float(len(y) / sr) if sr else None
        except ImportError:
            return None
    except Exception:
        return None


def _speaker_num_from_json(json_path: Path, dataset_type: str) -> int | None:
    """dataset_type에 따라 JSON에서 화자 수 반환."""
    try:
        if dataset_type == "synthetic":
            with open(json_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    n = data.get("num_speakers")
                    if n is not None:
                        return int(n)
            return None

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        if dataset_type == "multispeaker_single":
            return 1
        if dataset_type == "kdomainconf":
            return data.get("metadata", {}).get("speaker_num")
        return None
    except Exception:
        return None


def _iter_wav_files(data_path: Path):
    """wav/01~10/*.wav 또는 data_path/*.wav"""
    wav_base = data_path / "wav"
    if wav_base.exists():
        for sub in sorted(wav_base.iterdir()):
            if sub.is_dir():
                for f in sub.glob("*.wav"):
                    if f.is_file():
                        yield f
    else:
        for f in sorted(data_path.glob("*.wav")):
            if f.is_file():
                yield f


def _build_stem_to_path(data_path: Path, ext: str) -> dict[str, Path]:
    """{ext}/01~10 또는 data_path 에서 stem -> Path 맵 생성"""
    result = {}
    base = data_path / ext
    if base.exists():
        for sub in base.iterdir():
            if sub.is_dir():
                for f in sub.glob(f"*.{ext}"):
                    if f.is_file():
                        result[f.stem] = f
    for f in data_path.glob(f"*.{ext}"):
        if f.is_file():
            result[f.stem] = f
    return result


def create_path_files(
    data_dir: str,
    output_dir: str | None = None,
    manifest: bool = False,
    manifest_only: bool = False,
    dataset_type: str = "kdomainconf",
    train_val_split: float | None = None,
    split_seed: int = 42,
):
    """WAV와 RTTM 경로 리스트 파일 생성. manifest=True면 manifest.json도 생성. manifest_only=True면 manifest만 생성."""
    data_path = Path(data_dir).resolve()
    if not data_path.exists():
        print(f"Error: {data_dir} does not exist")
        return

    if output_dir is None:
        out_path = data_path
    else:
        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(_iter_wav_files(data_path), key=lambda f: f.stem)
    if not wav_files:
        print(f"Error: No WAV files found in {data_dir} (wav/01~10 or *.wav)")
        return

    print("RTTM stem 맵 수집 중...", flush=True)
    rttm_map = _build_stem_to_path(data_path, "rttm")
    print(f"  rttm: {len(rttm_map):,}개")

    wav_paths = []
    rttm_paths = []

    for wav_file in wav_files:
        base_name = wav_file.stem
        wav_paths.append(str(wav_file.resolve()))
        rttm_file = rttm_map.get(base_name)
        if rttm_file:
            rttm_paths.append(str(rttm_file.resolve()))
        else:
            rttm_paths.append("")

    if not manifest_only:
        audio_list_file = out_path / "audio_file_path_list.txt"
        rttm_list_file = out_path / "rttm_file_path_list.txt"

        with open(audio_list_file, "w", encoding="utf-8") as f:
            f.write("\n".join(wav_paths) + "\n")

        with open(rttm_list_file, "w", encoding="utf-8") as f:
            f.write("\n".join(p if p else "" for p in rttm_paths) + "\n")

        print(f"Created: {audio_list_file} ({len(wav_paths)} WAV)")
        print(f"Created: {rttm_list_file} ({sum(1 for p in rttm_paths if p)} RTTM)")
        if len(wav_paths) != sum(1 for p in rttm_paths if p):
            print(f"  Warning: {len(wav_paths) - sum(1 for p in rttm_paths if p)} WAV files have no RTTM")

    if manifest or manifest_only:
        print("JSON stem 맵 수집 중...", flush=True)
        json_map = _build_stem_to_path(data_path, "json")
        print(f"  json: {len(json_map):,}개")

        entries = []
        for wav_file in wav_files:
            base = wav_file.stem
            wav_abs = str(wav_file.resolve())
            rttm_file = rttm_map.get(base)
            json_file = json_map.get(base)

            duration = _duration_from_json(json_file, dataset_type) if json_file else None
            if duration is None:
                duration = _duration_from_wav(wav_file)
            if duration is None:
                duration = 0.0

            num_speakers = _speaker_num_from_json(json_file, dataset_type) if json_file else None
            if num_speakers is None:
                num_speakers = 0

            rttm_path = str(rttm_file.resolve()) if rttm_file else None
            entry = {
                "audio_filepath": wav_abs,
                "offset": 0,
                "duration": round(duration, 4),
                "label": "infer",
                "text": "-",
                "num_speakers": num_speakers,
                "rttm_filepath": rttm_path,
                "uem_filepath": None,
                "ctm_filepath": None,
            }
            entries.append(entry)

        manifest_file = out_path / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"Created: {manifest_file} ({len(entries)} entries, dataset_type={dataset_type})")

        if train_val_split is not None and 0 < train_val_split < 1:
            shuffled = entries.copy()
            random.seed(split_seed)
            random.shuffle(shuffled)
            n_train = int(len(shuffled) * train_val_split)
            train_entries = shuffled[:n_train]
            val_entries = shuffled[n_train:]
            train_file = out_path / "train_manifest.json"
            val_file = out_path / "val_manifest.json"
            with open(train_file, "w", encoding="utf-8") as f:
                for e in train_entries:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            with open(val_file, "w", encoding="utf-8") as f:
                for e in val_entries:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            print(f"Created: {train_file} ({len(train_entries)} train, {train_val_split:.0%})")
            print(f"Created: {val_file} ({len(val_entries)} val, {1-train_val_split:.0%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create path list files (audio, rttm) and optionally NeMo manifest.json"
    )
    parser.add_argument(
        "--data_dir",
        default="/home/devsy/workspace/data/multispeaker_speech_synthesis_data/Training",
        help="Directory containing wav/01~10, rttm/01~10 (or *.wav, *.rttm)",
    )
    parser.add_argument("--output_dir", default=None, help="Output directory (default: same as data_dir)")
    parser.add_argument("--manifest", action="store_true", help="Also create manifest.json (NeMo-style)")
    parser.add_argument("--manifest_only", action="store_true", help="Create only manifest.json (skip audio/rttm path lists)")
    parser.add_argument(
        "--dataset_type",
        choices=["kdomainconf", "multispeaker_single", "synthetic"],
        default="kdomainconf",
        help="JSON 구조: kdomainconf, multispeaker_single, synthetic(JSONL: offset+duration, num_speakers)",
    )
    parser.add_argument(
        "--train_val_split",
        type=float,
        default=None,
        help="8:2 분할 시 0.8 지정. train_manifest.json, val_manifest.json 추가 생성",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="train/val 분할 시 셔플 시드 (default: 42)",
    )
    args = parser.parse_args()

    create_path_files(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        manifest=args.manifest,
        manifest_only=args.manifest_only,
        dataset_type=args.dataset_type,
        train_val_split=args.train_val_split,
        split_seed=args.split_seed,
    )
