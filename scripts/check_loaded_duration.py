#!/usr/bin/env python3
"""
train manifest의 각 항목이 실제로 로드될 때 몇 초가 되는지 검증.
duration=-1이면 전체 파일 로드 → 180초 초과 시 OOM 원인.
"""
import json
import os
import sys
from pathlib import Path

try:
    import soundfile as sf
except ImportError:
    sf = None

def get_actual_duration(entry):
    """manifest 항목이 로드될 때 실제 duration (초) 반환."""
    fp = entry.get("audio_filepath", entry.get("audio_file", ""))
    dur = entry.get("duration", -1)
    offset = entry.get("offset", 0) or 0
    
    # 경로 정규화
    fp = fp.replace("/mnt/data/workspace", "/home/devsy/workspace")
    if not os.path.exists(fp):
        return None, f"missing:{fp[:50]}"
    
    if sf is None:
        return None, "no_soundfile"
    
    try:
        info = sf.info(fp)
        total_frames = info.frames
        sr = info.samplerate
        total_dur = total_frames / sr
    except Exception as e:
        return None, str(e)[:40]
    
    # realworld: duration=180, offset 지정 → 180s만 로드
    if isinstance(dur, (int, float)) and dur > 0:
        return min(dur, total_dur - offset), None
    
    # synthetic: duration=-1 → 전체 파일 로드
    return total_dur, None


def main():
    manifest = sys.argv[1] if len(sys.argv) > 1 else "/home/devsy/workspace/synthetic_data/train_2to8spk_new2.json"
    
    with open(manifest) as f:
        entries = [json.loads(line) for line in f]
    
    over_180 = []
    errors = []
    for i, e in enumerate(entries):
        dur, err = get_actual_duration(e)
        if err:
            errors.append((i, err))
            continue
        if dur is not None and dur > 185:
            over_180.append((i, dur, e.get("audio_filepath", "")[:60]))
    
    print(f"Checked {len(entries)} entries")
    print(f"Errors: {len(errors)}")
    if errors[:5]:
        for i, err in errors[:5]:
            print(f"  [{i}] {err}")
    
    print(f"\n180s 초과 (OOM 원인 후보): {len(over_180)}개")
    for i, dur, path in over_180[:20]:
        print(f"  [{i}] {dur:.0f}s  {path}...")
    
    if over_180:
        print(f"\n*** 180초 초과 항목이 {len(over_180)}개 있습니다. 이 항목들이 OOM의 원인입니다. ***")
        print("해결: 해당 manifest 항목에 duration=180, offset=0 추가하거나, 180초로 잘라서 사용")


if __name__ == "__main__":
    main()
