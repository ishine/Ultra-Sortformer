#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kaddress_speech 또는 kemergency_speech JSON 파일을 RTTM 포맷으로 변환하는 스크립트.

기본 사용법:
    python3 generate_rttm.py \
        --source /home/ubuntu/workspace/aihub/kaddress_speech \
        --output /home/ubuntu/workspace/aihub/kaddress_speech_rttm \
        --dataset_type kaddress
    
    python3 generate_rttm.py \
        --source /home/ubuntu/workspace/aihub/kemergency_speech \
        --output /home/ubuntu/workspace/aihub/kemergency_speech_rttm \
        --dataset_type kemergency
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

# multiprocessing temp가 루트(/) 대신 여유 공간 있는 디스크에 생성되도록 (import 전 설정)
for _cand in [os.path.expanduser("~/workspace"), os.path.expanduser("~"), "/tmp"]:
    if os.path.exists(_cand):
        _tmpdir = os.path.join(_cand, ".rttm_tmp")
        try:
            os.makedirs(_tmpdir, exist_ok=True)
            os.environ["TMPDIR"] = _tmpdir
            break
        except PermissionError:
            continue

from multiprocessing import Pool, cpu_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="kaddress_speech JSON을 RTTM 파일로 변환"
    )
    parser.add_argument(
        "--source",
        default="/home/ubuntu/workspace/aihub/kaddress_speech",
        help="JSON 파일이 위치한 최상위 디렉토리",
    )
    parser.add_argument(
        "--output",
        default="/home/ubuntu/workspace/aihub/kaddress_speech_rttm",
        help="생성된 RTTM을 저장할 디렉토리",
    )
    parser.add_argument(
        "--subset",
        choices=["Training", "Validation", "all"],
        default="all",
        help="변환할 서브셋 지정 (기본: all)",
    )
    parser.add_argument(
        "--dataset_type",
        choices=["kaddress", "kemergency", "korean_dialect", "kchildren_counseling", "kchildren_context", "kbroadcast", "kdomainconf", "multispeaker_single"],
        default="kaddress",
        help="데이터셋 타입 (기본: kaddress). multispeaker_single: JSON 1개=발화 1개, 기타정보.SpeechStart/SpeechEnd, 기본정보.NumberOfSpeaker",
    )
    return parser.parse_args()


def load_json(json_path: Path) -> Dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def derive_recording_id(data: Dict[str, Any], json_path: Path, dataset_type: str) -> str:
    """데이터셋 타입에 따라 recording ID 추출"""
    if dataset_type == "kaddress":
        file_name = (
            data.get("info", {}).get("file_name")
            or json_path.stem  # info.file_name 이 없으면 JSON 파일명 사용
        )
        return os.path.splitext(file_name)[0]
    elif dataset_type == "kemergency":
        # kemergency는 JSON 파일명을 recording ID로 사용
        return json_path.stem
    elif dataset_type == "korean_dialect":
        # korean_dialect는 fileName 필드 또는 JSON 파일명 사용
        return data.get("fileName") or json_path.stem
    elif dataset_type == "kchildren_counseling":
        # kchildren_counseling은 info.ID 또는 JSON 파일명 사용
        return data.get("info", {}).get("ID") or json_path.stem
    elif dataset_type == "kchildren_context":
        # kchildren_context는 media_info.id 또는 JSON 파일명 사용
        return data.get("media_info", {}).get("id") or json_path.stem
    elif dataset_type == "kbroadcast":
        # kbroadcast는 metadata.title 또는 JSON 파일명 사용
        return data.get("metadata", {}).get("title") or json_path.stem
    elif dataset_type == "kdomainconf":
        # kdomainconf_speech는 metadata.title 또는 JSON 파일명 사용
        return data.get("metadata", {}).get("title") or json_path.stem
    elif dataset_type == "multispeaker_single":
        # 다화자 음성합성: 파일정보.FileName(확장자 제외) = recording ID
        fname = data.get("파일정보", {}).get("FileName") or json_path.name
        return str(Path(fname).with_suffix(""))
    else:
        return json_path.stem


def time_to_seconds(time_str: str) -> float:
    """HH:MM:SS.mmm 또는 MM:SS.mmm 형식의 시간 문자열을 초로 변환"""
    try:
        parts = time_str.split(":")
        if len(parts) == 3:
            # HH:MM:SS.mmm 형식
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            # MM:SS.mmm 형식
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        return float(time_str)
    except:
        return 0.0

def dialogs_to_rttm_lines(
    dialogs: List[Dict[str, Any]], recording_id: str, dataset_type: str, speaker_info: str = None
) -> List[str]:
    """데이터셋 타입에 따라 대화 데이터를 RTTM 라인으로 변환"""
    lines: List[str] = []
    
    if dataset_type == "kaddress":
        for dialog in dialogs:
            start = float(dialog.get("startPoint", 0.0))
            end = float(dialog.get("endPoint", start))
            duration = max(0.0, end - start)
            speaker = dialog.get("speakerID", "unknown")

            line = (
                f"SPEAKER {recording_id} 1 "
                f"{start:.3f} {duration:.3f} <NA> <NA> "
                f"{speaker} <NA> <NA>"
            )
            lines.append(line)
    
    elif dataset_type == "kemergency":
        for utterance in dialogs:
            # kemergency는 밀리초 단위이므로 초로 변환
            start_ms = float(utterance.get("startAt", 0.0))
            end_ms = float(utterance.get("endAt", start_ms))
            start = start_ms / 1000.0  # 밀리초를 초로 변환
            end = end_ms / 1000.0
            duration = max(0.0, end - start)
            speaker = str(utterance.get("speaker", "unknown"))

            line = (
                f"SPEAKER {recording_id} 1 "
                f"{start:.3f} {duration:.3f} <NA> <NA> "
                f"{speaker} <NA> <NA>"
            )
            lines.append(line)
    
    elif dataset_type == "korean_dialect":
        for segment in dialogs:
            # korean_dialect는 HH:MM:SS.mmm 형식
            start_time_str = segment.get("startTime", "00:00:00.000")
            end_time_str = segment.get("endTime", start_time_str)
            start = time_to_seconds(start_time_str)
            end = time_to_seconds(end_time_str)
            duration = max(0.0, end - start)
            
            # 각 segment/sentence의 speakerId 사용 (sentences 배열에 있음)
            # segments에는 speakerId가 없을 수 있으므로 fallback
            speaker = segment.get("speakerId", speaker_info if speaker_info else "speaker0")

            line = (
                f"SPEAKER {recording_id} 1 "
                f"{start:.3f} {duration:.3f} <NA> <NA> "
                f"{speaker} <NA> <NA>"
            )
            lines.append(line)
    
    elif dataset_type == "kchildren_counseling":
        for audio_segment in dialogs:
            # kchildren_counseling은 MM:SS.mmm 형식
            start_time_str = audio_segment.get("start", "00:00.000")
            end_time_str = audio_segment.get("end", start_time_str)
            start = time_to_seconds(start_time_str)
            end = time_to_seconds(end_time_str)
            duration = max(0.0, end - start)
            
            # type: "Q" = 상담사, "A" = 아동
            audio_type = audio_segment.get("type", "unknown")
            if audio_type == "Q":
                speaker = "counselor"  # 상담사
            elif audio_type == "A":
                speaker = "child"  # 아동
            else:
                speaker = "unknown"

            line = (
                f"SPEAKER {recording_id} 1 "
                f"{start:.3f} {duration:.3f} <NA> <NA> "
                f"{speaker} <NA> <NA>"
            )
            lines.append(line)
    
    elif dataset_type == "kchildren_context":
        for sentence in dialogs:
            # kchildren_context는 HH:MM:SS.mmm 형식 (공백 제거 필요)
            start_time_str = sentence.get("start", "00:00:00.000").strip()
            end_time_str = sentence.get("end", start_time_str).strip()
            start = time_to_seconds(start_time_str)
            end = time_to_seconds(end_time_str)
            duration = max(0.0, end - start)
            
            # speaker 필드 사용 (예: "화자_001")
            speaker = sentence.get("speaker", "unknown").strip()

            line = (
                f"SPEAKER {recording_id} 1 "
                f"{start:.3f} {duration:.3f} <NA> <NA> "
                f"{speaker} <NA> <NA>"
            )
            lines.append(line)
    
    elif dataset_type == "kbroadcast":
        # speaker 리스트에서 유효한 speaker ID 추출
        valid_speaker_ids = set()
        if speaker_info:  # speaker_info에 speaker 리스트가 전달됨
            for spk in speaker_info:
                spk_id = spk.get("id") if isinstance(spk, dict) else str(spk)
                if spk_id:
                    valid_speaker_ids.add(str(spk_id))
        
        for utterance in dialogs:
            speaker_id = str(utterance.get("speaker_id", "unknown")).strip()
            
            # speaker_id가 "?"이거나 유효한 speaker 리스트에 없으면 제외
            if speaker_id == "?" or speaker_id == "unknown":
                continue
            if valid_speaker_ids and speaker_id not in valid_speaker_ids:
                continue
            
            # kbroadcast는 초 단위 (이미 초 단위이므로 변환 불필요)
            start = float(utterance.get("start", 0.0))
            end = float(utterance.get("end", start))
            duration = max(0.0, end - start)

            line = (
                f"SPEAKER {recording_id} 1 "
                f"{start:.3f} {duration:.3f} <NA> <NA> "
                f"{speaker_id} <NA> <NA>"
            )
            lines.append(line)
    
    elif dataset_type == "kdomainconf":
        # speaker 리스트에서 유효한 speaker ID 추출
        valid_speaker_ids = set()
        if speaker_info:  # speaker_info에 speaker 리스트가 전달됨
            for spk in speaker_info:
                spk_id = spk.get("id") if isinstance(spk, dict) else str(spk)
                if spk_id:
                    valid_speaker_ids.add(str(spk_id))
        
        for utterance in dialogs:
            speaker_id = str(utterance.get("speaker_id", "unknown")).strip()
            
            # speaker_id가 "?"이거나 유효한 speaker 리스트에 없으면 제외
            if speaker_id == "?" or speaker_id == "unknown":
                continue
            if valid_speaker_ids and speaker_id not in valid_speaker_ids:
                continue
            
            # kdomainconf는 초 단위 (문자열 또는 숫자로 저장될 수 있음)
            start_val = utterance.get("start", 0.0)
            end_val = utterance.get("end", start_val)
            start = float(start_val) if start_val is not None else 0.0
            end = float(end_val) if end_val is not None else start
            duration = max(0.0, end - start)

            line = (
                f"SPEAKER {recording_id} 1 "
                f"{start:.3f} {duration:.3f} <NA> <NA> "
                f"{speaker_id} <NA> <NA>"
            )
            lines.append(line)
    
    elif dataset_type == "multispeaker_single":
        for data in dialogs:
            start = float(data.get("기타정보", {}).get("SpeechStart", 0.0) or 0.0)
            end = float(data.get("기타정보", {}).get("SpeechEnd", start) or start)
            duration = max(0.0, end - start)
            speaker = str(data.get("기본정보", {}).get("NumberOfSpeaker", "unknown"))
            line = (
                f"SPEAKER {recording_id} 1 "
                f"{start:.3f} {duration:.3f} <NA> <NA> "
                f"{speaker} <NA> <NA>"
            )
            lines.append(line)
    
    return lines


def process_file(args: Tuple[Path, Path, str]) -> Tuple[bool, str]:
    json_path, subset_output_dir, dataset_type = args
    try:
        data = load_json(json_path)
    except Exception as err:
        return False, f"[오류] JSON 로드 실패 ({json_path}): {err}"

    # 데이터셋 타입에 따라 대화 데이터 추출
    if dataset_type == "kaddress":
        dialogs = data.get("dialogs", [])
    elif dataset_type == "kemergency":
        dialogs = data.get("utterances", [])
    elif dataset_type == "korean_dialect":
        transcription = data.get("transcription", {})
        # sentences 배열에 speakerId가 있음
        dialogs = transcription.get("sentences", [])
        # sentences가 없으면 segments 사용 (fallback)
        if not dialogs:
            dialogs = transcription.get("segments", [])
    elif dataset_type == "kchildren_counseling":
        # kchildren_counseling은 list -> list -> audio 구조
        dialogs = []
        question_list = data.get("list", [])
        for question_item in question_list:
            item_list = question_item.get("list", [])
            for item in item_list:
                audio_segments = item.get("audio", [])
                dialogs.extend(audio_segments)
    elif dataset_type == "kchildren_context":
        # kchildren_context는 media_info.sentence_list 구조
        dialogs = data.get("media_info", {}).get("sentence_list", [])
    elif dataset_type == "kbroadcast":
        # kbroadcast는 utterance 배열 구조
        dialogs = data.get("utterance", [])
    elif dataset_type == "kdomainconf":
        # kdomainconf_speech는 utterance 배열 구조
        dialogs = data.get("utterance", [])
    elif dataset_type == "multispeaker_single":
        dialogs = [data]
    else:
        dialogs = []
    
    if not dialogs:
        return True, f"[건너뜀] 대화 없음 ({json_path.name})"

    recording_id = derive_recording_id(data, json_path, dataset_type)
    
    # 화자 정보 전달 (fallback용 또는 필터링용)
    speaker_info = None
    if dataset_type == "korean_dialect":
        speakers = data.get("speaker", [])
        if speakers and len(speakers) > 0:
            speaker_info = speakers[0].get("speakerId", "speaker0")
    elif dataset_type == "kbroadcast":
        # kbroadcast는 speaker 배열 전체를 전달 (유효한 speaker ID 필터링용)
        speaker_info = data.get("speaker", [])
    elif dataset_type == "kdomainconf":
        # kdomainconf_speech는 speaker 배열 전체를 전달 (유효한 speaker ID 필터링용)
        speaker_info = data.get("speaker", [])
    
    lines = dialogs_to_rttm_lines(dialogs, recording_id, dataset_type, speaker_info)
    output_path = subset_output_dir / f"{recording_id}.rttm"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    return True, f"[생성] {output_path.relative_to(subset_output_dir.parent)}"


def main() -> None:
    args = parse_args()
    source_root = Path(args.source).resolve()
    output_root = Path(args.output).resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"소스 디렉토리를 찾을 수 없습니다: {source_root}")

    subsets = []
    if args.subset == "all":
        for candidate in ["Training", "Validation"]:
            subset_path = source_root / candidate
            if subset_path.exists():
                subsets.append((candidate, subset_path))
        # multispeaker_single: Training/Validation 없으면 source_root 자체를 한 서브셋으로
        if not subsets and args.dataset_type == "multispeaker_single":
            if list(source_root.rglob("*.json")):
                subsets.append(("", source_root))
    else:
        subset_path = source_root / args.subset
        if not subset_path.exists():
            raise FileNotFoundError(
                f"{args.subset} 디렉토리를 찾을 수 없습니다: {subset_path}"
            )
        subsets.append((args.subset, subset_path))

    if not subsets:
        raise RuntimeError("처리할 서브셋이 없습니다.")

    total_files = 0
    pool_size = min(cpu_count(), 16)
    for subset_name, subset_dir in subsets:
        if args.dataset_type == "multispeaker_single":
            json_files = sorted(subset_dir.rglob("*.json"))
        else:
            json_files = sorted(subset_dir.glob("*.json"))
        if not json_files:
            print(f"[경고] {subset_name} 내 JSON 파일이 없습니다: {subset_dir}")
            continue

        subset_output_dir = output_root / subset_name
        # ext4 단일 디렉토리 한계(~1500만 파일): Training에 이미 1900만+ 파일 있으면
        # rttm 하위디렉토리에 저장 (Training/rttm/*.rttm)
        if args.dataset_type == "multispeaker_single" and subset_name == "Training":
            subset_output_dir = subset_output_dir / "rttm"
        subset_output_dir.mkdir(parents=True, exist_ok=True)

        tasks = [(json_path, subset_output_dir, args.dataset_type) for json_path in json_files]

        # multiprocessing 시도, 실패 시 단일 프로세스로 fallback
        try:
            with Pool(pool_size) as pool:
                for success, message in pool.imap_unordered(process_file, tasks):
                    if success:
                        total_files += 1
                    print(message)
        except (PermissionError, OSError) as e:
            print(f"[경고] Multiprocessing 실패, 단일 프로세스로 실행: {e}")
            # 단일 프로세스로 실행
            for task in tasks:
                success, message = process_file(task)
                if success:
                    total_files += 1
                print(message)

        print(f"[완료] {subset_name}: 누적 {total_files}개 RTTM 생성")

    print(f"RTTM 생성이 완료되었습니다. 총 {total_files}개 파일.")


if __name__ == "__main__":
    main()

