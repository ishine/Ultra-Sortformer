#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
문장 단위 다중 화자 오디오 세션 시뮬레이터

기존 NeMo `MultiSpeakerSimulator`의 모든 로직은 그대로 두고,
문장(utterance) 단위로만 합성되도록 `_build_sentence` 부분만 바꾼 래퍼 스크립트입니다.

- **기존과 동일**: 세션 길이, 침묵/오버랩 분포, 화자 지배도, 증강, 출력 형식 등 전부
- **달라진 점**: 단어 단위 샘플링 대신, 매니페스트의 한 항목(문장 전체)을 통째로 붙여서 발화 구성

기존 YAML 설정 파일을 그대로 사용할 수 있고,
필요하면 몇 개 인자만 CLI에서 덮어쓸 수 있습니다.

사용 예시 (가장 기본):

    cd /home/devsy/workspace/NeMo
    python ../scripts/sentence_level_multispeaker_simulator.py \
      --manifest_filepath /home/devsy/workspace/test_manifest_10speakers_100audio.json \
      --output_dir /home/devsy/workspace/simulated_sentence_level

필요 시 세션 관련 파라미터만 추가로 덮어쓰기:

    python ../scripts/sentence_level_multispeaker_simulator.py \
      --manifest_filepath /home/devsy/workspace/test_manifest_10speakers_100audio.json \
      --output_dir /home/devsy/workspace/simulated_sentence_level \
      --output_filename sim_sentence_level \
      --num_speakers 5 \
      --num_sessions 10 \
      --session_length 90
"""

import os
import random
import sys
import time
import argparse
from typing import Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf

# NeMo 소스 경로를 PYTHONPATH에 추가
SCRIPT_DIR = os.path.dirname(__file__)
NEMO_ROOT = os.path.join(SCRIPT_DIR, "..", "NeMo")
if os.path.exists(NEMO_ROOT):
    sys.path.insert(0, NEMO_ROOT)

from nemo.collections.asr.data.data_simulation import MultiSpeakerSimulator
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.asr.parts.utils.data_simulation_utils import (
    normalize_audio,
    perturb_audio,
    per_speaker_normalize,
    get_split_points_in_alignments,
)
from nemo.utils import logging


class SentenceLevelMultiSpeakerSimulator(MultiSpeakerSimulator):
    """
    문장 단위 다중 화자 오디오 세션 시뮬레이터.

    - `MultiSpeakerSimulator`를 상속
    - `_generate_session` 등 나머지 로직은 그대로 사용
    - **오직 `_build_sentence`만** 문장(utterance) 전체를 붙이는 방식으로 수정
    """

    def _build_sentence(
        self,
        speaker_turn: int,
        speaker_ids: List[str],
        speaker_wav_align_map: Dict[str, list],
        max_samples_in_sentence: int,
    ):
        """
        한 화자의 한 턴(turn)에 대해,
        - 매니페스트에서 문장 단위의 오디오를 통째로 가져와
        - 여러 문장을 순서대로 이어 붙여 하나의 `self._sentence`를 구성.

        기존 구현은 `load_speaker_sample` + `_add_file` 조합으로
        단어 단위로 자르고 이어 붙였다면,
        여기서는 매니페스트의 한 row(=문장 전체)를 사용합니다.
        """
        sr = self._params.data_simulator.sr
        max_turn_sec = self._params.data_simulator.session_params.get("max_turn_duration_sec", None)
        if max_turn_sec is not None and max_turn_sec > 0:
            max_samples_in_sentence = min(max_samples_in_sentence, int(max_turn_sec * sr))

        # 1) 문장 개수 샘플링
        # max_sentences_per_turn이 있으면 1~N 균등 (기본 3). 0이면 음이항
        max_sent = self._params.data_simulator.session_params.get("max_sentences_per_turn", None)
        if max_sent is not None and max_sent >= 1:
            num_sentences = np.random.randint(1, int(max_sent) + 1)
        else:
            num_sentences = (
                np.random.negative_binomial(
                    self._params.data_simulator.session_params.sentence_length_params[0],
                    self._params.data_simulator.session_params.sentence_length_params[1],
                )
                + 1
            )

        # 2) 내부 버퍼 초기화 (기존과 동일한 필드 사용)
        self._sentence = torch.zeros(0, dtype=torch.float64, device=self._device)
        self._text = ""
        self._words, self._alignments = [], []

        sentence_count = 0
        total_samples = 0

        # 3) 문장 단위로 반복해서 붙이기
        while sentence_count < num_sentences and total_samples < max_samples_in_sentence:
            # 화자 ID 선택 (기존 스키마 유지)
            speaker_id = speaker_ids[speaker_turn]
            # `speaker_wav_align_map`의 키는 보통 str 이라 안전하게 str 변환
            speaker_id_str = str(speaker_id)

            if speaker_id_str not in speaker_wav_align_map:
                break

            manifest_list = speaker_wav_align_map[speaker_id_str]
            if not manifest_list:
                break

            # 하나의 문장(매니페스트 row)을 랜덤 선택
            audio_manifest = manifest_list[np.random.randint(0, len(manifest_list))]

            try:
                # 전체 오디오 파일 읽기 (문장 전체 사용)
                segment = AudioSegment.from_file(audio_file=audio_manifest["audio_filepath"])
                audio = torch.from_numpy(segment.samples).to(self._device)
                sr = segment.sample_rate

                # 스테레오 → 모노
                if audio.ndim > 1:
                    audio = torch.mean(audio, 1, False).to(self._device)

                # (선택) 샘플레이트 차이가 나면 간단한 리샘플링
                target_sr = self._params.data_simulator.sr
                if sr != target_sr:
                    ratio = target_sr / sr
                    new_len = int(len(audio) * ratio)
                    idx = torch.linspace(0, len(audio) - 1, new_len).long()
                    audio = audio[idx]
                    sr = target_sr

                # 세션 내에서 이 턴이 차지할 수 있는 최대 샘플 수
                remaining = max_samples_in_sentence - total_samples
                if remaining <= 0:
                    break

                # 남은 공간보다 길면 잘라서 사용
                if len(audio) > remaining:
                    audio = audio[:remaining]

                # 세그먼트 증강 (기존 로직 그대로 사용)
                if self._params.data_simulator.segment_augmentor.add_seg_aug:
                    audio = perturb_audio(audio, sr, self.segment_augmentor, device=self._device)

                # 오디오 붙이기
                self._sentence = torch.cat((self._sentence, audio), dim=0)

                # 텍스트 붙이기 (있을 때만)
                if "text" in audio_manifest:
                    if self._text:
                        self._text += " " + str(audio_manifest["text"])
                    else:
                        self._text = str(audio_manifest["text"])

                # 단어 / alignment 붙이기 (있을 때만)
                if "words" in audio_manifest and "alignments" in audio_manifest:
                    offset_time = total_samples / float(sr)
                    words = audio_manifest["words"]
                    aligns = audio_manifest["alignments"]
                    for w, a in zip(words, aligns):
                        if w:  # 빈 문자열 제외
                            self._words.append(w)
                            self._alignments.append(offset_time + float(a))

                # 카운터 갱신
                added = len(audio)
                total_samples += added
                sentence_count += 1

            except Exception as e:
                logging.warning(
                    f"Sentence-level: 오디오 로드 실패: {audio_manifest.get('audio_filepath', 'unknown')}, {e}"
                )
                continue

        # 4) 화자별 정규화 (기존 로직 동일) - per_speaker_normalize 사용
        if (
            self._params.data_simulator.session_params.normalize
            and self._sentence.numel() > 0
            and torch.max(torch.abs(self._sentence)) > 0
        ):
            splits = get_split_points_in_alignments(
                words=self._words,
                alignments=self._alignments,
                split_buffer=self._params.data_simulator.session_params.split_buffer,
                sr=self._params.data_simulator.sr,
                sentence_audio_len=len(self._sentence),
            )
            self._sentence = per_speaker_normalize(
                sentence_audio=self._sentence,
                splits=splits,
                speaker_turn=speaker_turn,
                volume=self._volume,
                device=self._device,
            )


def load_base_config(config_file: str | None) -> OmegaConf:
    """
    기본 NeMo `data_simulator.yaml`을 불러오거나,
    없으면 최소 cfg를 생성.
    """
    if config_file is not None and os.path.exists(config_file):
        return OmegaConf.load(config_file)

    # NeMo 기본 conf 경로 시도
    default_cfg_path = os.path.join(NEMO_ROOT, "tools", "speech_data_simulator", "conf", "data_simulator.yaml")
    if os.path.exists(default_cfg_path):
        return OmegaConf.load(default_cfg_path)

    # 그래도 없으면 최소 설정 생성 (원래 값과 동일한 구조)
    logging.warning("기본 NeMo data_simulator.yaml을 찾지 못해 최소 설정으로 진행합니다.")
    cfg = OmegaConf.create(
        {
            "data_simulator": {
                "manifest_filepath": "???",
                "sr": 16000,
                "random_seed": 42,
                "multiprocessing_chunksize": 10000,
                "session_config": {
                    "num_speakers": 4,
                    "num_sessions": 10,
                    "session_length": 600,
                },
                "session_params": {
                    "max_audio_read_sec": 20.0,
                    "max_sentences_per_turn": 3,
                    "sentence_length_params": [0.4, 0.05],
                    "dominance_var": 0.11,
                    "min_dominance": 0.05,
                    "turn_prob": 0.875,
                    "min_turn_prob": 0.5,
                    "mean_silence": 0.15,
                    "mean_silence_var": 0.01,
                    "per_silence_var": 900,
                    "per_silence_min": 0.0,
                    "per_silence_max": -1,
                    "mean_overlap": 0.1,
                    "mean_overlap_var": 0.01,
                    "per_overlap_var": 900,
                    "per_overlap_min": 0.0,
                    "per_overlap_max": -1,
                    "start_window": True,
                    "window_type": "hamming",
                    "window_size": 0.05,
                    "start_buffer": 0.1,
                    "split_buffer": 0.1,
                    "release_buffer": 0.1,
                    "normalize": True,
                    "normalization_type": "equal",
                    "normalization_var": 0.1,
                    "min_volume": 0.75,
                    "max_volume": 1.25,
                    "end_buffer": 0.5,
                },
                "outputs": {
                    "output_dir": "???",
                    "output_filename": "multispeaker_session",
                    "overwrite_output": True,
                    "output_precision": 3,
                },
                "background_noise": {
                    "add_bg": False,
                    "background_manifest": None,
                    "num_noise_files": 10,
                    "snr": 60,
                },
                "segment_augmentor": {
                    "add_seg_aug": False,
                },
                "session_augmentor": {
                    "add_sess_aug": False,
                },
                "speaker_enforcement": {
                    "enforce_num_speakers": True,
                    "enforce_time": [0.25, 0.75],
                },
            }
        }
    )
    return cfg


def main():
    parser = argparse.ArgumentParser(description="문장 단위 다중 화자 오디오 세션 시뮬레이터")
    parser.add_argument(
        "--manifest_filepath",
        type=str,
        required=True,
        help="입력 단일화자 manifest 파일 경로 (기존 JSON 그대로 사용 가능)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="출력 디렉토리 (wav/json/rttm 저장 위치)",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        help="출력 파일 이름 prefix (None이면 YAML 설정 사용)",
    )
    parser.add_argument(
        "--num_speakers",
        type=int,
        default=None,
        help="세션당 화자 수 (None이면 YAML 설정 사용)",
    )
    parser.add_argument(
        "--num_sessions",
        type=int,
        default=None,
        help="생성할 세션 수 (None이면 YAML 설정 사용)",
    )
    parser.add_argument(
        "--session_length",
        type=float,
        default=None,
        help="세션 길이(초) (None이면 YAML 설정 사용)",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="사용할 data_simulator.yaml 경로 (기본: NeMo/tools/speech_data_simulator/conf/data_simulator.yaml)",
    )
    parser.add_argument(
        "--no_enforce_speakers",
        action="store_true",
        help="세션 길이(session_length)를 엄격히 지키려면 사용. 이 옵션을 쓰면 5명이 다 나오기 전에 세션이 끝날 수 있음.",
    )
    parser.add_argument("--max_turn_sec", type=float, default=None, help="한 턴 최대 길이(초), 예: 18")
    parser.add_argument(
        "--max_sentences_per_turn",
        type=int,
        default=3,
        help="한 턴에 넣을 최대 문장 수. 1~N 균등 분포 (기본 3). 0이면 음이항 분포 사용",
    )
    parser.add_argument("--mean_silence", type=float, default=None, help="침묵 비율 (0~1). 기본 0.15")
    parser.add_argument("--mean_overlap", type=float, default=None, help="중첩 비율 (0~1, 0.15 이하 권장). 기본 0.1")
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="랜덤 시드. 지정 안 하면 매 실행마다 다른 시드 사용(다양성 보장). 재현 필요 시 숫자 지정.",
    )

    args = parser.parse_args()

    # 1) 기본 cfg 로드 (기존 YAML 사용)
    cfg = load_base_config(args.config_file)

    # 1-1) random_seed: 미지정 시 매 실행마다 다른 시드 (sil/ov 다르게 여러 번 생성 시 화자 다양성 보장)
    if args.random_seed is not None:
        cfg.data_simulator.random_seed = args.random_seed
        seed = args.random_seed
    else:
        # np.random.seed는 0 ~ 2**32-1 범위만 허용
        raw = int(time.time() * 1000) + random.randint(0, 99999)
        seed = raw % (2**32)
        cfg.data_simulator.random_seed = seed
        print(f"random_seed 미지정 → 시드 {seed} 사용 (실행마다 다른 화자/문장 조합)", flush=True)

    # 2) 필수 값 덮어쓰기
    cfg.data_simulator.manifest_filepath = args.manifest_filepath
    cfg.data_simulator.outputs.output_dir = args.output_dir

    # 2-1) 사용된 시드를 출력 폴더에 저장
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    seed_file = os.path.join(out_dir, "random_seed.txt")
    with open(seed_file, "w", encoding="utf-8") as f:
        f.write(f"random_seed={seed}\n")
        f.write(f"manifest={args.manifest_filepath}\n")
        f.write(f"mean_silence={args.mean_silence}\n")
        f.write(f"mean_overlap={args.mean_overlap}\n")
        f.write(f"num_speakers={args.num_speakers}\n")
        f.write(f"num_sessions={args.num_sessions}\n")
        f.write(f"session_length={args.session_length}\n")
    print(f"시드 저장: {seed_file}", flush=True)

    # 3) 선택 값들만 필요할 때 덮어쓰기
    if args.output_filename is not None:
        cfg.data_simulator.outputs.output_filename = args.output_filename
    if args.num_speakers is not None:
        cfg.data_simulator.session_config.num_speakers = args.num_speakers
    if args.num_sessions is not None:
        cfg.data_simulator.session_config.num_sessions = args.num_sessions
    if args.session_length is not None:
        cfg.data_simulator.session_config.session_length = args.session_length
    if args.no_enforce_speakers:
        if "speaker_enforcement" not in cfg.data_simulator:
            cfg.data_simulator["speaker_enforcement"] = {}
        cfg.data_simulator.speaker_enforcement.enforce_num_speakers = False
    if args.max_turn_sec is not None and args.max_turn_sec > 0:
        if "session_params" not in cfg.data_simulator:
            cfg.data_simulator["session_params"] = {}
        cfg.data_simulator.session_params.max_turn_duration_sec = args.max_turn_sec
    if args.max_sentences_per_turn is not None:
        if "session_params" not in cfg.data_simulator:
            cfg.data_simulator["session_params"] = {}
        cfg.data_simulator.session_params.max_sentences_per_turn = args.max_sentences_per_turn
    if args.mean_silence is not None and 0 <= args.mean_silence < 1:
        cfg.data_simulator.session_params.mean_silence = args.mean_silence
    if args.mean_overlap is not None and 0 <= args.mean_overlap < 1:
        cfg.data_simulator.session_params.mean_overlap = args.mean_overlap
        # mean_overlap=0 이거나 var가 Beta 제약(mean*(1-mean))을 넘으면 var=0으로 고정
        mv = cfg.data_simulator.session_params.mean_overlap_var
        mm = args.mean_overlap
        if mv <= 0 or mv >= mm * (1 - mm):
            cfg.data_simulator.session_params.mean_overlap_var = 0.0

    # 4) 시뮬레이터 실행 (RIR는 여기서는 사용하지 않고 base 클래스만 사용)
    simulator = SentenceLevelMultiSpeakerSimulator(cfg=cfg)
    simulator.generate_sessions()

    logging.info(f"문장 단위 합성 완료! 결과는 {cfg.data_simulator.outputs.output_dir}에 저장되었습니다.")


if __name__ == "__main__":
    main()
