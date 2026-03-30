#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentence-level multi-speaker session simulator (NeMo MultiSpeakerSimulator subclass).

Only `_build_sentence` differs from NeMo: concatenates whole manifest rows (utterances)
instead of word-level chunks. Silence/overlap/session layout follow `data_simulator.yaml`
(via `--config_file` or NeMo `tools/speech_data_simulator/conf/data_simulator.yaml`).

Example:
    cd NeMo && python ../scripts/sentence_level_multispeaker_simulator.py \\
      --manifest_filepath /path/to/manifest.json \\
      --output_dir /path/to/out
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
from omegaconf import OmegaConf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NEMO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "NeMo"))
if os.path.isdir(NEMO_ROOT):
    sys.path.insert(0, NEMO_ROOT)

from nemo.collections.asr.data.data_simulation import MultiSpeakerSimulator
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.asr.parts.utils.data_simulation_utils import (
    get_split_points_in_alignments,
    per_speaker_normalize,
    perturb_audio,
)

def linear_resample_audio(audio: torch.Tensor, sr_in: float, sr_out: float) -> torch.Tensor:
    """Nearest-index resample (same as previous inline implementation)."""
    if sr_in == sr_out or len(audio) == 0:
        return audio
    ratio = sr_out / sr_in
    new_len = int(len(audio) * ratio)
    idx = torch.linspace(0, len(audio) - 1, new_len).long()
    return audio[idx]


def load_utterance_mono_tensor(
    audio_filepath: str,
    target_sr: int,
    device: torch.device,
) -> torch.Tensor:
    """Load full file, mono, resampled to target_sr. Raises on I/O or decode errors."""
    segment = AudioSegment.from_file(audio_file=audio_filepath)
    audio = torch.from_numpy(segment.samples).to(device)
    file_sr = float(segment.sample_rate)
    if audio.ndim > 1:
        audio = torch.mean(audio, dim=1)
    return linear_resample_audio(audio, file_sr, float(target_sr))


def om_ensure_subtree(cfg_node: Any, key: str) -> None:
    """Ensure cfg_node[key] exists as a mutable mapping (OmegaConf-compatible)."""
    if key not in cfg_node or cfg_node[key] is None:
        cfg_node[key] = {}


def load_base_config(config_file: str | None) -> OmegaConf:
    if config_file is not None and os.path.isfile(config_file):
        return OmegaConf.load(config_file)
    default_cfg_path = os.path.join(
        NEMO_ROOT, "tools", "speech_data_simulator", "conf", "data_simulator.yaml"
    )
    if os.path.isfile(default_cfg_path):
        return OmegaConf.load(default_cfg_path)
    raise FileNotFoundError(
        "data_simulator.yaml not found. Use --config_file /path/to/data_simulator.yaml "
        f"or place NeMo at {NEMO_ROOT} so the default exists:\n  {default_cfg_path}"
    )


class SentenceLevelMultiSpeakerSimulator(MultiSpeakerSimulator):
    """MultiSpeakerSimulator with utterance-level `_build_sentence` only."""

    def _build_sentence(
        self,
        speaker_turn: int,
        speaker_ids: List[str],
        speaker_wav_align_map: Dict[str, list],
        max_samples_in_sentence: int,
    ) -> None:
        ds = self._params.data_simulator
        target_sr = int(ds.sr)

        max_turn_sec = ds.session_params.get("max_turn_duration_sec", None)
        if max_turn_sec is not None and max_turn_sec > 0:
            max_samples_in_sentence = min(max_samples_in_sentence, int(max_turn_sec * target_sr))

        max_sent = ds.session_params.get("max_sentences_per_turn", None)
        if max_sent is not None and max_sent >= 1:
            num_sentences = int(np.random.randint(1, int(max_sent) + 1))
        else:
            slp = ds.session_params.sentence_length_params
            num_sentences = int(np.random.negative_binomial(slp[0], slp[1]) + 1)

        self._sentence = torch.zeros(0, dtype=torch.float64, device=self._device)
        self._text = ""
        self._words, self._alignments = [], []

        sentence_count = 0
        total_samples = 0

        while sentence_count < num_sentences and total_samples < max_samples_in_sentence:
            speaker_id_str = str(speaker_ids[speaker_turn])
            if speaker_id_str not in speaker_wav_align_map:
                break
            manifest_list = speaker_wav_align_map[speaker_id_str]
            if not manifest_list:
                break

            audio_manifest = manifest_list[int(np.random.randint(0, len(manifest_list)))]
            path = audio_manifest["audio_filepath"]

            audio = load_utterance_mono_tensor(path, target_sr, self._device)

            remaining = max_samples_in_sentence - total_samples
            if remaining <= 0:
                break
            if len(audio) > remaining:
                audio = audio[:remaining]

            if ds.segment_augmentor.add_seg_aug:
                audio = perturb_audio(
                    audio, float(target_sr), self.segment_augmentor, device=self._device
                )

            self._sentence = torch.cat((self._sentence, audio), dim=0)

            if "text" in audio_manifest:
                t = str(audio_manifest["text"])
                if self._text:
                    self._text += " " + t
                else:
                    self._text = t

            if "words" in audio_manifest and "alignments" in audio_manifest:
                offset_time = total_samples / float(target_sr)
                for w, a in zip(audio_manifest["words"], audio_manifest["alignments"]):
                    if w:
                        self._words.append(w)
                        self._alignments.append(offset_time + float(a))

            total_samples += len(audio)
            sentence_count += 1

        if (
            ds.session_params.normalize
            and self._sentence.numel() > 0
            and torch.max(torch.abs(self._sentence)) > 0
        ):
            splits = get_split_points_in_alignments(
                words=self._words,
                alignments=self._alignments,
                split_buffer=ds.session_params.split_buffer,
                sr=target_sr,
                sentence_audio_len=len(self._sentence),
            )
            self._sentence = per_speaker_normalize(
                sentence_audio=self._sentence,
                splits=splits,
                speaker_turn=speaker_turn,
                volume=self._volume,
                device=self._device,
            )


def _apply_cli_overrides(cfg: OmegaConf, args: argparse.Namespace) -> None:
    ds = cfg.data_simulator
    if args.num_speakers is not None:
        ds.session_config.num_speakers = args.num_speakers
    if args.num_sessions is not None:
        ds.session_config.num_sessions = args.num_sessions
    if args.session_length is not None:
        ds.session_config.session_length = args.session_length
    if args.mean_silence is not None and 0 <= args.mean_silence < 1:
        om_ensure_subtree(ds, "session_params")
        ds.session_params.mean_silence = args.mean_silence
    if args.mean_overlap is not None and 0 <= args.mean_overlap < 1:
        om_ensure_subtree(ds, "session_params")
        ds.session_params.mean_overlap = args.mean_overlap
        mv = ds.session_params.mean_overlap_var
        mm = float(args.mean_overlap)
        if mv <= 0 or mv >= mm * (1 - mm):
            ds.session_params.mean_overlap_var = 0.0
    if args.max_sentences_per_turn is not None and args.max_sentences_per_turn >= 1:
        om_ensure_subtree(ds, "session_params")
        ds.session_params.max_sentences_per_turn = args.max_sentences_per_turn


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sentence-level multi-speaker session simulator")
    p.add_argument("--manifest_filepath", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--config_file", type=str, default=None)
    p.add_argument("--num_speakers", type=int, default=None)
    p.add_argument("--num_sessions", type=int, default=None)
    p.add_argument("--session_length", type=float, default=None)
    p.add_argument("--mean_silence", type=float, default=None)
    p.add_argument("--mean_overlap", type=float, default=None)
    p.add_argument(
        "--max_sentences_per_turn",
        "--max_sent",
        type=int,
        default=3,
        metavar="N",
        help="Per turn: concat 1..N random utterances (not negative-binomial count). Default 3.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = load_base_config(args.config_file)

    ds = cfg.data_simulator
    ds.manifest_filepath = args.manifest_filepath
    ds.outputs.output_dir = args.output_dir

    _apply_cli_overrides(cfg, args)

    SentenceLevelMultiSpeakerSimulator(cfg=cfg).generate_sessions()


if __name__ == "__main__":
    main()
