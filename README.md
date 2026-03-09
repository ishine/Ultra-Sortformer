# Streaming Sortformer Speaker Diarization

NeMo 기반 Streaming Sortformer 화자 다이어리제이션 프로젝트입니다.

## 구조

```
workspace/
├── configs/                 # NeMo 학습용 YAML 설정
├── NeMo/                    # NeMo 포크 (별도 클론, 수정된 sortformer_diar_models 포함)
├── scripts/                 # 추론, 데이터 생성 등 유틸리티
├── streaming_sortformer_diar_train/  # 트레이닝 설정 (hparams)
└── synthetic_data/          # 합성 데이터 params (실제 오디오는 제외)
```

> **NeMo**: 약 5.9GB로 Git에 포함되지 않습니다. [NeMo](https://github.com/NVIDIA/NeMo)를 별도 클론한 뒤 `sortformer_diar_models.py` 등 수정된 파일을 적용하세요.

## 주요 스크립트

| 스크립트 | 설명 |
|---------|------|
| `scripts/diarize_inference.py` | Sortformer 모델로 화자 다이어리제이션 추론 |
| `scripts/extend_sortformer_4spk_to_5spk.py` | 4spk → 5spk/Nspk 모델 확장 (직교 초기화) |
| `scripts/create_path_files.py` | 경로 파일 생성 |
| `scripts/generate_rttm.py` | RTTM 파일 생성 |
| `scripts/sentence_level_multispeaker_simulator.py` | 멀티스피커 시뮬레이터 |

## 요구사항

- NeMo (scripts에서 `NeMo` 경로 참조)
- PyTorch, omegaconf

## 사용법

### 추론
```bash
python scripts/diarize_inference.py --model_path <path_to.nemo> --audio_dir <dir> --output_dir <dir>
```

### 4spk → 5spk 모델 확장
```bash
python scripts/extend_sortformer_4spk_to_5spk.py --src <4spk.nemo> --dst_config <5spk_config> --out <5spk.nemo>
```

## 제외된 항목 (Git)

- `data/`, `synthetic_data/` (오디오): 대용량 데이터
- `*.nemo`, `models/`: 모델 체크포인트
- `output/`, `results/`: 추론 결과
- `venv/`: 가상환경
- `mago-speaker-diarization/`: 외부 프로젝트
