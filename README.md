# Ultra-Sortformer

**NVIDIA Sortformer** 화자 다이어리제이션 모델을 **4명 → 8명**으로 확장한 실험 프로젝트입니다.

기존 Sortformer는 최대 4명의 화자만 구분할 수 있어, 5~8명까지 확장 가능하도록 모델 구조를 수정하고 학습한 과정을 담고 있습니다.

## Hugging Face

8명 확장 모델은 Hugging Face에 공개됩니다:

> 🔗 **모델**: [Hugging Face - Ultra-Sortformer 8spk](https://huggingface.co/) *(업로드 후 링크 추가)*  
> 📂 **실험 코드/과정**: [GitHub - Ultra-Sortformer](https://github.com/LilDevsy0117/Ultra-Sortformer)

## 프로젝트 구조

```
├── configs/                 # NeMo 학습용 YAML 설정
├── scripts/                 # 추론, 모델 확장, 데이터 생성 등
├── streaming_sortformer_diar_train/  # 5spk, 8spk 트레이닝 실험 (hparams)
└── NeMo/                    # NeMo 포크 (별도 클론, 수정된 sortformer_diar_models)
```

> **NeMo**: 약 5.9GB로 Git에 포함되지 않습니다. [NeMo](https://github.com/NVIDIA/NeMo)를 별도 클론한 뒤 수정된 `sortformer_diar_models.py` 등을 적용하세요.

## 주요 스크립트

| 스크립트 | 설명 |
|---------|------|
| `scripts/diarize_inference.py` | Sortformer 모델로 화자 다이어리제이션 추론 |
| `scripts/extend_sortformer_4spk_to_5spk.py` | 4spk → 5spk/Nspk 모델 확장 (직교 초기화) |
| `scripts/create_path_files.py` | 경로 파일 생성 |
| `scripts/generate_rttm.py` | RTTM 파일 생성 |
| `scripts/sentence_level_multispeaker_simulator.py` | 멀티스피커 합성 데이터 시뮬레이터 |

## 사용법

### 추론
```bash
python scripts/diarize_inference.py --model_path <path_to.nemo> --audio_dir <dir> --output_dir <dir>
```

### 4spk → 8spk 모델 확장
```bash
python scripts/extend_sortformer_4spk_to_5spk.py --src <4spk.nemo> --dst_config <8spk_config> --out <8spk.nemo>
```

## 요구사항

- NeMo (scripts에서 `NeMo` 경로 참조)
- PyTorch, omegaconf

## 라이선스

NVIDIA NeMo 기반. 관련 라이선스를 확인하세요.
