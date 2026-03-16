---
language: en
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
base_model: nvidia/diar_streaming_sortformer_4spk-v2.1
tags:
  - speaker-diarization
  - diarization
  - speech
  - nemo
  - sortformer
  - streaming
  - multilingual
---

# Ultra Diar Streaming Sortformer (5-Speaker)

This model extends **NVIDIA Streaming Sortformer** speaker diarization from **4 speakers to 5 speakers**. The original [diar_streaming_sortformer_4spk-v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) supports up to 4 speakers; this model expands the capability to handle 5 speakers through fine-tuning and architectural modifications.

## Model Details

- **Base model**: [nvidia/diar_streaming_sortformer_4spk-v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)
- **Extension**: 4spk → 5spk
- **Framework**: NeMo (NVIDIA)
- **Version**: 1.0

## Code & Training

The experimental pipeline, training scripts, and inference code will be made public on GitHub at a later date. Currently available only on Hugging Face.

### Training

This model was trained on **2× NVIDIA H100 GPUs**. We use synthetic data with 2–5 speakers.

## Usage

This model requires the **NVIDIA NeMo toolkit** to train, fine-tune, or perform diarization. Install NeMo after installing Cython and the latest PyTorch.

### Install NeMo

```bash
apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython packaging
pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]
```

### Quick Start: Run Diarization

```python
from nemo.collections.asr.models import SortformerEncLabelModel

# Load model from Hugging Face
diar_model = SortformerEncLabelModel.from_pretrained("devsy0117/ultra_diar_streaming_sortformer_5spk_v1")
diar_model.eval()

# Streaming parameters (recommended for best performance)
diar_model.sortformer_modules.chunk_len = 340
diar_model.sortformer_modules.chunk_right_context = 40
diar_model.sortformer_modules.fifo_len = 40
diar_model.sortformer_modules.spkcache_update_period = 300

# Run diarization
predicted_segments = diar_model.diarize(audio=["/path/to/your/audio.wav"], batch_size=1)

for segment in predicted_segments[0]:
    print(segment)
```

### Loading the Model

```python
from nemo.collections.asr.models import SortformerEncLabelModel

# Option 1: Load directly from Hugging Face
diar_model = SortformerEncLabelModel.from_pretrained("devsy0117/ultra_diar_streaming_sortformer_5spk_v1")

# Option 2: Load from a downloaded .nemo file
diar_model = SortformerEncLabelModel.restore_from(
    restore_path="/path/to/ultra_diar_streaming_sortformer_5spk_v1.0.nemo",
    map_location="cuda",
    strict=False,
)

diar_model.eval()
```

### Input Format

- Single audio file: `audio_input="/path/to/multispeaker_audio.wav"`
- Multiple files: `audio_input=["/path/to/audio1.wav", "/path/to/audio2.wav"]`

## Evaluation Results

Comparison with the base model ([diar_streaming_sortformer_4spk-v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)) on AliMeeting and AMI benchmarks.

### Evaluation Parameters

| Parameter | Value |
|-----------|-------|
| Post-processing | None |
| Collar | 0.25 s |
| Ignore overlap | False |
| Chunk size | 340 frames |
| Batch size | 1 |

### AliMeeting (test)

| Model | DER | FA | MISS | CER | Spk_Count_Acc |
|-------|-----|----|------|-----|---------------|
| diar_streaming_sortformer_4spk-v2.1 (base) | **11.03%** | 0.40% | 9.93% | 0.70% | **95.00%** |
| ultra_diar_streaming_sortformer_5spk_v1.0 (ours) | **5.85%** | 1.03% | 3.80% | 1.01% | 65.00% |

### AMI IHM (test)

| Model | DER | FA | MISS | CER | Spk_Count_Acc |
|-------|-----|----|------|-----|---------------|
| diar_streaming_sortformer_4spk-v2.1 (base) | **26.05%** | 0.50% | 23.51% | 2.03% | **93.75%** |
| ultra_diar_streaming_sortformer_5spk_v1.0 (ours) | **10.98%** | 1.48% | 7.79% | 1.71% | 68.75% |

### AMI SDM (test)

| Model | DER | FA | MISS | CER | Spk_Count_Acc |
|-------|-----|----|------|-----|---------------|
| diar_streaming_sortformer_4spk-v2.1 (base) | **28.29%** | 0.82% | 23.76% | 3.72% | **93.75%** |
| ultra_diar_streaming_sortformer_5spk_v1.0 (ours) | **14.33%** | 2.09% | 8.33% | 3.91% | 87.50% |

### CallHome (test)

| Model | eng DER | deu DER | jpn DER | spa DER | zho DER | eng Spk_Acc | deu Spk_Acc | jpn Spk_Acc | spa Spk_Acc | zho Spk_Acc |
|-------|---------|---------|---------|---------|---------|-------------|-------------|-------------|-------------|-------------|
| diar_streaming_sortformer_4spk-v2.1 (base) | **4.94%** | **6.70%** | **10.03%** | 23.27% | **7.15%** | 83.57% | 80.83% | 79.17% | 63.57% | 72.86% |
| ultra_diar_streaming_sortformer_5spk_v1.0 (ours) | 7.39% | 6.98% | 10.59% | **17.92%** | 9.24% | **87.86%** | **86.67%** | **83.33%** | **72.14%** | **72.86%** |

> **Note**: The base model (v2.1) is trained for up to 4 speakers. The lower `Spk_Count_Acc` of this model on AliMeeting and AMI IHM reflects sessions with ≤4 speakers being predicted as 5, a trade-off from extending to 5-speaker support. DER improves significantly due to reduced MISS rate.

## License

This model is a derivative of NVIDIA Sortformer, licensed under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

**Attribution**: Licensed by NVIDIA Corporation under the NVIDIA Open Model License.
