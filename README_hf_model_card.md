---
language: en
license: other
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

# Ultra Diar Streaming Sortformer (8-Speaker)

This model extends **NVIDIA Streaming Sortformer** speaker diarization from **4 speakers to 8 speakers**. The original [diar_streaming_sortformer_4spk-v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) supports up to 4 speakers; this model expands the capability to handle 5–8 speakers through fine-tuning and architectural modifications.

## Model Details

- **Base model**: [nvidia/diar_streaming_sortformer_4spk-v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)
- **Extension**: 4spk → 8spk
- **Framework**: NeMo (NVIDIA)
- **Version**: 1.0.0

## Code & Training

The experimental pipeline, training scripts, and inference code will be made public on GitHub at a later date. Currently available only on Hugging Face.

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

# Load model from Hugging Face (requires Hugging Face token for gated models)
diar_model = SortformerEncLabelModel.from_pretrained("devsy0117/ultra_diar_streaming_sortformer_8spk_v1.0.0")
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

# Option 1: Load directly from Hugging Face (requires Hugging Face token)
diar_model = SortformerEncLabelModel.from_pretrained("devsy0117/ultra_diar_streaming_sortformer_8spk_v1.0.0")

# Option 2: Load from a downloaded .nemo file
diar_model = SortformerEncLabelModel.restore_from(
    restore_path="/path/to/ultra_diar_streaming_sortformer_8spk_v1.0.0.nemo",
    map_location="cuda",
    strict=False,
)

diar_model.eval()
```

### Input Format

- Single audio file: `audio_input="/path/to/multispeaker_audio.wav"`
- Multiple files: `audio_input=["/path/to/audio1.wav", "/path/to/audio2.wav"]`

## License

This model is a derivative of NVIDIA Sortformer, licensed under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

**Attribution**: Licensed by NVIDIA Corporation under the NVIDIA Open Model License.
