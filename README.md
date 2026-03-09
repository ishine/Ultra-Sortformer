# Ultra Diar Streaming Sortformer (8-Speaker)

This project extends **NVIDIA Streaming Sortformer** speaker diarization from **4 speakers to 8 speakers**. The original [diar_streaming_sortformer_4spk-v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) supports up to 4 speakers; we modified the model architecture and trained it to handle 5–8 speakers.

## Hugging Face

The 8-speaker model is available on Hugging Face:

> 🔗 **Model**: [devsy0117/ultra_diar_streaming_sortformer_8spk_v1.0.0](https://huggingface.co/devsy0117/ultra_diar_streaming_sortformer_8spk_v1.0.0)  
> 📂 **Code & experiments**: [GitHub - Ultra-Sortformer](https://github.com/LilDevsy0117/Ultra-Sortformer) *(will be public later)*

### Model Versions

| Version | Date       | Val F1 | Notes                    |
|---------|------------|--------|--------------------------|
| v1.0.0  | 2025-03-09 | 0.9884 | First release (best_09884) |

## Project Structure

```
├── configs/                 # NeMo training YAML configs
├── scripts/                 # Inference, model extension, data generation
├── streaming_sortformer_diar_train/  # 5spk, 8spk training experiments (hparams)
├── test/                    # Sample audio for testing
└── NeMo/                    # NeMo fork (clone separately, modified sortformer_diar_models)
```

> **NeMo**: ~5.9GB, not included in this repo. Clone [NeMo](https://github.com/NVIDIA/NeMo) separately and apply the modified `sortformer_diar_models.py` etc.

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/diarize_inference.py` | Speaker diarization inference with Sortformer |
| `scripts/extend_sortformer_4spk_to_5spk.py` | Extend 4spk → 5spk/Nspk (orthogonal initialization) |
| `scripts/test_diar_model.py` | Quick test with `test/` audio (uses HF model) |
| `scripts/create_path_files.py` | Path file generation |
| `scripts/generate_rttm.py` | RTTM file generation |
| `scripts/sentence_level_multispeaker_simulator.py` | Multispeaker synthetic data simulator |
| `scripts/ckpt_to_nemo.py` | Convert .ckpt to .nemo for Hugging Face upload |

## Usage

### Quick Start (from Hugging Face)

```python
from nemo.collections.asr.models import SortformerEncLabelModel

diar_model = SortformerEncLabelModel.from_pretrained("devsy0117/ultra_diar_streaming_sortformer_8spk_v1.0.0")
diar_model.eval()

# Streaming parameters (recommended)
diar_model.sortformer_modules.chunk_len = 340
diar_model.sortformer_modules.chunk_right_context = 40
diar_model.sortformer_modules.fifo_len = 40
diar_model.sortformer_modules.spkcache_update_period = 300

predicted_segments = diar_model.diarize(audio=["/path/to/your/audio.wav"], batch_size=1)
for segment in predicted_segments[0]:
    print(segment)
```

### Test script (uses `test/` audio)

```bash
python scripts/test_diar_model.py
```

### Inference (full script)

```bash
python scripts/diarize_inference.py --model_path <path_to.nemo> --audio_dir <dir> --output_dir <dir>
```

### Extend 4spk → 8spk

```bash
python scripts/extend_sortformer_4spk_to_5spk.py --src <4spk.nemo> --dst_config <8spk_config> --out <8spk.nemo>
```

### Convert checkpoint to .nemo

```bash
python scripts/ckpt_to_nemo.py --ckpt path/to/best.ckpt --out ultra_diar_streaming_sortformer_8spk_v1.0.0.nemo
```

## Requirements

- **NeMo**: `pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]`
- PyTorch, Cython, libsndfile1, ffmpeg

## Model Updates

When you release a new version:

1. Update `CHANGELOG.md` with the new version and changes
2. Update the "Model Versions" table above
3. Update `VERSION` file
4. Upload the new model to Hugging Face
5. Update `README_hf_model_card.md` and re-upload
6. (Optional) Create a git tag: `git tag v1.1.0 && git push origin v1.1.0`

## License

This model is a derivative of NVIDIA Sortformer, licensed under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

**Attribution**: Licensed by NVIDIA Corporation under the NVIDIA Open Model License.
