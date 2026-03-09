# Hugging Face Model Card Content (English)

Copy the content below into your Hugging Face model card (README.md).

---

## Short version (if space is limited)

# Ultra-Sortformer (8-Speaker)

Extends NVIDIA Sortformer from 4 to 8 speakers for speaker diarization.

**GitHub (code, training, inference)**: https://github.com/LilDevsy0117/Ultra-Sortformer

---

## Full version

# Ultra-Sortformer (8-Speaker)

This model extends **NVIDIA Sortformer** speaker diarization from **4 speakers to 8 speakers**. The original Sortformer supports up to 4 speakers; this model expands the capability to handle 5–8 speakers through fine-tuning and architectural modifications.

## Model Details

- **Base model**: NVIDIA Sortformer (streaming speaker diarization)
- **Extension**: 4spk → 8spk
- **Framework**: NeMo (NVIDIA)
- **Version**: 1.0.0 (see [CHANGELOG](https://github.com/LilDevsy0117/Ultra-Sortformer/blob/main/CHANGELOG.md) for updates)

## GitHub

For the full experimental pipeline, training scripts, inference code, and hyperparameters:

- **Repository**: [https://github.com/LilDevsy0117/Ultra-Sortformer](https://github.com/LilDevsy0117/Ultra-Sortformer)
- Inference script: `scripts/diarize_inference.py`
- Model extension script: `scripts/extend_sortformer_4spk_to_5spk.py`
- Training configs: 5spk, 8spk experiments in `streaming_sortformer_diar_train/`

## Usage

```python
# Inference script (requires NeMo)
from nemo.collections.asr.models import SortformerEncLabelModel

model = SortformerEncLabelModel.restore_from("path/to/ultra_sortformer_8spk.nemo")
# Run diarization on your audio...
```

Or via command line:

```bash
python scripts/diarize_inference.py --model_path <path_to.nemo> --audio_dir <dir> --output_dir <dir>
```

## License

Based on NVIDIA NeMo. See the relevant license for terms.

---
