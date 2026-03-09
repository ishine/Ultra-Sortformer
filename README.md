# Ultra-Sortformer

This project extends **NVIDIA Sortformer** speaker diarization from **4 speakers to 8 speakers**. The original Sortformer supports up to 4 speakers; we modified the model architecture and trained it to handle 5–8 speakers.

## Hugging Face

The 8-speaker model is available on Hugging Face:

> 🔗 **Model**: [Hugging Face - Ultra-Sortformer 8spk](https://huggingface.co/) *(add link after upload)*  
> 📂 **Code & experiments**: [GitHub - Ultra-Sortformer](https://github.com/LilDevsy0117/Ultra-Sortformer)

### Model Versions

| Version | Date       | Val F1 | Notes                    |
|---------|------------|--------|--------------------------|
| v1.0.0  | 2025-03-09 | 0.9884 | First release (best_09884) |

## Project Structure

```
├── configs/                 # NeMo training YAML configs
├── scripts/                 # Inference, model extension, data generation
├── streaming_sortformer_diar_train/  # 5spk, 8spk training experiments (hparams)
└── NeMo/                    # NeMo fork (clone separately, modified sortformer_diar_models)
```

> **NeMo**: ~5.9GB, not included in this repo. Clone [NeMo](https://github.com/NVIDIA/NeMo) separately and apply the modified `sortformer_diar_models.py` etc.

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/diarize_inference.py` | Speaker diarization inference with Sortformer |
| `scripts/extend_sortformer_4spk_to_5spk.py` | Extend 4spk → 5spk/Nspk (orthogonal initialization) |
| `scripts/create_path_files.py` | Path file generation |
| `scripts/generate_rttm.py` | RTTM file generation |
| `scripts/sentence_level_multispeaker_simulator.py` | Multispeaker synthetic data simulator |
| `scripts/ckpt_to_nemo.py` | Convert .ckpt to .nemo for Hugging Face upload |

## Usage

### Inference
```bash
python scripts/diarize_inference.py --model_path <path_to.nemo> --audio_dir <dir> --output_dir <dir>
```

### Extend 4spk → 8spk
```bash
python scripts/extend_sortformer_4spk_to_5spk.py --src <4spk.nemo> --dst_config <8spk_config> --out <8spk.nemo>
```

### Convert checkpoint to .nemo (for Hugging Face)
```bash
python scripts/ckpt_to_nemo.py --ckpt path/to/best.ckpt --out ultra_sortformer_8spk_v1.0.0.nemo
```

## Requirements

- NeMo (referenced in scripts)
- PyTorch, omegaconf

## Model Updates

When you release a new version:

1. Update `CHANGELOG.md` with the new version and changes
2. Update the "Model Versions" table above
3. Update `VERSION` file
4. Upload the new model to Hugging Face
5. (Optional) Create a git tag: `git tag v1.1.0 && git push origin v1.1.0`

## License

Based on NVIDIA NeMo. See the relevant license for terms.
