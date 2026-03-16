# Ultra-Sortformer: Extending NVIDIA Sortformer to N Speakers

[![Model on Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/devsy0117/ultra_diar_streaming_sortformer_8spk_v1.0.0)

This project documents the ongoing journey of extending **NVIDIA's Streaming Sortformer** speaker diarization model from 4 speakers toward 8 speakers — without retraining from scratch. The core idea is to surgically expand the output layer using orthogonal initialization, then fine-tune with differential learning rates to preserve existing knowledge while teaching the model new speakers.

> **Current status**: 5spk ✅ → 6spk 🔄 → 7spk (planned) → 8spk (planned)

---

## Table of Contents

1. [Background](#background)
2. [Architecture Overview](#architecture-overview)
3. [Extension Journey](#extension-journey)
   - [Step 1: Output Layer Extension (4 → 5spk)](#step-1-output-layer-extension-4--5spk)
   - [Step 2: Split Learning Rate Training](#step-2-split-learning-rate-training)
   - [Step 3: Layer Repeat Experiments (LLM Neuroanatomy)](#step-3-layer-repeat-experiments-llm-neuroanatomy)
   - [Step 4: Extending to 6–8 Speakers (In Progress)](#step-4-extending-to-6-7-8-speakers-in-progress-)
4. [Evaluation Results](#evaluation-results)
5. [Scripts](#scripts)
6. [Usage](#usage)
7. [Training](#training)
8. [Requirements](#requirements)

---

## Background

NVIDIA's [diar_streaming_sortformer_4spk-v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) is a streaming speaker diarization model based on the Sortformer architecture. It uses a **FastConformer encoder** (17 layers) followed by a **Transformer encoder** (18 layers) to produce per-frame speaker activity predictions. The final output layer is a single linear layer mapping from hidden states to N speaker probabilities.

The model is capable of real-time streaming diarization using a chunk-based speaker cache mechanism.

**Problem**: The model is hard-limited to 4 speakers. Any audio with 5+ speakers gets misidentified.

**Goal**: Extend the model to handle 5, 6, and 8 speakers while preserving performance on the original 2–4 speaker scenarios.

---

## Architecture Overview

```
Audio Input
    │
    ▼
┌─────────────────────────────┐
│  AudioToMelSpectrogramPreprocessor  │
└─────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  FastConformer Encoder       │  ← 17 layers, d_model=512
│  (NEST Encoder)              │    Subsampling factor: 8
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  Transformer Encoder         │  ← 18 layers, d_model=192
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  SortformerModules           │  ← Speaker Cache + Attention
│  (Streaming Speaker Cache)   │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  single_hidden_to_spks       │  ← Linear(192, N_spk)  ← KEY LAYER
└──────────────────────────────┘
    │
    ▼
  Per-frame speaker activity predictions  [batch, time, N_spk]
```

The output layer `single_hidden_to_spks` is a simple `nn.Linear(192, N_spk)`. Extending the model to handle more speakers means expanding this layer from N to M outputs.

---

## Extension Journey

### Step 1: Output Layer Extension (4 → 5spk)

**Script**: `scripts/extend_output_layer.py`

The naive approach — random weight initialization for new speaker neurons — catastrophically degrades performance on existing speakers. Instead, we use **SVD-based orthogonal initialization**.

#### How it works

Given the existing weight matrix `W ∈ ℝ^{N×H}` (N speakers, H=192 hidden dims):

```python
# SVD decomposition of existing weights
U, S, Vh = torch.linalg.svd(W, full_matrices=True)

# New speaker neurons = right singular vectors orthogonal to existing ones
# Vh[N], Vh[N+1], ... are orthogonal to the column space of W
new_row = Vh[N]  # fully orthogonal to existing N speakers

# Normalize to match existing neuron norms
avg_norm = W.norm(dim=1).mean()
new_row = new_row * (avg_norm / new_row.norm())
```

This ensures new speaker neurons start as far as possible from existing ones in the representation space, minimizing interference during fine-tuning.

The extended model is saved in **split mode** for differential learning rates:

```
single_hidden_to_spks_base  (N_base speakers)   ← preserved weights
single_hidden_to_spks_new   (N_new speakers)    ← new/extended weights
```

```bash
python scripts/extend_output_layer.py \
    --src diar_streaming_sortformer_4spk-v2.1.nemo \
    --dst-spk 5 \
    --out diar_streaming_sortformer_5spk_orthogonal.nemo
```

---

### Step 2: Split Learning Rate Training

**Key Insight**: When fine-tuning the extended model, using the same learning rate for all output neurons causes the model to "forget" its existing 2–4 speaker accuracy while trying to learn the 5th speaker.

#### The Problem

Early experiments showed that after fine-tuning with a uniform learning rate:
- val_2spk → val_4spk: accuracy degraded significantly
- The model started predicting 5 speakers on 3–4 speaker audio

#### The Solution: Differential Learning Rates

We split the output layer and apply different learning rates:

| Component | Learning Rate | Reason |
|-----------|-------------|--------|
| `single_hidden_to_spks_base` (spks 1–N) | `1e-5` | Preserve existing knowledge |
| `single_hidden_to_spks_new` (spk N+1) | `1e-4` | 10× higher: fast adaptation |
| All other parameters | `1e-5` | Normal fine-tuning |

This is implemented by overriding `setup_optimizer_param_groups` in `sortformer_diar_models.py`:

```python
def setup_optimizer_param_groups(self):
    sm = self.sortformer_modules
    n_base = getattr(sm, 'n_base_spks', 0)
    new_lr = self._cfg.get('optim_new_lr', None)

    if n_base > 0 and new_lr is not None and hasattr(sm, 'single_hidden_to_spks_new'):
        new_params = list(sm.single_hidden_to_spks_new.parameters())
        new_param_ids = {id(p) for p in new_params}
        base_params = [p for p in self.parameters() if id(p) not in new_param_ids]
        self._optimizer_param_groups = [
            {"params": base_params},
            {"params": new_params, "lr": new_lr},
        ]
```

#### Training Command (5spk example)

```bash
python NeMo/examples/speaker_tasks/diarization/neural_diarizer/streaming_sortformer_diar_train.py \
  --config-path=/path/to/conf \
  --config-name=streaming_sortformer_diarizer_4spk-v2.yaml \
  +init_from_nemo_model=diar_streaming_sortformer_5spk_orthogonal.nemo \
  model.train_ds.manifest_filepath=/path/to/train.json \
  model.validation_ds.manifest_filepath=/path/to/val.json \
  model.train_ds.session_len_sec=180 \
  model.validation_ds.session_len_sec=180 \
  model.max_num_of_spks=5 \
  +model.sortformer_modules.n_base_spks=4 \
  model.lr=1e-5 \
  +model.optim_new_lr=1e-4 \
  batch_size=4 \
  trainer.devices=2
```

---

### Step 3: Layer Repeat Experiments (LLM Neuroanatomy)

Inspired by [dnhkng's RYS-XLarge](https://github.com/dnhkng/GlitchHunter) — which reached #1 on the Open LLM Leaderboard by duplicating 7 middle layers of Qwen2-72B without any weight modification — we applied the same technique to Sortformer's Transformer encoder.

**Script**: `scripts/layer_repeat_experiment.py`

#### Concept

For a model with L layers, configuration (i, j) means:
```
Original:   0 → 1 → 2 → ... → i → ... → j → ... → L
Repeated:   0 → 1 → ... → i → ... → j → i → ... → j → j+1 → ... → L
```
The block [i, j] is executed twice. No weights are changed.

#### Findings on Sortformer (18-layer Transformer)

We ran all (start, end) combinations with block sizes 1–5 across 712 real-world test samples:

- **DER**: No block combination improved DER. All configurations degraded it slightly.
- **Spk_Count_Acc**: Layers 14–17 showed significant improvement (+3.4%) when duplicated.

This suggests the Transformer encoder in Sortformer (18 layers) is too compact for the functional anatomy separation seen in large LLMs. The reasoning circuits are entangled throughout the stack.

```
Transformer encoder functional map (hypothesis):
  Layers 0–13 : encoding + cross-speaker attention
  Layers 14–17: speaker count judgment (repeatable without major DER damage)
```

#### Heatmap Visualization

```bash
# Run full sweep
python scripts/layer_repeat_experiment.py \
    --model-path model.nemo \
    --manifest eval_manifest.json \
    --encoder transformer \
    --block-size-min 1 \
    --block-size-max 18 \
    --output results/layer_repeat.json

# Visualize
python scripts/plot_layer_heatmap.py \
    --input results/layer_repeat.json \
    --metric DER
```

---

### Step 4: Extending to 6, 7, 8 Speakers (In Progress 🔄)

The same orthogonal extension + split learning rate process is being applied iteratively to reach 6, 7, and 8 speakers. Each step starts from the previous model:

```
4spk (NVIDIA baseline)
  └─ extend_output_layer.py → fine-tune (split LR)
       └─ 5spk ✅
            └─ extend_output_layer.py → fine-tune (split LR)
                 └─ 6spk 🔄
                      └─ extend_output_layer.py → fine-tune (split LR)
                           └─ 7spk (planned)
                                └─ ...
                                     └─ 8spk (planned)
```

For each step:
```bash
# 1. Extend output layer
python scripts/extend_output_layer.py \
    --src <N>spk_model.nemo \
    --dst-spk $((N+1)) \
    --out $((N+1))spk_split_output.nemo

# 2. Fine-tune with split learning rate
python NeMo/examples/.../streaming_sortformer_diar_train.py \
    +init_from_nemo_model=$((N+1))spk_split_output.nemo \
    model.max_num_of_spks=$((N+1)) \
    +model.sortformer_modules.n_base_spks=$N \
    model.lr=1e-5 \
    +model.optim_new_lr=1e-4
```

Training data for each step uses 200 sessions per dataset (synthetic 2–Nspk + AliMeeting + AMI IHM + AMI SDM), ensuring no overlap with data used in previous steps.

---

## Synthetic Training Data

All synthetic data is generated from **Korean TTS speech** (`multispeaker_speech_synthesis_data`) using `scripts/sentence_level_multispeaker_simulator.py`. The simulator creates multi-speaker sessions by randomly interleaving single-speaker utterances with controlled silence and overlap ratios.

### Source Data

| Source | #Speakers | #Utterances | Language |
|--------|-----------|-------------|----------|
| `multispeaker_speech_synthesis_data/Training` | 8,666,803 utterances | — | Korean |
| `multispeaker_speech_synthesis_data/Validation` | 1,225,244 utterances | — | Korean |

### Generated Datasets

#### Training Sets (180s sessions)

| Dataset | Sessions | Duration | Speakers | Silence | Overlap | Status |
|---------|----------|----------|----------|---------|---------|--------|
| `training_1000sess_2spk_180s_sil0.1_ov0.05` | 1,000 | 180s | 2 | ~10% | ~5% | ✅ |
| `training_1000sess_3spk_180s_sil0.1_ov0.05` | 1,000 | 180s | 3 | ~10% | ~5% | ✅ |
| `training_1000sess_4spk_180s_sil0.1_ov0.05` | 1,000 | 180s | 4 | ~10% | ~5% | ✅ |
| `training_1000sess_5spk_180s_sil0.1_ov0.05` | 1,000 | 180s | 5 | ~10% | ~5% | ✅ |
| `training_1000sess_6spk_180s_sil0.1_ov0.05` | 1,000 | 180s | 6 | ~10% | ~5% | ✅ |
| `training_1000sess_7spk_180s_sil0.1_ov0.05` | 1,000 | 180s | 7 | ~10% | ~5% | ✅ |
| `training_1000sess_8spk_180s_sil0.1_ov0.05` | 1,000 | 180s | 8 | ~10% | ~5% | ✅ |
| `training_1000sess_2spk_180s_sil0.1_ov0.15` | 1,000 | 180s | 2 | ~10% | ~15% | ✅ |
| `training_1000sess_3spk_180s_sil0.1_ov0.15` | 1,000 | 180s | 3 | ~10% | ~15% | ✅ |
| `training_1000sess_4spk_180s_sil0.1_ov0.15` | 1,000 | 180s | 4 | ~10% | ~15% | ✅ |
| `training_1000sess_5spk_180s_sil0.1_ov0.15` | 1,000 | 180s | 5 | ~10% | ~15% | ✅ |
| `training_1000sess_6spk_180s_sil0.1_ov0.15` | 1,000 | 180s | 6 | ~10% | ~15% | ✅ |
| `training_1000sess_7spk_180s_sil0.1_ov0.15` | 1,000 | 180s | 7 | ~10% | ~15% | ✅ |
| `training_1000sess_8spk_180s_sil0.1_ov0.15` | 1,000 | 180s | 8 | ~10% | ~15% | 🔄 |

#### Validation Sets (90s sessions)

| Dataset | Sessions | Duration | Speakers | Silence | Overlap |
|---------|----------|----------|----------|---------|---------|
| `validation_100sess_2spk_90s_sil0.1_ov0.05` | 100 | 90s | 2 | ~10% | ~5% |
| `validation_100sess_3spk_90s_sil0.1_ov0.05` | 100 | 90s | 3 | ~10% | ~5% |
| `validation_100sess_4spk_90s_sil0.1_ov0.05` | 100 | 90s | 4 | ~10% | ~5% |
| `validation_100sess_5spk_90s_sil0.1_ov0.05` | 100 | 90s | 5 | ~10% | ~5% |
| `validation_100sess_6spk_90s_sil0.1_ov0.05` | 100 | 90s | 6 | ~10% | ~5% |
| `validation_100sess_7spk_90s_sil0.1_ov0.05` | 100 | 90s | 7 | ~10% | ~5% |
| `validation_100sess_8spk_90s_sil0.1_ov0.05` | 100 | 90s | 8 | ~10% | ~5% |

### Overlap Ratio Comparison

Two overlap variants were generated to study model robustness to overlapping speech:

| Variant | Mean Overlap | Mean Silence | Training Purpose |
|---------|-------------|--------------|-----------------|
| `ov0.05` | ~5% | ~9% | Standard training (low overlap) |
| `ov0.15` | ~15% | ~9% | Higher overlap — harder conditions |

> Overlap and silence ratios are **means** across sessions; individual sessions vary significantly (std ~9–10%).

### Generating New Datasets

```bash
python scripts/sentence_level_multispeaker_simulator.py \
    --manifest_filepath /path/to/single_speaker_manifest.json \
    --output_dir /path/to/output_dir \
    --num_speakers 6 \
    --num_sessions 1000 \
    --session_length 180 \
    --mean_silence 0.1 \
    --mean_overlap 0.15
```

### Training Manifest Composition

For 5spk and 6spk fine-tuning, manifests are assembled by randomly sampling from the above datasets while ensuring **no overlap with previously used samples**:

| Model | Train | Val | Datasets Used |
|-------|-------|-----|---------------|
| 5spk v1 | 250 | ~60 | synthetic 2–5spk (ov0.05), AliMeeting, AMI IHM, AMI SDM |
| 5spk splitlr | 800 | 200 | synthetic 2–5spk (ov0.05), AliMeeting, AMI IHM, AMI SDM (100 each) |
| 6spk v1 | 800 | 200 | synthetic 2–6spk (ov0.05), AliMeeting, AMI IHM, AMI SDM (100 each) |
| 6spk v2 | 1,600 | 400 | Same sources, doubled (200 each), no overlap with prior runs |

---

## Evaluation Results

All evaluations use:
- Post-processing: onset=0.25s, offset=0.25s, min_duration_on=0.0s, min_duration_off=0.0s
- Collar: 0.25s
- Ignore overlap: False

### Model Progression Summary

| Model | Max Spk | Val F1 | Base |
|-------|---------|--------|------|
| diar_streaming_sortformer_4spk-v2.1 | 4 | — | NVIDIA (baseline) |
| sortformer_5spk_splitlr_1e5_1e4 | 5 | 0.9583 | Extended from 4spk |
| sortformer_6spk_splitlr_1e5_1e4_v2 | 6 | — | Extended from 5spk |
| ultra_diar_streaming_sortformer_8spk_v1.0.0 | 8 | 0.9884 | Extended from 4spk |

### Synthetic Validation (2–8 speakers)

| Dataset | 4spk baseline | 5spk model | 6spk v2 |
|---------|-------------|-----------|---------|
| val_2spk DER | 15.26% | 0.04% | **0.00%** |
| val_3spk DER | 22.07% | 1.07% | **0.48%** |
| val_4spk DER | 25.19% | 2.01% | **0.88%** |
| val_5spk DER | 28.16% | 4.09% | **2.38%** |
| val_6spk DER | 34.59% | 9.96% | **4.22%** |
| val_2spk Spk_Acc | 100% | 100% | 99% |
| val_3spk Spk_Acc | 67% | 93% | **99%** |
| val_4spk Spk_Acc | 54% | 92% | **98%** |
| val_5spk Spk_Acc | 0% | 77% | **81%** |
| val_6spk Spk_Acc | 0% | 0% | **69%** |

### Real-World Datasets

| Dataset | 4spk baseline | 5spk model | 6spk v2 |
|---------|-------------|-----------|---------|
| AliMeeting DER | 11.03% | 5.85% | **6.28%** |
| AMI IHM DER | 26.05% | 10.98% | 13.58% |
| AMI SDM DER | 28.29% | 14.33% | **14.81%** |
| CallHome ENG DER | 4.94% | 7.39% | 7.89% |
| CallHome DEU DER | 6.70% | 6.98% | 7.68% |
| CallHome JPN DER | 10.03% | 10.59% | 11.06% |
| CallHome SPA DER | 23.27% | 17.92% | 19.38% |
| CallHome ZHO DER | 7.15% | 9.24% | 9.24% |

> **Note**: 4spk baseline shows high DER on multi-speaker real-world data because it caps at 4 speakers, causing many speaker confusion errors.

---

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/extend_output_layer.py` | Extend N→M speaker output layer with SVD orthogonal init |
| `scripts/eval.py` | Evaluate DER/Spk_Count_Acc across all datasets |
| `scripts/layer_repeat_experiment.py` | LLM Neuroanatomy-style layer block repeat experiment |
| `scripts/plot_layer_heatmap.py` | Visualize layer repeat results as DER heatmap |
| `scripts/merge_layer_repeat_results.py` | Merge results from 2-GPU parallel layer sweep |
| `scripts/sentence_level_multispeaker_simulator.py` | Synthetic multispeaker data generator |
| `scripts/expand_transformer_layers.py` | Permanently expand transformer encoder layers |
| `scripts/ckpt_to_nemo.py` | Convert `.ckpt` checkpoint to `.nemo` for HF upload |
| `scripts/test_diar_model.py` | Quick test with sample audio |

---

## Usage

### Quick Start (from Hugging Face)

```python
from nemo.collections.asr.models import SortformerEncLabelModel

# 8-speaker model
diar_model = SortformerEncLabelModel.from_pretrained(
    "devsy0117/ultra_diar_streaming_sortformer_8spk_v1.0.0"
)
diar_model.eval()

# Recommended streaming parameters
diar_model.sortformer_modules.chunk_len = 340
diar_model.sortformer_modules.chunk_right_context = 40
diar_model.sortformer_modules.fifo_len = 40
diar_model.sortformer_modules.spkcache_update_period = 300

predicted_segments = diar_model.diarize(
    audio=["/path/to/audio.wav"],
    batch_size=1
)
for segment in predicted_segments[0]:
    print(segment)
```

### Evaluation

```bash
# Evaluate across all datasets
python scripts/eval.py \
    --model-path /path/to/model.nemo \
    --batch-size 4 \
    --devices 2

# Evaluate on a specific manifest
python scripts/eval.py \
    --model-path /path/to/model.nemo \
    --manifest /path/to/manifest.json \
    --batch-size 4
```

### Extend a Model to More Speakers

```bash
# Extend 5spk → 6spk (or any N → M)
python scripts/extend_output_layer.py \
    --src /path/to/5spk_model.nemo \
    --dst-spk 6 \
    --out /path/to/6spk_model.nemo
```

### Generate Synthetic Training Data

```bash
python scripts/sentence_level_multispeaker_simulator.py \
    --manifest_filepath /path/to/single_speaker_manifest.json \
    --output_dir /path/to/output \
    --num_speakers 6 \
    --num_sessions 1000 \
    --session_length 180 \
    --mean_silence 0.1 \
    --mean_overlap 0.15
```

### Layer Repeat Experiment (2-GPU parallel)

```bash
# GPU 0: first half of manifest
CUDA_VISIBLE_DEVICES=0 python scripts/layer_repeat_experiment.py \
    --model-path model.nemo \
    --manifest eval_half0.json \
    --block-size-min 1 --block-size-max 18 \
    --output results/layer_repeat_half0.json

# GPU 1: second half
CUDA_VISIBLE_DEVICES=1 python scripts/layer_repeat_experiment.py \
    --model-path model.nemo \
    --manifest eval_half1.json \
    --block-size-min 1 --block-size-max 18 \
    --output results/layer_repeat_half1.json

# Merge and visualize
python scripts/merge_layer_repeat_results.py \
    --inputs results/layer_repeat_half0.json results/layer_repeat_half1.json \
    --output results/layer_repeat_merged.json

python scripts/plot_layer_heatmap.py \
    --input results/layer_repeat_merged.json \
    --metric DER
```

---

## Training

### NeMo Modifications

This project requires modifications to NeMo's Sortformer implementation:

**`nemo/collections/asr/models/sortformer_diar_models.py`**
- Added `setup_optimizer_param_groups()` override for differential learning rates

**`nemo/collections/asr/modules/sortformer_modules.py`**
- Added `n_base_spks` parameter to enable split output layers (`single_hidden_to_spks_base` + `single_hidden_to_spks_new`)

**`nemo/collections/asr/data/audio_to_diar_label.py`**
- Fixed `_eesd_train_collate_fn` to handle mixed 1D/2D (mono/stereo) audio tensors in batches

### Training Configuration

Key YAML settings (`configs/streaming_sortformer_diarizer_4spk-v2.yaml`):

```yaml
model:
  max_num_of_spks: 6       # Set to target speaker count
  lr: 1e-5                 # Base learning rate
  # optim_new_lr: 1e-4     # Add via CLI: +model.optim_new_lr=1e-4

  sortformer_modules:
    num_spks: ${model.max_num_of_spks}
    # n_base_spks: 5       # Add via CLI: +model.sortformer_modules.n_base_spks=5
```

---

## Requirements

```bash
# NeMo (clone separately, ~5.9GB)
git clone https://github.com/NVIDIA/NeMo.git
pip install -e NeMo/[asr]

# Other dependencies
pip install torch torchaudio
pip install soundfile librosa
pip install pyannote.metrics
```

---

## License

This model is a derivative of NVIDIA Sortformer, licensed under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

**Attribution**: Based on work by NVIDIA Corporation.
