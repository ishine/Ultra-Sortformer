# Ultra-Sortformer: Extending NVIDIA Sortformer to N Speakers

[![5spk Model on Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/devsy0117/ultra_diar_streaming_sortformer_5spk_v1)
[![8spk Model on Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/devsy0117/ultra_diar_streaming_sortformer_8spk_v1)

This project extends **NVIDIA's Streaming Sortformer** speaker diarization from the **4-speaker** baseline to **5- and 8-speaker** models. The approach is to expand the output layer with **orthogonal initialization**, then **fine-tune** with **split learning rates** so existing behavior stays stable while new speaker dimensions are learned.

**Released models (Hugging Face)**  
- [ultra_diar_streaming_sortformer_5spk_v1](https://huggingface.co/devsy0117/ultra_diar_streaming_sortformer_5spk_v1) — up to 5 speakers  
- [ultra_diar_streaming_sortformer_8spk_v1](https://huggingface.co/devsy0117/ultra_diar_streaming_sortformer_8spk_v1) — up to 8 speakers (4→8 extension + training)

---

## Table of Contents

1. [Background](#background)
2. [Architecture Overview](#architecture-overview)
3. [Extension Journey](#extension-journey)
   - [Step 1: Output Layer Extension (4 → 5spk)](#step-1-output-layer-extension-4--5spk)
   - [Step 2: Split Learning Rate Training](#step-2-split-learning-rate-training)
   - [Step 3: Layer Expansion Experiments](#step-3-layer-expansion-experiments)
   - [Step 4: Extending to 8 Speakers](#step-4-extending-to-8-speakers)
4. [Evaluation Results](#evaluation-results)
5. [Synthetic Training Data](#synthetic-training-data)
6. [Training](#training)
7. [Requirements](#requirements)

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
│  Preprocessor               │
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

### Step 3: Layer Expansion Experiments

Inspired by [dnhkng's RYS-XLarge](https://github.com/dnhkng/GlitchHunter) — which reached #1 on the Open LLM Leaderboard by duplicating middle layers of Qwen2-72B without any weight modification — we experimented with **layer repeat** and **permanent layer duplication** on Sortformer's 18-layer Transformer encoder.

**Scripts**: `scripts/layer_repeat_experiment.py`, `scripts/expand_transformer_layers.py`

We ran layer repeat (re-executing a block [i, j] twice) across all (start, end) combinations on 65 real-world test samples, and also tested permanent duplication of blocks like L8-9, L14-17. Key observations:

- **Layers 0–7** handle acoustic encoding; **layers 8–9** are the core diarization reasoning block (most sensitive to modification); **layers 14–17** drive speaker count judgment.
- **L14-17 duplication** dramatically boosted Speaker Count Accuracy on real-world data (e.g., AMI IHM: 43.75% → 87.50%), but increased DER without fine-tuning.
- Layer expansion without subsequent fine-tuning reliably increases MISS at the junction between original and copied blocks.

These experiments served as an architectural analysis tool; no layer-expanded models were used for final training.

---

### Step 4: Extending to 8 Speakers

The **8-speaker** model starts from the NVIDIA **4spk** checkpoint, extends the Sortformer output head with the same orthogonal / split-layer pipeline as above, and is **fine-tuned** with split learning rates (`~1e-5` base, `~1e-4` on new head parameters) on mixed synthetic + real meeting data.

- **Hugging Face**: [devsy0117/ultra_diar_streaming_sortformer_8spk_v1](https://huggingface.co/devsy0117/ultra_diar_streaming_sortformer_8spk_v1)  
- **Weights**: `ultra_diar_streaming_sortformer_8spk_v1.0.nemo` (same checkpoint as the Hub release)

Example training command shape (paths and manifests depend on your setup):

```bash
python NeMo/examples/speaker_tasks/diarization/neural_diarizer/streaming_sortformer_diar_train.py \
  --config-path=/path/to/conf/neural_diarizer \
  --config-name=streaming_sortformer_diarizer_4spk-v2.yaml \
  +init_from_nemo_model=/path/to/prior_8spk_checkpoint.nemo \
  model.train_ds.manifest_filepath=/path/to/train.json \
  model.validation_ds.manifest_filepath=/path/to/val.json \
  +model.sortformer_modules.n_base_spks=4 \
  model.lr=1e-5 \
  +model.optim_new_lr=1e-4 \
  exp_manager.name=sortformer_8spk_run \
  exp_manager.exp_dir=/path/to/logs
```

---

## Synthetic Training Data

All synthetic data is generated from **Korean TTS speech** using `scripts/sentence_level_multispeaker_simulator.py`, a customized version of NeMo's [`multispeaker_simulator.py`](https://github.com/NVIDIA/NeMo/blob/main/tools/speech_data_simulator/multispeaker_simulator.py). The original script was modified to operate at the **sentence level** — interleaving complete single-speaker utterances rather than splitting at the word/phoneme level — which better reflects natural conversational turn-taking. It creates multi-speaker sessions with controlled silence and overlap ratios.

### Source Data

Single-speaker utterances are sourced from the **[다화자 음성합성 데이터 (Multi-speaker Speech Synthesis Dataset)](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=542)** provided by [AI-Hub](https://www.aihub.or.kr) (한국지능정보사회진흥원, NIA). This dataset contains recordings from 3,400+ Korean speakers across diverse age groups (10s–60s), totaling 10,152 hours of speech.

| Source | #Utterances | Language |
|--------|-------------|----------|
| `multispeaker_speech_synthesis_data/Training` | 8,666,803 | Korean |
| `multispeaker_speech_synthesis_data/Validation` | 1,225,244 | Korean |

### Generated Datasets

Two overlap variants were generated for 2–8 speakers to study robustness to overlapping speech:

- **`ov0.05`** — 1,000 training sessions × 2–8 spk (180s), 100 validation sessions × 2–8 spk (90s), ~5% overlap
- **`ov0.15`** — 1,000 training sessions × 2–8 spk (180s), ~15% overlap (harder conditions)

All sessions use ~10% mean silence. Overlap and silence ratios are means; individual sessions vary (std ~9–10%).


---

## Evaluation Results

Comparisons below use the NVIDIA base [`diar_streaming_sortformer_4spk-v2.1`](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1).

### Evaluation Parameters

| Parameter | Value |
|-----------|-------|
| Post-processing | None |
| Collar | 0.25s |
| Ignore overlap | False |
| Chunk size | 340 frames |
| Batch size | 1 |

### `ultra_diar_streaming_sortformer_8spk_v1.0`

#### AliMeeting (test)

| Model | DER | FA | MISS | CER | Spk_Count_Acc |
|-------|-----|----|------|-----|---------------|
| diar_streaming_sortformer_4spk-v2.1 (base) | 11.03% | 0.40% | 9.93% | 0.70% | 95.00% |
| ultra_diar_streaming_sortformer_8spk_v1.0 | **5.69%** | 1.12% | 3.89% | 0.68% | 100.00% |

#### AMI IHM (test)

| Model | DER | FA | MISS | CER | Spk_Count_Acc |
|-------|-----|----|------|-----|---------------|
| diar_streaming_sortformer_4spk-v2.1 (base) | 26.05% | 0.50% | 23.51% | 2.03% | 93.75% |
| ultra_diar_streaming_sortformer_8spk_v1.0 | **10.87%** | 1.53% | 7.89% | 1.44% | 81.25% |

#### AMI SDM (test)

| Model | DER | FA | MISS | CER | Spk_Count_Acc |
|-------|-----|----|------|-----|---------------|
| diar_streaming_sortformer_4spk-v2.1 (base) | 28.29% | 0.82% | 23.76% | 3.72% | 93.75% |
| ultra_diar_streaming_sortformer_8spk_v1.0 | **15.61%** | 2.33% | 8.23% | 5.05% | 75.00% |

#### CallHome (test)

| Model | eng | deu | jpn | spa | zho |
|-------|-----|-----|-----|-----|-----|
| diar_streaming_sortformer_4spk-v2.1 (base) DER | 4.94% | 6.70% | 10.03% | 23.27% | 7.15% |
| ultra_diar_streaming_sortformer_8spk_v1.0 DER | 8.20% | 7.70% | 11.11% | **18.24%** | 10.16% |
| diar_streaming_sortformer_4spk-v2.1 (base) Spk_Acc | 83.57% | 80.83% | 79.17% | 63.57% | 72.86% |
| ultra_diar_streaming_sortformer_8spk_v1.0 Spk_Acc | **92.86%** | **90.00%** | **89.17%** | **70.00%** | **75.00%** |

> **Note**: Extending to 8 speakers changes speaker-count behavior on low-speaker or short clips; interpret `Spk_Count_Acc` next to DER. See the model card on Hugging Face for details.

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
