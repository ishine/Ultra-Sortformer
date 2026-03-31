# Ultra-Sortformer: Extending NVIDIA Sortformer to N Speakers

[![5spk Model on Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/devsy0117/ultra_diar_streaming_sortformer_5spk_v1)
[![8spk Model on Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/devsy0117/ultra_diar_streaming_sortformer_8spk_v1)

This repository records the **transition from a fixed 4-speaker cap to a configurable N-speaker Streaming Sortformer** (**N > 4**). The pattern is always the same: grow the speaker head with **SVD-based orthogonal initialization**, keep **base vs. new** weights in separate modules, and **fine-tune with split learning rates** so behavior on 2–4 speaker audio stays stable while the extra dimensions learn. You can target any **N** supported by your data and VRAM; we publish checkpoints for **N = 5** and **N = 8** as reference points.

**Released models (Hugging Face)**  
- [ultra_diar_streaming_sortformer_5spk_v1](https://huggingface.co/devsy0117/ultra_diar_streaming_sortformer_5spk_v1) — **N = 5**  
- [ultra_diar_streaming_sortformer_8spk_v1](https://huggingface.co/devsy0117/ultra_diar_streaming_sortformer_8spk_v1) — **N = 8**

---

## Table of Contents

1. [Background](#background)
2. [Architecture Overview](#architecture-overview)
3. [Extension Journey](#extension-journey)
   - [Step 1: Output Layer Extension (4 → N)](#step-1-output-layer-extension-4--n)
   - [Step 2: Split Learning Rate Training](#step-2-split-learning-rate-training)
   - [Step 3: Layer Expansion Experiments](#step-3-layer-expansion-experiments)
   - [Step 4: Scaling to Larger N (Example: 8 Speakers)](#step-4-scaling-to-larger-n-example-8-speakers)
4. [Evaluation Results](#evaluation-results)
5. [Synthetic Training Data](#synthetic-training-data)
   - [Prerequisites](#synthesis-prerequisites)
   - [How it differs from stock NeMo](#how-it-differs-from-stock-nemo)
   - [Configuration](#synthesis-configuration)
   - [CLI](#synthesis-cli)
   - [Session length and speaker enforcement](#session-length-and-speaker-enforcement)
   - [Outputs](#synthesis-outputs)
   - [Example command](#synthesis-example)
6. [Training](#training)
7. [Requirements](#requirements)

---

## Background

NVIDIA's [diar_streaming_sortformer_4spk-v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) is a streaming speaker diarization model based on the Sortformer architecture. It uses a **FastConformer encoder** (17 layers) followed by a **Transformer encoder** (18 layers) to produce per-frame speaker activity predictions. The final output layer is a single linear mapping from hidden states to **four** speaker probabilities.

The model supports real-time streaming diarization with a chunk-based speaker cache.

**Problem**: The public checkpoint is **hard-limited to four simultaneous speakers**. Scenes with more talkers are handled poorly once you only relabel data without widening the head.

**Goal**: Make **`max_num_of_spks = N` with N > 4** a first-class training target, without sacrificing the 2–4 speaker regime. This README walks through the mechanics; concrete runs in the repo use **N = 5** (smallest jump) and **N = 8** (larger jump from the same base).

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

The diagram shows the **stock** layout: one `single_hidden_to_spks` linear, `nn.Linear(192, N_spk)`. After extension in this repo, checkpoints use **`single_hidden_to_spks_base`** and **`single_hidden_to_spks_new`** (same total width **N_spk**, split for differential learning rates; see Step 1–2). Moving from NVIDIA’s **N_spk = 4** head to **N_spk = N** means adding **N − 4** rows (or **N_new** rows when extending from any **N_base**).

---

## Extension Journey

### Step 1: Output Layer Extension (4 → N)

**Script**: `scripts/extend_output_layer.py`

We treat the baseline as **N_base = 4** speakers and grow the matrix to **N = N_base + N_new**. The first shipped milestone is **N = 5** (**N_new = 1**); the **N = 8** model adds **N_new = 4** rows in one shot with the same procedure.

Random initialization for the new rows **destroys** accuracy on existing speakers. We instead use **SVD-based orthogonal initialization** so new logits start orthogonal to the subspace spanned by the original weights.

#### How it works

Let the existing weight matrix be `W ∈ ℝ^{N_base×H}` (H = 192):

```python
# SVD decomposition of existing weights
U, S, Vh = torch.linalg.svd(W, full_matrices=True)

# New speaker rows = right singular directions beyond the first N_base
# Vh[N_base], Vh[N_base+1], ... are orthogonal to the row space of W
new_row = Vh[N_base]  # first new speaker; repeat indexing for additional speakers

# Normalize to match typical row norms
avg_norm = W.norm(dim=1).mean()
new_row = new_row * (avg_norm / new_row.norm())
```

*The SVD story above is **intuition**: new rows are built from singular-vector directions of `W` so added logits start largely complementary to the existing head. It is not a formal subspace-orthogonality proof.*

Repeat for each new speaker index until the head reaches the target **N**.

The extended checkpoint is saved in **split** form so optimizers can treat base and new rows differently:

```
single_hidden_to_spks_base  (N_base speakers)   ← frozen / low LR
single_hidden_to_spks_new   (N_new speakers)    ← higher LR
```

---

### Step 2: Split Learning Rate Training

**Key insight**: A **single** learning rate on the whole expanded head tends to **erase** the old 2–4 speaker solution while the new dimensions still fit.

#### What we saw early on

- Synthetic **val_2spk–val_4spk** quality dropped under uniform LR.
- The network **over-used** the new capacity (e.g., predicting the full trained **N** on 3–4 speaker clips).

#### Mitigation: differential learning rates

| Component | Learning rate | Role |
|-----------|---------------|------|
| `single_hidden_to_spks_base` (speakers 1…N_base) | `1e-5` | Preserve the pretrained head |
| `single_hidden_to_spks_new` (added speakers) | `1e-4` | Faster adaptation on new dimensions |
| Rest of the model | `1e-5` | Standard fine-tuning |

Implemented via `setup_optimizer_param_groups` in `sortformer_diar_models.py`:

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

---

### Step 3: Layer Expansion Experiments

Inspired by [dnhkng's RYS-XLarge](https://github.com/dnhkng/GlitchHunter) — which reportedly reached #1 on the Open LLM Leaderboard at the time of that work, by duplicating middle layers of Qwen2-72B without weight modification — we experimented with **layer repeat** and **permanent layer duplication** on Sortformer's 18-layer Transformer encoder.

**Scripts**: `scripts/layer_repeat_experiment.py`, `scripts/expand_transformer_layers.py`

We ran layer repeat (re-executing a block [i, j] twice) across all (start, end) combinations on 65 real-world test samples, and also tested permanent duplication of blocks like L8-9, L14-17. Key observations:

- **Layers 0–7** handle acoustic encoding; **layers 8–9** are the core diarization reasoning block (most sensitive to modification); **layers 14–17** drive speaker count judgment.
- **L14-17 duplication** dramatically boosted Speaker Count Accuracy on real-world data (e.g., AMI IHM: 43.75% → 87.50%), but increased DER without fine-tuning.
- Layer expansion without subsequent fine-tuning reliably increases MISS at the junction between original and copied blocks.

These experiments served as an architectural analysis tool; no layer-expanded models were used for final training.

---

### Step 4: Scaling to Larger N (Example: 8 Speakers)

The **N = 8** release uses the **same pipeline**: start from NVIDIA **4-spk**, extend the Sortformer head with orthogonal / split weights (**N_base = 4**, **N_new = 4**), then fine-tune with **~1e-5** on the bulk of parameters and **~1e-4** on `single_hidden_to_spks_new` on mixed synthetic + real meeting data.

- **Hugging Face**: [devsy0117/ultra_diar_streaming_sortformer_8spk_v1](https://huggingface.co/devsy0117/ultra_diar_streaming_sortformer_8spk_v1)  
- **Weights**: `ultra_diar_streaming_sortformer_8spk_v1.nemo` (same artifact as the Hub upload)

For a different **N**, align `model.max_num_of_spks`, training manifests, and `n_base_spks` with your expansion checkpoint and patched NeMo setup (see [Training](#training)).

---

## Synthetic Training Data

All synthetic multi-speaker sessions are built from **single-speaker Korean TTS utterances** using `scripts/sentence_level_multispeaker_simulator.py`. The tool subclasses NeMo’s [`MultiSpeakerSimulator`](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/data/data_simulation.py) (same pipeline as [`multispeaker_simulator.py`](https://github.com/NVIDIA/NeMo/blob/main/tools/speech_data_simulator/multispeaker_simulator.py)): session layout, silence/overlap sampling, RTTM/JSON/CTM export, and optional augmentations follow NeMo’s `data_simulator.yaml`. Only **`_build_sentence`** is overridden so each “sentence” is built from **whole manifest rows** (full utterances), not word- or sub-word chunks—closer to natural turn-taking than the stock simulator.

### Source Data

Single-speaker utterances come from the **[다화자 음성합성 데이터 (Multi-speaker Speech Synthesis Dataset)](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=542)** on [AI-Hub](https://www.aihub.or.kr) (NIA). It spans 3,400+ Korean speakers (10s–60s), ~10k hours.

| Split | Approx. #Utterances | Language |
|--------|---------------------|----------|
| `multispeaker_speech_synthesis_data/Training` | 8,666,803 | Korean |
| `multispeaker_speech_synthesis_data/Validation` | 1,225,244 | Korean |

Build a **NeMo-style JSON manifest** listing `audio_filepath`, `speaker` (or compatible id), and optionally `text`, `words`, `alignments` for labels. The simulator groups rows by speaker id to sample per turn.

### Synthesis prerequisites

1. **Install NeMo** (see [Requirements](#requirements)). The simulator imports `nemo` from your environment; it looks for `NeMo/tools/speech_data_simulator/conf/data_simulator.yaml` **only if** a sibling `NeMo/` directory exists next to this repo. If you use a **pip-only** install, pass `--config_file` pointing to that YAML (e.g. from a checkout or a copied file).
2. **System audio libraries**: `libsndfile1` and `ffmpeg` (listed under Requirements) are required for decoding/writing audio in practice.
3. The script sets **`CUDA_VISIBLE_DEVICES=""`** so generation runs on **CPU only** (avoids driver/PyTorch CUDA mismatches on headless or older drivers). It is slower than GPU but predictable across environments.

### How it differs from stock NeMo

| Aspect | Stock `MultiSpeakerSimulator` | `SentenceLevelMultiSpeakerSimulator` |
|--------|------------------------------|--------------------------------------|
| Turn content | Word-aligned slices; reads audio in chunks up to `max_audio_read_sec` | One or more **entire** utterances per turn (mono, resampled to `sr`) |
| Turn length cap | Word-count target from `sentence_length_params` | **`max_sentences_per_turn`**: uniform **1…N** utterances per turn (CLI default **N = 3**). If unset in YAML and not overridden, falls back to negative binomial on **utterance count** (often too long—prefer explicit **N**) |
| Optional YAML | Same | `session_params.max_turn_duration_sec` caps samples per turn when set |

### Synthesis configuration

- **Base config**: `NeMo/tools/speech_data_simulator/conf/data_simulator.yaml` (or pass `--config_file`).
- Important YAML knobs (not all exposed on CLI):
  - `session_config.{num_speakers,num_sessions,session_length}` — target speakers per session, session count, **nominal** duration in **seconds**.
  - `session_params.{mean_silence,mean_overlap,...}` — global silence/overlap **means** (per-session values vary).
  - `speaker_enforcement.enforce_num_speakers` — if `true`, NeMo may **continue past `session_length`** and **pad** the waveform until every speaker has spoken; real duration can exceed `session_length`. Set `enforce_num_speakers: false` in YAML if you need a hard cap at the cost of possibly missing speakers in a session.
  - `sr`, `outputs.output_filename`, augmentors, background noise, etc. — unchanged from NeMo.

### Synthesis CLI

| Argument | Role |
|----------|------|
| `--manifest_filepath` | Input NeMo JSON manifest (single-speaker rows with speaker id). |
| `--output_dir` | Output directory for `.wav`, `.rttm`, `.json`, `params.yaml`, etc. |
| `--config_file` | Optional override YAML (defaults to NeMo’s `data_simulator.yaml`). |
| `--num_speakers` | Override `session_config.num_speakers`. |
| `--num_sessions` | Override `session_config.num_sessions`. |
| `--session_length` | Override nominal session length (**seconds**). |
| `--mean_silence` | Session mean silence ratio in **[0, 1)**. |
| `--mean_overlap` | Session mean overlap ratio in **[0, 1)**; invalid `mean_overlap_var` is clamped for stability. |
| `--max_sentences_per_turn` / `--max_sent` | Max utterances concatenated in one speaker turn; each run draws **uniformly from 1…N** (default **N = 3**). |

### Session length and speaker enforcement

`session_length` is a **target** timeline length in samples (`session_length × sr`). With **`enforce_num_speakers: true`** (NeMo default), the generator can **extend** the buffer so late speakers still get turns. For utterance-level simulation, combine **`--max_sent`** (small **N**) with YAML tuning (`enforce_num_speakers`, optional `max_turn_duration_sec`) if you need durations close to the nominal cap.

### Synthesis outputs

Per session index `i`: `multispeaker_session_i.wav`, `multispeaker_session_i.rttm`, `multispeaker_session_i.json` (and CTM if enabled), plus a copied **`params.yaml`** under `--output_dir`. Merge session manifests into NeMo diarization train/val JSON with your own tooling. Under `scripts/`, this repo includes `sentence_level_multispeaker_simulator.py` (synthesis) and `inference.py` (minimal HF Sortformer diarize example).

### Synthesis example

From the repository root (with `NeMo` installed and discoverable as above):

```bash
python scripts/sentence_level_multispeaker_simulator.py \
  --manifest_filepath /path/to/manifest.json \
  --output_dir /path/to/synthetic_run \
  --num_speakers 8 \
  --num_sessions 1000 \
  --session_length 180 \
  --mean_silence 0.10 \
  --mean_overlap 0.05 \
  --max_sent 3
```

Adjust paths, speaker count, session count, and overlap/silence means to match your experiment grid.

### Generated datasets (this project)

Synthetic grids for **2–8 speakers** used two mean-overlap settings: **`ov0.05`** (~5%) and **`ov0.15`** (~15%, harder). Both used comparable mean silence (~10%); overlap and silence are **session means**, so per-session values vary.

---

## Evaluation Results

Head-to-head metrics for **ultra_diar_streaming_sortformer_8spk_v1**, **ultra_diar_streaming_sortformer_5spk_v1**, NVIDIA [`diar_streaming_sortformer_4spk-v2.1`](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1), and **pyannote(mago_mstudio)** — per-dataset DER / FA / MISS / CER / Spk_Count_Acc, synthetic `val_*` splits, AMI / AliMeeting / CallHome, Korean eval corpora (with AI Hub source links), and **total / total (real)** rankings — live in **[`results/benchmark_results.md`](results/benchmark_results.md)** as the single benchmark document.

### Evaluation parameters

| Parameter | Value |
|-----------|-------|
| Post-processing | None |
| Collar | 0.25s |
| Ignore overlap | False |
| Chunk size | 340 frames |
| Batch size | 1 |

> **Note**: Extending to more speakers changes speaker-count behavior on low-speaker or short clips; interpret `Spk_Count_Acc` alongside DER. See the Hugging Face model cards for discussion.

---

## Training

These changes are **not** in upstream NVIDIA NeMo as of typical `main` installs. To reproduce split-head checkpoints and differential learning rates, apply the edits below to your NeMo source tree (or use a fork that already includes them).

### NeMo Modifications

This project requires modifications to NeMo's Sortformer implementation:

**`nemo/collections/asr/models/sortformer_diar_models.py`**
- Added `setup_optimizer_param_groups()` override for differential learning rates

**`nemo/collections/asr/modules/sortformer_modules.py`**
- Added `n_base_spks` parameter to enable split output layers (`single_hidden_to_spks_base` + `single_hidden_to_spks_new`)

### Training Configuration

Key model settings (same keys as in NeMo’s example config  
`examples/speaker_tasks/diarization/conf/neural_diarizer/streaming_sortformer_diarizer_4spk-v2.yaml` inside a NeMo checkout):

```yaml
model:
  max_num_of_spks: 6       # Set to your target N
  lr: 1e-5                 # Base learning rate
  # optim_new_lr: 1e-4     # Higher LR for single_hidden_to_spks_new (split-head fine-tuning)

  sortformer_modules:
    num_spks: ${model.max_num_of_spks}
    # n_base_spks: 4       # Base speaker count when using split output layers
```

---

## Requirements

System packages (Debian/Ubuntu; use `sudo` if you are not root):

```bash
sudo apt-get update && sudo apt-get install -y libsndfile1 ffmpeg
```

Python (recommended: a fresh virtual environment). `nemo_toolkit[asr]` pulls in PyTorch and most ASR dependencies declared by NeMo; install a **CUDA-enabled PyTorch wheel first** from [pytorch.org](https://pytorch.org) if you train or run on GPU.

```bash
pip install Cython packaging
pip install "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"
```

**What this covers:** NeMo ASR stack from `main`, including typical dependencies such as PyTorch-related packages as required by NeMo’s install. It replaces a separate `git clone` + `pip install -e NeMo/[asr]` for **importing** NeMo in Python.

**Ultra-Sortformer / split head:** A plain `pip install` from NVIDIA NeMo does **not** include `n_base_spks`, the split output layers (`single_hidden_to_spks_base` / `_new`), or the `setup_optimizer_param_groups` behavior described under [Training](#training). Match this repo’s training or checkpoints only after applying those NeMo changes (or using an equivalent fork).

**Not included above (add if you need them):**

- **`pyannote.metrics`** — used for DER / benchmark-style evaluation in this project’s docs and scripts.
- **`librosa`** — only if your own preprocessing or tooling uses it.
- **Sibling `NeMo/` clone** — optional but convenient for examples that reference `NeMo/examples/...` paths and for the default `data_simulator.yaml` path used by `sentence_level_multispeaker_simulator.py` (or always pass `--config_file`).

---

## License

**Model checkpoints** published from this work (e.g. on Hugging Face) are derivatives of NVIDIA Sortformer and fall under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
