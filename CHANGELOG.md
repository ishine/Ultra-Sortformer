# Changelog

All notable model versions are documented here.

---

## [Unreleased]

### In Progress
- 6spk v2 model training (`sortformer_6spk_splitlr_1e5_1e4_v2`)
- Layer repeat experiment (full 18-layer sweep) on 6spk model
- Synthetic data generation with higher overlap (sil=10%, overlap=15%) for 2–6spk

---

## [5spk-v1] - 2026-03

### Model: `sortformer_5spk_splitlr_1e5_1e4`

#### Architecture
- Base: `diar_streaming_sortformer_4spk-v2.1` (NVIDIA)
- Output layer extended 4 → 5 using SVD orthogonal initialization
- Split output: `single_hidden_to_spks_base` (4 spk) + `single_hidden_to_spks_new` (1 spk)

#### Training
- Differential LR: base speakers `lr=1e-5`, new speaker `lr=1e-4`
- Data: synthetic 2–5spk (200 each) + AliMeeting (200) + AMI IHM/SDM (200 each), 180s sessions
- val_f1_acc: 0.9583

#### Key Results (vs 4spk baseline)
| Dataset | 4spk DER | 5spk DER |
|---------|----------|----------|
| val_5spk | 28.16% | **4.09%** |
| AliMeeting | 11.03% | **5.85%** |
| AMI IHM | 26.05% | **10.98%** |

---

## [6spk-v1] - 2026-03

### Model: `sortformer_6spk_splitlr_1e5_1e4`

#### Architecture
- Base: `sortformer_5spk_splitlr_1e5_1e4`
- Output layer extended 5 → 6 using SVD orthogonal initialization
- Split: `single_hidden_to_spks_base` (5 spk) + `single_hidden_to_spks_new` (1 spk)

#### Training
- Differential LR: base speakers `lr=1e-5`, new speaker `lr=1e-4`
- Data: synthetic 2–6spk (100 each) + AliMeeting (100) + AMI IHM/SDM (100 each), 180s sessions

#### Fixes Applied
- Fixed NeMo `_eesd_train_collate_fn` bug for mixed mono/stereo audio batches

---

## [6spk-v2] - 2026-03

### Model: `sortformer_6spk_splitlr_1e5_1e4_v2`

#### Improvements over v1
- Doubled training data: 200 sessions per dataset (1600 train / 400 val total)
- Excluded all data already used in 5spk training

#### Key Results
| Dataset | 5spk DER | 6spk v2 DER |
|---------|----------|-------------|
| val_5spk | 4.09% | **2.38%** |
| val_6spk | 9.96% | **4.22%** |
| AliMeeting | 5.85% | 6.28% |
| AMI SDM | 14.33% | **14.81%** |

---

## [1.0.0] - 2025-03-09

### Model: `ultra_diar_streaming_sortformer_8spk_v1.0.0`

#### Architecture
- Base: `diar_streaming_sortformer_4spk-v2.1` (NVIDIA)
- Output layer extended 4 → 8 using SVD orthogonal initialization

#### Training
- 7000 sessions, 180s, 2–8 speakers (Korean synthetic data)
- val_f1_acc: 0.9884

#### Released on Hugging Face
- [devsy0117/ultra_diar_streaming_sortformer_8spk_v1.0.0](https://huggingface.co/devsy0117/ultra_diar_streaming_sortformer_8spk_v1.0.0)

---

## Versioning

We use [Semantic Versioning](https://semver.org/):

- **MAJOR** (1.x.x): Architecture change, breaking compatibility
- **MINOR** (x.1.x): Improved performance, new training run
- **PATCH** (x.x.1): Bug fix, minor tweak
