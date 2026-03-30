# Sortformer diarization benchmark

Summary of streaming Sortformer and baseline diarization models on a fixed multi-dataset evaluation protocol. **Lower DER is better.** Rankings use **total (real)** DER (real-world corpora only; synthetic `val_*` splits excluded from that aggregate).

| Rank | Model | total DER | total Spk_Count_Acc | total (real) DER | total (real) Spk_Count_Acc |
|------|-------|-----------|---------------------|------------------|---------------------------|
| 1 | ultra_diar_streaming_sortformer_8spk_v1 | 9.40% | 76.75% | 13.53% | 72.77% |
| 2 | ultra_diar_streaming_sortformer_5spk_v1 | 11.53% | 66.81% | 13.53% | 75.62% |
| 3 | diar_streaming_sortformer_4spk-v2.1 | 21.98% | 62.03% | 16.96% | 77.15% |
| 4 | pyannote(mago_mstudio) | 22.90% | 41.01% | 19.00% | 48.27% |

---

## Datasets

| Dataset | Description | Language |
|---------|-------------|----------|
| val_2spk ~ val_8spk | Synthetic validation (2–8 speakers, 90 s, silence / overlap) | Korean |
| alimeeting | AliMeeting meeting speech | Chinese |
| ami_ihm_test | AMI IHM (individual headset) test | English |
| ami_sdm_test | AMI SDM (single distant mic) test | English |
| callhome_eng | CallHome English | English |
| callhome_deu | CallHome German | German |
| callhome_jpn | CallHome Japanese | Japanese |
| callhome_spa | CallHome Spanish | Spanish |
| callhome_zho | CallHome Chinese | Chinese |
| kdomainconf_test30 | kdomainconf 5-speaker, 30 samples | Korean |
| kdomainconf_val_3_4spk_test30 | kdomainconf validation 3–4 speakers, 30 samples | Korean |
| kaddress_test30 | kaddress read speech, 30 samples | Korean |
| kemergency_test30 | kemergency domain, 30 samples | Korean |

## ultra_diar_streaming_sortformer_8spk_v1

Checkpoint: `streaming_sortformer_diar_train/sortformer_8spk_from4_splitlr_v4/checkpoints/sortformer_8spk_from4_splitlr_v4.nemo`

| dataset | FA | MISS | CER | DER | Spk_Count_Acc |
|---------|-----|------|-----|-----|---------------|
| val_2spk | 0.01% | 0.04% | 0.04% | 0.09% | 99.00% |
| val_3spk | 0.09% | 0.06% | 0.51% | 0.66% | 97.00% |
| val_4spk | 0.00% | 0.22% | 0.99% | 1.21% | 91.00% |
| val_5spk | 0.03% | 0.15% | 0.99% | 1.17% | 88.00% |
| val_6spk | 0.08% | 0.45% | 2.04% | 2.57% | 77.00% |
| val_7spk | 0.14% | 0.48% | 3.35% | 3.98% | 71.00% |
| val_8spk | 0.36% | 0.84% | 5.38% | 6.58% | 62.00% |
| alimeeting | 1.12% | 3.89% | 0.68% | 5.69% | 100.00% |
| ami_ihm_test | 1.53% | 7.89% | 1.44% | 10.87% | 81.25% |
| ami_sdm_test | 2.33% | 8.23% | 5.05% | 15.61% | 75.00% |
| callhome_eng | 6.97% | 0.93% | 0.30% | 8.20% | 92.86% |
| callhome_deu | 4.56% | 2.32% | 0.82% | 7.70% | 90.00% |
| callhome_jpn | 5.65% | 3.50% | 1.96% | 11.11% | 89.17% |
| callhome_spa | 5.85% | 7.06% | 5.33% | 18.24% | 70.00% |
| callhome_zho | 6.14% | 1.41% | 2.61% | 10.16% | 75.00% |
| kdomainconf_test30 | 6.59% | 3.26% | 7.13% | 16.98% | 23.33% |
| kdomainconf_val_3_4spk_test30 | 7.28% | 3.82% | 10.37% | 21.47% | 23.33% |
| kaddress_test30 | 0.00% | 6.67% | 0.22% | 6.89% | 60.00% |
| kemergency_test30 | 15.55% | 11.32% | 2.57% | 29.43% | 93.33% |
| **total** | - | - | - | **9.40%** | **76.75%** |
| **total (real)** | - | - | - | **13.53%** | **72.77%** |


## ultra_diar_streaming_sortformer_5spk_v1

Checkpoint: `streaming_sortformer_diar_train/sortformer_5spk_splitlr_1e5_1e4/checkpoints/sortformer_5spk_splitlr_1e5_1e4.nemo`

| dataset | FA | MISS | CER | DER | Spk_Count_Acc |
|---------|-----|------|-----|-----|---------------|
| val_2spk | 0.01% | 0.02% | 0.01% | 0.04% | 100.00% |
| val_3spk | 0.00% | 0.01% | 1.05% | 1.07% | 93.00% |
| val_4spk | 0.05% | 0.02% | 1.94% | 2.01% | 92.00% |
| val_5spk | 0.14% | 0.28% | 3.67% | 4.09% | 77.00% |
| val_6spk | 0.49% | 0.41% | 9.06% | 9.96% | 0.00% |
| val_7spk | 0.50% | 0.97% | 15.48% | 16.95% | 0.00% |
| val_8spk | 0.66% | 1.63% | 20.33% | 22.62% | 0.00% |
| alimeeting | 1.03% | 3.80% | 1.01% | 5.85% | 65.00% |
| ami_ihm_test | 1.48% | 7.79% | 1.71% | 10.98% | 68.75% |
| ami_sdm_test | 2.09% | 8.33% | 3.91% | 14.33% | 87.50% |
| callhome_eng | 5.73% | 1.29% | 0.37% | 7.39% | 87.86% |
| callhome_deu | 3.80% | 2.56% | 0.62% | 6.98% | 86.67% |
| callhome_jpn | 4.47% | 4.35% | 1.77% | 10.59% | 83.33% |
| callhome_spa | 5.25% | 7.58% | 5.09% | 17.92% | 72.14% |
| callhome_zho | 5.39% | 1.58% | 2.26% | 9.24% | 72.86% |
| kdomainconf_test30 | 6.06% | 3.48% | 7.85% | 17.39% | 46.67% |
| kdomainconf_val_3_4spk_test30 | 6.84% | 3.34% | 11.52% | 21.70% | 36.67% |
| kaddress_test30 | 0.00% | 7.74% | 0.00% | 7.74% | 100.00% |
| kemergency_test30 | 15.64% | 11.28% | 5.33% | 32.26% | 100.00% |
| **total** | - | - | - | **11.53%** | **66.81%** |
| **total (real)** | - | - | - | **13.53%** | **75.62%** |


## diar_streaming_sortformer_4spk-v2.1

Checkpoint: `diar_streaming_sortformer_4spk-v2.1/diar_streaming_sortformer_4spk-v2.1.nemo`

| dataset | FA | MISS | CER | DER | Spk_Count_Acc |
|---------|-----|------|-----|-----|---------------|
| val_2spk | 0.00% | 15.21% | 0.05% | 15.26% | 100.00% |
| val_3spk | 0.04% | 15.41% | 6.62% | 22.07% | 67.00% |
| val_4spk | 0.20% | 15.19% | 9.81% | 25.19% | 54.00% |
| val_5spk | 0.13% | 15.69% | 12.33% | 28.16% | 0.00% |
| val_6spk | 0.16% | 15.98% | 18.46% | 34.59% | 0.00% |
| val_7spk | 0.14% | 16.33% | 24.08% | 40.56% | 0.00% |
| val_8spk | 0.09% | 16.48% | 27.58% | 44.15% | 0.00% |
| alimeeting | 0.40% | 9.93% | 0.70% | 11.03% | 95.00% |
| ami_ihm_test | 0.50% | 23.51% | 2.03% | 26.05% | 93.75% |
| ami_sdm_test | 0.82% | 23.76% | 3.72% | 28.29% | 93.75% |
| callhome_eng | 1.84% | 2.85% | 0.25% | 4.94% | 83.57% |
| callhome_deu | 1.08% | 5.01% | 0.61% | 6.70% | 80.83% |
| callhome_jpn | 1.69% | 6.71% | 1.63% | 10.03% | 79.17% |
| callhome_spa | 2.75% | 18.76% | 1.76% | 23.27% | 63.57% |
| callhome_zho | 1.45% | 4.43% | 1.27% | 7.15% | 72.86% |
| kdomainconf_test30 | 2.96% | 11.65% | 13.23% | 27.84% | 0.00% |
| kdomainconf_val_3_4spk_test30 | 3.19% | 11.44% | 11.09% | 25.73% | 70.00% |
| kaddress_test30 | 0.00% | 10.79% | 0.00% | 10.79% | 100.00% |
| kemergency_test30 | 6.72% | 12.54% | 2.40% | 21.67% | 93.33% |
| **total** | - | - | - | **21.98%** | **62.03%** |
| **total (real)** | - | - | - | **16.96%** | **77.15%** |


## pyannote(mago_mstudio)

Artifact: `models/mstudio/speaker_diarization/m-dia` (pyannote-based Mago MStudio pipeline)

| dataset | FA | MISS | CER | DER | Spk_Count_Acc |
|---------|-----|------|-----|-----|---------------|
| val_2spk | 0.00% | 13.34% | 11.52% | 24.86% | 68.00% |
| val_3spk | 0.00% | 12.99% | 13.13% | 26.13% | 61.00% |
| val_4spk | 0.00% | 12.87% | 13.46% | 26.34% | 41.00% |
| val_5spk | 0.02% | 13.01% | 13.69% | 26.71% | 20.00% |
| val_6spk | 0.00% | 12.93% | 17.66% | 30.59% | 9.00% |
| val_7spk | 0.00% | 13.17% | 21.19% | 34.37% | 1.00% |
| val_8spk | 0.00% | 13.17% | 24.85% | 38.02% | 0.00% |
| alimeeting | 2.28% | 5.38% | 7.16% | 14.82% | 40.00% |
| ami_ihm_test | 1.76% | 6.79% | 3.58% | 12.13% | 25.00% |
| ami_sdm_test | 2.00% | 8.26% | 5.01% | 15.27% | 31.25% |
| callhome_eng | 1.80% | 7.35% | 6.70% | 15.86% | 58.57% |
| callhome_deu | 1.31% | 10.50% | 5.30% | 17.11% | 45.00% |
| callhome_jpn | 1.94% | 13.50% | 11.88% | 27.32% | 40.83% |
| callhome_spa | 2.93% | 14.33% | 7.63% | 24.89% | 47.86% |
| callhome_zho | 2.54% | 6.93% | 9.78% | 19.25% | 50.71% |
| kdomainconf_test30 | 2.58% | 10.45% | 2.66% | 15.69% | 50.00% |
| kdomainconf_val_3_4spk_test30 | 3.59% | 9.95% | 7.70% | 21.24% | 40.00% |
| kaddress_test30 | 0.10% | 11.72% | 2.55% | 14.37% | 93.33% |
| kemergency_test30 | 5.70% | 15.91% | 8.43% | 30.04% | 56.67% |
| **total** | - | - | - | **22.90%** | **41.01%** |
| **total (real)** | - | - | - | **19.00%** | **48.27%** |
