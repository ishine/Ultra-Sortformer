# Sortformer Evaluation Results

## 데이터셋 설명

| dataset | 설명 | 언어 |
|---------|------|------|
| val_2spk ~ val_8spk | 합성 검증 데이터 (2~8화자, 90초, silence/overlap 포함) | 한국어 |
| alimeeting | AliMeeting 회의 음성 | 중국어 |
| ami_ihm_test | AMI 코퍼스 IHM (개별 헤드셋 마이크) test | 영어 |
| ami_sdm_test | AMI 코퍼스 SDM (단일 원거리 마이크) test | 영어 |
| callhome_eng | CallHome 영어 | 영어 |
| callhome_deu | CallHome 독일어 | 독일어 |
| callhome_jpn | CallHome 일본어 | 일본어 |
| callhome_spa | CallHome 스페인어 | 스페인어 |
| callhome_zho | CallHome 중국어 | 중국어 |
| kdomainconf_test30 | kdomainconf 5화자 30개 샘플 | 한국어 |
| kdomainconf_val_3_4spk_test30 | kdomainconf validation 3~4화자 30개 샘플 | 한국어 |
| kaddress_test30 | kaddress 주소 음성 30개 샘플 | 한국어 |
| kemergency_test30 | kemergency 긴급 음성 30개 샘플 | 한국어 |

## diar_streaming_sortformer_4spk-v2.1

Model: `/home/devsy/workspace/diar_streaming_sortformer_4spk-v2.1/diar_streaming_sortformer_4spk-v2.1.nemo`

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

## sortformer_5spk_full_v1

Model: `/home/devsy/workspace/streaming_sortformer_diar_train/sortformer_5spk_full_v1/checkpoints/sortformer_5spk_full_v1.nemo`

| dataset | FA | MISS | CER | DER | Spk_Count_Acc |
|---------|-----|------|-----|-----|---------------|
| val_2spk | 0.01% | 0.00% | 0.01% | 0.01% | 100.00% |
| val_3spk | 0.01% | 0.00% | 0.61% | 0.62% | 94.00% |
| val_4spk | 0.06% | 0.00% | 0.83% | 0.88% | 97.00% |
| val_5spk | 0.08% | 0.21% | 2.53% | 2.81% | 83.00% |
| val_6spk | 0.49% | 0.29% | 8.14% | 8.93% | 0.00% |
| val_7spk | 0.42% | 0.75% | 14.80% | 15.97% | 0.00% |
| val_8spk | 0.91% | 1.22% | 19.72% | 21.85% | 0.00% |
| alimeeting | 3.51% | 2.74% | 4.47% | 10.73% | 15.00% |
| ami_ihm_test | 2.43% | 5.87% | 3.51% | 11.81% | 56.25% |
| ami_sdm_test | 3.61% | 7.35% | 8.87% | 19.84% | 18.75% |
| callhome_eng | 5.38% | 1.66% | 1.82% | 8.86% | 33.57% |
| callhome_deu | 2.80% | 3.33% | 1.80% | 7.93% | 42.50% |
| callhome_jpn | 3.21% | 6.01% | 3.82% | 13.05% | 40.83% |
| callhome_spa | 4.09% | 17.56% | 3.62% | 25.27% | 40.00% |
| callhome_zho | 5.37% | 2.16% | 4.90% | 12.43% | 28.57% |
| kdomainconf_test30 | 7.13% | 3.05% | 10.51% | 20.69% | 76.67% |
| kdomainconf_val_3_4spk_test30 | 9.46% | 2.72% | 10.98% | 23.16% | 16.67% |
| kaddress_test30 | 0.00% | 8.22% | 0.00% | 8.22% | 100.00% |
| kemergency_test30 | 15.75% | 12.02% | 7.14% | 34.91% | 86.67% |

## sortformer_5spk_full_v1--val_f1_acc=0.9559-epoch=16

Model: `streaming_sortformer_diar_train/sortformer_5spk_full_v1/checkpoints/sortformer_5spk_full_v1--val_f1_acc=0.9559-epoch=16.ckpt`

학습 데이터:

| 데이터셋 | Train | Val |
|----------|-------|-----|
| AMI | 200 | 50 |
| 합성_2spk | 200 | 50 |
| 합성_3spk | 200 | 50 |
| 합성_4spk | 200 | 50 |
| 합성_5spk | 200 | 50 |

train: `train_sampled_2to5spk_250.json` | val: `val_sampled_2to5spk_250.json`

| dataset | FA | MISS | CER | DER | Spk_Count_Acc |
|---------|-----|------|-----|-----|---------------|
| val_2spk | 0.04% | 0.01% | 0.10% | 0.15% | 98.00% |
| val_3spk | 0.24% | 0.00% | 0.65% | 0.90% | 94.00% |
| val_4spk | 0.18% | 0.16% | 1.61% | 1.95% | 91.00% |
| val_5spk | 0.10% | 0.59% | 3.56% | 4.25% | 74.00% |
| val_6spk | 0.15% | 0.89% | 9.97% | 11.01% | 0.00% |
| val_7spk | 0.20% | 2.04% | 15.41% | 17.65% | 0.00% |
| val_8spk | 0.33% | 3.02% | 20.31% | 23.66% | 0.00% |
| alimeeting | 2.55% | 3.83% | 2.12% | 8.50% | 65.00% |
| ami_ihm_test | 1.28% | 10.27% | 7.19% | 18.74% | 68.75% |
| ami_sdm_test | 2.05% | 13.33% | 6.28% | 21.66% | 75.00% |
| callhome_eng | 5.91% | 1.37% | 0.51% | 7.79% | 88.57% |
| callhome_deu | 4.40% | 2.73% | 1.05% | 8.18% | 80.00% |
| callhome_jpn | 4.66% | 5.00% | 2.00% | 11.66% | 85.00% |
| callhome_spa | 4.99% | 17.15% | 2.11% | 24.26% | 66.43% |
| callhome_zho | 6.14% | 1.92% | 2.59% | 10.64% | 72.14% |
| kdomainconf_test30 | 5.99% | 4.02% | 9.75% | 19.76% | 66.67% |
| kdomainconf_val_3_4spk_test30 | 6.47% | 3.98% | 13.37% | 23.83% | 30.00% |
| kaddress_test30 | 0.00% | 7.15% | 0.00% | 7.15% | 100.00% |
| kemergency_test30 | 13.30% | 12.42% | 3.84% | 29.57% | 93.33% |

## sortformer_5spk_full_v1_expanded_L10-16

Model: `/home/devsy/workspace/streaming_sortformer_diar_train/sortformer_5spk_full_v1/checkpoints/sortformer_5spk_full_v1_expanded_L10-16.nemo`

| dataset | FA | MISS | CER | DER | Spk_Count_Acc |
|---------|-----|------|-----|-----|---------------|
| val_2spk | 0.00% | 0.00% | 0.03% | 0.03% | 100.00% |
| val_3spk | 0.00% | 0.08% | 0.55% | 0.63% | 93.00% |
| val_4spk | 0.01% | 8.80% | 1.76% | 10.57% | 96.00% |
| val_5spk | 0.01% | 15.37% | 2.50% | 17.88% | 35.00% |
| val_6spk | 0.15% | 15.04% | 7.08% | 22.28% | 0.00% |
| val_7spk | 0.02% | 16.76% | 11.40% | 28.19% | 0.00% |
| val_8spk | 0.16% | 15.69% | 15.48% | 31.33% | 0.00% |
| alimeeting | 3.17% | 3.08% | 3.31% | 9.55% | 30.00% |
| ami_ihm_test | 2.44% | 7.72% | 3.87% | 14.03% | 81.25% |
| ami_sdm_test | 3.27% | 10.23% | 7.82% | 21.33% | 37.50% |
| callhome_eng | 5.04% | 1.72% | 1.93% | 8.69% | 37.86% |
| callhome_deu | 2.68% | 3.48% | 2.15% | 8.30% | 35.83% |
| callhome_jpn | 3.11% | 6.35% | 4.05% | 13.51% | 41.67% |
| callhome_spa | 3.98% | 18.01% | 3.39% | 25.38% | 35.71% |
| callhome_zho | 5.12% | 2.52% | 4.63% | 12.28% | 29.29% |
| kdomainconf_test30 | 6.18% | 20.84% | 11.06% | 38.07% | 70.00% |
| kdomainconf_val_3_4spk_test30 | 8.99% | 13.13% | 10.65% | 32.77% | 20.00% |
| kaddress_test30 | 0.00% | 8.38% | 0.00% | 8.38% | 100.00% |
| kemergency_test30 | 15.26% | 11.68% | 7.86% | 34.79% | 80.00% |

## sortformer_5spk_splitlr_1e5_1e4

Model: `/home/devsy/workspace/streaming_sortformer_diar_train/sortformer_5spk_splitlr_1e5_1e4/checkpoints/sortformer_5spk_splitlr_1e5_1e4.nemo`

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

## sortformer_6spk_splitlr_1e5_1e4

Model: `/home/devsy/workspace/streaming_sortformer_diar_train/sortformer_6spk_splitlr_1e5_1e4/checkpoints/sortformer_6spk_splitlr_1e5_1e4.nemo`

| dataset | FA | MISS | CER | DER | Spk_Count_Acc |
|---------|-----|------|-----|-----|---------------|
| val_2spk | 0.00% | 0.01% | 0.09% | 0.10% | 98.00% |
| val_3spk | 0.00% | 0.17% | 0.64% | 0.81% | 96.00% |
| val_4spk | 0.00% | 0.13% | 1.30% | 1.43% | 93.00% |
| val_5spk | 0.10% | 0.24% | 2.58% | 2.93% | 79.00% |
| val_6spk | 0.33% | 0.15% | 4.88% | 5.36% | 66.00% |
| val_7spk | 0.44% | 0.22% | 8.86% | 9.52% | 0.00% |
| val_8spk | 0.53% | 0.54% | 14.19% | 15.25% | 0.00% |
| alimeeting | 1.12% | 3.68% | 1.44% | 6.24% | 75.00% |
| ami_ihm_test | 1.62% | 7.22% | 1.35% | 10.19% | 37.50% |
| ami_sdm_test | 2.44% | 7.74% | 8.92% | 19.11% | 43.75% |
| callhome_eng | 6.03% | 1.30% | 0.47% | 7.79% | 85.00% |
| callhome_deu | 4.01% | 2.64% | 0.81% | 7.47% | 84.17% |
| callhome_jpn | 4.71% | 4.21% | 1.92% | 10.83% | 85.83% |
| callhome_spa | 5.18% | 7.81% | 6.05% | 19.04% | 67.14% |
| callhome_zho | 5.51% | 1.61% | 2.18% | 9.30% | 72.14% |
| kdomainconf_test30 | 6.39% | 3.65% | 10.90% | 20.94% | 26.67% |
| kdomainconf_val_3_4spk_test30 | 7.22% | 4.03% | 11.79% | 23.04% | 26.67% |
| kaddress_test30 | 0.00% | 7.30% | 0.00% | 7.30% | 93.33% |
| kemergency_test30 | 16.95% | 11.24% | 5.32% | 33.51% | 100.00% |

## sortformer_6spk_splitlr_1e5_1e4_v2

Model: `/home/devsy/workspace/streaming_sortformer_diar_train/sortformer_6spk_splitlr_1e5_1e4_v2/checkpoints/sortformer_6spk_splitlr_1e5_1e4_v2.nemo`

| dataset | FA | MISS | CER | DER | Spk_Count_Acc |
|---------|-----|------|-----|-----|---------------|
| val_2spk | 0.00% | 0.00% | 0.00% | 0.00% | 99.00% |
| val_3spk | 0.02% | 0.00% | 0.47% | 0.48% | 99.00% |
| val_4spk | 0.10% | 0.00% | 0.77% | 0.88% | 98.00% |
| val_5spk | 0.25% | 0.09% | 2.05% | 2.38% | 81.00% |
| val_6spk | 0.25% | 0.06% | 3.91% | 4.22% | 69.00% |
| val_7spk | 0.42% | 0.71% | 8.24% | 9.37% | 0.00% |
| val_8spk | 0.58% | 1.66% | 12.25% | 14.49% | 0.00% |
| alimeeting | 1.09% | 3.76% | 1.43% | 6.28% | 80.00% |
| ami_ihm_test | 1.67% | 7.43% | 4.47% | 13.58% | 43.75% |
| ami_sdm_test | 2.26% | 7.47% | 5.09% | 14.81% | 50.00% |
| callhome_eng | 5.86% | 1.32% | 0.72% | 7.89% | 82.86% |
| callhome_deu | 3.80% | 2.74% | 1.14% | 7.68% | 80.00% |
| callhome_jpn | 4.33% | 4.66% | 2.07% | 11.06% | 87.50% |
| callhome_spa | 5.06% | 8.39% | 5.92% | 19.38% | 65.00% |
| callhome_zho | 5.15% | 1.70% | 2.38% | 9.24% | 72.14% |
| kdomainconf_test30 | 6.68% | 3.31% | 9.56% | 19.55% | 33.33% |
| kdomainconf_val_3_4spk_test30 | 7.76% | 3.29% | 10.90% | 21.95% | 36.67% |
| kaddress_test30 | 0.00% | 7.04% | 0.00% | 7.04% | 93.33% |
| kemergency_test30 | 16.42% | 11.27% | 3.58% | 31.26% | 96.67% |

## sortformer_6spk_v2_expanded_L14-17

Model: `sortformer_6spk_v2_expanded_L14-17.nemo`

| dataset | FA | MISS | CER | DER | Spk_Count_Acc |
|---------|-----|------|-----|-----|---------------|
| val_2spk | 0.00% | 0.00% | 0.00% | 0.00% | 100.00% |
| val_3spk | 0.00% | 0.07% | 0.77% | 0.84% | 97.00% |
| val_4spk | 0.00% | 0.95% | 0.69% | 1.64% | 98.00% |
| val_5spk | 0.00% | 1.40% | 2.24% | 3.64% | 71.00% |
| val_6spk | 0.11% | 2.96% | 4.11% | 7.18% | 52.00% |
| val_7spk | 0.01% | 6.31% | 6.40% | 12.71% | 0.00% |
| val_8spk | 0.00% | 10.42% | 9.51% | 19.93% | 0.00% |
| alimeeting | 1.06% | 5.36% | 1.17% | 7.59% | 90.00% |
| ami_ihm_test | 1.22% | 10.86% | 3.18% | 15.26% | 87.50% |
| ami_sdm_test | 1.39% | 12.14% | 4.21% | 17.74% | 87.50% |
| callhome_eng | 6.19% | 1.33% | 0.48% | 8.00% | 90.00% |
| callhome_deu | 4.00% | 2.84% | 0.80% | 7.64% | 90.83% |
| callhome_jpn | 4.71% | 4.72% | 1.98% | 11.40% | 88.33% |
| callhome_spa | 5.16% | 9.08% | 5.52% | 19.76% | 69.29% |
| callhome_zho | 5.27% | 2.18% | 3.12% | 10.57% | 72.86% |
| kdomainconf_test30 | 5.53% | 5.06% | 9.78% | 20.38% | 16.67% |
| kdomainconf_val_3_4spk_test30 | 6.44% | 7.11% | 10.15% | 23.69% | 36.67% |
| kaddress_test30 | 0.00% | 7.09% | 0.01% | 7.10% | 86.67% |
| kemergency_test30 | 16.55% | 11.22% | 3.20% | 30.97% | 100.00% |

## sortformer_6spk_v2_expanded_L16-17

Model: `sortformer_6spk_v2_expanded_L16-17.nemo`

| dataset | FA | MISS | CER | DER | Spk_Count_Acc |
|---------|-----|------|-----|-----|---------------|
| val_2spk | 0.00% | 0.00% | 0.02% | 0.03% | 99.00% |
| val_3spk | 0.09% | 0.00% | 0.54% | 0.63% | 98.00% |
| val_4spk | 0.00% | 0.12% | 0.95% | 1.07% | 97.00% |
| val_5spk | 0.19% | 0.34% | 1.76% | 2.29% | 82.00% |
| val_6spk | 0.05% | 1.04% | 3.98% | 5.07% | 59.00% |
| val_7spk | 0.08% | 4.24% | 6.38% | 10.70% | 0.00% |
| val_8spk | 0.04% | 5.13% | 10.73% | 15.90% | 0.00% |
| alimeeting | 0.92% | 4.96% | 1.16% | 7.04% | 80.00% |
| ami_ihm_test | 1.14% | 11.26% | 2.57% | 14.98% | 62.50% |
| ami_sdm_test | 1.29% | 12.93% | 5.93% | 20.14% | 75.00% |
| callhome_eng | 6.05% | 1.33% | 0.45% | 7.83% | 87.86% |
| callhome_deu | 3.79% | 2.86% | 0.62% | 7.27% | 91.67% |
| callhome_jpn | 4.56% | 4.75% | 1.66% | 10.97% | 89.17% |
| callhome_spa | 5.07% | 8.64% | 5.51% | 19.23% | 70.00% |
| callhome_zho | 5.14% | 1.90% | 2.03% | 9.07% | 75.71% |
| kdomainconf_test30 | 5.96% | 3.99% | 8.95% | 18.90% | 43.33% |
| kdomainconf_val_3_4spk_test30 | 6.75% | 5.74% | 10.43% | 22.92% | 33.33% |
| kaddress_test30 | 0.00% | 7.02% | 0.00% | 7.02% | 96.67% |
| kemergency_test30 | 16.37% | 11.20% | 3.23% | 30.79% | 100.00% |

## sortformer_6spk_v2_expanded_L8-9

Model: `sortformer_6spk_v2_expanded_L8-9.nemo`

| dataset | FA | MISS | CER | DER | Spk_Count_Acc |
|---------|-----|------|-----|-----|---------------|
| val_2spk | 0.00% | 0.00% | 0.00% | 0.00% | 100.00% |
| val_3spk | 0.03% | 0.00% | 0.40% | 0.43% | 99.00% |
| val_4spk | 0.16% | 0.00% | 0.70% | 0.86% | 97.00% |
| val_5spk | 0.31% | 0.05% | 2.14% | 2.50% | 75.00% |
| val_6spk | 0.39% | 0.45% | 3.59% | 4.43% | 71.00% |
| val_7spk | 0.70% | 1.48% | 8.26% | 10.43% | 0.00% |
| val_8spk | 0.43% | 2.51% | 12.23% | 15.16% | 0.00% |
| alimeeting | 1.18% | 4.02% | 1.27% | 6.48% | 75.00% |
| ami_ihm_test | 1.83% | 7.00% | 2.95% | 11.78% | 56.25% |
| ami_sdm_test | 2.57% | 7.66% | 6.98% | 17.21% | 56.25% |
| callhome_eng | 6.20% | 1.27% | 0.73% | 8.20% | 86.43% |
| callhome_deu | 4.07% | 2.68% | 1.24% | 7.99% | 82.50% |
| callhome_jpn | 4.53% | 4.56% | 1.84% | 10.93% | 85.83% |
| callhome_spa | 5.26% | 8.16% | 5.97% | 19.39% | 65.00% |
| callhome_zho | 5.37% | 1.69% | 2.67% | 9.73% | 68.57% |
| kdomainconf_test30 | 6.48% | 3.77% | 10.68% | 20.94% | 16.67% |
| kdomainconf_val_3_4spk_test30 | 7.95% | 3.60% | 10.02% | 21.57% | 33.33% |
| kaddress_test30 | 0.00% | 7.12% | 0.00% | 7.12% | 93.33% |
| kemergency_test30 | 17.17% | 11.27% | 3.25% | 31.68% | 96.67% |

## sortformer_6spk_v2_expanded_L8-9_L14-17

Model: `sortformer_6spk_v2_expanded_L8-9_L14-17.nemo`

| dataset | FA | MISS | CER | DER | Spk_Count_Acc |
|---------|-----|------|-----|-----|---------------|
| val_2spk | 0.00% | 0.00% | 0.00% | 0.00% | 100.00% |
| val_3spk | 0.00% | 0.05% | 0.78% | 0.83% | 96.00% |
| val_4spk | 0.01% | 1.53% | 0.65% | 2.19% | 97.00% |
| val_5spk | 0.02% | 1.79% | 1.98% | 3.79% | 68.00% |
| val_6spk | 0.01% | 4.82% | 4.19% | 9.02% | 50.00% |
| val_7spk | 0.03% | 8.73% | 6.50% | 15.25% | 0.00% |
| val_8spk | 0.01% | 12.03% | 8.79% | 20.83% | 0.00% |
| alimeeting | 1.13% | 5.55% | 1.38% | 8.05% | 85.00% |
| ami_ihm_test | 1.21% | 12.32% | 2.16% | 15.69% | 75.00% |
| ami_sdm_test | 1.36% | 13.63% | 5.76% | 20.76% | 81.25% |
| callhome_eng | 6.44% | 1.34% | 0.52% | 8.30% | 90.71% |
| callhome_deu | 4.28% | 2.75% | 0.86% | 7.89% | 90.00% |
| callhome_jpn | 4.92% | 4.61% | 2.03% | 11.56% | 88.33% |
| callhome_spa | 5.34% | 8.64% | 5.42% | 19.40% | 69.29% |
| callhome_zho | 5.57% | 2.02% | 2.73% | 10.32% | 74.29% |
| kdomainconf_test30 | 5.37% | 6.87% | 8.59% | 20.83% | 10.00% |
| kdomainconf_val_3_4spk_test30 | 6.51% | 9.30% | 9.96% | 25.77% | 40.00% |
| kaddress_test30 | 0.00% | 7.14% | 0.01% | 7.15% | 86.67% |
| kemergency_test30 | 16.42% | 11.67% | 3.17% | 31.26% | 96.67% |

## sortformer_7spk_splitlr_1e5_1e4

Model: `streaming_sortformer_diar_train/sortformer_7spk_splitlr_1e5_1e4/checkpoints/sortformer_7spk_splitlr_1e5_1e4.nemo`

| dataset | FA | MISS | CER | DER | Spk_Count_Acc |
|---------|-----|------|-----|-----|---------------|
| val_2spk | 0.00% | 0.00% | 0.01% | 0.01% | 100.00% |
| val_3spk | 0.03% | 0.01% | 0.41% | 0.45% | 96.00% |
| val_4spk | 0.10% | 0.00% | 0.62% | 0.73% | 96.00% |
| val_5spk | 0.09% | 0.14% | 1.92% | 2.15% | 83.00% |
| val_6spk | 0.26% | 0.17% | 3.13% | 3.56% | 68.00% |
| val_7spk | 0.59% | 0.24% | 5.70% | 6.53% | 58.00% |
| val_8spk | 0.70% | 0.30% | 9.42% | 10.42% | 0.00% |
| alimeeting | 1.09% | 3.60% | 1.79% | 6.48% | 85.00% |
| ami_ihm_test | 1.83% | 6.88% | 6.29% | 15.01% | 37.50% |
| ami_sdm_test | 2.36% | 8.19% | 7.35% | 17.90% | 56.25% |
| callhome_eng | 6.72% | 1.22% | 0.65% | 8.59% | 91.43% |
| callhome_deu | 4.46% | 2.51% | 0.76% | 7.72% | 90.00% |
| callhome_jpn | 5.32% | 4.07% | 1.82% | 11.21% | 90.00% |
| callhome_spa | 5.33% | 12.07% | 3.70% | 21.11% | 67.86% |
| callhome_zho | 6.14% | 1.47% | 3.17% | 10.77% | 75.00% |
| kdomainconf_test30 | 7.32% | 2.99% | 8.69% | 19.00% | 16.67% |
| kdomainconf_val_3_4spk_test30 | 8.40% | 3.31% | 10.85% | 22.56% | 26.67% |
| kaddress_test30 | 0.00% | 7.81% | 1.99% | 9.80% | 93.33% |
| kemergency_test30 | 18.27% | 11.16% | 4.00% | 33.43% | 96.67% |

