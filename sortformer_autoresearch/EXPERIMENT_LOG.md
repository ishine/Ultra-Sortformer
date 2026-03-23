# Sortformer Autoresearch 실험 기록

> 4-speaker Streaming Sortformer를 8-speaker로 확장하면서 DER을 최소화하는 자동화 실험 기록.
>
> **DER**(Diarization Error Rate)은 "누가 언제 말했는가"를 얼마나 틀렸는지 나타내는 지표. 0.29면 전체 발화의 29%에서 오류. 낮을수록 좋다. DER = FA + MISS + CER.
> - **FA**(False Alarm): 아무도 안 말하는데 "말하고 있다"고 판정한 비율
> - **MISS**: 말하고 있는데 놓친 비율
> - **CER**(Confusion Error): 말한 건 잡았지만 화자를 잘못 배정한 비율

- 브랜치: `sortformer-autoresearch/mar20`
- 기간: 2026-03-20 ~ 2026-03-23 (진행 중)
- GPU: NVIDIA H100 80GB x2 (DDP), 학습 2000 steps 고정

---

## DER 변화 요약

| Phase | 핵심 실험 | DER | FA | MISS | CER | 무엇이 바뀌었나 |
|-------|----------|-----|-----|------|-----|----------------|
| 출발 | baseline | 0.289 | 0.04 | 0.20 | 0.05 | 기본 4→8spk 확장 |
| 1 | exp15 | 0.260 | 0.05 | 0.17 | 0.04 | Threshold 최적화 |
| 2 | exp19 | 0.243 | 0.05 | 0.16 | 0.04 | Focal Loss + GELU + LayerScale |
| 3 | (21회 discard) | 0.243 | - | - | - | 하이퍼파라미터 미세조정 소진 |
| 4 | exp48 | 0.240 | 0.05 | 0.15 | 0.04 | 후처리 파이프라인 구축 |
| 5 | exp56 | 0.211 | 0.04 | 0.10 | 0.07 | **ALiBi 돌파구** + avg smoothing |
| 6 | exp65 | **0.203** | 0.04 | 0.10 | 0.06 | Decorrelation Loss로 CER↓ |

**총 개선: 0.289 → 0.203 (DER 29.7% 감소)**

---

## Phase 1: Threshold 튜닝 (baseline ~ exp16)

### baseline — DER 0.289

기본 코드 그대로 실행해서 출발점을 잡았다.

4spk 모델을 8spk로 확장하는 방법:
- **SVD 직교 초기화**: 기존 4명 화자의 가중치를 수학적으로 분해해서, 그 방향과 수직인 새 방향으로 5~8번 화자의 초기 가중치를 만든다. 기존 화자와 겹치지 않게 출발하도록.
- **Split LR(분리 학습률)**: 기존 화자(1~4)는 이미 잘 학습되어 있으니 낮은 학습률(1e-5)로, 새 화자(5~8)는 처음이니 높은 학습률(1e-4)로 학습.
- Loss는 **BCE**(각 프레임마다 "말한다/안 한다"를 판정하는 기본 손실 함수)를 사용하고, **ATS**(화자를 처음 말한 순서로 정렬)와 **PIL**(모든 화자 번호 조합 중 최적을 선택) 두 가지 방식을 50:50으로 섞어서 학습.

### exp1~16 — DER 0.289 → 0.260

스트리밍 추론 시 "이 프레임에서 화자가 말하고 있다"고 판단하는 **threshold**(기준값)를 조정했다. 기본 0.5에서 낮추면 더 많이 잡고(MISS↓), 높이면 덜 잡는다(FA↓).

핵심 발견: 기존 화자(1~4)와 새 화자(5~8)에 **서로 다른 threshold**를 써야 한다. 기존 화자는 예측이 정확하니 높게(0.54), 새 화자는 불확실하니 낮게(0.44). 16번 실험해서 이 조합을 찾았다.

---

## Phase 2: Loss 및 아키텍처 대규모 변경 (exp18~19)

### exp18 — DER 0.250

Threshold 조정만으로는 한계. 한 번에 5가지를 동시에 바꿨다:

1. **Focal Loss**: 기존 BCE는 모든 프레임을 동등하게 학습하는데, Focal Loss는 "모델이 이미 잘 맞추는 쉬운 프레임"의 가중치를 줄이고 "헷갈리는 어려운 프레임"에 집중한다. 여러 명이 겹쳐서 말하는 구간처럼 판단이 어려운 곳에서 더 열심히 학습하게 된다.
2. **LayerScale**: Transformer 각 레이어의 출력에 학습 가능한 스케일링 값을 곱해서, 레이어마다 "내 출력을 얼마나 반영할지"를 스스로 조절하게 하는 기법. 깊은 네트워크의 학습 안정성 향상.
3. **GELU**: 기존 ReLU(0 이하를 완전 차단) 대신 부드럽게 차단하는 활성화 함수. Transformer에서 ReLU보다 성능이 좋은 경우가 많다.
4. **SVD+noise**: 새 화자 초기화에 작은 랜덤 노이즈를 추가해서 다양성 부여.
5. **PIL/ATS 비율 55:45**: PIL(유연한 화자 배정)의 비중을 살짝 높임.

### exp19 — DER 0.243

Focal Loss의 **gamma**(어려운 예제에 얼마나 집중할지)를 2.0→1.5로 줄였다. 너무 어려운 것에만 몰아주면 오히려 불안정해서, 적당히 집중하는 1.5가 최적.

---

## Phase 3: 정체기 (exp20~40) — 21회 연속 실패

exp19가 잘 되니까 비슷한 파라미터를 하나씩 조금씩 바꿔봤다. gamma ±0.01, LR ±10%, warmup, grad clip, seed, session 길이 등. **21회 연속 전부 실패.** 이 접근이 완전히 소진됐다는 신호였고, 사람이 개입해서 전략을 바꿨다.

시도한 것 중 주요한 것들:
- **TV Loss**: 시간축에서 예측이 급변하면 페널티를 주는 보조 loss → 효과 없음
- **Label smoothing**: 정답을 0/1 대신 0.02/0.98로 부드럽게 → Focal Loss와 상성 나쁨
- gamma 1.25~1.65 범위 전부 탐색 → 1.5가 최적 확정

---

## Phase 4: 후처리 파이프라인 구축 (exp42~48) — DER 0.243 → 0.240

MISS가 DER의 66%를 차지하는 주요 병목. 모델 학습은 건드리지 않고, 모델이 이미 출력한 예측값을 시간축으로 다듬어서 MISS를 줄이는 방향으로 전환. eval 시에만 적용되고 학습에는 영향 없다.

최종 확정된 후처리 4단계:

```
raw preds → Median Filter → Morph Close → Gap Fill → Avg Smooth → final preds
```

1. **Temporal Median Filter (k=9)**: 각 프레임 예측값을 앞뒤 4프레임씩 총 9개의 중앙값으로 교체. 갑자기 튀는 노이즈를 제거. "말함-안말함-말함-말함-말함" → "말함-말함-말함-말함-말함".
2. **Morphological Close (k=5)**: 이미지 처리에서 빌려온 기법. 팽창(주변에 1 있으면 채움) → 수축(주변에 0 있으면 빼냄) 순서로 적용. 발화 중간의 작은 구멍을 메운다.
3. **Interior Gap Fill (max 6프레임)**: 양쪽에 발화가 있는 짧은 침묵을 발화로 채우기. "말함—쉼—말함" → "말함—말함—말함". 짧은 쉼을 silence로 잘못 잡은 것을 복구.
4. **Temporal Avg Smooth (k=11)**: 주변 11프레임의 산술 평균으로 교체. 전체적으로 매끄럽게 다듬기. (Phase 5에서 추가)

이 단계에서 참고할 만한 실험: exp44에서 **SwiGLU**(FFN을 게이트 구조로 바꾸는 기법. 입력을 두 갈래로 나눠 하나는 "열고 닫기", 하나는 "값"으로 사용)를 시도했는데, MISS가 0.13까지 줄었지만 FA/CER 증가로 전체 DER은 상승. SwiGLU 자체의 효과는 확인됨.

---

## Phase 5: ALiBi 돌파구 (exp49~57) — DER 0.240 → 0.211

### 실패한 시도들

- **exp49 — Logit Mixing**: 8명 화자 출력 뒤에 Linear(8→8)를 추가해서 화자 간 관계를 학습시키려 했으나 효과 없음.
- **exp50 — 초기화 변경**: SVD+복사 혼합 초기화 → 기존보다 악화.
- **exp51 — Adapter Layer**: 기존 가중치는 두고 작은 보조 네트워크(큰 차원→작은 차원→큰 차원)를 Transformer 블록마다 병렬로 붙임 → 효과 없음.

### exp52 — ALiBi ★ 전체 실험 최대 돌파구 (DER 0.213)

기존 Transformer는 **위치 정보가 전혀 없어서** "100프레임 전"이나 "바로 옆 프레임"을 동등하게 취급했다. 하지만 화자 다이어리제이션에서는 가까운 프레임이 훨씬 중요하다.

**ALiBi (Attention with Linear Biases)**: Attention 점수에 "거리 × 기울기"만큼 감점을 주는 기법. 가까운 프레임은 감점이 적고, 먼 프레임은 많이 감점. 각 attention head마다 기울기(slope)가 다르며, 이 slope이 학습 가능(learnable).

```
일반 attention:  점수 = Query·Key
ALiBi attention: 점수 = Query·Key - slope × |거리|
→ 가까운 프레임: 감점 작음 → 높은 attention
→ 먼 프레임: 감점 큼 → 낮은 attention
```

결과: **단일 실험으로 DER 0.0264 감소** (전체 67회 중 최대). MISS 0.15→0.11.

이후 avg smoothing 커널을 k=5~13까지 시도해서 k=11에서 최적(DER 0.2111). 다만 CER이 0.04→0.07로 올라가는 부작용 발생.

---

## Phase 6: CER 감소 전략 (exp58~67) — DER 0.211 → 0.203

ALiBi로 MISS는 크게 줄었지만, CER(화자 혼동)이 0.04→0.07로 악화. 서로 다른 화자를 구별하는 능력을 개선해야 했다.

### 실패한 시도들

- **exp58~61 — SwiGLU 조합**: ALiBi와 SwiGLU를 합쳤더니 VRAM만 14.7→17.1GB 늘고 DER 악화. 이 모델 크기에서는 SwiGLU가 과도.
- **exp60 — decorr λ=0.015**: 너무 강한 제약 → 효과 없음.
- **exp63~64 — Speaker Count Head**: "지금 몇 명이 동시에 말하는가"를 추가로 학습시키는 보조 head. MISS는 0.09까지 줄었지만 CER 0.08로 악화. 보조 head가 메인 학습을 방해.

### exp62, exp65 — Decorrelation Loss (DER 0.203, 현재 최고)

**Decorrelation Loss**: 화자 출력 레이어에서 각 화자의 가중치 벡터끼리 내적했을 때, 다른 화자끼리 내적이 0에 가깝도록(= 서로 수직이 되도록) 페널티를 주는 보조 loss. 화자 가중치가 비슷해지면 혼동이 생기니까, 서로 다른 방향을 유지하도록 강제하는 것.

λ=0.005(약하게)가 효과적. CER 0.07→0.06으로 감소 성공. MISS와 FA는 유지.

### 진행 중

- **exp66 — RoPE**: ALiBi 대신 **RoPE(Rotary Position Embedding)**을 시도. ALiBi가 거리에 "감점"을 주는 방식이라면, RoPE는 Query와 Key 벡터를 위치에 따라 "회전"시키는 방식. 가까운 프레임끼리 회전 각도가 비슷해서 자연스럽게 높은 attention.
- **exp67 — Local Attention**: ALiBi는 먼 프레임을 깎지만 완전히 차단하지는 않는다. 아예 ±96프레임(약 7.7초) 밖은 참조 자체를 못 하도록 강하게 제한. 현재 실행 중.

---

## 현재 최적 모델 (exp65)

### baseline 대비 변경된 것

| 영역 | 변경 내용 |
|------|----------|
| Loss | BCE → **Focal Loss** (gamma=1.5) |
| 활성화 함수 | ReLU → **GELU** |
| Transformer | **LayerScale** 추가 (레이어별 학습 가능 스케일링) |
| Attention | **ALiBi** 추가 (거리 기반 감점, learnable slope) |
| 초기화 | SVD → **SVD + noise** (0.05) |
| 보조 Loss | **Decorrelation Loss** (화자 가중치 직교화, λ=0.005) |
| Loss 비율 | ATS:PIL 50:50 → **45:55** |
| Threshold | 전체 0.50 → base **0.54** / new **0.44** |

### 후처리 파이프라인 (eval 시에만)

```
Median Filter (k=9) → Morph Close (k=5) → Gap Fill (max 6f) → Avg Smooth (k=11)
```

---

## 교훈

1. **하이퍼파라미터 미세조정은 빠르게 소진된다** — 21회 연속 실패로 입증
2. **아키텍처 변경이 가장 큰 점프를 만든다** — exp18(Loss+구조 변경), exp52(ALiBi)가 최대 개선
3. **ALiBi가 diarization에 매우 효과적** — "가까운 프레임이 중요하다"는 직관이 이 task에 딱 맞음
4. **한 번에 너무 많이 바꾸면 실패** — exp58(4가지 동시)은 실패, exp62(decorr 단독)은 성공
5. **MISS와 CER은 trade-off** — MISS를 줄이면 CER이 올라가는 경향. 균형이 중요
6. **보조 Loss는 약하게** — λ=0.005 성공, λ=0.015 실패
7. **DDP에서 조기 종료는 rank0만 판단 후 broadcast** — `debiased`가 랭크마다 달라 한쪽만 `break`하면 NCCL 타임아웃. 재활성화 시 `dist.broadcast(stop_tensor, src=0)` 경로 사용
8. **Progressive unfreeze (exp69)** — 500/1000/12–17 스케줄 단독 적용 시 DER **0.2209**로 exp65(0.2032) 대비 악화 → 1순위 추가 실험(스케줄 변형) 또는 2순위(overlap loss) 검토
