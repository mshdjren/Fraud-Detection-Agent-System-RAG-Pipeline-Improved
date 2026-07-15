# README 교체 섹션 — 학습 데이터 생성 / 모델 학습 / 평가 체계

> 기존 README의 `2️⃣ 학습 데이터 생성`, `5️⃣ LLM Analyzer 경량화 및 평가 체계` 를 대체하는 섹션입니다.
> 표의 **빈칸 = 미실행**(실험 후 채움), 각 수치에는 측정 출처(run/노트북)를 명시했습니다.

---

## 2️⃣ 학습 데이터 생성 — Mining + Prompt v2 + 3중 Rejection Sampling

### 설계 원칙

> **"규칙으로 정확히 계산되는 신호는 코드가, 서술만 LLM이."**
> LLM 출력 key 수 = 검증 불가능한 자유도. 자유도를 줄이고, 남은 자유도는 **사실 검증 가능한 형태**로 강제한다.

데이터 생성은 `TEACHER_TASK` 스위치 하나로 **agent evaluator / forwardRAG 두 모드를 단일 노트북**(NB01-04 통합)에서 생성합니다. 샘플링·z-table·balance·eval 분리·진단 로직은 **완전히 동일**하고, prompt(`common/prompts.py`)·출력 key·rejection 기준만 분기합니다.

| | forwardRAG (baseline) | Agent Evaluator |
|---|---|---|
| 출력 key | `{cls, confidence, reasoning}` (3-key) | `{cls, sufficiency, what_missing}` (3-key) |
| LLM 자유 출력 | `reasoning` | `what_missing` **only** |
| deterministic 주입 | `confidence` (nd 재라벨) | `sufficiency` (nd-band), `cls`(confident 구간) |
| 산출물 | `training_data_forward{,_eval}.jsonl` | `training_data_mined{,_eval}.jsonl` |

### (1) Prompt v2 — "통계는 시스템이 계산하고, 해석만 남긴다"

논문(Multi-Cluster Memory Bank)의 판정 메커니즘상 **이웃의 원시 feature는 점수 계산에 등장하지 않습니다**(거리로만 기여). v1 프롬프트는 이 원칙을 3군데서 위반하고 있었고, 이를 제거한 것이 v2입니다.

| 위반 | v1 (문제) | v2 (교체) | 근거 |
|---|---|---|---|
| ① 이웃 원시 덤프 | top-3 이웃 전체 feature (~150-200 tok) | `top-k agrees: … \| differs: …` 다이제스트 1줄 (~15 tok) | 3B조차 교차대조 실패 — 이웃 전원이 `source=Ads`인데 "source deviates" 환각. 원시 덤프 = latency 비용 + 오답 유도 재료 |
| ② 원시 통계표 | mean/std/min/max → 모델이 z를 암산 | **Deviation Table**: `purchase_value: z=+0.79 [ok]` 사전계산 | 소형 모델 산술 신뢰도 낮음. z 사전제공 시 고유율 18.9→75.8% (NB04 실측) |
| ③ 판정/설명 재료 미분리 | 5-bullet Decision criteria | nd band 가이드만 유지(캘리브레이션 앵커) | cls는 nd가 결정(det) — 자유도만 열어줌 |

**레이아웃 = `[정적 instruction 선두][가변 TEST/FIT/RETRIEVAL 후미]`** — llama.cpp prefix KV cache 정합 목적. 실측 전체 ~230 tok(v1 대비 절반), 그중 **정적 ~120 tok이 캐시 대상**.

`prompts.py`는 **학습·서빙 단일 소스**입니다. 블록 순서 변경은 NB 재생성 → NB02 재학습과 같은 사이클에서만 반영해 train-serve 정합을 유지합니다.

### (2) Pipeline-aware Hard Negative Mining

파이프라인이 실제 틀리는 3가지 오류 패턴을 정의하고 학습 데이터에 반영:
- **Type 1** 클러스터 매칭 실패(misrouted) · **Type 2** 최근접 정상 벡터 검색 오류(cross-cluster) · **Type 3** 거리 차이 모호(boundary)

샘플 배분: `mining 0.4 / boundary 0.4 / easy 0.2` × `oversample 1.7` — 특정 구간 편중에 따른 학습 신호 왜곡 방지. boundary 하한은 sufficiency band와 정합(`nd∈[0.3, 0.55)`)시켜 **ambiguous 표본**을 확보합니다.

### (3) 3중 Rejection Sampling — cls-GT / grounded / dedup

teacher(Qwen2.5-14B)가 **GT hint 없이 순수 추론**(agent evaluator는 blind) → 사후 필터로 거릅니다. GT-conditioned 사후 생성은 reason↔label 인과를 파괴(rationalization)하므로 금지.

| 필터 | 내용 | 해결한 실패 모드 |
|---|---|---|
| ① **cls-GT** | teacher 판정 ≠ GT → 기각 | teacher 오답(정답률 60~65%)이 학습 신호 오염 |
| ② **grounded** | 인용 feature가 **실제로** 이탈/정합인지 코드 검증 | cls만 거르는 rejection이 boilerplate 통과 → student 암기 |
| ③ **dedup cap** | soft-key(정규화 40자) 기준 키당 ≤2% | 최빈 문장 점유 붕괴 |

**②가 핵심 교정입니다.** v1에서 "feature 인용율 71.8%"는 함정이었습니다 — 최빈 2·3위가 "within range" 류 **정합 서술**이라 인용율 검사를 통과했기 때문. 인용율이 아니라 **인용의 사실 방향**을 채점해야 합니다.

**agent 전용 — deterministic 주입 + coupling 강제:**

```
sufficiency = ambiguous  if 0.3 ≤ nd < 0.55  else confident      # local_analyzer와 동일 상수
cls         = nd threshold(0.5)          [confident 구간]
            = teacher의 evidence 판정     [ambiguous 구간] ← GT rejection으로 검증
what_missing: confident → null 강제 / ambiguous → grounded 서술 필수
```

- `sufficiency`는 **서빙(local_analyzer)과 동일하게 프롬프트 후미에 `[SYSTEM-COMPUTED]` 라인으로 주입** — 빌더가 아닌 후미 append라 이중 주입이 없고 train-serve 입력이 바이트 단위로 정합.
- **ambiguous에서만 cls를 evidence로 판정**하는 것이 agent의 존재 이유입니다(nd가 신뢰 불가한 구간). confident는 nd 복사.

**what_missing 스펙 — 반구조화 `"<direction>: <evidence>"`** (기계 파싱 가능 → planner가 게이트 결정의 confirm/veto 소프트 신호로 소비):

| direction | 의미 | grounding 검증 기준 | 인용 대상 |
|---|---|---|---|
| `cluster_mismatch` | 배정 클러스터 persona 이탈 → **reroute** | 인용 feature가 실제 `[DEVIANT]` | **numeric**(purchase_value/age) |
| `neighbor_insufficient` | 이웃이 흩어짐/불일치 → **expand** | `[spread]` 또는 `differs ≥ agrees` | `differs` 목록의 실제 feature명 |

> **categorical을 cluster_mismatch 근거로 유도하지 않습니다** — boundary에서 categorical 8종 전부 nd-통제 partial corr `+0.000` / AUROC `0.500`(무신호, NB01 실측). categorical은 `neighbor_insufficient`의 differ 증거로만 자연 등장시킵니다.

### 📊 데이터 생성 결과

**Table 2-1. Prompt v2 + grounded rejection 적용 효과 (forward, NB04)**

| 지표 | v1 (cls-GT only) | **v2 (+Deviation Table +grounded)** | 목표 |
|---|---:|---:|---:|
| reasoning 고유율 | 18.9% | **75.8%** | >40% |
| top-1 문장 점유율 | 28.2% | **4.6%** | <10% |
| confidence ↔ \|nd−0.5\| 상관 | — | **0.985** | ≈1.0 |
| confidence std | — | **15.8** | >10 (평균회귀 방지) |

**Table 2-2. cls별 분리 진단 (forward, n=1000)**

| | n | 고유율 | top1 | soft-top1 | numeric 인용 | categorical 인용 |
|---|---:|---:|---:|---:|---:|---:|
| ABNORMAL | 500 | 92.2% | 1.2% | 7.0% | 56.0% | 51.4% |
| NORMAL | 500 | 59.4% | 9.2% | **40.8%** | 99.8% | 12.2% |

> NORMAL은 exact-string상 다양해 보이나 **soft-top1 40.8%**로 구조적 템플릿 잔존 — B-1(lowest-\|z\| 강제)/B-2(categorical 유도) 대상. (ABNORMAL은 grounded rejection이 케이스별 이탈 feature를 강제해 자연 분산 — **v2 원리 검증 성공 사례**)

**Table 2-3. 재라벨 confidence calibration — nd 기반 라벨링의 본질적 상한 (forward, n=1000)**

| conf bin | n | 평균 conf | nd-GT 일치율 | gap |
|---|---:|---:|---:|---:|
| [50, 57) | 184 | 0.53 | 0.62 | +0.09 |
| [57, 65) | 204 | 0.60 | 0.77 | +0.17 |
| [65, 84) | 203 | 0.71 | 0.87 | +0.17 |
| [84, 92) | 166 | 0.87 | 0.94 | +0.07 |
| [92, 98) | 243 | 0.93 | 0.99 | +0.06 |
| **ECE(라벨 함수)** | | | | **0.109** |

> 전 bin gap 양수 = 라벨이 **일관되게 보수적**(과신 학습 없음). ECE 0.109의 실체 = tanh 스케일 보수성 + **nd 반대편 정답 레코드 15.5%**(teacher cls==GT인데 nd는 반대). 이것이 서빙 ECE의 **비교 기준선**입니다.

**Table 2-4. Agent evaluator 데이터 정합성 (F6, train n=556)**

| 지표 | v1 | **v2 (blind teacher + direction grounding)** | 판정 |
|---|---:|---:|---|
| coupling 위반 (confident⟺null) | — | **0/556** | ✅ |
| wm 형식 준수율 (`<direction>: <evidence>`) | — | **100%** | ✅ |
| grounding 정합 (전체) | 42.6% | **100%** | ✅ |
| ┗ `cluster_mismatch` (persona 이탈 실재) | 37.2%(cat) | **74/74 = 100%** | ✅ |
| ┗ `neighbor_insufficient` (spread/differs 실재) | — | **16/16 = 100%** | ✅ |
| **ambiguous cls: evidence vs GT** | — | **100.0%** | ✅ |
| **ambiguous cls: nd-baseline vs GT** | — | 64.4% | — |
| evidence ≠ nd (agent 개입 지점) | — | **32/90** | ✅ agent 가치 |

> **핵심 결과**: ambiguous 구간에서 evidence 판정이 nd 판정을 32회 뒤집었고 **전부 GT와 일치**(100% vs 64.4%). "ambiguous에선 nd가 아니라 evidence로 cls를 결정한다"는 설계가 데이터로 정당화됐습니다.

**Table 2-5. Rejection 수율 (agent, 최신 run)**

| 단계 | 건수 | 비고 |
|---|---:|---|
| 생성 | 1,700 | oversample ×1.7 |
| ① cls-GT 기각 | 0 | (confident는 nd 주입, ambiguous만 GT 필터) |
| ② grounded 기각 | 653 | 38% — v2 규칙 강화의 대가 |
| ③ dedup 기각 | 374 | teacher 템플릿 반복 |
| 통과 | 673 | N=189 / AB=484 |
| balance 후 채택 | **378** | ambiguous **71** ← *표본 부족, 개선 대상* |

---

## 3️⃣ 모델 학습 — Knowledge Distillation (SFT → GRPO)

### 구성

| | 설정 |
|---|---|
| Teacher | Qwen2.5-14B-Instruct (AWQ) |
| Student | Qwen2.5-0.5B / 3B-Instruct |
| 방법 | PEFT/LoRA → SFT (TRL) → GRPO |
| 서빙 | GGUF Q4_K_M, llama-cpp-python (CPU, e2-standard-2) |
| 산출물 | `{forwardRAG_,}adapter_{sft,grpo}/gguf/` (task별 분리 저장) |

**학습 target = 저장된 3-key JSON 전체**이나, 자유 생성 대상은 1개(`what_missing` / `reasoning`)입니다. `sufficiency`·`cls`(confident)·`confidence`는 deterministic 값이므로 **학습은 "표에서 골라 서술하는 과제"로 격하**됩니다 — 이것이 0.5B 전환을 정당화하는 근거입니다.

> **train-serve 정합**: instruction 텍스트는 모델이 일부 key만 예측하더라도 **서빙과 verbatim 일치**해야 합니다. eval set은 `TRAINING_JSONL_PATH.replace(".jsonl","_eval.jsonl")` 규칙으로 자동 정합(과거 stale `mined_eval`을 forward 학습에 붙여 eval loss 4.07이 나온 사고의 재발 방지).

### GRPO Reward 설계

SFT는 토큰 우도를 최적화하므로 **템플릿 암기를 보상**합니다(실제로 그 결과가 관측됨). grounding을 **직접** 밀어올리려면 reward로 최적화해야 합니다.

| reward 항 | 정의 | 검증 가능성 |
|---|---|---|
| `R_cls` | ambiguous 구간 cls == GT | GT (verifiable) |
| `R_grounded` | 인용 feature의 사실 방향 일치 (deviant⟺ABNORMAL) | Deviation Table (코드 계산) |
| `R_direction` | direction ↔ deterministic 신호 일치 (`cluster_mismatch`⟺DEVIANT 존재 / `neighbor_insufficient`⟺spread\|differs≥agrees) | retrieval digest (코드 계산) |
| `R_format` | `"<direction>: <evidence>"` 형식 + JSON 유효성 | 정규식 |
| `− P_template` | soft-key 기준 배치 내 중복 빈도 페널티 | dedup key |
| `− P_length` | 출력 토큰 초과 페널티 (decode latency 직결) | 토큰 수 |

> reward가 **전부 코드로 검증 가능**한 항으로만 구성됩니다("deterministic한 건 전부 코드가" 원칙의 reward 레벨 적용). GT-free 항(`R_grounded`, `R_direction`, `R_format`)만으로도 학습 가능해 서빙 분포로 확장 여지가 있습니다.
>
> ⚠️ **용어 주의**: 여기서의 hard negative(파이프라인 실패 유형: misroute/cross-cluster/boundary)는 reward 맥락의 hard negative(False Negative)와 **다른 개념**입니다.

### 📊 학습 결과

**Table 3-1. SFT vs GRPO (동일 eval set)**

| 지표 | SFT | GRPO | Δ |
|---|---:|---:|---:|
| AUROC | | | |
| ECE | | | |
| P1 grounded-consistency | | | |
| P4 다양성(고유율) | | | |
| direction 균형 (cm:ni) | | | |
| 파싱 성공률 | | | |

**Table 3-2. Student 모델 크기 비교 (3B vs 0.5B)**

| | 3B (Q4) | 0.5B (Q4) | 판정 기준 |
|---|---:|---:|---|
| judge 점수 | | | 0.5B ≥ 3B×0.9 → 교체 |
| P1 grounded-consistency | | | |
| cls acc / FNR | | | |
| latency (e2-standard-2, 단건) | | | |
| 메모리 (GGUF) | ~2.0 GB | ~0.4 GB | |

**Table 3-3. 모델 크기 × grounding 곡선 (teacher, NB01 실측)**

| teacher | grounding ratio | teacher acc (boundary) | fusion Δ |
|---|---:|---:|---:|
| Qwen2.5-7B | 1.00 | 0.575 | +0.016 |
| Qwen2.5-14B | 1.91 | 0.617 | +0.016 |
| QwQ-32B | | | |

> **두 열의 분리가 이 프로젝트의 직접 증거입니다** — grounding(설명 품질)은 모델 스케일에 반응하지만, `fusion Δ`(판별 기여)는 **모델과 무관하게 고정**. 즉 *설명은 모델로, 판별은 정보량으로*.

---

## 4️⃣ 평가 체계 — 우선순위 지표 + LLM-as-Judge

### 평가 축 설계 (NB03 Block 0 = 정본)

인용율·loss 같은 **통과하기 쉬운 지표**가 아니라, 사실 정합을 우선 채점하도록 우선순위를 고정했습니다.

| 우선순위 | 지표 | 무엇을 검증하나 | 왜 필요한가 |
|---|---|---|---|
| **P1** | grounded-consistency | 인용 feature가 **실제로** deviant/ok인지 | 인용율만 보면 정합 서술도 통과 (v1 오판의 교정) |
| **P2** | teacher-parity | teacher 대비 인용 feature Jaccard + 방향 일치 | teacher 상한 대비 위치 (fallback 오염을 Jaccard 0.000으로 포착한 전례) |
| **P3** | [agent] wm 형식 준수율 + direction 분포<br>[forward] confidence ECE | 기계 파싱 가능성 / 캘리브레이션 | planner 소비 가능성 / "평균 회귀" 교정 여부 |
| **P4** | 다양성 (고유율 / top1 / soft-top1) | 템플릿 암기 붕괴 | soft-top1이 진짜 신호 (exact-string은 tail만 달라도 통과) |
| **P5** | cls acc / FNR | 판정 정확도 | det 모드 서빙에선 **참고 지표** (nd threshold가 결정) |

**데이터셋 검증 블록(Block 1/F6)** — 모델 평가와 **같은 잣대로 학습 데이터 자체**를 채점: direction 균형 / grounding(numeric+categorical 분리) / coupling. "모델이 나쁜가, 데이터가 나쁜가"의 분기를 먼저 확정합니다.

**LLM-as-Judge** — GT가 없는 `what_missing` 품질을 이산 rubric(0/0.5/1)으로 채점하되, **judge 신뢰성 자체를 먼저 검증**해야 지표로 채택합니다:
- judge 점수 ↔ GT 정오답 상관 (>0 이어야 유효)
- 동일 입력 2회 채점 일치율
- 순서 교환 편향(position bias)
- rubric 축: relevance / consistency / specificity + **actionability**(gap 방향 서술 ↔ deterministic 신호[misroute/plateau] 일치율 = "planner에 실질 도움"의 조작적 정의)

### 📊 평가 결과

**Table 4-1. Block 0 우선순위 평가 — Agent vs forwardRAG (student, n=100)**

| 지표 | Agent Evaluator | forwardRAG | 해석 |
|---|---:|---:|---|
| 파싱 성공 | 100/100 | 95/100 | forward 5% JSON 이탈 → GBNF로 해소 예정 |
| **P1** grounded-consistency | **35.7%** (5/14) | **95.5%** (84/88) | forward 압승 |
| **P2** feature-Jaccard | 0.412 | **0.720** | |
| **P3** 형식 준수율 / ECE | 100% (75/75) | ECE **0.120** (mean 77.3, std 11.7) | forward ECE ≈ 라벨 상한(0.109) |
| **P4** 고유율 / top1 | 75.0% / 10.0% | **97.9% / 2.1%** | |
| **P5** cls acc / FNR | 0.760 / 0.327 | **0.811 / 0.261** | FNR은 nd threshold 이슈(직교) |
| direction 분포 | cm 57 : ni 3 ⚠ | n/a | **degenerate — reroute만 트리거** |

> **판독**: forward는 A/B 진행 가능 상태. agent의 `what_missing`이 병목 — student가 grounding "관계"가 아니라 **템플릿을 암기**(프롬프트의 RETRIEVAL digest 라인을 그대로 복사하는 사례 관측). 단 forward의 95.5%는 **과제 난이도 차이**도 포함합니다(reasoning = z-table에서 최대이탈 feature 선택·재서술 = grounded selection / wm = "검색 gap의 방향+근거" = 더 추상적).
>
> ※ forward의 "P2 direction 일치 100%"는 artifact입니다(forward엔 direction이 없어 양쪽 모두 None → 무조건 일치). 신호 아님.

**Table 4-2. 시스템 지표 — Agent vs forwardRAG (동일 eval set, 동일 harness)**

| 축 | forwardRAG (baseline) | Agent System |
|---|---:|---:|
| AUROC | | |
| FNR / FPR | | |
| **LLM 호출량** | 100% (전수) | ambiguous만 (목표 30~40%) |
| latency mean / p95 | | |
| 서빙 confidence ECE | | (n/a — det) |
| 경로별 기여 (no-loop/expand/reroute) | n/a | |

**Table 4-3. LLM-as-Judge (신뢰성 검증 + 채점)**

| 항목 | 값 | 채택 기준 |
|---|---:|---|
| judge ↔ GT 상관 | | > 0 |
| 재현성 (2회 일치율) | | |
| 순서 교환 편향 | | |
| **judge avg — Agent (3B)** | | |
| **judge avg — Agent (0.5B)** | | 3B의 ≥90% |
| **judge avg — forwardRAG** | | |
| actionability | | |

**Table 4-4. 휴리스틱 baseline + 신호 융합 천장 (NB01 실측, boundary n=11,440)**

| 신호 | AUROC | 판정 |
|---|---:|---|
| top5_spread 단독 | 0.403 | 무신호 |
| persona_dev 단독 | 0.496 | 무신호 |
| **nd only** | 0.779 | |
| **nd + std(top3)** | **0.857** (Δ **+0.077**) | ✅ spread는 **코드 융합**에서 가치 실현 |
| nd + std + gap | 0.857 (Δ +0.077) | 포화 |
| + persona (purchase_value, age) | 0.873 (Δ +0.016) | ⚠ 정제 천장 닫힘 |
| + persona ALL features | 0.873 (Δ +0.016) | ⚠ 동일 — 추가 정제 무익 |

**Table 4-5. nd 통제 후 feature별 추가 정보량 (boundary)**

| feature | AUROC | partial(\|dev\|, label \| nd) |
|---|---:|---:|
| purchase_value | 0.437 | −0.103 |
| age | 0.419 | −0.122 |
| sex / source / browser / IP_country / month / weekday | **0.500** | **+0.000** |

> **결론**: categorical 6종은 nd 통제 후 **정보량 0**. → ① `cluster_mismatch` 근거를 numeric으로 한정한 설계의 실증 근거 ② evaluator의 **판별** 역할 확대는 정제가 아니라 **graph(새 차원)로만** 가능.
>
> 보강 증거: `corr(spread, ambiguous) = −0.184` — sufficiency는 spread를 반영하지 않는 **nd-band 재진술**. → 코드 게이트가 정답이고 LLM은 서술만.

---

## 🗂 의사결정 기록 (추가분)

| 결정 | 근거 |
|---|---|
| ❌ Teacher rationalization (GT-conditioned 사후 wm 생성) | reason↔label 인과 파괴 → rejection sampling(GT-free 생성 + GT 필터)으로 대체 |
| ❌ 인용율(citation rate)을 품질 지표로 사용 | "within range" 정합 서술이 통과 → **grounded-consistency**(사실 방향)로 교체 |
| ❌ categorical을 cluster_mismatch 근거로 유도 | nd 통제 partial corr +0.000 / AUROC 0.500 (Table 4-5) — 무신호 유도는 hallucination 증식 |
| ❌ `neighbor_insufficient`에 lowest-\|z\| 인용 강제 | 개념 불일치 — 검색 품질 신호에 persona-fit 근거를 붙이는 것 |
| ❌ eval_loss 기반 best-checkpoint 선택 | 템플릿 암기를 보상 → **P1 grounded-consistency 기반 early-stop**으로 전환 |
| ✅ blind teacher + ambiguous cls GT-rejection | evidence-GT 100% vs nd 64.4% (Table 2-4) |
| ✅ prompt v2 (z-table + digest + 정적 선두) | 고유율 18.9→75.8%, 프롬프트 ~230 tok(절반), 정적 120 tok 캐시 가능 |

---

## 🔜 남은 작업

- [ ] **ambiguous 표본 확대** — 현재 wm 학습 표본 71개(최우선). boundary 하한 `0.4→0.3`(sufficiency band 정합), balance 완화, oversample↑
- [ ] NORMAL wm 템플릿 쏠림 (soft-top1 40.8~60%) — B-1/B-2 재생성
- [ ] SFT 재학습(P1 early-stop, 과적합 억제: epoch↓/LoRA rank↓) → GRPO
- [ ] `CLS_ND_THRESHOLD` 0.5→**0.42** 재캘리브레이션 (직교 레버, FNR 0.327 대응)
- [ ] Judge 신뢰성 검증 → Table 4-3 채움
- [ ] 동일 eval set A/B 배치 (`PIPELINE_MODE=forward`) → Table 4-2 채움
- [ ] Latency: KV prefix cache 실측 / ISA(AVX2) 재빌드 / GBNF grammar / assistant-prefix 이관

### 📈 필요 시각화 (README 이미지 슬롯)

| 그림 | 내용 | 상태 |
|---|---|---|
| Fig 1 | LangGraph StateGraph (mermaid 엔드포인트) | |
| Fig 2 | **Rejection funnel** — 생성 1700 → grounded/dup 기각 → balance → 채택 (Sankey) | |
| Fig 3 | **Calibration curve** — 라벨 함수(ECE 0.109) vs 서빙 모델 | |
| Fig 4 | **다양성 산점** — 고유율/soft-top1: v1 vs v2, agent vs forward | |
| Fig 5 | **Fusion AUROC bar** — nd → +spread → +persona (천장 닫힘 시각화) | |
| Fig 6 | Streamlit agent trace 타임라인 / latency breakdown | |
| Fig 7 | Locust 부하 그래프 (P95 / 에러율) | |
