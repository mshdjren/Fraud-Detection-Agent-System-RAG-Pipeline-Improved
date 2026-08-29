# Fraud Detection AgentRAG Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/GKE-Kubernetes-326CE5?logo=kubernetes" />
  <img src="https://img.shields.io/badge/Elasticsearch-8.x-005571?logo=elasticsearch" />
  <img src="https://img.shields.io/badge/LangGraph-StateGraph-1C3C3C" />
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi" />
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python" />
</p>

## 📌 프로젝트 개요

**Multi-Cluster Memory Bank 이상탐지 알고리즘(EAAI 2025)을 운영 가능한 수준의 AgentRAG 시스템으로 확장한 엔드투엔드 파이프라인 개발 프로젝트**입니다.

Tabular 이상 거래를 "retrieval → 거리 기반 이상탐지 → 판정 원인 설명" 흐름으로 처리하며, Vector Indexing · Router(클러스터 라우팅) · Retriever(HNSW) · Evaluator(LLM) · Planner · Orchestrator의 6개 MSA로 GKE에 배포했습니다. 정적 forwardRAG(전수 LLM 추론 baseline)에서 출발해, **검색 결과를 평가하고 선택적으로 재검색하는 Agent 시스템**으로 확장한 과정과 그 측정 근거가 이 프로젝트의 핵심입니다.

### 🎯 핵심 문제 해결 (검증 완료 지표)

| 문제 | 해결 방안 | 성과 |
|------|---------|------|
| 벡터 검색 병목 | Percolate Query Bucketization (3단계 고도화) | Router Recall@5 +9%p, P99 latency −25% |
| 메모리 제약 | Coreset Sampling 10% 운영점 | 메모리 −89% (2.8GB→0.31GB), AUROC 손실 0.4%p 이내, median latency −46% |
| Teacher 오답이 학습 신호 오염 | Rejection Sampling (GT-free 생성 → GT 필터) | teacher 정답률 60~65% 감안 ×1.7 oversample로 품질·수량 동시 확보 |
| SFT-GRPO Training으로 llm 역할 분배 | 과제 설계 원인 정량 규명 → Deterministic-LLM 역할 분리 | what_missing 생성 타이밍 11%→100%, 실제 이탈 feature 지목 정합도 70% |
| Deterministic-planner 역할 분담 | 2-stage — 결정론 게이트 + GRPO tool selector | 게이트 AUC: nd 스칼라 0.68~0.82 vs LLM 0.44~0.48 (3회) → 비용은 admission control로 분리 |
| Planner에 LLM이 필요한가 | 전수 counterfactual 오라클(130,883 rollout)로 **상한 먼저 측정** | 제약 디코딩 +0.014(학습 0회) / GRPO 추가 +0.017 / 상한 대비 15.3% |
| 재검색 turn 수 | λ_turn(1→2턴 가격) 유도 | 2턴 우세 state 0.3%(9/2,830), λ_turn/λ*=0.06 → **MAX_ROUND=1 측정 확정** |

---

## 🏗️ 시스템 아키텍처

```text
Test Transaction
      ↓
[Router]      Percolate Query → 클러스터 매칭 (rank1=percolate, rank2~5=centroid)
      ↓
[Retriever]   Filtered HNSW (cluster-scoped, k=3) → nd_before / 이웃 evidence
      ↓
[Evaluator]   LLM 1회 — wm_type(cluster_mismatch / neighbor_insufficient / accept)
      ↓
   sufficiency = f(nd_before)   ── confident(60.4%) → 종료, LLM 호출 없음
      ↓ ambiguous (39.6%)
┌─────────────────────────────────────────────────────────────┐
│ [Stage A] 결정론 게이트   |nd − τ| 상위 B%          비용 0    │  ← admission control
│ [Stage B] Planner(LLM)   think → tool + feature 선택         │  ← GRPO 학습 대상
│              contextual_knn(f≤2)  |  attribute_knn(f≤2)      │
└─────────────────────────────────────────────────────────────┘
      ↓
[Orchestrator] nd_final vs τ=0.42 → 최종 판정 + round_log/termination
```
<img src="picture/Langgraph.jpg" width="100%">

**Planner 재설계 (v2 → v3)** — 기존 `Selective Tool Gate + Gemini Planner`를
**2-stage(결정론 게이트 + 학습된 tool selector)** 로 교체했습니다. 근거는 §5️⃣.

| | v2 (이전) | v3 (현재) |
|---|---|---|
| 게이트 | rule search (`nd∈[0.48,0.60) ∧ plateau≤0.142`) | `|nd − τ|` 랭킹 상위 B% — 예산이 운영 파라미터 |
| Planner | Gemini Flash Lite, tool 3종 | Qwen2.5-0.5B GRPO, tool 2종 + feature 조합 |
| Tool | expand_within_cluster / reroute_cross_cluster / accept | contextual_knn(soft boost) / attribute_knn(hard filter) |
| 출력 | action 1개 | `{"think": 근거, "action": tool, "features": [...]}` |
| 비용 | reward에 λ·cost 스칼라화 | **admission control로 분리** (§5️⃣-C) |

`PIPELINE_MODE=forward` 전환 시 동일 인프라에서 **정적 forwardRAG baseline**([route]→[Retrieve]→[Analyzer] 1회)으로 동작합니다 — Agent 시스템의 가치를 같은 문제지에서 A/B 측정하기 위한 구조입니다.

---

## 🔬 핵심 기술 (Core Technologies)

### 1️⃣ Percolate Query-based Cluster Routing

최근접 정상 클러스터의 매칭 정확도 및 검색 속도 개선을 위해, 라우팅 쿼리를 **3단계에 걸쳐 고도화**했습니다.

---

#### **Version 1: Strict AND**

Decision Tree의 리프 노드 경로를 `bool.filter`의 range 조건으로 직접 변환하여 정밀 매칭합니다.

```python
# 예시: feature v1 ∈ [0.8, ∞) AND v3 ∈ [-1.2, ∞) → Cluster 63
{
  "query": {
    "bool": {
      "filter": [
        {"range": {"v1": {"gte": 0.8}}},
        {"range": {"v3": {"gte": -1.2}}}
      ]
    }
  }
}
```

- **의도**: Decision tree의 leaf 정보를 각 클러스터의 rules로 표현하고, 이를 직관적으로 percolate query로 매칭 
- **한계**: Strict AND 다수 결합으로 인해, Top-5 클러스터 매칭의 다양성이 떨어짐 (같은 1개의 클러스터만 매칭)

---

#### **Version 2: Centroid-Augmented Diversity**

Version1의 Top-1 매칭 결과에 **벡터 유사도 기반 Top-4 클러스터**를 추가하여 다양성을 보완합니다.

```python
# Phase 1: Strict AND로 Top-1 확보
top_1 = percolate_strict_and(test_vec)

# Phase 2: Centroid cosine distance로 Top-4 보완
top_4_nearest = sorted(
    cluster_centroids, 
    key=lambda c: 1 - cosine_similarity(test_vec, c)
)[:4]

# 최종 Top-5
candidates = [top_1] + top_4_nearest
```

- **의도**: Recall@5 및 Top-5 매칭 다양성 ↑
- **한계**: Strict AND에 비해 검색 지연 ↑

---

#### **Version 3: Optimized Bucketization**

Range 조건을 **Bucket index + Term filter**로 변환하여 후보군 pruning을 최적화합니다.

````python
# 1. Ingest: Range → Bucket index
split_points["v1"] = [-2.5, -1.0, 0.0, 1.0, 2.5]
# Rule: v1 ≤ 0.8 → bucket_3, direction="lte"

# 2. Query: Test vector → Candidate buckets
test_vec["v1"] = 0.05  # → bucket_3
tolerance = ±1         # → [2, 3, 4]
````

##### **최종 Query 구조**

````python
{
  "bool": {
    "must": [
      # 기존 percolate (range 매칭)
      {"percolate": {"field": "query", "document": {...}}},
      
      # Bucket pre-filter (term 매칭)
      {
        "bool": {
          "should": [
            # v1 feature
            {
              "must": [
                {"term": {"v1_direction": "lte"}},
                {"terms": {"v1_bucket": [2, 3, 4]}}
              ]
            },
            # ... 총 9개 features
          ],
          "minimum_should_match": 3  # 9개 중 3개
        }
      }
    ]
  }
}
````

- **의도**: Range 부동소수점 비교 → Term filter bitset 연산
- **한계**: Percolate query index ↑ 및 rules 수가 늘어났을때의 검색 지연 가능성

---

#### 📊 실험 결과 (Percoalte query routing)

Percolate query version에 따른 클러스터 정확도, 리콜, 단건 검색 지연 시간 및 AUROC 지표를 비교합니다.

| query version | Cluster acc / Recall@5 (Router) | Router MRR | Latency p99 (ms) |  Recall@5 (knn) / AUROC |
| :--- | :---: | :---: | :---: | :---: |
| **Strict AND (v1)** | 0.816 / 0.816 | 0.941 | 420.25 | 0.786 / 0.8726 |
| **V1 + aug** | **0.823 / 0.905** | **1.00** | 430.01 | **0.956 / 0.9125** |
| **V1 + aug + bucket** | 0.823 / 0.897 | 0.988 | **409.25** | 0.956 / 0.9037 |

---

### 2️⃣ Filtered KNN (Coreset Sampling 검증)
Coreset Sampling 최적 운영점 검증: 메모리-이상탐지 성능-검색 latency trade-off 분석을 통해, coreset sampling 비율 10%의 최적점을 도출하고자 하였습니다.

<table>
<tr>
<td width="50%" align="center">

<img src="picture/coreset_explain.jpg" width="100%">

**Coreset Sampling Algorithm**<br>[(from https://kmhana.tistory.com/6)](https://kmhana.tistory.com/6)

</td>
<td width="50%" align="center">

<img src="picture/coreset_tsne.jpg" width="100%">

**Cluster Distribution with coreset sampling ratio**<br>(t-SNE)

</td>
</tr>
</table>

---

#### 📊 실험 결과 (Coreset Sampling 검증)

**Croreset Sampling 비율 vs AUROC vs 검색 지연**

| Coreset % | Index Size (GB) | Docs Count | AUROC | Recall@5 (knn) | Median Latency (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 100 | 2.8 | 140,128 | 0.956 | **0.956** | 125 |
| **10** | 0.31 | 14,012 | **0.952** | 0.947 | 68 |
| 1 | 0.035 | 1,401 | 0.901 | 0.882 | **52** |

<table>
<tr>
<td width="50%" valign="top">

**Coreset Sampling 유무에 따른 Page Faults 발생 정도**<br>
*(Memory 2GB로 Node 제한 시)*

| Index sharding | 100% Index<br>(Page Faults/sec) | 10% Index<br>(Page Faults/sec) |
| :---: | :---: | :---: |
| **X** | 45.2 | 38.7 |
| **O** | 52.3 | **8.1** |

</td>
<td width="50%" valign="top" align="center">

<img src="picture/coreset_tradeoff.jpg" width="100%">
<p align="center"><i>AUROC and Inference Speed vs. Coreset Sampling Ratio</i></p>

</td>
</tr>
</table>

<br>

---

### 3️⃣ 학습 데이터 생성 — Prompt v2 + 3중 Rejection Sampling

Teacher(Qwen2.5-14B) 증류 데이터의 품질 문제를 **프롬프트와 필터를 2단계에 걸쳐 고도화**해 해결했습니다.
<p align="center"><img src="picture/streamlit_AgentRAG.jpg" width="100%"><br><i>AgentRAG streamlit UI</i></p>

<p align="center"><img src="picture/streamlit_forwardRAG.jpg" width="100%"><br><i>ForwardRAG streamlit UI</i></p>
---

#### **Version 1: GT-conditioned 생성 (rejection 없음)**

Teacher에게 GT를 알려주고 사후 설명을 생성하는 직관적 방식입니다.

```text
프롬프트: [이웃 원시 feature 덤프 ~200 tok] + [mean/std 통계표] + GROUND TRUTH: ABNORMAL
→ teacher: "reasoning을 GT에 맞춰 사후 합리화(rationalization)"
```

- **의도**: GT 조건화로 라벨-설명 일관성 확보
- **한계**: reason↔label 인과 파괴 — student가 관계가 아닌 **템플릿을 암기** (reasoning 고유율 18.9%, 최빈 문장 점유 28.2%, confidence 평균회귀 std<10). teacher 정답률 60~65%의 오답도 그대로 학습 신호 오염

---

#### **Version 2: Prompt v2 + GT-free 생성 → 3중 Rejection**

"통계는 시스템이 계산하고, 해석만 LLM에 남긴다" — 프롬프트에서 산술을 제거하고, blind 생성 후 코드로 거릅니다.

```text
[정적 instruction 선두 (~120 tok, KV prefix cache 대상)]
[Deviation Table]  purchase_value: z=+0.79 [ok] ...   ← z를 사전 계산 (모델 암산 제거)
[RETRIEVAL digest] top-k agrees: sex(3/3) | differs: browser(1/3) ...  ← 원시 덤프 15 tok 압축
```

| 필터 | 내용 | 해결한 실패 모드 |
|---|---|---|
| ① cls-GT | blind teacher 판정 ≠ GT → 기각 | teacher 오답(60~65%) 오염 |
| ② grounded | 인용 feature의 **사실 방향**을 코드 검증 | "인용율 71.8%"의 함정 — 정합 서술도 통과하던 문제 |
| ③ dedup cap | soft-key(40자) 키당 ≤2% | 최빈 문장 점유 붕괴 |

- **의도**: 검증 가능한 것(z, 합의)은 코드가 계산·필터 → LLM은 "표에서 골라 서술"만
- **한계**: grounded 기각 ~38%의 수율 비용 (oversample ×1.7~2.5로 상쇄) + rejection이 만드는 지표 착시(부록 참조)

---

#### 📊 실험 결과 (학습 데이터 생성)

| 지표 | v1 (GT-conditioned) | **v2 (+Rejection)** | 목표 |
|---|---:|---:|---:|
| reasoning 고유율 | 18.9% | **75.8%** | >40% |
| top-1 문장 점유율 | 28.2% | **4.6%** | <10% |
| confidence ↔ \|nd−0.5\| 상관 | — | **0.985** | ≈1.0 |
| ECE (라벨 함수) | — | **0.109** | 서빙 비교 기준선 |

<table>
<tr>
<td width="50%" align="center">

<img src="picture/mining_data.jpg" width="100%">

**Rejection Funnel (Sankey)**<br>생성 → parse/GT/grounded/dup 기각 → train 최종

</td>
<td width="50%" align="center">

<img src="picture/nb03-fig3.png" width="100%">

**다양성 산점 (고유율 × soft-top1)**<br>v1 → v2 개선 + student 붕괴점 대비

</td>
</tr>
</table>

---

### 4️⃣ Deterministic–LLM 역할 분리 — Agent Evaluator 출력 자유도 5키 → 1키

"어디까지 LLM에 맡길 것인가"를 감이 아니라 **단계별 실험(S1→S4)으로 측정해 축소**했습니다.

---

#### **Version 1: 3-key Evaluator (cls·sufficiency·what_missing 전부 LLM)**

ambiguous 구간(nd∈[0.3, 0.55))에서는 nd가 신뢰 불가하므로, LLM이 evidence로 cls까지 판정하게 합니다.

- **의도**: 검색 결과가 애매할 때 LLM evidence 판정으로 nd를 보완
- **한계**: 데이터는 완벽(coupling 위반 0/995)했으나 **0.5B가 판정과 서술을 동시에 못 함** — 아래 A/B에서 해소 불가 트레이드오프 확인

| SUF_BALANCE A/B | nd→cls 유지 | wm 품질(P1) | 판정 |
|---|---:|---:|---|
| ambiguous만 학습 (224) | **53.8%** ❌ | 92.7% | 판정 붕괴 |
| 전 밴드 학습 (1,014) | 94.7% | **79.6%** | 서술 희석 |

---

#### **Version 2: 1-key + Deterministic 판정 (최종 채택)**

판정·게이트를 전부 코드로 이관하고, LLM에는 자유 출력 1개만 남깁니다.

```python
sufficiency = "ambiguous" if 0.3 <= nd < 0.55 else "confident"   # 코드 (nd-band)
cls         = "ABNORMAL"  if nd >= 0.5 else "NORMAL"             # 코드 (nd threshold)
what_missing = LLM(prompt + "[SYSTEM-COMPUTED] sufficiency=...") # 유일한 LLM 자유 출력
```

| 신호 | 담당 | 근거 (실측) |
|---|---|---|
| sufficiency | 코드 (nd-band 주입) | 이산 판정의 SFT collapse — 주입 후 wm 생성 타이밍 11%→**100%** |
| cls | 코드 (nd threshold) | ambiguous에서도 student(0.703) < nd(0.791), **net gain −14** |
| confidence | 코드 (50+50·tanh 재라벨) | teacher 자체 conf는 평균회귀 (std<10) |
| what_missing | **LLM** (감사 주석) | 실제 이탈 feature 지목 정합도 70% |

- **의도**: "표에서 골라 서술" 과제로 격하 → **3B를 0.5B로 대체**하는 근거. instruction은 서빙과 verbatim 동일(train-serve 정합)
- **한계**: wm의 판정 기여는 기각됨(부록 F7) — 가치는 감사 주석 + 선택적 호출로 재정의

---

#### 📊 실험 결과 (역할 분리 — Step 0 직접 검정, eval n=300 / ambiguous 158)

| 측정 | 실측 | 판정 |
|---|---:|---|
| nd AUROC (ambiguous 밴드 내) | **0.832** | 밴드 안에서도 nd가 강신호 |
| student cls acc vs nd acc | 0.703 vs **0.791** | evidence < nd |
| net gain (student가 nd를 이긴 순증) | **−14** | ⛔ LLM 판정 개입은 손해 → det 전환 확정 |

---

### 5️⃣ Planner GRPO — 전수 counterfactual 오라클 기반 상한 측정 후 설계

Planner에 LLM을 넣기 전에 **"넣을 가치가 있는가"를 먼저 측정**했습니다. action space가 닫혀 있어(`tool 2종 × feature 조합`) **전수 counterfactual 오라클**을 만들 수 있었고 — 2,830 state × 46 후보 = **130,883 rollout**, 158분 — 그래서 "RL이 도달 가능한 상한의 몇 %를 회수했는가"를 직접 측정합니다. 문서검색 agent 연구가 tool별 전수 실행이 불가능해 못 하는 측정입니다.

#### A. Reward 커널 — 어디에 gradient를 줄 것인가

`nd`가 `dnd`만큼 움직였을 때의 반응으로 보면 네 커널은 **gradient의 지지집합**이 다릅니다.

| 커널 | 미분 형태 | EVPI | 여지>0 | 게이트 AUC (RF vs nd단독) |
|---|---|---:|---:|---|
| `R_margin` | `(2y−1)·1·dnd` — 균등 | +0.0357 | 30.3% | 0.702 vs 0.697 (**+0.005**) |
| `R_flip` | `(2y−1)·δ(nd−τ)·dnd` — τ에 Dirac | +0.0854 | 9.0% | 0.747 vs 0.681 (**+0.066**) |
| `R_resolve` | band 경계에 Dirac | +0.1228 | 12.5% | 0.700 vs 0.642 (+0.058) |
| `R_auroc` | 반대라벨 밀도 가중 | +0.0448 | 23.9% | 0.748 vs 0.813 (**−0.065**) |

**`R_flip` 채택** — 게이트 probe 격차가 유일하게 유의미하게 양수입니다(+0.066). `R_auroc`은 격차가 음수, 즉 **스칼라 임계값 하나가 25-feature RF보다 낫다 = LLM이 불필요**하다는 결론이 나오는 커널입니다.

**하이브리드 보정** — `R_flip`은 `{−1,0,+1}` 이산이라 flip 가능 state(9%) 밖에서 모든 후보가 동점입니다(동률 99.6%). 연속 성분을 섞어 지지집합을 넓혔습니다.

```
R = R_flip + α·R_margin        α = 2.5  (flip ±1이 margin 최대 ±0.2를 5배 압도)
→ 동률 99.6% → 64.2%,  EVPI +0.13 → +0.24
```

#### B. Turn 수 — λ_turn 유도로 1-turn 확정

`λ*`(검색할까 말까의 가격)와 같은 원리로 **`λ_turn`(한 번 더 할까의 가격)** 을 유도했습니다.

| 커널 | 1턴 최적 | 2턴 최적 | 한계이득 | λ_turn/λ* | 2턴 우세 state |
|---|---:|---:|---:|---:|---:|
| `R_margin` | +0.0242 | +0.0242 | +0.0000 | 0.00× | 23.4% |
| `R_auroc` | +0.0377 | +0.0551 | +0.0174 | 0.57× | 19.1% |
| `R_resolve` | +0.1007 | +0.1332 | +0.0325 | 0.44× | 3.3% |
| **`R_flip`** | **+0.0873** | **+0.0905** | **+0.0032** | **0.06×** | **0.3%** (9/2,830) |

**`MAX_ROUND=1` 확정.** "복잡해서 1턴"이 아니라 **2턴의 한계이득이 1턴의 6%로 측정됐기 때문**입니다. 이산 커널일수록 1턴에 결판난다는 구조적 관찰이기도 합니다(margin 23.4% → flip 0.3%).

#### C. 비용 — reward 스칼라화 대신 admission control

`stop`을 action space에 넣으면 라벨이 91% `stop`으로 편중되고(SFT eval 100% stop), 빼면 `−λ·n_calls`가 그룹 내 상수가 되어 **advantage에서 완전히 소거**됩니다(`mean_cost 0.14` 100 step 고정). λ 튜닝으로 풀리지 않는 구조입니다.

```
maximize E[gain]  s.t.  E[cost] ≤ B     ← 라그랑주 스칼라화가 아니라 제약으로
Stage A: |nd − τ| 랭킹 상위 B%          ← 예산은 운영 파라미터
Stage B: 그 예산 안에서 gain만 최적화    ← LLM
```

**근거 3가지**: ① λ 스칼라화는 파레토 곡선의 한 점만 주는데 운영 예산은 SLA/트래픽에 따라 바뀝니다 ② `nd` 단독 게이트 AUC 0.68~0.82 vs **LLM 게이트 0.443~0.481(3회 독립 실행 전부 무정보 이하)** ③ 비용 축을 분리하면 예산 곡선 전체를 보고할 수 있습니다. 비용 단위는 latency(ms)가 아니라 **호출 수**입니다 — 실측 latency(contextual 3,677ms vs global 722ms)가 손으로 찍은 cost(1.2 vs 1.5)와 역전돼 있었고 인덱스/구현 변경마다 바뀝니다.

#### D. 제약 디코딩 — format penalty를 구조적으로 제거

`{"think":"[` 까지 강제 → think 자유(28 tok) → `","action":...` 부터 오라클 후보 trie. **집합 밖 출력이 생성 자체로 불가능**해집니다.

| | 무제약 | 제약 |
|---|---:|---:|
| parse rate | 5.6% | **98.5~99.1%** |
| achieved | −0.00186 | **+0.01401** (학습 0회) |

#### 📊 실험 결과 (동일 eval set n=337)

| 축 | achieved | regret | parse | 해석 |
|---|---:|---:|---:|---|
| ① base + 제약 | +0.01401 | 0.190 | 99.1% | 제약만으로 얻은 것 (학습 0회) |
| ①' base 무제약 | −0.00186 | 0.206 | 5.6% | 제약 없으면 붕괴 |
| **② GRPO_ONLY** | **+0.03127** | **0.173** | 98.5% | 학습 순이득 **+0.01725** |
| ③ oracle 상한 | +0.20428 | 0.000 | 100% | 상한 대비 **15.3%** 달성 |

**학습 가능성 사전 검정(`[G3]`)** — GBDT/RF는 표 데이터에서 0.5B LLM보다 강하므로, 이들이 majority를 못 넘으면 학습이 무의미합니다. 학습 전 게이트로 씁니다.

| | majority | tree d=3 | RF-300 | leakage gap |
|---|---:|---:|---:|---:|
| 오라클 라벨 | 0.570 | **0.632** | 0.588 | −0.000 |

#### E. think — LLM 정당성의 직접 증거

출력이 `43-way 선택`이면 원리적으로 분류기로 환원되어 0.5B를 쓸 이유가 없습니다. `think`를 붙여 **"모델이 evidence를 읽는가"** 를 측정 가능하게 만들었습니다.

| | v5-a (구체 예시) | v5-b (`[` 강제) |
|---|---:|---:|
| 태그 인용 | 10.8% | **41.7%** |
| **없는 태그 지어냄** | **89.2%** | **0.8%** |
| grounded (태그 ∧ 필드 언급) | 6.7% | 21.7% |

> **🔧 정정** — 프롬프트에 진단-처방 힌트(`cluster_mismatch → attribute_knn`)를 명시했더니 모델이 83.5%로 따랐으나 **오라클 정합률은 61.2%(우연 수준)** 였고, 그 결과 `grounded=True` 케이스의 achieved가 오히려 낮았습니다(−0.168 vs +0.035). **설계 가설을 프롬프트로 주입하면 모델이 그 가설의 오류까지 학습합니다.** 처방 힌트 제거 후 재실험 중.


---

### 6️⃣ Tool 설계 — sparse 축 도입과 정량적 실패 진단

`contextual_knn`(dense pool 확장 + soft rerank)과 `attribute_knn`(exact term filter)을 **서로 다른 정보축**으로 설계했습니다. 전자는 `neighbor_insufficient`, 후자는 `cluster_mismatch`를 겨냥합니다.

**이론적 임계** — contextual의 pool 크기 `P`, 반환 수 `k`, 필터 선택도 `s = support/N`일 때

```
P·s ≥ k  →  contextual이 이미 필터 일치 top-k를 pool 안에 갖고 있음 → 두 tool 동일
임계 s* = k/P = 5/25 = 0.20  →  support* ≈ 24,600
```

**실측이 예측과 일치**합니다.

| support | n | Jaccard(attr, global) | flip_rate |
|---|---:|---:|---:|
| <1k | 165 | **0.32** | 0.156 |
| 1k–10k | 4,310 | **0.29** | 0.077 |
| 10k–25k | 7,499 | 0.48 | 0.096 |
| ≥25k | 9,586 | 0.58~0.68 | 0.12 |

저선택도 구간에서는 **이질 축이 실증**되지만, **후보의 45%가 support ≥ 25,000** 입니다. candidate feature의 단일 조합이 전부 고선택도라 그 구간에 도달하지 못합니다.

| 진단 | 값 | 의미 |
|---|---:|---|
| `Jaccard(contextual, attribute)` | 0.905 | 같은 이웃을 가져옴 |
| attribute 순기여 (EVPI 기여) | +0.0088 (3.4%) | 빼도 상한이 거의 안 변함 |
| base logprob 차이 | 0.031 | 모델은 두 tool을 동등하게 봄 |

> **원인은 tool 설계가 아니라 임베딩**입니다. AnoLLM 임베딩이 categorical feature를 이미 인코딩하고 있어, **같은 인덱스 안에서는 어떤 필터도 새 축이 되지 못합니다.** 코퍼스를 바꾸는 tool(사기 memory bank 조회 등)이 남은 탈출구입니다.

---

### 7️⃣ 평가 체계 — 결과 산출 포맷

세 산출물이 자동 생성됩니다. **아래 표의 수치는 planner v2(Gemini) 시점 스모크(n=20)이며, v3 GRPO planner 서빙 후 교체 예정**입니다.

#### (1) `results.md` — 컴포넌트 ablation (planner 무관 축)

Router/Retriever/양자화/재검색 인과를 축별로 분해합니다.

```
T1-A  Percolate      | variant | cand_recall@5 | routing_acc | MRR | P95(ms) | dist AUROC |
T1-B  Coreset        | coreset | docs | store(MB) | dist AUROC | recall@5 adj | ε-recall |
T1-C  Quantization   | variant | quant | oversample | working set(MB) | dist AUROC | cls@0.42 |
T1-D  재검색 인과분해 | run | n_res | AUROC pre→post | Δ | rescued | degraded | net_gain | dir_acc |
```

`T1-D`가 planner 교체의 직접 비교축입니다 — v2에서 `net_gain −2 ~ −6`, `dir_acc 0.48~0.57`로 **재검색이 순손실**이었고, 이것이 v3 재설계의 출발점입니다.

#### (2) `agent_eval_report.json` — 서빙 배치 추론 집계

```jsonc
{
  "run_id": "...", "pipeline_mode": "agent", "n_total": 20,
  "threshold_info":   { "auroc", "best_threshold", "best_fpr/tpr/fnr", "youden_j" },
  "agent_nd_dual":    { "n_researched", "cls_nd_threshold", "agent_nd_auroc_all" },
  "research_effect":  { "auroc": {"pre","post","delta"},
                        "by_view": {"fixed": {"overall": {"TP","FP","TN","FN","FNR","FPR"}}} },
  "planner_policy":   { "action_distribution", "search_rate", "gate_rate", "gate_reasons",
                        "model_action_distribution", "source_distribution",
                        "planner_latency_ms": {"p50","p95"}, "output_tokens", "mean_cost" },
  "evidence_novelty": { "mean_jaccard_round0_final", "mean_novelty",
                        "cross_cluster_escape_rate", "mean_support", "jaccard_by_action" },
  "budget_curve":     [ { "budget", "n_search", "accuracy", "delta_vs_no_search",
                          "added_latency_ms_total" } ],
  "verdict_change":   { "n_changed", "verdict_change_rate", "correct_flip_n", "wrong_flip_n" },
  "latency_breakdown":{ "base_retrieve_ms", "total_ms" },
  "agent_summary":    { "distance_auroc", "anomaly_auroc", "research_net_gain",
                        "research_flip_rate", "research_dir_acc", "agent_rag_fnr/fpr" }
}
```

**v3에서 핵심으로 볼 4개 블록**

| 블록 | 무엇을 답하는가 | v2 스모크 (n=20) |
|---|---|---|
| `budget_curve` | **§5-C의 예산 곡선** — LLM 호출 B%로 상한의 몇 %를 얻는가 | B=0% acc 0.65 → B=20% **0.75** (Δ+0.10) |
| `planner_policy` | tool 분포 / 게이트 발동 / p50·p95 latency / 평균 cost | `attribute 0.75 / stop 0.25`, gate 25%, p50 **10.3s** |
| `evidence_novelty` | **재검색이 실제로 새 근거를 가져왔는가** (Jaccard/escape/support) | v2 미계측 (`null`) — v3에서 채워짐 |
| `verdict_change` | 판정이 바뀐 비율과 그 방향 | 13.3% 변경, **correct_flip 2/2** |

> `evidence_novelty`가 v3의 신규 축입니다. outcome(`flip`)과 **검색 품질(novelty)** 을 분리 보고해야 "검색 개선 → 탐지 개선"의 인과를 주장할 수 있습니다.

#### (3) `agent_predictions.json` — 건별 trace

```jsonc
{ "classification", "confidence", "final_verdict",
  "primary_cluster_id", "top_5_clusters", "match_type", "top_1_distance",
  "planner_used", "agent_trace", "agent_rounds", "termination",
  "_round0_classification", "_nd_round0", "_nd_final", "_nd_n_rounds",
  "_researched", "_nd_source", "_wm_type", "_research_ms",
  "retrieved_original_indices", "latency_ms" }
```

`_nd_round0` → `_nd_final` 변화와 `retrieved_original_indices`가 있어, **오프라인에서 novelty/escape를 재계산**할 수 있습니다(재수집 불필요).

#### 학습 측 평가 (NB02b)

| 지표 | 정의 | 판정 기준 |
|---|---|---|
| `[G3]` CV acc | GBDT/RF가 오라클 라벨을 예측 | **> majority** 여야 학습 의미 있음 |
| `[B]` 3축 | base+제약 / 학습모델 / oracle 상한 | 학습 순이득 = ② − ① |
| `[G]` grounded × achieved | 근거 인용 여부별 성능 차 | **유의하면 "evidence를 읽는 것이 성능에 기여"** |
| `[T1]` 예산 곡선 | 게이트 B%별 achieved | 게이트 기여와 tool 선택 기여 분리 |


---

## 📎 부록 — Fraudecom 데이터 직접 검증 (Intuition)

위 설계 결정들의 근거가 된 원데이터 실험 (NB03 v3 Section 3, 모델 불필요).

**A-1. 신호 융합 천장 (boundary n=11,440)** — "거리가 유일 신호"의 실측

| 신호 | AUROC | 판정 |
|---|---:|---|
| top5_spread / persona_dev 단독 | 0.403 / 0.496 | 무신호 |
| **nd only** → **nd + std(top3)** | 0.779 → **0.857 (Δ+0.077)** | spread 가치는 **코드 융합**에서만 — LLM 재진술 활용 0% |
| + persona (useful/ALL) | 0.873 (Δ+0.016) | 정제 천장 닫힘 → 판별 확대는 graph(새 차원)로만 |

**A-2. nd 통제 후 feature별 추가 정보량** — categorical 6종 전부 partial corr **+0.000** / AUROC 0.500 → `cluster_mismatch` 근거를 numeric으로 한정한 설계의 실증. numeric(purchase_value/age)도 partial −0.10대(nd에 흡수).

**A-3. misroute 가설 기각 (F7)** — "배정 클러스터에서 크게 벗어나면 배정이 틀렸다"는 가설: max|z| AUROC **0.421**(재측정 0.514) — 오배정 샘플의 이탈이 오히려 작음. ambiguous 중 clean 59~77% — "ambiguous ⟺ 검색 결함" 전제도 붕괴. → wm의 판정 기여 기각의 결정 근거.

**A-4. reroute 재정렬 진단 (n=28,302)** — 전체는 재정렬 성격(Spearman 0.653, Δ+0.022)이나 **boundary에서 순위 보존이 깨지며(0.102) min-fusion Δ+0.062** → 6️⃣ nd min-fusion 서빙 반영의 직접 근거이자, 잔여 실험(nd_round0 vs nd_final)의 기대치.

**A-5. 지표 착시 2건 (정직한 기록)** — ① "evidence-GT 100%"는 rejection이 정답만 남긴 **순환논증**(무필터 Step 0에서 net −14로 반전) ② P1 100%(데이터)도 동어반복 — rejection 생존자의 성질. 평가는 반드시 **무필터 eval set**에서.

---

## 🗂 의사결정 기록 (채택 / 정정 / 기각)

| 결정 | 근거 |
|---|---|
| ❌ Teacher rationalization (GT-conditioned wm) | reason↔label 인과 파괴 → GT-free 생성 + 필터 |
| ❌ 인용율을 품질 지표로 | 정합 서술 통과 → grounded-consistency로 교체 |
| ❌ categorical을 cluster_mismatch 근거로 | partial corr +0.000 (부록 A-2) |
| ❌ eval_loss 기반 best-checkpoint | 템플릿 암기 보상 → P1 기반 early-stop |
| ❌ spread의 LLM 재진술 / retrieval_reliability 슬롯 | 활용 0% / 0/200 미발화 |
| ❌ Global centroid hard-anomaly gate / 단순 PatchCore reweight | precision 0.64~0.76 / FNR 0.16→0.26 순손실 |
| **🔧 정정** "ambiguous cls = evidence 판정" (evidence-GT 100%) | **순환논증** — rejection이 정답만 잔존. 무필터 Step 0 net −13/−14로 반전 → cls 전면 deterministic 전환 |
| **🔧 정정** "GRPO: AUROC +2.3%p, ECE −8%" | v4 무효(cls-gate 버그, 전 스텝 skip) → v5 무효(안전지대 + direction-flip hacking, P1 0%) → **v6 ladder reward로 재실험** (5️⃣ 진단 체인) |
| ✅ prompt v2 / deterministic sufficiency 주입 / Selective expand gate | 고유율 18.9→75.8% / wm 타이밍 11→100% / flip 3.2:1 |
| ❌ Planner에 `stop` 포함 (flat action space) | 라벨 91% stop 편중 → SFT argmax 붕괴(non-stop 정답률 0/20), GRPO 게이트 AUC 0.443~0.481 3회 연속 무정보 이하 → **2-stage 분리** |
| ❌ 비용을 reward에 λ 스칼라화 | `stop` 제거 시 `−λ·n_calls`가 그룹 상수 → advantage에서 소거(`mean_cost` 100 step 고정). λ 튜닝으로 불가 → **admission control** |
| ❌ latency(ms)를 cost 단위로 | 실측(contextual 3,677ms / global 722ms)이 손으로 찍은 1.2/1.5와 **역전** → 호출 수로 통일, latency는 SLA 지표로 분리 |
| ❌ `R_auroc` 커널 | K3 probe 격차 −0.065 — 스칼라 임계값이 25-feature RF보다 나음 = LLM 불필요 결론이 나오는 커널 |
| ❌ 2-turn 확장 | `[K1-T]` 2턴 우세 0.3%(9/2,830) — 이산 커널에서 판정은 1턴에 결판 |
| **🔧 정정** 진단-처방 힌트 프롬프트 주입 | 모델 정합률 83.5%인데 **오라클 정합률 61.2%(우연 수준)** → `grounded=True`의 achieved가 오히려 낮음(−0.168 vs +0.035). 가설을 주입하면 오류까지 학습 |
| **🔧 정정** `attribute_knn` = 이질 축 | 저선택도(support<10k)에서만 Jaccard 0.29. 후보 45%가 support≥25k(임계 s*=k/P) → 순기여 EVPI의 3.4%. **임베딩이 categorical을 이미 인코딩** |
| ✅ `R_flip + 2.5·R_margin` 하이브리드 | Dirac 커널의 좁은 지지집합을 균등 성분으로 확장 — 동률 99.6%→64.2%, EVPI +0.13→+0.24 |
| ✅ 제약 디코딩 (think 자유 + JSON trie) | parse 5.6%→98.5%, format/invalid penalty가 구조적으로 발생 불가 |
| ✅ 출력 자유도 5→1키 + 0.5B 전환 | 판정·진단은 코드, 서술만 LLM — 4단계 측정의 귀결 |

> **6회 독립 측정 + 4단계(S1→S4) 실험의 일관된 결론** — 이 도메인에서 판정 신호는 임베딩 거리(nd)가 유일하며, LLM의 판정·진단 개입은 순손실이다. 따라서 LLM의 자유 출력을 5키 → 1키로 축소해 3B를 0.5B로 대체하고, **agent의 가치를 "정확도 개선"이 아닌 "선택적 호출에 의한 비용 절감(이론치 ≈57%)"과 감사 주석(what_missing)으로 재정의**했다.

## 🔜 남은 작업

- [ ] **Planner v3 서빙 교체** — GRPO adapter → GGUF Q8 → `local_planner` 배포 → 7️⃣ 결과 표/JSON 실측 교체 (현재 표는 v2 Gemini 시점 스모크 n=20)
- [ ] 진단-처방 힌트 제거 후 재실험 — `grounded × achieved` 부호 확인 (§5️⃣-E)
- [ ] `evidence_novelty` 블록 실측 — 재검색 novelty/escape/support를 outcome과 분리 보고
- [ ] tool 축 재설계 검토 — 같은 인덱스 내 필터로는 새 축이 안 나옴(§6️⃣) → 코퍼스를 바꾸는 tool(사기 bank 조회) 타당성 측정
- [ ] Judge 신뢰성 3종 통과 / 3B vs 0.5B 비교 — `grounded rate`가 모델 스펙 함수인지 확인
- [ ] 하위 10% split 오라클로 EVPI 전이성 검증 (현재 상한은 상위 10% in-sample)

## 💻 기술 스택 & 실행

**Infra**: GKE(e2), Elasticsearch 8.x(ECK), GCS, Artifact Registry, Prometheus+Grafana · **ML**: Qwen2.5(0.5B/3B/14B), PEFT/LoRA, llama-cpp-python(GGUF Q4), Gemini Flash Lite(Planner) · **Serving**: FastAPI, LangGraph, MCP wrapper · **Training/Eval**: Colab L4 (NB01-04 → NB02 → NB03), Streamlit, Locust

```bash
# 모드 전환 (agent ↔ forwardRAG baseline)
kubectl patch configmap anomaly-config -n elastic --type merge \
  --patch '{"data": {"PIPELINE_MODE": "forward"}}'
kubectl rollout restart deployment/anomaly-orchestrator deployment/local-analyzer -n elastic

# 평가 (고정 eval set — 버전/모드 간 동일 문제지)
python evaluation/agent_eval/agent_batch_inference.py \
    --eval-source eval_set --n-samples 100 \
    --gcs-upload gs://fraud_ecomm/evaluation/agent-eval
# NB03 v3 Section 2.5 마지막 블록이 stage별 결과를 gs://fraud_ecomm/evaluation/{run_tag}/에 저장
```

```
apps/          # router, retriever, local_analyzer(evaluator|forward), controller, orchestrator
common/        # config, prompts(학습·서빙 단일 소스, 1key/3key 스위치), judge(2-key rubric)
evaluation/    # agent_eval, gt_loader, hard_negative_miner + {run_tag}/ 결과 아카이브
k8s/           # configmap 매니페스트 + batch jobs
notebooks/     # NB01-04(생성, EXPERIMENT_STAGE) → NB02(SFT·GRPO) → NB03(검증·Judge·Fig)
```

## 📚 참조

- Multi-Cluster Memory Bank 이상탐지 (EAAI 2025, 공동 1저자) — 알고리즘 기반
- PatchCore (CVPR 2022) — Coreset sampling, local density reweighting
- Elasticsearch Percolate Query / HNSW, LangGraph, MCP