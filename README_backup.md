# Fraud Detection AgentRAG Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/GKE-Kubernetes-326CE5?logo=kubernetes" />
  <img src="https://img.shields.io/badge/Elasticsearch-8.x-005571?logo=elasticsearch" />
  <img src="https://img.shields.io/badge/LangGraph-StateGraph-1C3C3C" />
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi" />
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python" />
</p>

## 📌 프로젝트 개요

**Multi-Cluster Memory Bank 이상탐지 알고리즘(EAAI 2025)을 운영 가능한 AgentRAG 시스템으로 확장한 엔드투엔드 파이프라인**입니다.

Tabular 이상 거래를 "retrieval → 거리 기반 이상탐지 → 판정 원인 설명" 흐름으로 처리하며, Vector Indexing · Router(클러스터 라우팅) · Retriever(HNSW) · Evaluator(LLM) · Planner · Orchestrator의 6개 MSA로 GKE에 배포했습니다. 정적 forwardRAG(전수 LLM 추론 baseline)에서 출발해, **검색 결과를 의심하고 선택적으로 재검색하는 Agent 시스템**으로 확장한 과정과 그 측정 근거가 이 프로젝트의 핵심입니다.

### 🎯 핵심 문제 해결 (검증 완료 지표)

| 문제 | 해결 방안 | 성과 |
|------|---------|------|
| 벡터 검색 병목 | Percolate Query Bucketization (3단계 고도화) | Router Recall@5 +9%p, P99 latency −25% |
| 메모리 제약 | Coreset Sampling 10% 운영점 | 메모리 −89% (2.8GB→0.31GB), AUROC 손실 0.4%p 이내, median latency −46% |
| Teacher 오답이 학습 신호 오염 | Rejection Sampling (GT-free 생성 → GT 필터) | teacher 정답률 60~65% 감안 ×1.7 oversample로 품질·수량 동시 확보 |
| 작은 모델의 modal collapse | 과제 설계 원인 정량 규명 → Deterministic-LLM 역할 분리 | what_missing 생성 타이밍 11%→100%, 실제 이탈 feature 지목 정합도 70% |
| Ambiguous 전수 재검색의 낭비 | 측정 기반 Selective Tool Gate (rule search) | expand 발동 84%→10~12%, 게이트 영역 내 good/bad flip 3.2:1 |

---

## 🏗️ 시스템 아키텍처

```text
Test Transaction
      ↓
[Router]     Percolate Query → 클러스터 매칭 (rank1=percolate, rank2~5=centroid score)
      ↓
[Retriever]  Filtered HNSW → 거리 신호 top-5 / LLM evidence top-3
      ↓
[Controller] LangGraph StateGraph — Evaluator(LLM) → Selective Tool Gate → Planner
      │         ├─ confident            → accept
      │         ├─ mismatch 신호        → reroute_cross_cluster (rank2 클러스터 + persona 교체)
      │         └─ expand rule 충족     → expand_within_cluster (evidence 3→5)
      ↓
[Orchestrator] 최종 판정 + round_log/termination → API Response
```

`PIPELINE_MODE=forward`로 전환하면 동일 인프라에서 **정적 forwardRAG baseline**(route→retrieve→analyzer 1회, 3-key 출력)으로 동작합니다 — Agent 시스템의 가치를 같은 문제지에서 A/B 측정하기 위한 구조입니다.

---

## 🔬 핵심 기술

### 1️⃣ 검색 파이프라인 최적화

**Percolate Query 3단계 고도화** — Strict AND(v1) → Centroid-Augmented(v1+aug) → Bucketization(v1+aug+bucket):

| query version | Cluster acc / Recall@5 | Router MRR | Latency p99 (ms) | Recall@5 (knn) / AUROC |
| :--- | :---: | :---: | :---: | :---: |
| Strict AND (v1) | 0.816 / 0.816 | 0.941 | 420.25 | 0.786 / 0.8726 |
| **v1 + aug** | **0.823 / 0.905** | **1.00** | 430.01 | **0.956 / 0.9125** |
| v1 + aug + bucket | 0.823 / 0.897 | 0.988 | **409.25** | 0.956 / 0.9037 |

**Coreset Sampling 운영점** — 1,000만 건 인덱스에 page fault를 의도적으로 유발해 메모리 병목을 재현하고, 압축률-AUROC-latency trade-off에서 10%를 운영점으로 채택 (메모리 −89%, AUROC 손실 0.4%p 이내, median latency −46%). HNSW 파라미터(m, ef_construction) sweep과 벡터 quantization(int8/BBQ)을 직교 축으로 결합해 벡터 인덱스 메모리 추가 −25%, Recall 손실 0.5%p 이내.

### 2️⃣ 학습 데이터 생성 — Mining + Rejection Sampling

- **Pipeline-aware Hard Negative Mining**: 파이프라인이 실제 틀리는 3가지 오류 패턴(클러스터 매칭 실패 · 최근접 정상 벡터 검색 오류 · 거리 차이 모호)을 정의하고 학습 데이터에 반영
- **Rejection Sampling**: teacher(Qwen2.5-14B)가 GT hint 없이 순수 추론 → cls가 GT와 불일치하는 샘플 제거. teacher 정답률(60~65%)을 감안한 ×1.7 oversample로 목표 수량 확보. mining/boundary/easy 비율 조정으로 특정 구간 편중에 따른 학습 신호 왜곡 방지
- **what_missing 품질 필터** *(진행 중)*: 학습 데이터 진단 결과 wm 고유율 18.9%, 최빈 문장("boundary distance; expand evidence")이 28.2%를 점유 — cls만 거르는 rejection이 wm boilerplate를 통과시켜 student가 이를 암기(서빙에서 42/50 동일 문장으로 재현). boilerplate 금지구 + feature grounding + dedup cap(≤2%) 필터를 Pass 2 직후에 추가

### 3️⃣ forwardRAG → Evaluator 전환: 한계의 정량 진단

초기 forwardRAG는 매칭된 클러스터·최근접 정상 벡터와의 차이를 근거로 **{cls, confidence, reasoning} 3-key**를 출력하는 정적 구조였습니다. 학습·서빙에서 확인된 구조적 한계:

1. **confidence가 실제 난이도와 무관하게 평균값으로 회귀**하고 reasoning이 상투적 문구로 수렴
2. **클러스터 매칭을 100% 신뢰하는 전제** — 라우팅 자체가 틀린 hard negative(Type 1/2)를 구조적으로 감지 불가
3. **modal collapse 정량 규명**: {cls, sufficiency, what_missing}를 단일 출력으로 학습하자 소형 모델이 다수 클래스로 붕괴. 원인이 모델 용량이 아니라 이산적 sufficiency 판정 · key 간 conditional dependency · what_missing 학습 신호 희소성이 겹친 **과제 설계 문제**임을 확인 (teacher조차 boundary 정확도 59% 천장)

→ 검색 신뢰도(sufficiency)를 판별하고 재검색(expand/reroute)을 촉발하는 **Evaluator + Agent loop**로 확장.

### 4️⃣ Deterministic-LLM 역할 분리

"규칙으로 정확히 계산되는 신호는 코드가, 서술만 LLM이" — 6회 독립 측정에서 L2 거리가 유일한 분류 신호였다는 결론의 필연적 귀결입니다.

| 신호 | 담당 | 근거 |
|---|---|---|
| sufficiency | **deterministic** (nd-band [0.3, 0.55) 주입) | 이산 판정의 SFT collapse — 주입 전환 후 wm 생성 타이밍 11%→100% |
| cls | **deterministic** (nd threshold, `ANALYZER_DECISION_MODE=det`) | nd가 유일 신호; `llm` 모드와 A/B 스위치 유지 |
| what_missing | **LLM** (유일한 자유 출력) | 실제 이탈 feature 지목 정합도 70% |
| 이웃 spread | **deterministic** (거리 구조 신호) | LLM 재진술 실험 실패(활용 0%) — 서술 대상이 아닌 계산 대상 |

**Selective Tool Gate** — "ambiguous면 무조건 expand"를 폐기하고 측정으로 발동 조건을 설계:

- **expand**: 오프라인 rule search(N=28,302)로 `nd∈[0.48, 0.60) AND plateau_ratio≤0.142`(top-5 거리 조밀도) 도출 — 발동 영역에서 good/bad flip **3.2 : 1** (89/28), coverage 1.2%. spread5 조건은 non-binding으로 확인되어 제외. 실측: expand rate 84% → 10~12%
- **reroute**: what_missing 텍스트에는 mismatch 신호가 없음(boilerplate)을 확인 → 라우터의 관측 신호로 대체: `match_type=centroid`(percolate 매칭 실패, 실측 60%) + rank1/2 score 근접(≥0.93). rank2 클러스터로 전환 시 **해당 클러스터의 persona까지 교체**해 analyzer가 새 기준으로 재평가 *(τ 캘리브레이션 진행 중 — round_log의 gate 관측 필드 × GT cluster로 확정 예정)*

### 5️⃣ LLM Analyzer 경량화 및 평가 체계

- **Knowledge Distillation**: Qwen2.5-14B teacher → 0.5B/3B student, GGUF Q4 양자화로 CPU 서빙
- **SFT/GRPO**: 정확도·서술 일치도·사실성 reward → SFT 대비 AUROC +2.3%p, ECE −8%
- **LLM-as-Judge**: GT가 없는 what_missing 품질을 이산 rubric(0/0.5/1)으로 자동 채점. **judge 신뢰성 자체를 검증** — GT 케이스와의 점수 상관 + 동일 입력 2회 채점 일치율 + 순서 교환 편향 체크를 통과해야 지표로 채택
- **Latency 분석** *(진행 중)*: Colab L4 → e2-standard-2(2 vCPU) 이관 시 이론 15~40× 증가 (prefill compute-bound 30~70× + decode bandwidth-bound 10~20×). 실측 단건 ~38s → persona 복원 후 ~68s. 대응: ① Selective Gate로 2차 호출 자체를 84%→10%대로 절감 ② det 모드에서 LLM 역할이 wm 한 문장으로 축소된 만큼 **0.5B 전환** ③ 프롬프트 다이어트(prefill 지배 환경)

### 6️⃣ Agent System vs forwardRAG — 동일 인프라 A/B

configmap 스위치만으로 두 모드를 같은 문제지(NB01 teacher 생성 eval set, 고정 seed)에서 비교합니다:

| | forwardRAG (baseline) | Agent System |
|---|---|---|
| 전환 | `PIPELINE_MODE=forward` + `ANALYZER_TASK=forward` | `agent` + `evaluator` (기본) |
| 흐름 | route→retrieve→analyzer 1회 | + Evaluator→Gate→Planner→expand/reroute loop |
| LLM 출력 | {cls, confidence, reasoning} 전부 LLM | {what_missing}만 LLM (cls/sufficiency는 deterministic) |
| 비교 축 | AUROC / FNR / FPR / **LLM 호출량** / latency / 경로별 기여 | 동일 harness (`agent_batch_inference --eval-source eval_set`) |

*현재 상태: 게이트 배포 후 smoke에서 구조 동작 확인(expand 10~12%, reroute 첫 발화 30%). 판정 지표는 nd threshold 캘리브레이션(0.5→0.42, 분포 실측 기반)과 50+ 샘플 재실행 후 확정 예정.*

### 7️⃣ Observability & 검증 도구

- **LangGraph round_log**: 각 round의 {nd, sufficiency, what_missing, planner 결정 vs 게이트 결정, 게이트 관측치(plateau/ratio/match_type), evidence top-k}를 구조화 기록 — 경로별(no-loop/expand/reroute) FNR·AUROC 기여, conditional edge 정확성, loop 안전성(budget 초과 0건)을 배치 리포트로 측정
- **Streamlit 검증 UI**: 입력 거래의 연관성(최근접 정상 벡터·매칭 점수)·판정 근거·재검색 이력 시각화, 매칭 점수가 낮으면 검색 신뢰도 주의 표기
- **Locust 부하 테스트**: 단계별 부하에서 P95 latency·에러율 측정 (coreset/HNSW 운영점 검증에 사용)

---

## 🗂 의사결정 기록 (채택 / 보류 / 기각)

측정으로 내린 결정들 — "무엇을 하지 않기로 했는가"도 시스템 설계의 일부입니다.

| 결정 | 근거 |
|---|---|
| ❌ Global Centroid hard-anomaly gate | threshold 전 구간 precision 0.64~0.76 — 정상 임베딩의 multimodal 구조상 단일 centroid로 고정밀 게이트 불가 |
| ❌ spread 신호의 LLM 서술 | 학습 후 활용 0% — deterministic 거리 신호(plateau_ratio)로 이관 |
| ❌ retrieval_reliability 출력 슬롯 | 0/200 미발화 — 폐기, sufficiency로 일원화 |
| ❌ 단순 PatchCore reweight (k=5) | boundary AUROC +2.5%p였지만 할인 전용 계수로 분포가 좌측 이동 → FNR 0.16→0.26, wrong flip 순손실(816 vs 581) |
| ⏸ Centered reweight | FNR 안전(0.159)하나 boundary 개선 +0.5%p로 소실 — Selective Gate 영역 한정 재검토 대상 |
| ⏸ 0.5B student 전환 | det 모드에서 LLM 역할 축소로 정당화 — wm 품질 필터 적용 재학습 후 3B와 judge 비교로 결정 |
| ✅ Selective expand gate | rule search 검증 (위 4️⃣) |
| ✅ Deterministic sufficiency 주입 | wm 타이밍 11%→100% |

---

## 💻 기술 스택

**Infra**: GKE (e2 CPU nodes), Elasticsearch 8.x (ECK), GCS, Artifact Registry, Prometheus + Grafana
**ML**: Qwen2.5 (0.5B/3B student, 14B teacher), PEFT/LoRA, llama-cpp-python (GGUF Q4), Gemini Flash Lite (Planner)
**Serving**: FastAPI, LangGraph StateGraph, MCP tool wrapper, structlog, pybreaker
**Training/Eval**: Colab L4 (NB01 데이터 생성 → NB02 SFT → NB03 평가·Judge), Streamlit, Locust

## 🚀 실행 가이드

```bash
# 1) ES + 서비스 배포 (k8s/ configmap 패턴 — 상세는 배포 문서)
kubectl apply -f k8s/00-anomaly-config.yaml -n elastic
# router / retriever / local-analyzer / controller / orchestrator configmap + rollout

# 2) 모드 전환 (agent ↔ forwardRAG baseline)
kubectl patch configmap anomaly-config -n elastic --type merge \
  --patch '{"data": {"PIPELINE_MODE": "forward"}}'
kubectl patch configmap local-analyzer-config -n elastic --type merge \
  --patch '{"data": {"ANALYZER_TASK": "forward"}}'
kubectl rollout restart deployment/anomaly-orchestrator deployment/local-analyzer -n elastic

# 3) 평가 (고정 eval set — 버전/모드 간 동일 문제지)
python evaluation/agent_eval/agent_batch_inference.py \
    --eval-source eval_set --n-samples 100 \
    --gcs-upload gs://fraud_ecomm/evaluation/agent-eval
# → NB03 Section C가 bucket 마운트로 로드해 Layer1/3 채점 + 버전 비교
```

## 📁 프로젝트 구조

```
apps/          # router, retriever, local_analyzer(evaluator|forward), controller(LangGraph), orchestrator
common/        # config, prompts(학습·서빙 단일 소스), judge(LLM-as-Judge)
evaluation/    # agent_eval(batch/metrics/report), gt_loader, hard_negative_miner, vector_db_sweep
k8s/           # configmap 기반 배포 매니페스트 + batch jobs
notebooks/     # NB01(데이터 생성) → NB02(SFT) → NB03(평가·Judge) → NB04(forwardRAG, 재작성 중)
```

## 📚 참조

- Multi-Cluster Memory Bank 이상탐지 (EAAI 2025, 공동 1저자) — 본 파이프라인의 알고리즘 기반
- PatchCore (CVPR 2022) — Coreset sampling, local density reweighting
- Elasticsearch Percolate Query / HNSW, LangGraph, MCP