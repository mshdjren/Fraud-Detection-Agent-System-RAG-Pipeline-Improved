# Agent Workflow 실험 설계 초안 (코드 미적용)

## 목적
본 문서는 기존 fraudecom 파이프라인의 코드 변경 없이, 향후 3개 연구 방향(멀티 analyzer agent, multi-step feedback reasoning, agent infra 최적화)의 실험 설계/검증 기준을 정리한다.

## 프로젝트 현황 요약
- 현재 파이프라인은 Router → Retriever → Analyzer → Orchestrator 구조이며, Router는 percolate 기반 클러스터 후보를 생성한다.
- Router는 실험 버전(v1~v14) 스위칭을 지원하며, centroid 보강/버킷화 실험 맥락이 남아 있다.
- Retriever는 cluster-filter 기반 검색과 skip_analyzer 경로(전체 검색)를 모두 지원한다.
- Orchestrator는 SKIP_ANALYZER 분기, retry/circuit breaker/cache 및 Prometheus 관측 지표를 제공한다.

## 3가지 계획에 대한 비판적 점검

### 1) 단일 analyzer → 멀티 agent ensemble
잠재 모순:
- Router/Retriever가 이미 강한 후보 압축을 수행 중인데, agent를 지나치게 분할하면 설명 다양성은 늘어도 latency/일관성 손실 가능.
- "임베딩 거리 기반" 판단과 "최근접 정상 데이터 증거" 판단은 상관관계가 높아 완전 독립 agent로 두면 정보 중복 가능.

수정 제안:
- 완전 병렬 다중 agent가 아니라 **역할 분리형 3-agent**로 제한:
  1) Distance agent: 수치 점수/불확실성
  2) Persona agent: cluster persona 기반 정성 해석
  3) Evidence agent: top-k normal 근거 대비 이상 포인트 추출
- 최종 결합은 가중합 + veto rule(고신뢰 충돌 시 human-readable conflict flag).

### 2) distance-confidence gap 기반 재검색 루프 + multi-step reasoning
잠재 모순:
- gap threshold 기반 루프와 "LLM multi-step"을 동시에 늘리면, 어느 요소가 성능 개선 원인인지 분리 불가(실험 교란).
- 재검색 반복은 recall 개선 가능하지만, 잘못 설계 시 false positive drift 및 tail latency 급증.

수정 제안:
- 단계적 ablation:
  - A: 기존 단일 루프(threshold 재검색만)
  - B: reasoning만 추가(재검색 없음)
  - C: reasoning + 재검색 결합
- 각 단계에서 latency budget(P95/P99 상한) 고정 후 비교.

### 3) agent workflow 인프라 최적화 (batching/latency/MCP)
잠재 모순:
- free-tier GKE에서 agent 간 통신을 늘리면 CPU throttling + network hop으로 오히려 악화 가능.
- MCP wrapping을 너무 일찍 도입하면 기능 실험보다 인터페이스 복잡도 검증이 먼저 되어 연구 목표가 흐려질 수 있음.

수정 제안:
- 우선 단일 프로세스 내부 오케스트레이션(함수콜)로 upper-bound 측정 → 이후 MCP/서비스 분리.
- batching은 "요청 배치"와 "agent 내부 후보 배치"를 구분해 별도 측정.

## 실험 목적/비교군/지표

### 계획 1 (멀티 analyzer agent)
- 목적: 성능(AUROC/ECE/설명충실도) 개선이 latency 증가를 정당화하는지 검증
- 비교군:
  - Baseline: 기존 단일 analyzer
  - Exp-1: 2-agent (distance + evidence)
  - Exp-2: 3-agent (distance + persona + evidence)
  - Exp-3: 3-agent + lightweight meta-judge
- 핵심 지표:
  - 판별: AUROC, AUPRC, F1@opt-threshold
  - calibration: ECE, Brier
  - 시스템: P50/P95/P99 latency, timeout율
  - 설명 품질: factuality/consistency(샘플링 수동평가 + 규칙 기반 체크)

### 계획 2 (feedback reasoning loop)
- 목적: 재검색 루프 + multi-step reasoning이 hard negative(Type1~3)에서 유의미한지 확인
- 비교군:
  - Baseline: loop 없음
  - Loop-only: threshold 기반 재검색
  - Reason-only: multi-step reasoning, 검색 고정
  - Hybrid: reasoning + 재검색
- 핵심 지표:
  - Hard negative type별 AUROC/FNR
  - confidence gap 축소량(전/후)
  - iteration당 성능 증가율(1회, 2회)
  - tail latency 및 비용(토큰/CPU time)

### 계획 3 (agent infra 최적화)
- 목적: free-tier 노드 제약에서 workflow 실용성 확보
- 비교군:
  - Monolithic orchestration
  - Microservice RPC
  - MCP-wrapped router/retriever
- 핵심 지표:
  - end-to-end latency breakdown(네트워크/추론/검색)
  - 처리량(QPS) vs 오류율
  - batch size sweep(1,2,4,8,16)
  - 비용지표(CPU throttling, memory peak)

## 적용 권장 핵심 기법
- 실험 설계: blocked evaluation (hard negative type stratified)
- 통계 검증: bootstrap CI, paired significance test
- 운영 안정화: early-exit 정책, max-iteration cap, timeout budget
- 관측성: 기존 Prometheus metric에 agent 단계별 histogram/counter 추가

## 진행 원칙
- 코드 적용 전, 위 실험 설계를 먼저 확정하고 성공 기준(Go/No-Go)을 수치로 고정한다.
- 변경은 최소 단위 PR로 분리해 원인-결과 해석 가능성을 확보한다.
