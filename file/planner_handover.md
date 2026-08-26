# Planner (GRPO/SFT) 인수인계 문서

## 0. 배경 한 줄 요약
fraudecom 이상탐지 AgentRAG 파이프라인에서, Evaluator(state 해석, SFT 전용)와 별도로
**Planner**(retrieval action 선택: stop/global_knn/contextual_knn)를 GRPO로 학습 중.
목표: "GRPO가 SFT 대비 의미 있는가"를 오라클 기반 실험으로 검증.

---

## 1. 전체 아키텍처

```
transaction → router(percolate, cluster_id+persona) → retriever(round0, top-3 이웃)
            → evaluator(wm_type/agrees/differs/deviant/rare/persona_match, SFT 고정)
            → Planner ← 여기가 이번 작업 범위
                 action ∈ {stop, global_knn, contextual_knn(features≤2)}
            → (필요시) 재검색 → 최종 판정
```

- **Evaluator**: state → 구조화 해석. SFT 전용(정답 mapping 존재, 과거 GRPO 실패 확인됨). 안 건드림.
- **Planner**: state → action. 정답이 실행 전엔 모름(counterfactual) → GRPO 후보.
- **Oracle**: 매 state당 모든 action을 미리 다 실행해 reward를 전수 계산해둔 lookup table.
  SFT 타깃(argmax reward)과 GRPO reward(lookup) 둘 다 이 오라클에서 나옴 — **재검색 없이 학습 가능**.
- **Reward**: `R = Δmargin − λ·cost`, `margin(nd,y) = (2y-1)(nd-τ)`.
  `cost`: stop=0, contextual_knn=1.2, global_knn=1.5(스코프 global 고정, V1 원칙).
  candidate features: `agrees ∪ persona_match`, 최대 2개 (evaluator 결정론적 evidence).

**내 의견**: 이 3단계 분리(Evaluator=SFT 해석기 / Planner=GRPO 정책 / Oracle=사전계산 reward table)는
설계 자체로는 견고하다. 문제가 된 건 설계가 아니라 **reward 함수 경제성(cost 스케일)과 action 3개
구조가 GRPO의 그룹 비교 메커니즘과 궁합이 안 맞았던 것** — 아래 §5 참고.

---

## 2. 구현 단계별 (파일별) + 의견

| 파일 | 역할 | 상태/의견 |
|---|---|---|
| `controller.py` `/rollout` | state 1개 → 전 action 전수 실행 → reward 계산 | round0(retrieve+analyze)는 실서빙과 동일 로직 재사용. **의견**: LangGraph 순차루프와 분리한 설계가 맞았음 — 여러 action을 같은 base state에서 병렬 비교해야 해서. |
| `agent_tools_mcp_server.py` | `global_knn`/`contextual_knn` MCP tool | mcp_native 경로만 구현(요청사항). |
| `retriever.py` | `should_boost`(soft feature boost) | **버그 2건 발견/수정**: ①`SEARCH_MODE` 기본값이 `should_boost`를 처리 못하는 분기로 새서 contextual_knn≡global_knn이 되던 버그(V2 corr=1.000으로 발견) ②should_boost를 ES 쿼리 레벨에서 점수 합산하면 `_score`가 1.0을 넘어 distance 역산 공식이 NaN을 뱉고 `JSONResponse(allow_nan=False)`가 500을 냄 → **풀 unboosted knn으로 넓게 pool 받아온 뒤 Python에서 재정렬**하는 방식으로 재설계(진짜 distance 보존). **의견**: 두 버그 다 "작동은 하는데 결과가 조용히 틀린" 유형이라 오라클 신뢰도에 큰 리스크였음 — V1~V3 진단 체계 없었으면 못 잡았을 것. |
| `ingest_gaussian_aug.py` | percolator/aug 인덱스 매핑 | `cluster_id` 외 전 필드 `index:false,doc_values:false` → contextual_knn의 term query가 전부 400. `RETRIEVAL_FILTERABLE_COLS` allowlist로 수정 코드 제공. 재bulk 필요(ES 필드 속성 불변, `_reindex`는 `vec`이 `_source` 제외라 불가). 사용자가 raw(비augmented) 인덱스로 우회 확인함. |
| `oracle_extractor.py` | 오라클 수집 배치 스크립트 | GTLoader 재사용, router+retriever 1쌍(base call)+`/rollout` 호출. 재시도/backoff, ES 매핑 완화책(필드 하나 빼고 재시도) 포함. |
| `oracle_backfill.py` | 기존 오라클에 `persona`/`neighbors` 필드만 추가 | 전체 재수집(~4h) 없이 router+retriever 1쌍만 다시 호출(~수십분). Planner prompt 재구성에 필요한데 처음 수집분엔 없었음. |
| `prompts.py` | Planner 전용 prompt | evaluator prompt와 완전 분리(`PLANNER_SYSTEM_PROMPT`/`build_planner_user_prompt`). v1(DECISION RULE 휴리스틱 포함) vs v2(evidence-only) 비교 실험 — **v2가 오히려 더 나쁨**(SFT non-stop 정확도 하락). **의견**: "grounding 안 된 서술이 해롭다"는 가설은 반증됨 — 오히려 휴리스틱이 0.5B에게 귀납적 바이어스로 도움이 됐을 가능성. |
| `02b_planner_sft_grpo_training.ipynb` | SFT+GRPO 학습 노트북 | evaluator NB2 구조 이식(데이터 통계, G3 identifiability probe, confusion matrix, telemetry). SFT→GRPO 런타임 연속성(GCS 재로드) 구조로 리팩터. |

---

## 3. Oracle 수집 요약 (간단)

- 대상: test set 앞 10%(2830개, hard_negative_miner.py와 동일 관례), action당 exhaustive rollout → 11052 rows.
- `wm_type` 분포(evaluator 버그 수정 후): accept/cluster_mismatch/neighbor_insufficient 3종 다 관측됨.
- **λ 스윕은 재수집 없이 사후 계산 가능** — `delta_margin`/`cost` 원본이 저장돼 있어 `reward_final = Δmargin - λ·cost`만 다시 계산 (매우 중요한 방법론, 반복 활용됨).
- **ambiguous-only 필터**: `wm_type.notna()`만 남김 (confident/SKIP_ANALYZER_ON_CONFIDENT 케이스 제거) — 2830→1121.
  실서빙에서 Planner는 sufficiency=="ambiguous"일 때만 호출되므로, confident 케이스는 학습에 불필요한 잡음.
- feature 조합은 `agrees∪persona_match`, 최대 2개까지만 실제 실행됨 — 3개 이상/다른 pool 정의/cluster-scope
  contextual은 재수집 필요(미시도).

---

## 4. 핵심 지표 정의 (실험 전체에서 반복 사용)

- **eval mean_reward**: eval set(자연분포) 각 케이스에 모델이 고른 action의 오라클 reward 평균. **함정**: stop은 항상 reward=0(정의상)이라, "거의 다 stop 예측"이면 오답이어도 0 근처로 나와 착시를 일으킴.
- **non_stop mean_reward / 정답률**: oracle≠stop인 케이스만 뗀 지표. **이게 진짜 지표** — 전체 mean_reward만 보면 mode collapse를 "baseline과 비슷하다"로 착각함.
- **group_diversity**: GRPO 그룹(같은 prompt, num_generations개 생성) 내 unique action 수 평균. 1=완전 확신(entropy collapse).
- **garbage_rate / valid_group_diversity**: JSON 파싱 실패/3종 밖 action을 diversity에서 분리 집계(높은 temperature 실험에서 "가짜 다양성" 착시 방지용으로 추가).
- **reward 성분 분해**(mean Δmargin vs mean cost): stop 쏠림이 margin 문제인지 cost 문제인지 구분.

---

## 5. 실험 히스토리 — 핵심 발견 (시간순 압축)

| # | 실험 | 핵심 수치/관찰 | Intuition |
|---|---|---|---|
| 1 | λ 스윕 (재수집 없이 사후계산) | λ=0.05에서 stop 90%/global 6%/contextual 4%; λ 낮출수록 contextual 비중↑ | reward 임계값이 stop 쏠림의 상당 부분을 설명 |
| 2 | global_knn Δmargin, y별 분리 | y=0: +0.063, y=1: -0.057 (거의 대칭) | 후보 풀이 클수록(global) 사기 샘플도 우연히 가까운 정상과 매칭되는 위험(distance concentration) — 정상 라우팅 정정 이득과 상쇄되어 순효과≈0. **global_knn은 구조적으로 거의 항상 짐** |
| 3 | G3 identifiability (tree/RF probe) | 구조화 feature만으로 majority baseline 이상 CV acc 나옴 | 0.5B의 표현력 문제가 아님 — 학습 역학 문제 |
| 4 | SFT epoch=3 (첫 균형 downsample) | stop confusion 완전 붕괴(10/240) | 완전균형(1:1:1)이 stop 사전확률을 지나치게 왜곡 |
| 5 | GRPO (다양한 샘플링: natural/majority/downsample/search_oversample) | **매번 100% stop 또는 100% 특정 action으로 collapse**, group_diversity는 대부분 건강(2~4) | 데이터 분포가 원인이 아님 — diversity 있어도 collapse |
| 6 | telemetry 추가(reward 성분 분해) | mean cost(빨강)가 mean Δmargin(초록)을 거의 항상 압도 | **cost가 margin을 구조적으로 이긴다** — GRPO가 정직하게 "stop이 유리"로 수렴 중이었을 가능성 |
| 7 | temperature/penalty 조합 실험 | temp=1.5+낮은 penalty → garbage 텍스트 폭증(다양성 착시) | penalty를 너무 낮추면 포맷 붕괴; greedy(eval) vs sampling(train) 디코딩 차이로 invalid_rate 불일치 발생(정상) |
| 8 | DAPO 논문 조사 | "그룹 reward 분산 0 → advantage 0 → gradient 0" (dynamic sampling 원리) | 우리 구조와 정확히 일치하는 실패 모드 |
| 9 | dynamic sampling 구현/디버깅 | round() 반올림이 n=1 데이터에서 hard cutoff처럼 작동 → BASE_REPEATS 절대곱 방식이 전체 스케일업(1009→3437) → mean-정규화 방식으로 재설계 | "smooth weighting"은 반복 카운트가 충분히 커야 의미 있음 |
| 10 | ambiguous-only 필터 + λ=0.02 SFT | **non-stop 정답률 64.7%, non-stop reward +0.0497 (최초 양수)** | λ=0.02가 "cost가 압도한다"는 가설을 가장 강하게 검증 — 그러나 stop 정확도 18%로 하락(라벨 분포가 이번엔 반대로 왜곡) |
| 11 | 같은 SFT 위 GRPO, lr=3e-6 vs lr=2e-5 vs epsilon-high | lr=2e-5: non-stop 97.1%지만 **stop 정확도 0%(전부 contextual로 collapse)** — 다른 방향의 동일 병리 | mean_reward 단독으로는 "collapse 방향이 바뀐 것"과 "진짜 개선"을 구분 못함 — confusion matrix 필수 |
| 12 | `MAJORITY_OVERSAMPLE_CAP` × λ 상호작용 | 같은 CAP=4가 λ=0.05에선 완만한 증강, λ=0.02에선(raw 소수 클래스 절대 개수가 작아져) `min()`이 상한에 걸려 사실상 1:1 균형이 됨 | **λ와 balancing 배율은 독립 레버가 아니라 함께 재조정해야 함** — 반복 확인된 원칙 |

---

## 6. 현재 상태 / 결론 후보

**구조적 원인 후보 (배제/유력 순으로 정리)**:
- ❌ entropy/group diversity collapse — 다수 실험에서 diversity 건강했는데도 collapse 반복, 배제.
- ❌ 프롬프트 grounding(v1 vs v2) — v2가 오히려 더 나쁨, 주요인 아님.
- ❌ GRPO config(temperature/KL/scale_rewards/epsilon_high) — 다 시도, 근본 해결 안 됨.
- ✅ **reward 경제성(cost ≫ 평균 margin) + action 3개 중 1개(stop)가 무위험 안전패라는 구조** — 가장 유력. GRPO의 그룹 상대비교 메커니즘이 "그룹에 stop이 섞이면 그쪽으로 끌림"을 반복.
- ⚠️ λ와 SFT balancing(majority CAP)의 상호작용 미해결 — 마지막 실험(λ=0.02, CAP=2)이 가장 균형 잡힌 결과(stop 79.5%/non-stop 18.2%)였으나 **모든 SFT가 epoch=1**이라 노이즈와 신호 미분리 상태.

**다음 담당자에게 제안하는 순서**:
1. epoch=1 고정을 풀고(epoch=2 등) 같은 (λ=0.02, CAP=2) 설정 재현 — 지금 결과가 노이즈인지 확인.
2. λ×CAP 그리드를 좁게(λ=0.02~0.035, CAP=1.5~3) 재탐색, 매번 **raw action 분포를 먼저 찍고** CAP 역산.
3. 이 SFT가 안정되면 그 위에서만 GRPO 재시도(lr은 3e-6~2e-5 사이 중간값), **confusion matrix를 mean_reward와 항상 같이 확인**.
4. 그래도 collapse 반복되면: SFT-only 서빙 확정 + GRPO 보류(PDF §13 "SFT로 충분했다" 결론 수용). Action space 확장(3-feature, cluster-scope 등)은 오라클 재수집 비용 때문에 이 결론 이후에만 고려.

**알려진 코드 버그(전부 수정 완료, 재발 방지용 기록)**: `apply_chat_template(return_dict=True)` 누락, `evaluate_planner` 지역변수 대신 전역변수 참조, reward key의 `action`/`features`가 리스트/중첩 시 unhashable, `GRPO_BATCH_SIZE`는 `num_generations`의 배수 필요, torchao 버전 monkey-patch 필요(Colab 환경), `majority` CAP에 float 대입 시 `pool * reps` TypeError(정수 캐스팅 필요), dynamic sampling 시 `round()`가 n=1 데이터에서 hard cutoff처럼 작동.
