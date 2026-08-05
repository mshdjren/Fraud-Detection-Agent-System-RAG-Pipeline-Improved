# 실험 결과 — Tier 1

- runs: **24**  ·  indices: **7**


### T1-A · Percolate 최적화 (CV C1)

| variant | version | bucket | cand_recall@5 | routing_acc | MRR | recall@5 | base P95(ms) | total P95(ms) | dist AUROC |
|---|---|---|---|---|---|---|---|---|---|
| v1 | v1 | false | 0.91 | 0.8 | 1 | 0.75 | 178.1 | 8451.8 | 0.8711 |
| v14 | v14 | false | 1 | 0.8 | 0.9467 | 0.92 | 159.1 | 8457 | 0.9157 |
| v14bucket | v14 | true | 1 | 0.42 | 0.7214 | 0.92 | 179.5 | 8601.3 | 0.8013 |

**vs `a-percolate-v1`** — `a-percolate-v14` recall@5 +9.0%p · P95 -10.7% · `a-percolate-v14bucket` recall@5 +9.0%p · P95 +0.8%

### T1-B · Coreset 압축 (CV C2)

| coreset | docs | store(MB) | working set(MB) | dist AUROC | **recall@5 adj** | recall@5 raw | ε-recall | base mean(ms) | base P95(ms) |
|---|---|---|---|---|---|---|---|---|---|
| 1% | 1228 | 0 | 2.8 | 0.7363 | 0.02 | 0.02 | 0.48 | 198.4 | 287.1 |
| 10% | 12281 | 61.8 | 28.5 | 0.7973 | 0.19 | 0.19 | 0.69 | 150.2 | 283 |
| 100% | 122810 | 616.8 | 284.8 | 0.8711 | 0.75 | 0.75 | 0.91 | 126.7 | 239.1 |

**10% vs 100%** — 메모리 -90.0% · AUROC -7.38%p

> **`recall@5 adj` 를 봐야 한다.** GT top-5 는 100% 인덱스 기준이라, 10% coreset
> 에서는 진짜 top-5 가 인덱스에 존재하지 않는다. `raw` 가 떨어지는 것은 성능
> 저하가 아니라 **구조적 필연**이며, coreset 커버리지로 보정한 `adj` 만 비교 가능하다.
>
> **latency 는 이 규모에서 의미 없다.** 100% 인덱스(≈285MB working set)가 docs 노드
> page cache(≈2Gi)에 전량 상주하므로 압축해도 page fault 가 발생하지 않는다.
> coreset 의 latency 이득은 **working set 이 page cache 를 초과할 때만** 발현된다.

### T1-C · Vector Quantization (CV C3)

| variant | quant | oversample | store(MB) | working set(MB) | **dist AUROC** | recall@5 | best_thr | cls@0.42 acc | cls@0.42 FPR | mean nd |
|---|---|---|---|---|---|---|---|---|---|---|
| bbqos3 | bbq_hnsw | 3 | 627.1 | 23.4 | 0.8711 | 0.75 | 1.516 | 0.76 | 0.434 | 0.643 |
| f32 | f32 | 0 | 616.8 | 284.8 | 0.8711 | 0.75 | 1.516 | 0.76 | 0.434 | 0.643 |
| int8os0 | int8_hnsw | 0 | 652.8 | 82.5 | 0.8599 | 0.75 | 2.173 | 0.47 | 1 | 0.835 |
| int8os3 | int8_hnsw | 3 | 652.8 | 82.5 | 0.8711 | 0.75 | 1.516 | 0.76 | 0.434 | 0.643 |

> **읽는 법**: `dist AUROC`(순위)와 `cls@0.42`(절대 임계)를 **반드시 분리해서** 본다.
> 양자화 잔차는 실제 최근접 거리와 맞먹는 크기로 **가법 상수처럼** 더해지므로
> (d_est² ≈ d_true² + ‖r‖²), 단조 변환이라 순위는 보존되고 스케일만 이동한다.
> → `int8os0`(rescore 없음)에서 **AUROC 는 유지되는데 `best_thr`/`cls@0.42` 만 무너지면**
> "거리값이 신호인 시스템에서 rescore 는 optional 이 아니라 필수"가 수치로 증명된다.


![t1c_quant](plots/t1c_quant.png)

### T1-D · 재검색 인과 분해

| run | n_res | AUROC pre | post | Δ | rescued | degraded | **net_gain** | Δnd mean | **dir_acc** | amb_prec | ms/call |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a-percolate-v1 | 35 | 0.8699 | 0.8653 | -0.0046 | 1 | 3 | -2 | -0.016 | 0.5714 | 0.1463 | 552.8 |
| a-percolate-v14 | 33 | 0.9107 | 0.9002 | -0.0105 | 1 | 4 | -3 | -0.013 | 0.5455 | 0.1522 | 514.1 |
| a-percolate-v14bucket | 33 | 0.8838 | 0.8416 | -0.0422 | 4 | 9 | -5 | 0.058 | 0.4848 | 0.1915 | 560.1 |
| b-coreset-cs1 | 1 | 0.7358 | 0.7358 | 0 | 0 | 0 | 0 | 0.198 | 0 | 0 | 442.7 |
| b-coreset-cs10 | 9 | 0.7971 | 0.7983 | 0.0012 | 0 | 1 | -1 | 0.154 | 0.1111 | 0.1111 | 528.1 |
| b-coreset-cs100 | 35 | 0.8699 | 0.8653 | -0.0046 | 1 | 3 | -2 | -0.016 | 0.5714 | 0.1463 | 513.7 |
| c-quant-bbqos3 | 35 | 0.8699 | 0.8653 | -0.0046 | 1 | 3 | -2 | -0.016 | 0.5714 | 0.1463 | 553.3 |
| c-quant-f32 | 35 | 0.8699 | 0.8653 | -0.0046 | 1 | 3 | -2 | -0.016 | 0.5714 | 0.1463 | 540 |
| c-quant-int8os0 | 1 | 0.8599 | 0.8599 | 0 | 0 | 0 | 0 | 0.152 | 0 | 0 | 480.6 |
| c-quant-int8os3 | 35 | 0.8699 | 0.8653 | -0.0046 | 1 | 3 | -2 | -0.016 | 0.5714 | 0.1463 | 534.6 |
| d-hnsw-s0 | 35 | 0.8699 | 0.8653 | -0.0046 | 1 | 3 | -2 | -0.016 | 0.5714 | 0.1463 | 560.8 |
| d-hnsw-s1 | 35 | 0.8699 | 0.8653 | -0.0046 | 1 | 3 | -2 | -0.016 | 0.5714 | 0.1463 | 528.8 |
| d-hnsw-s2 | 35 | 0.8699 | 0.8653 | -0.0046 | 1 | 3 | -2 | -0.016 | 0.5714 | 0.1463 | 570 |
| e-numcand-nc100 | 5 | 0.8283 | 0.8081 | -0.0202 | 0 | 2 | -2 | 0.017 | 0.4 | 0 | 617.8 |
| e-numcand-nc200 | 5 | 0.8283 | 0.8081 | -0.0202 | 0 | 2 | -2 | 0.017 | 0.4 | 0 | 551 |
| e-numcand-nc400 | 5 | 0.8283 | 0.8081 | -0.0202 | 0 | 2 | -2 | 0.017 | 0.4 | 0 | 536.5 |
| e-numcand-nc50 | 5 | 0.8283 | 0.8081 | -0.0202 | 0 | 2 | -2 | 0.017 | 0.4 | 0 | 558.4 |
| f-mcp-direct | 5 | 0.8283 | 0.8081 | -0.0202 | 0 | 2 | -2 | 0.017 | 0.4 | 0 | 135.3 |
| f-mcp-mcp | 5 | 0.8283 | 0.8081 | -0.0202 | 0 | 2 | -2 | 0.017 | 0.4 | 0 | 373.2 |
| f-mcp-mcpnative | 5 | 0.8283 | 0.8081 | -0.0202 | 0 | 2 | -2 | 0.017 | 0.4 | 0 | 598.8 |
| i-bucket-t0m7 | 35 | 0.8894 | 0.837 | -0.0524 | 5 | 11 | -6 | 0.077 | 0.4857 | 0.2041 | 536.3 |
| i-bucket-t1m3 | 34 | 0.8994 | 0.8386 | -0.0608 | 4 | 11 | -7 | 0.075 | 0.4706 | 0.1875 | 507.4 |
| i-bucket-t1m5 | 35 | 0.8894 | 0.837 | -0.0524 | 5 | 11 | -6 | 0.077 | 0.4857 | 0.2041 | 530.1 |
| i-bucket-t2m3 | 33 | 0.8838 | 0.8416 | -0.0422 | 4 | 9 | -5 | 0.058 | 0.4848 | 0.1915 | 545.8 |

> `net_gain = rescued − degraded`. verdict 가 하나도 안 바뀌어도
> `dir_acc`(Δnd 부호가 정답 방향인 비율)는 살아 있으므로,
> **"재검색 효과 없음"과 "효과를 측정 못 함"을 구분**할 수 있다.
> 0.5 근처면 재검색이 무작위, 0.8 이상이면 판정을 못 바꿨을 뿐 정보는 더한 것.

### T1-D-diag · 재검색 nd 갱신 진단

| run | n_res | mean|Δnd| | dir_acc | n_flip | ms/call |
|---|---|---|---|---|---|
| a-percolate-v1 | 35 | 0.0546 | 0.5714 | 4 | 552.8 |
| a-percolate-v14 | 33 | 0.0654 | 0.5455 | 5 | 514.1 |
| a-percolate-v14bucket | 33 | 0.1162 | 0.4848 | 13 | 560.1 |
| b-coreset-cs1 | 1 | 0.198 | 0 | 0 | 442.7 |
| b-coreset-cs10 | 9 | 0.1541 | 0.1111 | 1 | 528.1 |
| b-coreset-cs100 | 35 | 0.0546 | 0.5714 | 4 | 513.7 |
| c-quant-bbqos3 | 35 | 0.0546 | 0.5714 | 4 | 553.3 |
| c-quant-f32 | 35 | 0.0546 | 0.5714 | 4 | 540 |
| c-quant-int8os0 | 1 | 0.1519 | 0 | 0 | 480.6 |
| c-quant-int8os3 | 35 | 0.0546 | 0.5714 | 4 | 534.6 |
| d-hnsw-s0 | 35 | 0.0546 | 0.5714 | 4 | 560.8 |
| d-hnsw-s1 | 35 | 0.0546 | 0.5714 | 4 | 528.8 |
| d-hnsw-s2 | 35 | 0.0546 | 0.5714 | 4 | 570 |
| e-numcand-nc100 | 5 | 0.0614 | 0.4 | 2 | 617.8 |
| e-numcand-nc200 | 5 | 0.0614 | 0.4 | 2 | 551 |
| e-numcand-nc400 | 5 | 0.0614 | 0.4 | 2 | 536.5 |
| e-numcand-nc50 | 5 | 0.0614 | 0.4 | 2 | 558.4 |
| f-mcp-direct | 5 | 0.0614 | 0.4 | 2 | 135.3 |
| f-mcp-mcp | 5 | 0.0614 | 0.4 | 2 | 373.2 |
| f-mcp-mcpnative | 5 | 0.0614 | 0.4 | 2 | 598.8 |
| i-bucket-t0m7 | 35 | 0.1258 | 0.4857 | 16 | 536.3 |
| i-bucket-t1m3 | 34 | 0.1248 | 0.4706 | 15 | 507.4 |
| i-bucket-t1m5 | 35 | 0.1258 | 0.4857 | 16 | 530.1 |
| i-bucket-t2m3 | 33 | 0.1162 | 0.4848 | 13 | 545.8 |

> `mean|Δnd| == 0` 이고 `dir_acc` 가 `—` 면 **재검색 nd 가 갱신되지 않은 것**이다
> (효과 없음이 아니라 측정 불가). 원인은 `agent_predictions.json` 의
> `agent_rounds[].patchcore_status` 로 구분된다:
> `tool_failed`(예외 삼킴) / `insufficient_neighbors`(이웃<2) / `off`(env) / `ok`(실제 무변화).

### T2-D · HNSW m/ef sweep (CV C7)

| variant | m | ef_c | store(MB) | working set(MB) | segs | dist AUROC | recall@5 | cand_recall@5 | base mean(ms) | base P95(ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| s0 | 16 | 100 | 616.8 | 284.8 | 15 | 0.8711 | 0.75 | 0.91 | 123.7 | 216.5 |
| s1 | 32 | 200 | 617.1 | 299.8 | 13 | 0.8711 | 0.75 | 0.91 | 139.8 | 284.9 |
| s2 | 8 | 64 | 581.4 | 277.3 | 18 | 0.8711 | 0.75 | 0.91 | 154.4 | 337.8 |

> `m` 은 그래프 연결도(메모리·recall 동시 상승), `ef_construction` 은 build 품질.
> **여기서 고른 운영점을 stage 3 전체에 고정**한다 — 검색이 흔들리면
> MCP/toolgate 측정이 교란된다.

### T2-E · num_candidates 곡선

| num_candidates | dist AUROC | recall@5 | cand_recall@5 | base mean(ms) | base P95(ms) | total P95(ms) |
|---|---|---|---|---|---|---|
| 50 | 0.8283 | 0.7 | 0.9 | 317.4 | 889.9 | 29460.6 |
| 100 | 0.8283 | 0.7 | 0.9 | 160.7 | 300.4 | 30628.2 |
| 200 | 0.8283 | 0.7 | 0.9 | 164.5 | 281.8 | 30426.4 |
| 400 | 0.8283 | 0.7 | 0.9 | 212 | 449.6 | 29318.3 |

> **재적재 0** 인 유일한 검색 축 — 쿼리 파라미터라 ConfigMap patch 만으로 바뀐다.
> recall 이 평탄해지는 지점(knee)이 운영점이며, 그 이상은 latency 만 낸다.


![t2e_numcand](plots/t2e_numcand.png)

### T2-I · bucket prefilter 튜닝

| variant | tolerance | min_match | **routing_acc** | MRR | cand_recall@5 | recall@5 | base mean(ms) | base P95(ms) | dist AUROC |
|---|---|---|---|---|---|---|---|---|---|
| t0m7 | 0 | 7 | 0.34 | 0.6572 | 1 | 0.93 | 133.9 | 201.3 | 0.7884 |
| t1m3 | 1 | 3 | 0.35 | 0.6623 | 1 | 0.93 | 159.7 | 219.9 | 0.7884 |
| t1m5 | 1 | 5 | 0.34 | 0.6572 | 1 | 0.93 | 145.6 | 204.7 | 0.7884 |
| t2m3 | 2 | 3 | 0.42 | 0.7214 | 1 | 0.92 | 157.8 | 248 | 0.8013 |

> v14bucket 에서 `routing_acc` 0.80 → **0.25**, MRR 1.00 → 0.57 로 무너졌다.
> 그런데 `cand_recall@5` 는 1.0 유지 — **정답 클러스터는 top-5 에 계속 있는데
> top-1 서열만 흔들린다.** prefilter 가 후보를 거르는 게 아니라 percolate
> 점수 서열을 희석시킨다는 뜻이다.
> feature 9개 중 `min_match` 개만 ±`tolerance` 버킷 안에 들면 통과하므로,
> 느슨할수록 살아남는 룰이 많아진다. `t0m7`(가장 엄격)에서 routing_acc 가
> 회복되면 원인이 확정된다. `base P95` 도 함께 떨어져야 정상
> (prefilter 가 매칭 대상을 줄였다면 빨라져야 하는데 v14bucket 은 오히려 늘었다).

### T3-F · MCP transport 3-axis (CV C4)

| transport | mode | tool Δ mean(ms) | tool Δ P95(ms) | ctrl loop mean | ctrl loop P95 | **vs direct(ms)** | dist AUROC |
|---|---|---|---|---|---|---|---|
| direct | direct | 2334.2 | 7889.8 | 2333.6 | 7889.3 | +0.0 | 0.8283 |
| mcp | mcp | 2376.8 | 8186.9 | 2376 | 8185.9 | +42.6 | 0.8283 |
| mcpnative | mcp_native | 2397.3 | 8106.1 | 2396.7 | 8105.4 | +63.1 | 0.8283 |

> `direct` 를 기준선으로 한 **순수 transport overhead**. 세 경로 모두 동일
> wall-clock 구간(`mcp_transport_latency_seconds`)으로 감싸 측정된다.
> AUROC 가 세 행에서 동일해야 정상 — 다르면 transport 가 결과를 바꾼 것이므로 버그다.


![t3f_mcp](plots/t3f_mcp.png)

### T3-G · conditional re-retrieval (CV C6)

_(데이터 없음)_

> CV C6 은 "FNR −4.5%p, P99 증가 +8~12% 억제" 를 주장한다.
> **off 대비 on 의 FNR 감소와 P95 증가를 같이 봐야** trade-off 가 성립한다.

### T4-H · percolator 샤드 분산

_(데이터 없음)_

> percolate 는 등록 쿼리(5,725개)를 입력 문서에 MemoryIndex 로 실행하므로
> 비용이 **샤드당 쿼리 수**에 비례한다. 일반 역색인과 달리 CPU 작업이 실제로
> 나뉘어 near-linear 로 스케일한다 — **단 노드 CPU 코어 수까지만**
> (percolator 노드 limit 1800m ≈ 1.8코어 → 2배 부근에서 포화 예상).
> `recall`/`AUROC` 는 세 행에서 동일해야 정상 — 샤딩은 결과를 바꾸지 않는다.

### T6 · 부하 3단계 (light/medium/heavy)

_(locust 결과 없음 — run_locust.sh 미실행)_

### P1 · Prometheus 메모리 병목 (page fault)

| run | quant | working set(MB) | pf p50 | pf p95 | heap% | retriever p50 | retriever p95 | top1 dist p50 |
|---|---|---|---|---|---|---|---|---|
| a-percolate-v1 | f32 | 284.8 | — | — | — | — | — | — |
| a-percolate-v14 | f32 | 284.8 | — | — | — | — | — | — |
| a-percolate-v14bucket | f32 | 284.8 | — | — | — | — | — | — |
| b-coreset-cs1 | f32 | 2.8 | — | — | — | — | — | — |
| b-coreset-cs10 | f32 | 28.5 | — | — | — | — | — | — |
| b-coreset-cs100 | f32 | 284.8 | — | — | — | — | — | — |
| c-quant-bbqos3 | bbq_hnsw | 23.4 | — | — | — | — | — | — |
| c-quant-f32 | f32 | 284.8 | — | — | — | — | — | — |
| c-quant-int8os0 | int8_hnsw | 82.5 | — | — | — | — | — | — |
| c-quant-int8os3 | int8_hnsw | 82.5 | — | — | — | — | — | — |
| d-hnsw-s0 | f32 | 284.8 | — | — | — | — | — | — |
| d-hnsw-s1 | f32 | 299.8 | — | — | — | — | — | — |
| d-hnsw-s2 | f32 | 277.3 | — | — | — | — | — | — |
| e-numcand-nc100 | f32 | 284.8 | — | — | — | — | — | — |
| e-numcand-nc200 | f32 | 284.8 | — | — | — | — | — | — |
| e-numcand-nc400 | f32 | 284.8 | — | — | — | — | — | — |
| e-numcand-nc50 | f32 | 284.8 | — | — | — | — | — | — |
| f-mcp-direct | f32 | 284.8 | — | — | — | — | — | — |
| f-mcp-mcp | f32 | 284.8 | — | — | — | — | — | — |
| f-mcp-mcpnative | f32 | 284.8 | — | — | — | — | — | — |
| i-bucket-t0m7 | f32 | 284.8 | — | — | — | — | — | — |
| i-bucket-t1m3 | f32 | 284.8 | — | — | — | — | — | — |
| i-bucket-t1m5 | f32 | 284.8 | — | — | — | — | — | — |
| i-bucket-t2m3 | f32 | 284.8 | — | — | — | — | — | — |

> **압축의 latency 이득은 page fault 가 날 때만 발현된다.** working set 이
> docs 노드 page cache 안에 전부 들어가면 f32 도 pf=0 이라 bbq 와 차이가 없다.
> `pf p95` 가 전 행 0 이면 실험 조건이 성립하지 않은 것이다 —
> docs 노드 `resources.limits.memory` 를 낮춰 page cache 를 줄이거나
> aug 인덱스로 데이터를 키워야 한다.

### P2 · Prometheus MCP transport · TTFT/TPOT

| run | mode | mcp p50(ms) | **mcp p95(ms)** | vs direct | mcp err | TTFT p50(s) | TTFT p95(s) | TPOT p50(ms) | out tok |
|---|---|---|---|---|---|---|---|---|---|
| a-percolate-v1 | mcp_native | — | — | — | — | — | — | — | — |
| a-percolate-v14 | mcp_native | — | — | — | — | — | — | — | — |
| a-percolate-v14bucket | mcp_native | — | — | — | — | — | — | — | — |
| b-coreset-cs1 | mcp_native | — | — | — | — | — | — | — | — |
| b-coreset-cs10 | mcp_native | — | — | — | — | — | — | — | — |
| b-coreset-cs100 | mcp_native | — | — | — | — | — | — | — | — |
| c-quant-bbqos3 | mcp_native | — | — | — | — | — | — | — | — |
| c-quant-f32 | mcp_native | — | — | — | — | — | — | — | — |
| c-quant-int8os0 | mcp_native | — | — | — | — | — | — | — | — |
| c-quant-int8os3 | mcp_native | — | — | — | — | — | — | — | — |
| d-hnsw-s0 | mcp_native | — | — | — | — | — | — | — | — |
| d-hnsw-s1 | mcp_native | — | — | — | — | — | — | — | — |
| d-hnsw-s2 | mcp_native | — | — | — | — | — | — | — | — |
| e-numcand-nc100 | mcp_native | — | — | — | — | — | — | — | — |
| e-numcand-nc200 | mcp_native | — | — | — | — | — | — | — | — |
| e-numcand-nc400 | mcp_native | — | — | — | — | — | — | — | — |
| e-numcand-nc50 | mcp_native | — | — | — | — | — | — | — | — |
| f-mcp-direct | direct | — | — | — | — | — | — | — | — |
| f-mcp-mcp | mcp | — | — | — | — | — | — | — | — |
| f-mcp-mcpnative | mcp_native | — | — | — | — | — | — | — | — |
| i-bucket-t0m7 | mcp_native | — | — | — | — | — | — | — | — |
| i-bucket-t1m3 | mcp_native | — | — | — | — | — | — | — | — |
| i-bucket-t1m5 | mcp_native | — | — | — | — | — | — | — | — |
| i-bucket-t2m3 | mcp_native | — | — | — | — | — | — | — | — |

> **`mcp p95` 가 CV C4 의 진짜 측정값이다.** T3-F 의 `tool Δ` 는
> `total − base_retrieve` 라 analyzer 추론 수 초가 통째로 섞여 있어
> 수십 ms 짜리 transport overhead 를 잴 수 없다(2.7초 vs 12.5ms).
> `mcp err > 0` 이면 그 run 의 재검색 결과는 **전부 무효**다 —
> tool 이 실패하면 patchcore nd 가 계산되지 않아 Δnd 가 0 으로 나온다.


### t2e_numcand

![t2e_numcand](plots/t2e_numcand.png)


### t3f_mcp

![t3f_mcp](plots/t3f_mcp.png)


### t1e_pareto

![t1e_pareto](plots/t1e_pareto.png)


### t1f_latency

![t1f_latency](plots/t1f_latency.png)
