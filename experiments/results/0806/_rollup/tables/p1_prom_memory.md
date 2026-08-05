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
