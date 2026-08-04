### P2 · Prometheus MCP transport · TTFT/TPOT

| run | mode | mcp p50(ms) | **mcp p95(ms)** | vs direct | mcp err | TTFT p50(s) | TTFT p95(s) | TPOT p50(ms) | out tok |
|---|---|---|---|---|---|---|---|---|---|
| a-percolate-v1 | mcp_native | — | — | — | — | — | — | — | — |
| a-percolate-v14 | mcp_native | — | — | — | — | — | — | — | — |
| a-percolate-v14bucket | mcp_native | — | — | — | — | — | — | — | — |
| b-coreset-cs10 | mcp_native | — | — | — | — | — | — | — | — |
| c-quant-bbqos3 | mcp_native | — | — | — | — | — | — | — | — |
| c-quant-f32 | mcp_native | — | — | — | — | — | — | — | — |
| c-quant-int8os0 | mcp_native | — | — | — | — | — | — | — | — |
| c-quant-int8os3 | mcp_native | — | — | — | — | — | — | — | — |
| d-hnsw-s0 | mcp_native | — | — | — | — | — | — | — | — |
| d-hnsw-s1 | mcp_native | — | — | — | — | — | — | — | — |
| d-hnsw-s2 | mcp_native | — | — | — | — | — | — | — | — |
| e-numcand-nc50 | mcp_native | — | — | — | — | — | — | — | — |
| f-mcp-direct | direct | — | — | — | — | — | — | — | — |
| f-mcp-mcp | mcp | — | — | — | — | — | — | — | — |
| f-mcp-mcpnative | mcp_native | — | — | — | — | — | — | — | — |

> **`mcp p95` 가 CV C4 의 진짜 측정값이다.** T3-F 의 `tool Δ` 는
> `total − base_retrieve` 라 analyzer 추론 수 초가 통째로 섞여 있어
> 수십 ms 짜리 transport overhead 를 잴 수 없다(2.7초 vs 12.5ms).
> `mcp err > 0` 이면 그 run 의 재검색 결과는 **전부 무효**다 —
> tool 이 실패하면 patchcore nd 가 계산되지 않아 Δnd 가 0 으로 나온다.
