### T1-D-diag · 재검색 nd 갱신 진단

| run | n_res | mean|Δnd| | dir_acc | n_flip | ms/call |
|---|---|---|---|---|---|
| a-percolate-v1 | 35 | 0 | — | 0 | 67.5 |
| a-percolate-v14 | 33 | 0 | — | 0 | 60.6 |
| a-percolate-v14bucket | 33 | 0 | — | 0 | 63.1 |
| b-coreset-cs1 | 1 | 0 | — | 0 | 59.5 |
| b-coreset-cs10 | 9 | 0 | — | 0 | 106 |
| b-coreset-cs100 | 35 | 0 | — | 0 | 122.7 |
| c-quant-bbqos3 | 35 | 0 | — | 0 | 126.9 |
| c-quant-f32 | 35 | 0 | — | 0 | 126.9 |
| c-quant-int8os0 | 1 | 0 | — | 0 | 53.9 |
| c-quant-int8os3 | 35 | 0 | — | 0 | 127.9 |
| d-hnsw-s0 | 35 | 0 | — | 0 | 131.2 |
| d-hnsw-s1 | 35 | 0 | — | 0 | 126.2 |
| d-hnsw-s2 | 35 | 0 | — | 0 | 167.4 |
| e-numcand-nc50 | 5 | 0 | — | 0 | 106.8 |
| f-mcp-direct | 5 | 0.0614 | 0.4 | 2 | 203.4 |
| f-mcp-mcp | 5 | 0.0614 | 0.4 | 2 | 412.4 |
| f-mcp-mcpnative | 5 | 0 | — | 0 | 76.1 |

> `mean|Δnd| == 0` 이고 `dir_acc` 가 `—` 면 **재검색 nd 가 갱신되지 않은 것**이다
> (효과 없음이 아니라 측정 불가). 원인은 `agent_predictions.json` 의
> `agent_rounds[].patchcore_status` 로 구분된다:
> `tool_failed`(예외 삼킴) / `insufficient_neighbors`(이웃<2) / `off`(env) / `ok`(실제 무변화).
