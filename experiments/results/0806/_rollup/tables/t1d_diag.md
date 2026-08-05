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
