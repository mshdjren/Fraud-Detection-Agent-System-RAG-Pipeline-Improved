### T3-F · MCP transport 3-axis (CV C4)

| transport | mode | tool Δ mean(ms) | tool Δ P95(ms) | ctrl loop mean | ctrl loop P95 | **vs direct(ms)** | dist AUROC |
|---|---|---|---|---|---|---|---|
| direct | direct | 2334.2 | 7889.8 | 2333.6 | 7889.3 | +0.0 | 0.8283 |
| mcp | mcp | 2376.8 | 8186.9 | 2376 | 8185.9 | +42.6 | 0.8283 |
| mcpnative | mcp_native | 2397.3 | 8106.1 | 2396.7 | 8105.4 | +63.1 | 0.8283 |

> `direct` 를 기준선으로 한 **순수 transport overhead**. 세 경로 모두 동일
> wall-clock 구간(`mcp_transport_latency_seconds`)으로 감싸 측정된다.
> AUROC 가 세 행에서 동일해야 정상 — 다르면 transport 가 결과를 바꾼 것이므로 버그다.
