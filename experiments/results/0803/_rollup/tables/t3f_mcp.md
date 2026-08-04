### T3-F · MCP transport 3-axis (CV C4)

| transport | mode | tool Δ mean(ms) | tool Δ P95(ms) | ctrl loop mean | ctrl loop P95 | **vs direct(ms)** | dist AUROC |
|---|---|---|---|---|---|---|---|
| direct | direct | 2354.2 | 7980.2 | 2353.7 | 7979.6 | +0.0 | 0.8283 |
| mcp | mcp | 2622.9 | 8665.2 | 2622.1 | 8664.2 | +268.6 | 0.8283 |
| mcpnative | mcp_native | 2331.3 | 7989.8 | 2330.8 | 7989.2 | -22.9 | 0.8283 |

> `direct` 를 기준선으로 한 **순수 transport overhead**. 세 경로 모두 동일
> wall-clock 구간(`mcp_transport_latency_seconds`)으로 감싸 측정된다.
> AUROC 가 세 행에서 동일해야 정상 — 다르면 transport 가 결과를 바꾼 것이므로 버그다.
