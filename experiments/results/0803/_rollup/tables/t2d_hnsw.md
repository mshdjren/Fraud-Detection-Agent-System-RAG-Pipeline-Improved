### T2-D · HNSW m/ef sweep (CV C7)

| variant | m | ef_c | store(MB) | working set(MB) | segs | dist AUROC | recall@5 | cand_recall@5 | base mean(ms) | base P95(ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| s0 | 16 | 100 | 616.8 | 284.8 | 14 | 0.8699 | 0.75 | 0.91 | 196.7 | 324.6 |
| s1 | 32 | 200 | 617 | 299.8 | 15 | 0.8699 | 0.75 | 0.91 | 187.7 | 293.6 |
| s2 | 8 | 64 | 586.9 | 277.3 | 17 | 0.8699 | 0.75 | 0.91 | 183.7 | 298.4 |

> `m` 은 그래프 연결도(메모리·recall 동시 상승), `ef_construction` 은 build 품질.
> **여기서 고른 운영점을 stage 3 전체에 고정**한다 — 검색이 흔들리면
> MCP/toolgate 측정이 교란된다.
