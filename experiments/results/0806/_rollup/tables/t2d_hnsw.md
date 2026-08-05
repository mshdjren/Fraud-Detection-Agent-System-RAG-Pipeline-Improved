### T2-D · HNSW m/ef sweep (CV C7)

| variant | m | ef_c | store(MB) | working set(MB) | segs | dist AUROC | recall@5 | cand_recall@5 | base mean(ms) | base P95(ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| s0 | 16 | 100 | 616.8 | 284.8 | 15 | 0.8711 | 0.75 | 0.91 | 123.7 | 216.5 |
| s1 | 32 | 200 | 617.1 | 299.8 | 13 | 0.8711 | 0.75 | 0.91 | 139.8 | 284.9 |
| s2 | 8 | 64 | 581.4 | 277.3 | 18 | 0.8711 | 0.75 | 0.91 | 154.4 | 337.8 |

> `m` 은 그래프 연결도(메모리·recall 동시 상승), `ef_construction` 은 build 품질.
> **여기서 고른 운영점을 stage 3 전체에 고정**한다 — 검색이 흔들리면
> MCP/toolgate 측정이 교란된다.
