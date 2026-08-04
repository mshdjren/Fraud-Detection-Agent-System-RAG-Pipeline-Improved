### T2-E · num_candidates 곡선

| num_candidates | dist AUROC | recall@5 | cand_recall@5 | base mean(ms) | base P95(ms) | total P95(ms) |
|---|---|---|---|---|---|---|
| 50 | 0.8283 | 0.7 | 0.9 | 221.8 | 310.1 | 31005.4 |

> **재적재 0** 인 유일한 검색 축 — 쿼리 파라미터라 ConfigMap patch 만으로 바뀐다.
> recall 이 평탄해지는 지점(knee)이 운영점이며, 그 이상은 latency 만 낸다.
