### T1-A · Percolate 최적화 (CV C1)

| variant | version | bucket | cand_recall@5 | routing_acc | MRR | recall@5 | base P95(ms) | total P95(ms) | dist AUROC |
|---|---|---|---|---|---|---|---|---|---|
| v1 | v1 | false | 0.91 | 0.8 | 1 | 0.75 | 177.3 | 8243.5 | 0.8699 |
| v14 | v14 | false | 1 | 0.8 | 0.9467 | 0.92 | 185.9 | 8547.9 | 0.9105 |
| v14bucket | v14 | true | 1 | 0.42 | 0.7214 | 0.92 | 213.8 | 8366.3 | 0.8836 |

**vs `a-percolate-v1`** — `a-percolate-v14` recall@5 +9.0%p · P95 +4.9% · `a-percolate-v14bucket` recall@5 +9.0%p · P95 +20.6%
