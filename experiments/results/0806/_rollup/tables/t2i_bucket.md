### T2-I · bucket prefilter 튜닝

| variant | tolerance | min_match | **routing_acc** | MRR | cand_recall@5 | recall@5 | base mean(ms) | base P95(ms) | dist AUROC |
|---|---|---|---|---|---|---|---|---|---|
| t0m7 | 0 | 7 | 0.34 | 0.6572 | 1 | 0.93 | 133.9 | 201.3 | 0.7884 |
| t1m3 | 1 | 3 | 0.35 | 0.6623 | 1 | 0.93 | 159.7 | 219.9 | 0.7884 |
| t1m5 | 1 | 5 | 0.34 | 0.6572 | 1 | 0.93 | 145.6 | 204.7 | 0.7884 |
| t2m3 | 2 | 3 | 0.42 | 0.7214 | 1 | 0.92 | 157.8 | 248 | 0.8013 |

> v14bucket 에서 `routing_acc` 0.80 → **0.25**, MRR 1.00 → 0.57 로 무너졌다.
> 그런데 `cand_recall@5` 는 1.0 유지 — **정답 클러스터는 top-5 에 계속 있는데
> top-1 서열만 흔들린다.** prefilter 가 후보를 거르는 게 아니라 percolate
> 점수 서열을 희석시킨다는 뜻이다.
> feature 9개 중 `min_match` 개만 ±`tolerance` 버킷 안에 들면 통과하므로,
> 느슨할수록 살아남는 룰이 많아진다. `t0m7`(가장 엄격)에서 routing_acc 가
> 회복되면 원인이 확정된다. `base P95` 도 함께 떨어져야 정상
> (prefilter 가 매칭 대상을 줄였다면 빨라져야 하는데 v14bucket 은 오히려 늘었다).
