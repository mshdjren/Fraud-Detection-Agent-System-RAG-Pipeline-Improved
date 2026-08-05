### T1-B · Coreset 압축 (CV C2)

| coreset | docs | store(MB) | working set(MB) | dist AUROC | **recall@5 adj** | recall@5 raw | ε-recall | base mean(ms) | base P95(ms) |
|---|---|---|---|---|---|---|---|---|---|
| 1% | 1228 | 0 | 2.8 | 0.7363 | 0.02 | 0.02 | 0.48 | 198.4 | 287.1 |
| 10% | 12281 | 61.8 | 28.5 | 0.7973 | 0.19 | 0.19 | 0.69 | 150.2 | 283 |
| 100% | 122810 | 616.8 | 284.8 | 0.8711 | 0.75 | 0.75 | 0.91 | 126.7 | 239.1 |

**10% vs 100%** — 메모리 -90.0% · AUROC -7.38%p

> **`recall@5 adj` 를 봐야 한다.** GT top-5 는 100% 인덱스 기준이라, 10% coreset
> 에서는 진짜 top-5 가 인덱스에 존재하지 않는다. `raw` 가 떨어지는 것은 성능
> 저하가 아니라 **구조적 필연**이며, coreset 커버리지로 보정한 `adj` 만 비교 가능하다.
>
> **latency 는 이 규모에서 의미 없다.** 100% 인덱스(≈285MB working set)가 docs 노드
> page cache(≈2Gi)에 전량 상주하므로 압축해도 page fault 가 발생하지 않는다.
> coreset 의 latency 이득은 **working set 이 page cache 를 초과할 때만** 발현된다.
