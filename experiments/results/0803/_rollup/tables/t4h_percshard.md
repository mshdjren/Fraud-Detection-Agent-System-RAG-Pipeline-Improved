### T4-H · percolator 샤드 분산

_(데이터 없음)_

> percolate 는 등록 쿼리(5,725개)를 입력 문서에 MemoryIndex 로 실행하므로
> 비용이 **샤드당 쿼리 수**에 비례한다. 일반 역색인과 달리 CPU 작업이 실제로
> 나뉘어 near-linear 로 스케일한다 — **단 노드 CPU 코어 수까지만**
> (percolator 노드 limit 1800m ≈ 1.8코어 → 2배 부근에서 포화 예상).
> `recall`/`AUROC` 는 세 행에서 동일해야 정상 — 샤딩은 결과를 바꾸지 않는다.
