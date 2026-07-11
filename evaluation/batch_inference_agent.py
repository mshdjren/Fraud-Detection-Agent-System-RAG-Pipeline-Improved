"""
agent_smoke_inference.py — Agent loop 단건/소량 스모크 추론 (scratch)
====================================================================
목적:
  orchestrator(agentic) 의 /detect 로 test case 1건(또는 N건)을 보내,
  [deterministic pre-router → deep_retrieve → analyzer → planner 루프]
  가 의도대로 1회 동작하고 response가 정상 출력되는지 "확인"한다.
  (Recall/AUROC 등 평가가 아니라, agent loop가 끝까지 도는지 검증이 우선)

설계 원칙:
  ★ 평가(MetricCalculator) 없음. GT 라벨 비교 없음.
  ★ GTLoader는 그대로 재사용 (load_all + get_test_case) → 입력 TestCase 생성.
  ★ evaluation_metrics.py / gt_loader.py / batch_inference.py 무변경.
  ★ agent 전용 response 필드(track, planner_used, agent_trace)를 출력/저장.

Usage:
  # 단건 (가장 기본 — 1개 요청이 agent loop를 의도대로 지나는지 확인)
  python agent_smoke_inference.py \
      --experiment-case pca_64_k100 --percolate-version v14 \
      --n-samples 1 --output-dir /results

  # 소량 (loop/planner 동작 분포 확인)
  python agent_smoke_inference.py --n-samples 20 --output-dir /results
"""

import os
import json
import time
import argparse
import asyncio
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional

import httpx
import structlog

# gt_loader 그대로 재사용 (입력 TestCase 생성 전용)
from gt_loader import GTLoader, VALID_EXPERIMENT_CASES

logger = structlog.get_logger()

# agent orchestrator 서비스 URL (router/retriever batch와 달리 /detect 단일 진입점)
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://anomaly-orchestrator-agentic:80")

# planner가 고를 수 있는 follow-up action (분포 집계용)
FOLLOWUP_ACTIONS = (
    "accept_verdict", "expand_within_cluster", "reroute_cross_cluster", "compare_features",
)


# ──────────────────────────────────────────────────────────
# /detect 단건 호출
# ──────────────────────────────────────────────────────────
async def call_agent_detect(client: httpx.AsyncClient, test_case: Dict, timeout: int) -> Optional[Dict]:
    try:
        t = time.perf_counter()
        resp = await client.post(f"{ORCHESTRATOR_URL}/detect", json=test_case, timeout=timeout)
        resp.raise_for_status()
        result = resp.json()
        result["_client_latency_ms"] = (time.perf_counter() - t) * 1000
        return result
    except httpx.TimeoutException:
        logger.warning("agent_detect_timeout")
        return None
    except Exception as e:
        logger.error("agent_detect_failed", error=str(e))
        return None


# ──────────────────────────────────────────────────────────
# 단건 출력 + 의도된 loop 동작 점검
# ──────────────────────────────────────────────────────────
def _check_and_print(idx: int, result: Dict) -> bool:
    trace = result.get("agent_trace", [])
    track = result.get("track")
    # 의도된 동작: trace가 존재하고, deep이면 finalize로 종료 / fast면 pre_router:fast로 종료
    ok = bool(trace) and (
        (track == "deep" and "finalize" in trace) or
        (track == "fast")
    )

    print("\n" + "─" * 70)
    print(f"[sample idx={idx}]  track={track}  planner_used={result.get('planner_used')}")
    print(f"  agent_trace : {' → '.join(str(s) for s in trace)}")
    print(f"  verdict     : {result.get('classification')} (conf={result.get('confidence')})")
    print(f"  top1_dist   : {result.get('top_1_distance')}")
    print(f"  cluster     : {result.get('primary_cluster_id')}  match_type={result.get('match_type')}")
    rsn = (result.get("reasoning") or "")
    print(f"  reasoning   : {rsn[:200]}")
    lat = result.get("latency_ms")
    cli = result.get("_client_latency_ms")
    print(f"  latency_ms  : {lat}  (client={cli:.0f}ms)" if isinstance(cli, (int, float)) else f"  latency_ms : {lat}")
    if not ok:
        print(f"  ⚠️  agent_trace 비정상 (loop 미완료 의심): {trace}")
    return ok


# ──────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(
        description="Agent loop smoke inference (scratch) — /detect 단건/소량 검증",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiment-case", type=str, default="pca_64_k100",
                        choices=VALID_EXPERIMENT_CASES)
    parser.add_argument("--percolate-version", type=str, default="v14")
    parser.add_argument("--coreset-percentage", type=str, default="100")
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output-dir", type=str, default="./results")
    args = parser.parse_args()

    # GTLoader 재사용 — gt_data는 사용하지 않음 (입력 TestCase 생성만)
    loader = GTLoader(
        experiment_case=args.experiment_case,
        coreset_percentage=args.coreset_percentage,
    )
    test_data, test_emb, _gt = loader.load_all()
    n_total = len(test_data)

    start = max(0, args.start_index)
    end = min(start + args.n_samples, n_total)
    indices = list(range(start, end))

    print("=" * 70)
    print("AGENT LOOP SMOKE INFERENCE (scratch, no evaluation)")
    print("=" * 70)
    print(f"  orchestrator     : {ORCHESTRATOR_URL}")
    print(f"  experiment_case  : {args.experiment_case}")
    print(f"  percolate_version: {args.percolate_version}")
    print(f"  n_samples        : {len(indices)} (idx {start}..{end - 1} / total {n_total})")

    predictions: List[Dict] = []
    n_err = 0
    action_dist: Counter = Counter()

    async with httpx.AsyncClient() as client:
        for idx in indices:
            tc = loader.get_test_case(
                idx, test_data, test_emb,
                percolate_version=args.percolate_version,
                experiment_case=args.experiment_case,
            )
            result = await call_agent_detect(client, tc, args.timeout)
            if result is None:
                n_err += 1
                predictions.append({"idx": idx, "error": True})
                print(f"\n[sample idx={idx}] ❌ ERROR (no response)")
                continue

            ok = _check_and_print(idx, result)
            result["_smoke_ok"] = ok
            predictions.append({"idx": idx, **result})

            for step in result.get("agent_trace", []):
                if step in FOLLOWUP_ACTIONS:
                    action_dist[step] += 1

    # 저장 (agent 전용 — 기존 predictions.json과 분리)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "agent_predictions.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2, default=str)

    n_ok = sum(1 for p in predictions if p.get("_smoke_ok"))
    n_planner = sum(1 for p in predictions if p.get("planner_used"))

    print("\n" + "=" * 70)
    print(f"[agent-smoke] DONE  n={len(predictions)}  ok={n_ok}  errors={n_err}")
    print(f"  planner_used : {n_planner}/{len(predictions)}")
    print(f"  action_dist  : {dict(action_dist)}")
    print(f"  saved        : {pred_path}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
