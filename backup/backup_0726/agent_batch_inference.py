"""
agent_batch_inference.py — Agent Loop 전체 평가용 배치 추론
===========================================================
batch_inference_agent.py(smoke) 를 확장하여:
  1. GT 라벨 비교 (AUROC, FNR 계산 가능)
  2. round 0 판정 저장 (_round0_classification) → verdict change 추적
  3. distance band 분류 → boundary zone 분리 측정
  4. agent_eval_metrics.AgentMetricCalculator 연동

설계 원칙:
  - GTLoader / get_test_case 무변경 재사용
  - 기존 evaluation_metrics.MetricCalculator의 calculate_distance_based_auroc
    / calculate_anomaly_auroc도 그대로 호출 가능
  - FORCE_MOCK_ANALYZER 모드에서도 동작 (loop 검증 목적)

Usage:
  python agent_batch_inference.py \\
    --experiment-case pca_64_k100 \\
    --percolate-version v14 \\
    --n-samples 200 \\
    --output-dir /results/agent_eval

  # boundary zone만 타깃 (start-index를 조정하거나 --boundary-only)
  python agent_batch_inference.py --n-samples 500 --boundary-only
"""

import os
import json
import time
import argparse
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

import httpx
import structlog

from gt_loader import GTLoader, VALID_EXPERIMENT_CASES
from agent_eval_metrics import (
    AgentMetricCalculator,
    build_llm_scores,
    find_best_threshold,
    BOUNDARY_LOW, BOUNDARY_HIGH,
)

logger = structlog.get_logger()

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://anomaly-orchestrator:80")

# ★ [mode 분기] controller/orchestrator와 동일한 PIPELINE_MODE 분기 —
#   "agent"  : S4 evaluator (1-key wm, det cls/suf) — 학습/평가 stem training_data_mined
#   "forward": S2 forwardRAG (3-key cls/conf/reasoning) — stem training_data_forward
#   ⚠ 서버(orchestrator PIPELINE_MODE)와 이 값이 일치해야 함 — 불일치 시 종료 사유
#     (termination=="forward" 여부)로 감지해 경고를 남긴다.
PIPELINE_MODE_DEFAULT = os.getenv("PIPELINE_MODE", "agent").lower()


def _resolve_eval_blob(blob: str, pipeline_mode: str) -> str:
    """NB01-04(v3) stage stem 정합 — S2/S4 eval jsonl 자동 매핑.
    사용자가 --eval-jsonl-gcs로 mined 경로만 준 상태에서 forward 모드를 켜면
    같은 디렉토리의 forward stem으로 치환한다 (반대 방향도 동일)."""
    if pipeline_mode == "forward" and "training_data_mined_eval" in blob:
        blob = blob.replace("training_data_mined_eval", "training_data_forward_eval")
    elif pipeline_mode == "agent" and "training_data_forward_eval" in blob:
        blob = blob.replace("training_data_forward_eval", "training_data_mined_eval")
    return blob


# ============================================================
# /detect 호출 — round 0 판정까지 저장
# ============================================================

async def call_agent_detect(
    client: httpx.AsyncClient,
    test_case: Dict,
    timeout: int,
) -> Optional[Dict]:
    """
    orchestrator /detect 호출.
    응답에 _round0_classification 추가:
      analyzer_history가 있으면 round 0의 cls를 저장.
      없으면 최종 classification으로 fallback.
    """
    try:
        t = time.perf_counter()
        resp = await client.post(
            f"{ORCHESTRATOR_URL}/detect", json=test_case, timeout=timeout
        )
        resp.raise_for_status()
        result = resp.json()
        result["_client_latency_ms"] = (time.perf_counter() - t) * 1000

        # ── round 0 판정 추출 ────────────────────────────────
        # ★ 수정: controller round_log(agent_rounds)가 이제 응답에 실려
        # 오므로 round 0의 실제 cls를 사용한다. 기존 fallback(최종 cls를
        # r0으로 저장)은 verdict_change_rate가 항상 0으로 나오는 원인이었음.
        rounds = result.get("agent_rounds") or []
        r0 = (rounds[0].get("cls") if rounds else None) \
             or result.get("classification")   # rounds 없을 때만 fallback
        result["_round0_classification"] = r0

        # ★ [NB v3 — 잔여 실험 B] controller finalize가 round_log에 남기는
        #   {"round": "final", nd_round0, nd_final} 엔트리 소비 —
        #   reroute/expand가 nd를 실제로 개선하는지(min-fusion)의 유일한 실측 경로.
        _fin = next((r for r in rounds if r.get("round") == "final"), None)
        result["_nd_round0"] = (_fin or {}).get("nd_round0")
        result["_nd_final"]  = (_fin or {}).get("nd_final")
        result["_nd_n_rounds"] = (_fin or {}).get("n_rounds")   # 1이면 재검색 없음

        # ── ★ [batch_inference V5.1 포팅] router-retriever 상세 지표 소스 ──
        #   top_5_clusters(순위 보존)에서 두 dict를 만든다:
        #   cluster_distances      — 하위호환 키 (score fallback 포함, MRR fallback용)
        #   cluster_centroid_dists — 거리 계열만 (ε-recall 전용; score는 유사도라 제외)
        _top5 = result.get("top_5_clusters") or []
        _cds, _cds_dist = {}, {}
        for _c in _top5:
            _cid = _c.get("cluster_id")
            if _cid is None:
                continue
            _d = _c.get("centroid_distance", _c.get("distance"))
            if _d is not None:
                _cds_dist[int(_cid)] = float(_d)
            _v = _d if _d is not None else _c.get("score")
            if _v is not None:
                _cds[int(_cid)] = float(_v)
        result["top_k_clusters"] = _top5             # MRR/candidate recall 입력 호환
        result["cluster_distances"] = _cds
        if _cds_dist:
            result["cluster_centroid_dists"] = _cds_dist

        # ── distance band 분류 ──────────────────────────────
        dist = float(result.get("top_1_distance") or 0)
        if dist < BOUNDARY_LOW:
            band = "normal_clear"
        elif dist < BOUNDARY_HIGH:
            band = "boundary"
        else:
            band = "abnormal_clear"
        result["_distance_band"] = band

        return result

    except httpx.TimeoutException:
        logger.warning("agent_detect_timeout")
        return None
    except Exception as e:
        logger.error("agent_detect_failed", error=str(e))
        return None


# ============================================================
# eval_set 모드 — NB01 teacher 생성 eval jsonl 기반
# ============================================================

EVAL_JSONL_GCS = os.getenv(
    "EVAL_JSONL_GCS",
    "tree-search/analyzer_ft/pca_64_k100/hf_0713/training_data_mined_eval.jsonl",
)
GCS_BUCKET = os.getenv("GCS_BUCKET_NAME", "fraud_ecomm")


def _download_eval_jsonl(gcs_blob_path: str, local_dir: str = "/tmp/eval_set") -> str:
    """NB01이 GCS에 저장한 eval jsonl을 컨테이너 로컬로 다운로드.
    gt_loader와 동일한 google.cloud.storage 스택 사용 (evaluation-sa 권한 재사용)."""
    from google.cloud import storage
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    local_path = str(Path(local_dir) / Path(gcs_blob_path).name)
    storage.Client().bucket(GCS_BUCKET).blob(gcs_blob_path) \
        .download_to_filename(local_path)
    print(f"  eval_jsonl     : gs://{GCS_BUCKET}/{gcs_blob_path} → {local_path}")
    return local_path


def load_eval_set_indices(args, gt_data, n_total: int):
    """
    eval jsonl 레코드 → test idx 매핑 + n_samples 층화 서브샘플.

    매핑 전략 (2단):
      1) metadata["test_idx"] 존재 시 그대로 사용 (NB01 1줄 추가분 — 신규 생성분)
      2) fallback: top1_orig_index 역매핑.
         gt_loader의 top5_indices와 NB01의 gt_knn_indices_576d는 동일 컬럼이므로
         top5_indices[idx][0] == metadata.top1_orig_index 매칭이 유효.
         후보 복수 시 gt_label 일치 → |top1_l2_distance − top5_dist[idx][0]| 최소로 tie-break.
         탐색 범위는 eval 구간(테스트셋 끝 EVAL_RATIO)으로 제한.

    Returns: (indices, eval_meta)  — eval_meta[idx] = NB 메타(teacher 응답/nd/top_k)
    """
    local = _download_eval_jsonl(
        _resolve_eval_blob(args.eval_jsonl_gcs, args.pipeline_mode))
    records = [json.loads(l) for l in open(local, encoding="utf-8") if l.strip()]
    print(f"  eval records   : {len(records)}")

    labels    = gt_data["label"]
    top5_idx  = np.asarray(gt_data["gt_top5_indices"])          # (N,5) original_index
    top5_dist = np.asarray(gt_data.get("gt_top5_dist"))          # (N,5) 576d 거리
    eval_start = int(n_total * (1 - args.eval_ratio))         # 끝 EVAL_RATIO 구간

    # top1_orig_index → 후보 test idx 사전 (eval 구간만)
    cand_by_oi: Dict[int, List[int]] = {}
    for i in range(eval_start, n_total):
        cand_by_oi.setdefault(int(top5_idx[i][0]), []).append(i)

    indices, eval_meta = [], {}
    n_direct = n_fallback = n_miss = 0
    for r in records:
        m = r.get("metadata", {})
        idx = m.get("test_idx")
        if idx is not None:                                    # 1) 직접 매핑
            idx = int(idx); n_direct += 1
        else:                                                  # 2) 역매핑 fallback
            cands = cand_by_oi.get(int(m.get("top1_orig_index", -1)), [])
            cands = [c for c in cands if int(labels[c]) == int(m.get("gt_label", -9))]
            if not cands:
                n_miss += 1
                continue
            if len(cands) > 1 and top5_dist is not None and m.get("top1_l2_distance") is not None:
                cands = sorted(cands, key=lambda c:
                               abs(float(top5_dist[c][0]) - float(m["top1_l2_distance"])))
            idx = cands[0]; n_fallback += 1
        if idx in eval_meta:                                   # 중복 매핑 방지
            continue
        indices.append(idx)
        nb_nd = m.get("top1_l2_dist_norm")
        eval_meta[idx] = {
            "nb_nd":   float(nb_nd) if nb_nd is not None else None,
            "top_k":   m.get("top_k"),
            "teacher": r.get("assistant"),                     # teacher 응답 (NB03 참조용)
            "gt_cluster_id": m.get("gt_cluster_id"),
        }
    print(f"  idx 매핑       : direct={n_direct} fallback={n_fallback} miss={n_miss}")
    if n_miss:
        print(f"  ⚠ {n_miss}건 매핑 실패 — NB01 metadata에 test_idx 추가 후 재생성 권장")

    # n_samples 층화 서브샘플 (eval set의 50:50 유지)
    if args.n_samples and args.n_samples < len(indices):
        rng = np.random.default_rng(args.sample_seed)
        pos = [i for i in indices if labels[i] == 1]
        neg = [i for i in indices if labels[i] == 0]
        n_pos = min(len(pos), args.n_samples // 2)
        n_neg = min(len(neg), args.n_samples - n_pos)
        indices = sorted(rng.choice(pos, n_pos, replace=False).tolist()
                         + rng.choice(neg, n_neg, replace=False).tolist())
        print(f"  층화 샘플      : {len(indices)} (pos={n_pos}, neg={n_neg}, seed={args.sample_seed})")
    return indices, eval_meta


def _nd_category(nd: Optional[float]) -> str:
    """NB nd 기준 카테고리 파생 (NB03 카테고리 히트맵 호환)."""
    if nd is None:
        return "uncat"
    if BOUNDARY_LOW <= nd < BOUNDARY_HIGH:
        return "boundary"
    return "easy_low" if nd < BOUNDARY_LOW else "easy_high"


# ============================================================
# 메인 배치 루프
# ============================================================

async def run_batch(args) -> List[Dict]:
    loader = GTLoader(
        experiment_case=args.experiment_case,
        coreset_percentage=str(args.coreset_percentage),
    )
    _t_wall0 = time.perf_counter()          # ★ 총 실험 시간 (forwardRAG 비교용)
    test_data, test_emb, gt_data = loader.load_all()
    gt_labels   = gt_data["label"]        # np.ndarray, 0=정상/1=이상
    # ★ [E1 수정] gt_loader의 실제 키는 "gt_cluster_id" (구 batch_inference L627과 동일)
    #   — "gt_cluster_tree" 단독 조회로 None → _gt_cluster 미부착 + routing acc/MRR/ε 전멸
    gt_clusters = gt_data.get("gt_cluster_tree", gt_data.get("gt_cluster_id"))
    n_total     = len(test_data)

    # ── 샘플 선택: smoke | eval_set ──────────────────────────
    eval_meta: Dict[int, Dict] = {}
    if args.eval_source == "eval_set":
        # NB01 teacher 생성 eval jsonl (테스트셋 끝 10%, 50:50) 기반
        indices, eval_meta = load_eval_set_indices(args, gt_data, n_total)
    else:
        # smoke: 범위 or 랜덤 N_SAMPLE
        if args.sample_random:
            rng = np.random.default_rng(args.sample_seed)
            indices = sorted(rng.choice(
                n_total, min(args.n_samples, n_total), replace=False).tolist())
        else:
            start = max(0, args.start_index)
            end   = min(start + args.n_samples, n_total)
            indices = list(range(start, end))

    # --boundary-only: boundary zone 샘플만 필터 (GT label 기준이 아닌 거리 기준)
    # 실제 거리 정보는 추론 전에는 없으므로, gt_data의 top5_dist[i][0]로 사전 필터
    if args.boundary_only:
        gt_dists = gt_data.get("top5_dist")  # (N, 5) 거리
        if gt_dists is not None:
            n_before = len(indices)
            indices = [
                i for i in indices
                if BOUNDARY_LOW <= float(gt_dists[i][0]) < BOUNDARY_HIGH
            ]
            logger.info("boundary_only_filter",
                        n_before=n_before, n_after=len(indices))

    print("=" * 70)
    print("AGENT LOOP EVALUATION INFERENCE")
    print("=" * 70)
    print(f"  orchestrator    : {ORCHESTRATOR_URL}")
    print(f"  experiment_case : {args.experiment_case}")
    print(f"  eval_source     : {args.eval_source}")
    print(f"  pipeline_mode   : {args.pipeline_mode}  (서버 PIPELINE_MODE와 일치해야 함)")
    print(f"  n_samples       : {len(indices)} / {n_total}")
    print(f"  boundary_only   : {args.boundary_only}")

    predictions: List[Dict] = []
    n_err = 0

    async with httpx.AsyncClient() as client:
        for i, idx in enumerate(indices):
            tc = loader.get_test_case(
                idx, test_data, test_emb,
                percolate_version=args.percolate_version,
                experiment_case=args.experiment_case,
            )
            result = await call_agent_detect(client, tc, args.timeout)
            if result is None:
                n_err += 1
                predictions.append({
                    "idx": idx,
                    "error": True,
                    "_gt_label": int(gt_labels[idx]),
                })
                continue

            result["idx"]       = idx
            result["_gt_label"] = int(gt_labels[idx])
            # ── GT cluster 연결 (gt_loader.gt_cluster_tree) → routing 정확도 측정 ──
            if gt_clusters is not None:
                result["_gt_cluster"] = int(gt_clusters[idx])
            # ── eval_set 모드: NB 메타 부착 (nd 카테고리 / teacher 참조 / NB nd) ──
            if idx in eval_meta:
                em = eval_meta[idx]
                result["_category"] = _nd_category(em["nb_nd"])
                result["_nb_nd"]    = em["nb_nd"]      # 오프라인(576d GT-kNN) nd
                result["_teacher"]  = em["teacher"]    # teacher 응답 (NB03 Section B 참조)
            else:
                result["_category"] = "smoke"
            predictions.append(result)

            # 진행 상황 출력 (10건마다)
            if (i + 1) % 10 == 0 or i == 0:
                ok_n = sum(1 for p in predictions if not p.get("error"))
                logger.info("batch_progress",
                            done=i + 1, total=len(indices),
                            ok=ok_n, err=n_err)

    print(f"\n[batch] DONE  n={len(predictions)}  errors={n_err}")
    # ── 서버 mode 불일치 감지: forward 모드 요청인데 agent 응답(또는 반대) ──
    _n_fwd = sum(1 for p in predictions
                 if not p.get("error") and p.get("termination") == "forward")
    _n_ok  = sum(1 for p in predictions if not p.get("error"))
    if _n_ok:
        _fwd_ratio = _n_fwd / _n_ok
        if args.pipeline_mode == "forward" and _fwd_ratio < 0.5:
            print(f"⚠ mode 불일치 의심: --pipeline-mode=forward 인데 forward 종결 "
                  f"{_fwd_ratio:.0%} — orchestrator PIPELINE_MODE 확인")
        if args.pipeline_mode == "agent" and _fwd_ratio > 0.5:
            print(f"⚠ mode 불일치 의심: --pipeline-mode=agent 인데 forward 종결 "
                  f"{_fwd_ratio:.0%} — orchestrator PIPELINE_MODE 확인")

    wall = {"total_s": round(time.perf_counter() - _t_wall0, 1),
            "n": len(predictions),
            "per_sample_s": round((time.perf_counter() - _t_wall0) / max(len(predictions), 1), 2),
            "pipeline_mode": args.pipeline_mode}

    # ── router-retriever 상세 지표용 GT 배열 (indices 순서로 gather) ──
    gt_arrays = {}
    # ★ [E1 수정] 키 alias — gt_cluster_tree가 없으면 gt_cluster_id로 조회
    for _k, _aliases in (("gt_cluster_tree", ("gt_cluster_tree", "gt_cluster_id")),
                         ("gt_top5_cluster_ids",  ("gt_top5_cluster_ids",)),
                         ("gt_top5_cluster_dist", ("gt_top5_cluster_dist",)),
                         ("gt_top5_dist",         ("gt_top5_dist",))):
        _a = next((gt_data.get(_al) for _al in _aliases if gt_data.get(_al) is not None), None)
        if _a is not None:
            gt_arrays[_k] = np.asarray(_a)[indices]

    return predictions, gt_labels, indices, wall, gt_arrays

def calculate_path_conditioned_metrics(predictions: List[Dict], gt_sub: np.ndarray) -> Dict:
    """agent_trace 기반 경로별(no-loop/expand/reroute) AUROC·FNR breakdown.
    agent_eval_metrics.AgentMetricCalculator를 건드리지 않고 여기서 독립 계산."""
    from collections import defaultdict
    groups = defaultdict(list)
    for p, gt in zip(predictions, gt_sub):
        if p.get("error"):
            continue
        trace = p.get("agent_trace", [])
        tool = ("expand" if any(h.startswith("expand_within_cluster") for h in trace)
                else "reroute" if any(h.startswith("reroute_cross_cluster") for h in trace)
                else "none")
        pred = 1 if p.get("classification") == "ABNORMAL" else 0
        groups[tool].append((pred, int(gt)))

    out = {}
    for tool, pairs in groups.items():
        preds, gts = zip(*pairs)
        preds, gts = np.array(preds), np.array(gts)
        fn = ((preds == 0) & (gts == 1)).sum()
        pos = (gts == 1).sum()
        out[tool] = {"n": len(pairs), "fnr": float(fn / pos) if pos else None,
                     "accuracy": float((preds == gts).mean())}
    return out

# ============================================================
# Evaluation 실행 + 저장
# ============================================================

def run_evaluation(
    predictions: List[Dict],
    gt_labels:   np.ndarray,
    indices:     List[int],
    out_dir:     Path,
    experiment_case: str,
    pipeline_mode: str = "agent",
    gt_arrays: Optional[Dict] = None,      # ★ router-retriever 상세 지표용 GT 배열
    wall: Optional[Dict] = None,           # ★ 총 실험 시간 (run_batch가 측정)
) -> Dict:
    """
    배치 추론 결과를 받아 전체 평가 지표를 계산하고 저장.
    pipeline_mode="forward"면 agent 전용 지표(verdict change/action/nd-fusion/
    path-conditioned)는 건너뛴다 — controller/orchestrator의 mode 분기와 동일 원칙.
    """
    calc = AgentMetricCalculator()
    calc.set_experiment_case(experiment_case)

    # ── 인덱스 정렬 (indices 기준으로 gt 슬라이싱) ──────────
    gt_sub = gt_labels[indices]

    # ── 1. Distance-based AUROC (기존 MetricCalculator 재사용) ──
    dist_auroc = calc.calculate_distance_based_auroc(predictions, gt_sub)

    # ── 2. Distance Best Threshold 산출 ─────────────────────
    dist_scores = np.array([
        float(p.get("top_1_distance") or p.get("top1_distance") or 0)
        for p in predictions
    ])
    valid_mask = np.array([not p.get("error") for p in predictions])
    threshold_info = find_best_threshold(gt_sub, dist_scores, valid_mask)
    best_thr = threshold_info.get("best_threshold", 0.5)

    # ── 3. 동일 threshold에서 세 방법 비교 ─────────────────
    #   (a) Distance-only
    dist_at_thr = calc.calculate_classification_at_threshold(
        gt_sub, dist_scores, valid_mask, threshold=best_thr,
        method_name="distance_only",
    )

    #   (b) LLM / Agent RAG (binary 0/1)
    llm_scores, llm_valid = build_llm_scores(predictions)
    llm_at_thr = calc.calculate_classification_at_threshold(
        gt_sub, llm_scores, llm_valid, threshold=50.0,  # binary 기준점 50
        method_name="forward_rag" if pipeline_mode == "forward" else "agent_rag",
    )

    # ── 4. Anomaly AUROC (LLM score 기반) ───────────────────
    pred_labels = np.array([
        1 if p.get("classification") == "ABNORMAL" else 0
        for p in predictions
    ])
    anomaly_auroc = calc.calculate_anomaly_auroc(
        gt_sub[llm_valid], llm_scores[llm_valid], pred_labels[llm_valid]
    )

    # ── 5. Boundary Zone 분리 측정 ───────────────────────────
    boundary = calc.calculate_boundary_metrics(
        predictions, gt_sub, threshold=best_thr
    )

    # ── 6~7. Agent 전용 지표 — forward 모드는 루프가 없으므로 skip ──
    if pipeline_mode == "agent":
        vcr = calc.calculate_verdict_change_rate(predictions, gt_sub)
        action_dist = calc.calculate_action_distribution(predictions)
    else:
        vcr, action_dist = None, None

    # ── 8. Latency Breakdown (mode 공용 — forward_analyzer 키 자동 수집) ──
    latency = calc.calculate_agent_latency_breakdown(predictions)

    # ── 9. ★ Router-Retriever 상세 지표 (batch_inference V5 포팅, mode 공용) ──
    #   routing acc / MRR / candidate recall / routing·coreset ε-approximate recall
    rr_metrics: Dict = {}
    if gt_arrays:
        _gt_c = gt_arrays.get("gt_cluster_tree")
        if _gt_c is not None:
            _pred_c = np.array([
                int(p.get("primary_cluster_id", -1)) if not p.get("error") else -1
                for p in predictions])
            rr_metrics["routing_accuracy"] = calc.calculate_cluster_assignment_accuracy(
                _gt_c, _pred_c)
            rr_metrics["router_mrr"] = calc.calculate_router_mrr(predictions, _gt_c)
        if gt_arrays.get("gt_top5_cluster_ids") is not None:
            rr_metrics["candidate_recall"] = calc.calculate_candidate_recall(
                predictions, gt_arrays["gt_top5_cluster_ids"])
        if _gt_c is not None and gt_arrays.get("gt_top5_cluster_dist") is not None:
            rr_metrics["routing_epsilon_recall"] = calc.calculate_routing_epsilon_recall(
                predictions, _gt_c, gt_arrays["gt_top5_cluster_dist"], epsilon=0.1,
                gt_top5_cluster_ids=gt_arrays.get("gt_top5_cluster_ids"))  # ★ [E2] GT 테이블 lookup
        if gt_arrays.get("gt_top5_dist") is not None:
            rr_metrics["coreset_epsilon_recall"] = calc.calculate_coreset_epsilon_recall(
                predictions, gt_arrays["gt_top5_dist"], epsilon=0.1)

    # ── 10. ★ [재설계 요청 7] Agent nd 이중 리포트 ──────────────
    #   (1) agent_nd_auroc : agent loop가 최종 판정에 쓴 nd(재검색 시 patchcore,
    #       아니면 raw)의 연속 score AUROC — threshold 무관, 순위 품질.
    #   (2) cls_at_nd_threshold : 그 nd를 CLS_ND_THRESHOLD로 이분한 실제 판정의
    #       accuracy/FNR/FPR (best_threshold가 아니라 서빙과 동일한 고정 임계).
    #   추가로 재검색군(patchcore) vs 비재검색군(raw) 분리 AUROC — 두 척도가
    #   한 임계를 공유해도 되는지(미스캘리브레이션 여부)를 직접 진단.
    import os as _os
    CLS_ND_THRESHOLD = float(_os.getenv("CLS_ND_THRESHOLD", "0.5"))
    nd_dual: Dict = {}
    if pipeline_mode == "agent":
        def _final_entry(p):
            for r in (p.get("agent_rounds") or []):
                if r.get("round") == "final":
                    return r
            return {}
        _nd, _y, _researched = [], [], []
        for p, lbl in zip(predictions, gt_sub):
            if p.get("error"):
                continue
            fe = _final_entry(p)
            v = fe.get("nd_final")
            if v is None:
                continue
            _nd.append(float(v)); _y.append(int(lbl))
            _researched.append(bool(fe.get("researched")))
        _nd = np.array(_nd); _y = np.array(_y); _researched = np.array(_researched)

        def _auroc(scores, labels):
            if len(labels) < 3 or len(set(labels.tolist())) < 2:
                return None
            try:
                from sklearn.metrics import roc_auc_score
                return round(float(roc_auc_score(labels, scores)), 4)
            except Exception:
                # rank 기반 fallback (sklearn 부재 시)
                order = scores.argsort()
                ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(scores) + 1)
                n_pos = labels.sum(); n_neg = len(labels) - n_pos
                if n_pos == 0 or n_neg == 0:
                    return None
                auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
                return round(float(auc), 4)

        def _cls_at(scores, labels, thr):
            if len(labels) == 0:
                return None
            pred = (scores >= thr).astype(int)
            tp = int(((pred == 1) & (labels == 1)).sum()); fn = int(((pred == 0) & (labels == 1)).sum())
            fp = int(((pred == 1) & (labels == 0)).sum()); tn = int(((pred == 0) & (labels == 0)).sum())
            n = len(labels)
            return {
                "threshold": thr, "n": n,
                "accuracy": round((tp + tn) / max(n, 1), 4),
                "fnr": round(fn / max(tp + fn, 1), 4),
                "fpr": round(fp / max(fp + tn, 1), 4),
                "recall": round(tp / max(tp + fn, 1), 4),
                "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            }

        nd_dual = {
            "n": int(len(_nd)),
            "n_researched": int(_researched.sum()),
            "n_raw": int((~_researched).sum()),
            "cls_nd_threshold": CLS_ND_THRESHOLD,
            # (1) 연속 nd AUROC (threshold 무관)
            "agent_nd_auroc_all": _auroc(_nd, _y),
            # (2) 고정 임계(CLS_ND_THRESHOLD) 판정 지표 = 서빙 실현값
            "cls_at_nd_threshold": _cls_at(_nd, _y, CLS_ND_THRESHOLD),
            # (3) 척도 분리 진단 — 두 분포가 한 임계를 공유해도 되는가
            "researched_nd_auroc": _auroc(_nd[_researched], _y[_researched]) if _researched.any() else None,
            "raw_nd_auroc":        _auroc(_nd[~_researched], _y[~_researched]) if (~_researched).any() else None,
            "researched_mean_nd":  round(float(_nd[_researched].mean()), 4) if _researched.any() else None,
            "raw_mean_nd":         round(float(_nd[~_researched].mean()), 4) if (~_researched).any() else None,
        }

    # ── 결과 취합 ────────────────────────────────────────────
    report = {
        "experiment_case":    experiment_case,
        "pipeline_mode":      pipeline_mode,
        "wall_time":          wall,                    # ★ 총/건당 실험 시간
        "n_total":            len(predictions),
        "n_error":            sum(1 for p in predictions if p.get("error")),

        "threshold_info":     threshold_info,   # best threshold + ROC 곡선
        "dist_auroc":         dist_auroc,
        "anomaly_auroc":      anomaly_auroc,
        "agent_nd_dual":      nd_dual,               # ★ [요청 7] nd 이중 리포트
        "distance_at_threshold": dist_at_thr,   # 비교 표용
        "agent_at_threshold": llm_at_thr,       # 비교 표용
        "boundary":           boundary,
        "verdict_change":     vcr,
        "action_distribution": action_dist,
        "latency_breakdown":  latency,
        "router_retriever":   rr_metrics,              # ★ 상세 검색 계층 지표
        "path_conditioned":   (calculate_path_conditioned_metrics(predictions, gt_sub)
                               if pipeline_mode == "agent" else None),
        "agent_summary":      calc.get_agent_summary(),
    }

    # ── 저장 ────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "agent_predictions.json").write_text(
        json.dumps(predictions, indent=2, default=str)
    )
    (out_dir / "agent_eval_report.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    logger.info("evaluation_saved", out_dir=str(out_dir))
    return report


# ============================================================
# Entry Point
# ============================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Agent Loop Evaluation Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiment-case",     type=str, default="pca_64_k100",
                        choices=VALID_EXPERIMENT_CASES)
    parser.add_argument("--percolate-version",   type=str, default="v14")
    parser.add_argument("--coreset-percentage",  type=int, default=100)
    parser.add_argument("--n-samples",           type=int, default=200)
    parser.add_argument("--start-index",         type=int, default=0)
    parser.add_argument("--timeout",             type=int, default=300)
    parser.add_argument("--output-dir",          type=str,
                        default=f"./results/agent_eval/{os.getenv('RUN_NAME', 'local')}")
    parser.add_argument("--boundary-only",       action="store_true",
                        help="boundary zone 샘플(0.3~0.55)만 대상으로 실행")
    parser.add_argument("--pipeline-mode",       type=str, default=PIPELINE_MODE_DEFAULT,
                        choices=["agent", "forward"],
                        help="agent=S4 evaluator 루프 / forward=S2 forwardRAG baseline "
                             "(서버 orchestrator PIPELINE_MODE와 일치 필요)")
    parser.add_argument("--eval-source",         type=str, default="smoke",
                        choices=["smoke", "eval_set"],
                        help="smoke: 테스트셋 범위/랜덤 N_SAMPLE | "
                             "eval_set: NB01 teacher 생성 eval jsonl(GCS) 기반")
    parser.add_argument("--eval-jsonl-gcs",      type=str, default=EVAL_JSONL_GCS,
                        help="eval_set 모드에서 다운로드할 GCS blob 경로 "
                             "(bucket 제외, env EVAL_JSONL_GCS)")
    parser.add_argument("--eval-ratio",          type=float,
                        default=float(os.getenv("EVAL_RATIO", "0.1")),
                        help="eval 구간 비율 — NB01 EVAL_RATIO와 동일해야 역매핑 유효")
    parser.add_argument("--sample-random",       action="store_true",
                        help="smoke 모드에서 범위 대신 랜덤 N_SAMPLE 추출")
    parser.add_argument("--sample-seed",         type=int, default=42)
    parser.add_argument("--report-only",         type=str, default=None,
                        metavar="PRED_JSON",
                        help="기존 agent_predictions.json으로 평가만 재실행")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    # -- 기존 predictions JSON으로 평가만 재실행 --
    if args.report_only:
        preds = json.loads(Path(args.report_only).read_text())
        loader = GTLoader(experiment_case=args.experiment_case,
                          coreset_percentage=str(args.coreset_percentage))
        _, _, gt_data = loader.load_all()
        indices = [p["idx"] for p in preds if not p.get("error")]
        report  = run_evaluation(preds, gt_data["label"], indices,
                                 out_dir, args.experiment_case,
                                 pipeline_mode=args.pipeline_mode)
        print(json.dumps(report["agent_summary"], indent=2))
        return

    # -- 배치 추론 + 평가 --
    predictions, gt_labels, indices, wall, gt_arrays = await run_batch(args)
    report = run_evaluation(
        predictions, gt_labels, indices, out_dir, args.experiment_case,
        pipeline_mode=args.pipeline_mode, gt_arrays=gt_arrays, wall=wall,
    )

    # 요약 출력
    s = report["agent_summary"]
    _mode_label = "forwardRAG(S2)" if args.pipeline_mode == "forward" else "AgentRAG(S4)"
    print("\n" + "=" * 60)
    print(f"[EVALUATION SUMMARY — {_mode_label}]")
    print(f"  distance AUROC       : {report['dist_auroc'].get('distance_auroc')}")
    print(f"  {_mode_label:20s} AUROC : {report['anomaly_auroc'].get('auroc')}")
    print(f"  best_threshold       : {report['threshold_info'].get('best_threshold')}")
    print(f"  distance FNR@thr     : {report['distance_at_threshold'].get('fnr')}")
    print(f"  model FNR@thr        : {report['agent_at_threshold'].get('fnr')}")
    print(f"  boundary FNR         : {s.get('boundary_fnr')}")

    # ── ★ 총 실험 시간 + 요소별 시간 (forwardRAG ↔ agent 비교의 공통 축) ──
    print(f"\n  [시간] total={wall['total_s']}s  n={wall['n']}  "
          f"per-sample={wall['per_sample_s']}s")
    _lb = report.get("latency_breakdown", {})
    for _k, _lab in (("base_retrieve_ms", "router+retriever(base)"),
                     ("forward_analyzer_ms", "forward analyzer"),
                     ("controller_loop_ms", "agent loop(controller)"),
                     ("tool_delta_ms", "analyzer+planner+tool Δ"),
                     ("total_ms", "end-to-end")):
        _st = _lb.get(_k)
        if isinstance(_st, dict) and _st.get("mean") is not None:
            print(f"    {_lab:26s}: mean {_st['mean']}ms  p95 {_st['p95']}ms")

    # ── ★ Router-Retriever 상세 (batch_inference V5 포팅 — mode 공용) ──
    _rr = report.get("router_retriever") or {}
    if _rr:
        print("\n  [router-retriever]")
        _ra = (_rr.get("routing_accuracy") or {})
        print(f"    routing acc          : {_ra.get('accuracy', _ra.get('cluster_assignment_accuracy'))}")
        print(f"    router MRR           : {(_rr.get('router_mrr') or {}).get('router_mrr')}")
        print(f"    candidate recall@5   : {(_rr.get('candidate_recall') or {}).get('candidate_recall_top5')}")
        print(f"    routing ε-recall(10%): "
              f"{(_rr.get('routing_epsilon_recall') or {}).get('routing_epsilon_recall_10pct')}")
        print(f"    coreset ε-recall(10%): "
              f"{(_rr.get('coreset_epsilon_recall') or {}).get('coreset_epsilon_recall_10pct')}")

    # ── ★ Agent 전용 지표 (forward 모드는 skip — 루프 없음) ──
    if args.pipeline_mode == "agent":
        print(f"\n  verdict change rate  : {s.get('verdict_change_rate')}")
        _ad = report.get("action_distribution") or {}
        print(f"  expand / reroute rate: {_ad.get('expand_rate')} / {_ad.get('reroute_rate')}"
              f"  (deep n={_ad.get('deep_track_n')})")
        print(f"  wm direction         : cm {_ad.get('cm_n')} ({_ad.get('cm_ratio')})  "
              f"ni {_ad.get('ni_n')} ({_ad.get('ni_ratio')})  "
              f"none {_ad.get('wm_none_n')}  [ambiguous rounds={_ad.get('wm_rounds_n')}]")
    # ── ★ [요청 7] Agent nd 이중 리포트 ──────────────────────
    if args.pipeline_mode == "agent":
        _nd = report.get("agent_nd_dual") or {}
        _cls = _nd.get("cls_at_nd_threshold") or {}
        print(f"\n  [agent nd 이중 리포트] n={_nd.get('n')}  "
              f"(재검색 {_nd.get('n_researched')} / raw {_nd.get('n_raw')})")
        print(f"    (1) agent nd AUROC (연속, threshold 무관) : {_nd.get('agent_nd_auroc_all')}")
        print(f"    (2) cls@CLS_ND_THRESHOLD={_nd.get('cls_nd_threshold')} → "
              f"acc {_cls.get('accuracy')} / FNR {_cls.get('fnr')} / FPR {_cls.get('fpr')}")
        print(f"    (3) 척도분리 AUROC — 재검색군(patchcore) {_nd.get('researched_nd_auroc')} "
              f"(mean nd {_nd.get('researched_mean_nd')}) | "
              f"raw군 {_nd.get('raw_nd_auroc')} (mean nd {_nd.get('raw_mean_nd')})")
        print(f"       ↑ 두 AUROC/mean 차가 크면 한 임계 공유가 미스캘리브레이션 신호")

        # ── ★ [요청 5] round0/round1 재검색 시간 ──
        _r0 = [p.get("latency_ms", {}).get("base_retrieve") for p in predictions
               if not p.get("error") and isinstance(p.get("latency_ms"), dict)
               and p["latency_ms"].get("base_retrieve") is not None]
        _r1 = []
        for p in predictions:
            for r in (p.get("agent_rounds") or []):
                if isinstance(r.get("round"), int) and r.get("research_ms") is not None:
                    _r1.append(float(r["research_ms"]))
        import statistics as _st2
        if _r0:
            print(f"    round0 retrieve      : mean {_st2.mean(_r0):.1f}ms  n={len(_r0)}")
        if _r1:
            print(f"    round1 재검색(tool)   : mean {_st2.mean(_r1):.1f}ms  n={len(_r1)}")
        else:
            print(f"    round1 재검색(tool)   : (재검색 발동 0건 또는 research_ms 미기록)")
    # routing 정확도: 서빙 primary_cluster_id vs GT cluster (gt_cluster_tree)
    _r = [(p.get("primary_cluster_id"), p.get("_gt_cluster"))
          for p in predictions if not p.get("error")
          and p.get("_gt_cluster") is not None and p.get("_gt_cluster") >= 0]
    if _r:
        _acc = sum(int(a == b) for a, b in _r) / len(_r)
        print(f"  routing accuracy     : {_acc:.4f}  (n={len(_r)})")
    print(f"  expand rate          : {s.get('expand_rate')}")
    print(f"  latency P95 (total)  : {s.get('latency_total_p95_ms')} ms")
    print(f"  saved → {out_dir}")
    print("=" * 60)

    # ── NB03 소비용 GCS export ─────────────────────────────
    # judge/agent-system 평가는 Colab NB03에서 수행하므로(GKE는 teacher
    # 리소스 제약), predictions/report를 버킷으로 올려 NB03이 bucket
    # 마운트로 읽게 한다.
    # if args.gcs_upload:
    #     import subprocess
    #     for fn in ("agent_predictions.json", "agent_eval_report.json"):
    #         fp = out_dir / fn
    #         if fp.exists():
    #             dst = args.gcs_upload.rstrip("/") + "/" + fn
    #             r = subprocess.run(["gsutil", "cp", str(fp), dst],
    #                                capture_output=True, text=True)
    #             print(f"  gcs → {dst}" if r.returncode == 0
    #                   else f"  ⚠ gcs upload 실패({fn}): {r.stderr.strip()[:120]}")


if __name__ == "__main__":
    asyncio.run(main())
