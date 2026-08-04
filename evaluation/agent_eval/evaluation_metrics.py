"""
Evaluation Metrics - V5.0
==========================

V4.1 → V5.0 변경사항:

  ★ calculate_retrieval_recall_dual(): original_index 기반 recall@5
    - 기존: predictions["top_k_results"][*]["id"] 비교
    - 변경: predictions["retrieved_original_indices"] 직접 비교
            (batch_inference.py가 SearchResult.original_index 추출해 저장)
    - GT(gt_top5_raw/adj)도 original_index 공간 (gt_loader v5)

  ★ calculate_router_metrics(): purity/support/leaf_id 제거
    - top_k_clusters: [{cluster_id, rank, score}]만 참조

  ★ calculate_distance_based_auroc():
    - top1_distance / top_1_distance 양쪽 필드명 지원

  ★ calculate_latency_breakdown():
    - SKIP_ANALYZER=true 시 analyzer_lat 빈 배열 처리
"""

import numpy as np
from typing import Dict, List, Optional
import structlog

try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = structlog.get_logger()


class MetricCalculator:

    def __init__(self):
        self.experiment_case    = "unknown"
        self.coreset_percentage = "100"
        self._summary: Dict     = {}

    def set_experiment_case(self, v: str):
        self.experiment_case = v

    def set_coreset_percentage(self, v: str):
        self.coreset_percentage = v

    # ──────────────────────────────────────────────────────────
    # Cluster Assignment Accuracy
    # ──────────────────────────────────────────────────────────

    def calculate_cluster_assignment_accuracy(
        self, gt_clusters: np.ndarray, pred_clusters: np.ndarray
    ) -> Dict:
        valid = (gt_clusters >= 0) & (pred_clusters >= 0)
        n_valid = int(valid.sum())
        if n_valid == 0:
            return {"cluster_assignment_accuracy": None, "n_valid": 0}
        acc = float((gt_clusters[valid] == pred_clusters[valid]).mean())
        self._summary["cluster_accuracy"] = round(acc, 4)
        return {
            "cluster_assignment_accuracy": round(acc, 4),
            "n_valid":   n_valid,
            "n_correct": int((gt_clusters[valid] == pred_clusters[valid]).sum()),
        }

    # ──────────────────────────────────────────────────────────
    # Router Metrics (V5: purity/support/leaf_id 없음)
    # ──────────────────────────────────────────────────────────

    def calculate_router_metrics(
        self, predictions: List[Dict], gt_clusters: np.ndarray
    ) -> Dict:
        """
        ★ V5: top_k_clusters = [{cluster_id, rank, score}]
        """
        n = len(predictions)
        recall_hits = 0
        score_gaps  = []

        for i, pred in enumerate(predictions):
            if pred.get("error"):
                continue
            gt_c = int(gt_clusters[i]) if i < len(gt_clusters) else -1
            if gt_c < 0:
                continue
            # top_5_clusters (orchestrator) 또는 top_k_clusters (router)
            top_k = pred.get("top_k_clusters", pred.get("top_5_clusters", []))
            cids  = [c["cluster_id"] for c in top_k]
            if gt_c in cids:
                recall_hits += 1
            scores = [c.get("score", 0.0) for c in top_k]
            if len(scores) >= 2:
                score_gaps.append(scores[0] - scores[1])

        recall_at_5 = recall_hits / max(n, 1)
        self._summary["router_recall_at_5"] = round(recall_at_5, 4)
        return {
            "recall_at_5":    round(recall_at_5, 4),
            "n_evaluated":    n,
            "n_recall_hits":  recall_hits,
            "score_gap_mean": round(float(np.mean(score_gaps)), 4) if score_gaps else None,
        }

    # ──────────────────────────────────────────────────────────
    # Retrieval Recall@5 (V5: original_index 기반)
    # ──────────────────────────────────────────────────────────

    def calculate_retrieval_recall_dual(
        self,
        predictions:              List[Dict],
        gt_top5_raw:              np.ndarray,
        gt_top5_coreset_adjusted: np.ndarray,
        coreset_coverage_rate:    float = 1.0,
        n_coreset_train:          int   = 0,
    ) -> Dict:
        """
        Recall@5 이중 계산 (V5: original_index 기반).

        ★ 핵심:
          predictions[i]["retrieved_original_indices"]:
            batch_inference.py가 retriever SearchResult.original_index 추출해 저장.
          gt_top5_raw[i]:
            gt_loader v5가 kNN row_indices → original_index 변환한 GT.

          두 값이 동일 original_index 공간 → 직접 set intersection 비교.

          예:
            retrieved_original_indices[i] = [100, 200, 300, 400, 500]
            gt_top5_raw[i]                = [100, 150, 300, 450, 600]
            hit = {100, 300} ∩ gt ≠ ∅ → True

        fallback: "retrieved_original_indices" 없으면 "results"에서 original_index 추출.
        """
        hits_raw = []
        hits_adj = []
        n_empty_gt = 0

        for i, pred in enumerate(predictions):
            if pred.get("error") or i >= len(gt_top5_raw):
                continue

            # ★ V5: retrieved_original_indices 우선 사용
            retrieved = pred.get("retrieved_original_indices")
            if not retrieved:
                # fallback: results에서 직접 추출
                retrieved = [
                    r.get("original_index", -1)
                    for r in pred.get("results", pred.get("top_k_results", []))
                ]
            retrieved_set = set(retrieved) - {-1}

            # ── Raw GT ──────────────────────────────────────────
            gt_raw_set = set(gt_top5_raw[i].tolist()) - {-1}
            if gt_raw_set:
                hits_raw.append(int(bool(retrieved_set & gt_raw_set)))

            # ── Coreset-Adjusted GT ─────────────────────────────
            gt_adj_set = set(gt_top5_coreset_adjusted[i].tolist()) - {-1}
            if not gt_adj_set:
                n_empty_gt += 1
                continue
            hits_adj.append(int(bool(retrieved_set & gt_adj_set)))

        recall_raw = float(np.mean(hits_raw)) if hits_raw else 0.0
        recall_adj = float(np.mean(hits_adj)) if hits_adj else 0.0

        self._summary["recall@5_raw"] = round(recall_raw, 4)
        self._summary["recall@5_adj"] = round(recall_adj, 4)

        logger.info(
            "recall_computed_v5",
            recall_raw=round(recall_raw, 4),
            recall_adj=round(recall_adj, 4),
            n_raw=len(hits_raw), n_adj=len(hits_adj),
            n_empty_gt=n_empty_gt,
            index_space="original_index",
        )
        return {
            "recall@5_raw":               round(recall_raw, 4),
            "recall@5_coreset_adjusted":  round(recall_adj, 4),
            "recall@5_delta":             round(recall_adj - recall_raw, 4),
            "n_evaluated_raw":            len(hits_raw),
            "n_evaluated_adjusted":       len(hits_adj),
            "n_empty_gt_after_filtering": n_empty_gt,
            "coreset_coverage_rate":      coreset_coverage_rate,
            "n_coreset_train":            n_coreset_train,
            "index_space":                "original_index (= ES _id = parquet original_index)",
        }

    # ──────────────────────────────────────────────────────────
    # Router MRR (Mean Reciprocal Rank)
    # ──────────────────────────────────────────────────────────

    def calculate_router_mrr(
        self,
        predictions: List[Dict],
        gt_clusters: np.ndarray,
    ) -> Dict:
        """
        Router가 반환한 top_k_clusters를 거리순 정렬했을 때,
        gt_cluster_id가 몇 번째 rank에 위치하는지 측정.
        
        MRR = mean(1 / rank), rank는 1-indexed
        
        predictions[i]["cluster_distances"]: {cluster_id: distance}
        """
        reciprocal_ranks = []
        n_not_found = 0
        
        for i, pred in enumerate(predictions):
            if pred.get("error") or i >= len(gt_clusters):
                continue
            
            gt_c = int(gt_clusters[i])
            if gt_c < 0:
                continue
            
            # ★ [수정] rank 소스 우선순위: router가 정렬해 반환한 top_k 리스트
            #   (rank1=percolate, rank2~5=centroid score — score는 클수록 좋아
            #   거리 오름차순 정렬이 의미론적으로 틀림) → 없을 때만 거리 정렬 fallback.
            top_k = pred.get("top_k_clusters") or pred.get("top_5_clusters") or []
            if top_k:
                order = [c.get("cluster_id") for c in top_k]
            else:
                cluster_dists = pred.get("cluster_distances", {})
                if not cluster_dists:
                    continue
                order = [cid for cid, _ in sorted(cluster_dists.items(), key=lambda x: x[1])]

            # gt_cluster_id의 rank 찾기 (1-indexed)
            rank = None
            for idx, cid in enumerate(order, start=1):
                if cid == gt_c:
                    rank = idx
                    break
            
            if rank is not None:
                reciprocal_ranks.append(1.0 / rank)
            else:
                n_not_found += 1
        
        mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
        self._summary["router_mrr"] = round(mrr, 4)
        
        return {
            "router_mrr": round(mrr, 4),
            "n_evaluated": len(reciprocal_ranks),
            "n_not_found": n_not_found,
        }

    # ──────────────────────────────────────────────────────────
    # Candidate Recall (Top-5 Cluster IDs)
    # ──────────────────────────────────────────────────────────

    def calculate_candidate_recall(
        self,
        predictions: List[Dict],
        gt_top5_cluster_ids: np.ndarray,
    ) -> Dict:
        """
        라우터가 반환한 top_k_clusters 중 하나라도
        gt_top5_cluster_ids에 포함되는지 확인.
        
        gt_top5_cluster_ids[i]: PCA 공간 내 centroid 기준 Top-5 정답셋
        """
        hits = []
        
        for i, pred in enumerate(predictions):
            if pred.get("error") or i >= len(gt_top5_cluster_ids):
                continue
            
            # 라우터가 반환한 클러스터 ID들
            top_k = pred.get("top_k_clusters", pred.get("top_5_clusters", []))
            routed_ids = set(c["cluster_id"] for c in top_k)
            
            # GT Top-5 정답셋
            gt_ids = set(gt_top5_cluster_ids[i].tolist()) - {-1}
            
            if not gt_ids:
                continue
            
            # 교집합 존재 여부
            hits.append(int(bool(routed_ids & gt_ids)))
        
        recall = float(np.mean(hits)) if hits else 0.0
        self._summary["candidate_recall_top5"] = round(recall, 4)
        
        return {
            "candidate_recall_top5": round(recall, 4),
            "n_evaluated": len(hits),
        }

    # ──────────────────────────────────────────────────────────
    # Routing ε-Approximate Recall
    # ──────────────────────────────────────────────────────────

    def calculate_routing_epsilon_recall(
        self,
        predictions: List[Dict],
        gt_clusters: np.ndarray,
        gt_top5_cluster_dist: np.ndarray,
        epsilon: float = 0.1,
        gt_top5_cluster_ids: np.ndarray = None,
    ) -> Dict:
        """
        라우팅된 클러스터의 거리가 실제 최단 거리 d* 대비
        (1+ε) 배 이내인지 확인.
        ★ [E2] router 응답에 거리 필드가 없고 score(유사도)만 있는 배포에서는
          score를 d*와 비교할 수 없다 → primary가 GT top5 클러스터 안에 있으면
          해당 rank의 gt_top5_cluster_dist 값을 routed 거리로 lookup(동일 PCA 공간),
          top5 밖이면 d5 초과가 확정이므로 fail(0)로 집계한다.
        
        gt_top5_cluster_dist[i][0]: PCA 공간 내 최단 거리 d*_pca
        
        Args:
            epsilon: 허용 오차 비율 (default=0.1 = 10%)
        """
        within_epsilon = []
        
        for i, pred in enumerate(predictions):
            if pred.get("error") or i >= len(gt_clusters):
                continue
            
            gt_c = int(gt_clusters[i])
            if gt_c < 0 or i >= len(gt_top5_cluster_dist):
                continue
            
            # ★ [수정] docstring 의도("실제 라우팅된 클러스터")와 달리 gt_c 키를
            #   조회하던 버그 → primary_cluster_id(실제 배정) 기준으로 교정.
            #   거리 계열 dict(cluster_centroid_dists)를 우선 사용 — score(유사도)를
            #   d*와 비교하면 의미가 깨짐. JSON 재로딩(report-only) 대비 str-key 방어.
            _pc = pred.get("primary_cluster_id")
            if _pc is None:
                continue
            # ① 거리 계열 dict가 있으면 그대로 사용 (score dict는 사용 금지)
            cluster_dists = pred.get("cluster_centroid_dists") or {}
            routed_dist = cluster_dists.get(_pc, cluster_dists.get(str(_pc)))
            # ② ★ [E2] 없으면 GT top5 테이블에서 lookup (같은 PCA 공간 거리)
            if routed_dist is None and gt_top5_cluster_ids is not None \
                    and i < len(gt_top5_cluster_ids):
                _ids = [int(x) for x in np.asarray(gt_top5_cluster_ids[i]).tolist()]
                if int(_pc) in _ids:
                    routed_dist = float(gt_top5_cluster_dist[i][_ids.index(int(_pc))])
                else:
                    within_epsilon.append(0)   # top5 밖 = d5 초과 확정 → fail
                    continue
            if routed_dist is None:
                continue
            
            # GT 최단 거리 (PCA 공간)
            d_star = gt_top5_cluster_dist[i][0] if len(gt_top5_cluster_dist[i]) > 0 else float('inf')
            
            if d_star == 0:
                # 완전 일치 케이스
                within_epsilon.append(int(routed_dist == 0))
            else:
                # ε-approximate 체크
                within_epsilon.append(int(routed_dist <= d_star * (1 + epsilon)))
        
        recall = float(np.mean(within_epsilon)) if within_epsilon else 0.0
        self._summary[f"routing_epsilon_recall_{int(epsilon*100)}pct"] = round(recall, 4)
        
        return {
            f"routing_epsilon_recall_{int(epsilon*100)}pct": round(recall, 4),
            "epsilon": epsilon,
            "n_evaluated": len(within_epsilon),
        }

    # ──────────────────────────────────────────────────────────
    # Coreset ε-Approximate Recall (Retrieval)
    # ──────────────────────────────────────────────────────────

    def calculate_coreset_epsilon_recall(
        self,
        predictions: List[Dict],
        gt_top5_dist: np.ndarray,
        epsilon: float = 0.1,
    ) -> Dict:
        """
        검색된 문서의 거리가 원본 576d 공간 내 최단 거리 d* 대비
        (1+ε) 배 이내인지 확인.
        
        predictions[i]["top1_distance"]: 검색된 1등 문서의 거리
        gt_top5_dist[i][0]: 원본 공간 내 최단 거리 d*
        
        Args:
            epsilon: 허용 오차 비율 (default=0.1 = 10%)
        """
        within_epsilon = []
        
        for i, pred in enumerate(predictions):
            if pred.get("error") or i >= len(gt_top5_dist):
                continue
            
            # 검색된 거리
            retrieved_dist = pred.get("top1_distance", pred.get("top_1_distance"))
            if retrieved_dist is None or retrieved_dist == float("inf"):
                continue
            
            # GT 최단 거리 (576d 공간)
            gt_dists = gt_top5_dist[i]
            if len(gt_dists) == 0:
                continue
            
            d_star = gt_dists[0]
            
            if d_star == 0:
                within_epsilon.append(int(retrieved_dist == 0))
            else:
                within_epsilon.append(int(retrieved_dist <= d_star * (1 + epsilon)))
        
        recall = float(np.mean(within_epsilon)) if within_epsilon else 0.0
        self._summary[f"coreset_epsilon_recall_{int(epsilon*100)}pct"] = round(recall, 4)
        
        return {
            f"coreset_epsilon_recall_{int(epsilon*100)}pct": round(recall, 4),
            "epsilon": epsilon,
            "n_evaluated": len(within_epsilon),
        }

    # ──────────────────────────────────────────────────────────
    # Distance-Based Anomaly AUROC
    # ──────────────────────────────────────────────────────────

    def calculate_distance_based_auroc(
        self, predictions: List[Dict], gt_labels: np.ndarray
    ) -> Dict:
        """
        ★ V5: top1_distance / top_1_distance 양쪽 지원
        """
        distances, labels = [], []
        for i, pred in enumerate(predictions):
            if pred.get("error") or i >= len(gt_labels):
                continue
            dist = pred.get("top1_distance", pred.get("top_1_distance"))
            if dist is None or dist == float("inf"):
                continue
            distances.append(float(dist))
            labels.append(int(gt_labels[i]))

        if not distances:
            return {"distance_auroc": None, "n_evaluated": 0}

        dist_arr  = np.array(distances)
        label_arr = np.array(labels)

        mean_normal = mean_anomaly = dist_ratio = None
        if label_arr.sum() > 0 and label_arr.sum() < len(label_arr):
            mean_normal  = float(np.mean(dist_arr[label_arr == 0]))
            mean_anomaly = float(np.mean(dist_arr[label_arr == 1]))
            dist_ratio   = round(mean_anomaly / mean_normal, 4) if mean_normal > 0 else None

        auroc = None
        if SKLEARN_AVAILABLE and label_arr.sum() > 0 and label_arr.sum() < len(label_arr):
            try:
                auroc = float(roc_auc_score(label_arr, dist_arr))
            except Exception as e:
                logger.warning("distance_auroc_failed", error=str(e))

        self._summary["distance_auroc"] = round(auroc, 4) if auroc else None
        return {
            "distance_auroc":    round(auroc, 4) if auroc is not None else None,
            "mean_normal_dist":  round(mean_normal, 4)  if mean_normal  is not None else None,
            "mean_anomaly_dist": round(mean_anomaly, 4) if mean_anomaly is not None else None,
            "distance_ratio":    dist_ratio,
            "n_evaluated":       len(distances),
        }

    # ──────────────────────────────────────────────────────────
    # Anomaly Detection AUROC (Full Pipeline)
    # ──────────────────────────────────────────────────────────

    def calculate_anomaly_auroc(
        self,
        gt_labels:   np.ndarray,
        pred_confs:  np.ndarray,
        pred_labels: np.ndarray,
    ) -> Dict:
        n   = len(gt_labels)
        acc = float((gt_labels == pred_labels).mean()) if n > 0 else 0.0
        auroc = None
        if SKLEARN_AVAILABLE and gt_labels.sum() > 0 and gt_labels.sum() < n:
            try:
                auroc = float(roc_auc_score(gt_labels, pred_confs))
            except Exception as e:
                logger.warning("anomaly_auroc_failed", error=str(e))

        tp = int(((pred_labels == 1) & (gt_labels == 1)).sum())
        fp = int(((pred_labels == 1) & (gt_labels == 0)).sum())
        tn = int(((pred_labels == 0) & (gt_labels == 0)).sum())
        fn = int(((pred_labels == 0) & (gt_labels == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-9)

        self._summary["anomaly_auroc"] = round(auroc, 4) if auroc else None
        return {
            "auroc":       round(auroc, 4) if auroc is not None else None,
            "accuracy":    round(acc, 4),
            "precision":   round(precision, 4),
            "recall":      round(recall, 4),
            "f1":          round(f1, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "n_evaluated": n,
        }

    # ──────────────────────────────────────────────────────────
    # Latency
    # ──────────────────────────────────────────────────────────

    def calculate_latency_metrics(self, latencies: np.ndarray) -> Dict:
        if len(latencies) == 0:
            return {}
        return {
            "latency_mean_ms": round(float(np.mean(latencies)), 2),
            "latency_p50_ms":  round(float(np.percentile(latencies, 50)), 2),
            "latency_p95_ms":  round(float(np.percentile(latencies, 95)), 2),
            "latency_p99_ms":  round(float(np.percentile(latencies, 99)), 2),
            "latency_max_ms":  round(float(np.max(latencies)), 2),
        }

    def calculate_latency_breakdown(
        self,
        router_lat:    np.ndarray,
        retriever_lat: np.ndarray,
        analyzer_lat:  np.ndarray,
    ) -> Dict:
        """
        ★ V5: SKIP_ANALYZER=true 시 analyzer_lat=[] 처리 (에러 없이 0 반환).
        """
        def _mean(a): return round(float(np.mean(a)), 2)  if len(a) > 0 else 0.0
        def _p95(a):  return round(float(np.percentile(a, 95)), 2) if len(a) > 0 else 0.0

        return {
            "router_mean_ms":    _mean(router_lat),
            "retriever_mean_ms": _mean(retriever_lat),
            "analyzer_mean_ms":  _mean(analyzer_lat),
            "router_p95_ms":     _p95(router_lat),
            "retriever_p95_ms":  _p95(retriever_lat),
            "analyzer_p95_ms":   _p95(analyzer_lat),
            "sum_mean_ms":       round(_mean(router_lat) + _mean(retriever_lat) + _mean(analyzer_lat), 2),
            "analyzer_skipped":  len(analyzer_lat) == 0,
        }

    # ──────────────────────────────────────────────────────────
    # Goals & Summary
    # ──────────────────────────────────────────────────────────

    def check_goals(self) -> Dict[str, bool]:
        goals = {}
        s = self._summary
        if "recall@5_adj"         in s: goals["recall@5_adj >= 0.8"]    = s["recall@5_adj"]         >= 0.8
        if "recall@5_raw"         in s: goals["recall@5_raw >= 0.7"]    = s["recall@5_raw"]         >= 0.7
        if "router_recall_at_5"   in s: goals["router_recall@5 >= 0.9"] = s["router_recall_at_5"]   >= 0.9
        if "cluster_accuracy"     in s: goals["cluster_acc >= 0.7"]     = s["cluster_accuracy"]     >= 0.7
        if s.get("distance_auroc") is not None: goals["distance_auroc >= 0.7"] = s["distance_auroc"] >= 0.7
        if s.get("anomaly_auroc")  is not None: goals["anomaly_auroc >= 0.8"]  = s["anomaly_auroc"]  >= 0.8
        # ★ 추가 goals (이 부분만 추가)
        if "router_mrr"              in s: goals["router_mrr >= 0.7"]         = s["router_mrr"]         >= 0.7
        if "candidate_recall_top5"   in s: goals["candidate_recall >= 0.85"]  = s["candidate_recall_top5"] >= 0.85
        return goals

    # ──────────────────────────────────────────────────────────
    # ★ [EXP] Research Effect — 재검색 전/후 nd 판정 변화의 인과 분해
    # ──────────────────────────────────────────────────────────
    #  입력은 controller finalize가 round_log에 남기는 {nd_round0, nd_final,
    #  researched} 3키(= agent_batch_inference가 _nd_round0/_nd_final/
    #  _researched로 승격한 값). 재검색이 verdict를 바꾸지 못하더라도 Δnd
    #  방향성/선택 품질은 살아 있으므로, "효과 없음"과 "측정 불가"를 구분한다.
    #
    #  임계 3-view (핵심 설계):
    #    fixed           : CLS_ND_THRESHOLD 고정 — 서빙 실제 효과
    #    pre_best_frozen : pre의 Youden 임계를 post에 그대로 적용
    #                      → 임계 재탐색 이득을 배제하고 nd 이동 효과만 격리 (주 지표)
    #    each_best       : pre/post 각자 best — 상한 성능(참고)
    def calculate_research_effect(
        self,
        predictions: List[Dict],
        labels,
        *,
        cls_threshold: float = 0.42,
        boundary: tuple = (0.3, 0.55),
        max_samples_listed: int = 50,
    ) -> Dict:
        import math

        lo, hi = float(boundary[0]), float(boundary[1])

        # ── 1) per-sample 추출 ────────────────────────────────
        rows: List[Dict] = []
        for i, (p, lb) in enumerate(zip(predictions, list(labels))):
            if p.get("error"):
                continue
            pre  = p.get("_nd_round0")
            post = p.get("_nd_final")
            # ★ 방어: harness가 _nd_* 승격에 실패해도 agent_rounds의 final
            #   엔트리에서 직접 복구한다(nd_dual과 동일 경로 → 반드시 성공).
            if pre is None or post is None:
                _fin = next((r for r in (p.get("agent_rounds") or [])
                             if isinstance(r, dict) and r.get("round") == "final"), {})
                pre  = _fin.get("nd_round0") if pre  is None else pre
                post = _fin.get("nd_final")  if post is None else post
                if p.get("_researched") is None:
                    p["_researched"] = bool(_fin.get("researched", False))
            if pre is None or post is None:
                continue
            pre, post = float(pre), float(post)
            rows.append({
                "idx":        int(p.get("idx", i)),
                "label":      int(lb),
                "nd_pre":     round(pre, 4),
                "nd_post":    round(post, 4),
                "delta_nd":   round(post - pre, 4),
                # band는 재검색 대상 선정 기준(= 재검색 전 nd)으로 정의한다
                "band":       "ambiguous" if (lo <= pre < hi) else "confident",
                "researched": bool(p.get("_researched", (post != pre))),
                "wm_type":    p.get("_wm_type"),
                "nd_source":  p.get("_nd_source"),
                "research_ms": p.get("_research_ms"),
            })

        if not rows:
            _has_rounds = sum(1 for p in predictions if p.get("agent_rounds"))
            return {"n": 0, "_note": "no (nd_round0, nd_final) pair found",
                    "_diag": {"n_predictions": len(predictions),
                              "n_with_agent_rounds": _has_rounds,
                              "hint": ("controller round_log의 final 엔트리 부재 — "
                                       "controller.py 재배포 여부 확인")}}

        pre_arr  = np.array([r["nd_pre"]  for r in rows], dtype=float)
        post_arr = np.array([r["nd_post"] for r in rows], dtype=float)
        y        = np.array([r["label"]   for r in rows], dtype=int)
        n = len(rows)

        # ── 2) 보조 함수 ──────────────────────────────────────
        def _auroc(scores):
            if len(set(y.tolist())) < 2:
                return None
            try:
                from sklearn.metrics import roc_auc_score
                return round(float(roc_auc_score(y, scores)), 4)
            except Exception:
                order = scores.argsort()
                ranks = np.empty_like(order, dtype=float)
                ranks[order] = np.arange(1, len(scores) + 1)
                n_pos = int(y.sum()); n_neg = n - n_pos
                if n_pos == 0 or n_neg == 0:
                    return None
                return round(float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2)
                                   / (n_pos * n_neg)), 4)

        def _best_thr(scores):
            """Youden J 최대점. sklearn 없으면 후보 스캔."""
            try:
                from sklearn.metrics import roc_curve
                fpr, tpr, thr = roc_curve(y, scores)
                j = tpr - fpr
                return float(thr[int(np.argmax(j))])
            except Exception:
                cands = np.unique(scores)
                best, best_j = float(cls_threshold), -1.0
                for t in cands:
                    pred = (scores >= t).astype(int)
                    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
                    fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
                    tpr = tp / max(tp + fn, 1); fpr = fp / max(fp + tn, 1)
                    if tpr - fpr > best_j:
                        best_j, best = tpr - fpr, float(t)
                return best

        def _conf(scores, thr, mask=None):
            m = np.ones(len(scores), dtype=bool) if mask is None else mask
            if m.sum() == 0:
                return {"n": 0}
            sc, yy = scores[m], y[m]
            pred = (sc >= thr).astype(int)
            tp = int(((pred == 1) & (yy == 1)).sum()); fn = int(((pred == 0) & (yy == 1)).sum())
            fp = int(((pred == 1) & (yy == 0)).sum()); tn = int(((pred == 0) & (yy == 0)).sum())
            npos, nneg = tp + fn, fp + tn
            return {
                "n": int(m.sum()), "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                "TPR": round(tp / npos, 4) if npos else None,
                "FNR": round(fn / npos, 4) if npos else None,
                "FPR": round(fp / nneg, 4) if nneg else None,
                "TNR": round(tn / nneg, 4) if nneg else None,
                "accuracy": round((tp + tn) / max(int(m.sum()), 1), 4),
            }

        def _wilson(k, tot, z=1.96):
            if tot == 0:
                return [None, None]
            ph = k / tot
            d  = 1 + z * z / tot
            c  = ph + z * z / (2 * tot)
            m  = z * math.sqrt(ph * (1 - ph) / tot + z * z / (4 * tot * tot))
            return [round(max(0.0, (c - m) / d), 4), round(min(1.0, (c + m) / d), 4)]

        amb_m  = np.array([r["band"] == "ambiguous" for r in rows])
        conf_m = ~amb_m
        res_m  = np.array([r["researched"] for r in rows])

        # ── 3) 임계 3-view ────────────────────────────────────
        thr_pre_best  = _best_thr(pre_arr)
        thr_post_best = _best_thr(post_arr)
        views = {
            "fixed":           {"pre": float(cls_threshold), "post": float(cls_threshold)},
            "pre_best_frozen": {"pre": thr_pre_best,         "post": thr_pre_best},
            "each_best":       {"pre": thr_pre_best,         "post": thr_post_best},
        }

        by_view: Dict = {}
        for vname, t in views.items():
            by_view[vname] = {
                "threshold_pre": round(t["pre"], 4), "threshold_post": round(t["post"], 4),
                "overall":   {"pre": _conf(pre_arr, t["pre"]),
                              "post": _conf(post_arr, t["post"])},
                "ambiguous": {"pre": _conf(pre_arr, t["pre"], amb_m),
                              "post": _conf(post_arr, t["post"], amb_m)},
                "confident": {"pre": _conf(pre_arr, t["pre"], conf_m),
                              "post": _conf(post_arr, t["post"], conf_m)},
            }
            # ★ 대조군 불변성: 재검색 안 한 confident군은 pre==post 여야 정상
            by_view[vname]["confident"]["_invariant_ok"] = (
                by_view[vname]["confident"]["pre"] == by_view[vname]["confident"]["post"]
            )

        # ── 4) 판정 전이 (주 지표 = pre_best_frozen) ──────────
        MAIN = "pre_best_frozen"
        t_main = views[MAIN]
        pred_pre  = (pre_arr  >= t_main["pre"]).astype(int)
        pred_post = (post_arr >= t_main["post"]).astype(int)
        ok_pre, ok_post = (pred_pre == y), (pred_post == y)

        rescued_m  = (~ok_pre) & ok_post          # 오답 → 정답
        degraded_m = ok_pre & (~ok_post)          # 정답 → 오답
        flip_m     = pred_pre != pred_post

        def _dump(mask):
            out = []
            for r, m in zip(rows, mask):
                if m:
                    out.append({k: r[k] for k in
                                ("idx", "label", "nd_pre", "nd_post", "delta_nd",
                                 "band", "wm_type", "researched")})
            return out[:max_samples_listed]

        def _band_count(mask):
            return {"ambiguous": int((mask & amb_m).sum()),
                    "confident": int((mask & conf_m).sum())}

        n_res = int(res_m.sum())
        transitions = {
            "main_view":      MAIN,
            "n_flip":         int(flip_m.sum()),
            "flip_rate":      round(float(flip_m.sum()) / n, 4),
            "rescued":  {"n": int(rescued_m.sum()),  "by_band": _band_count(rescued_m),
                         "samples": _dump(rescued_m)},
            "degraded": {"n": int(degraded_m.sum()), "by_band": _band_count(degraded_m),
                         "samples": _dump(degraded_m)},
            "stable_correct": int((ok_pre & ok_post).sum()),
            "stable_wrong":   int(((~ok_pre) & (~ok_post)).sum()),
            "net_gain":       int(rescued_m.sum()) - int(degraded_m.sum()),
        }

        # ── 5) Δnd 분포 + 방향 정확도 ─────────────────────────
        #   verdict가 안 바뀌어도 살아 있는 신호. label=1이면 nd↑, label=0이면 nd↓가 옳다.
        d = post_arr - pre_arr
        d_res = d[res_m] if n_res else np.array([])
        moved = res_m & (np.abs(d) > 1e-9)
        dir_ok = ((d[moved] > 0) == (y[moved] == 1)) if moved.sum() else np.array([])

        def _wm_delta(t):
            m = res_m & np.array([r["wm_type"] == t for r in rows])
            if m.sum() == 0:
                return {"n": 0}
            dd = d[m]
            return {"n": int(m.sum()), "mean": round(float(dd.mean()), 4),
                    "mean_abs": round(float(np.abs(dd).mean()), 4)}

        delta_nd = {
            "n_researched": n_res,
            "mean":     round(float(d_res.mean()), 4) if n_res else None,
            "median":   round(float(np.median(d_res)), 4) if n_res else None,
            "mean_abs": round(float(np.abs(d_res).mean()), 4) if n_res else None,
            "p10":      round(float(np.percentile(d_res, 10)), 4) if n_res else None,
            "p90":      round(float(np.percentile(d_res, 90)), 4) if n_res else None,
            "n_moved":  int(moved.sum()),
            "directional_accuracy": (round(float(dir_ok.mean()), 4)
                                     if len(dir_ok) else None),
            "by_wm_type": {"ni": _wm_delta("ni"), "cm": _wm_delta("cm")},
        }

        # ── 6) abstention 선택 품질 ───────────────────────────
        wrong_pre = ~ok_pre
        selection = {
            "n_ambiguous": int(amb_m.sum()),
            "ambiguous_precision": (round(float((amb_m & wrong_pre).sum() / amb_m.sum()), 4)
                                    if amb_m.sum() else None),
            "ambiguous_recall":    (round(float((amb_m & wrong_pre).sum() / wrong_pre.sum()), 4)
                                    if wrong_pre.sum() else None),
            "missed": int((conf_m & wrong_pre).sum()),   # confident인데 오답 = 기회 상실
        }

        # ── 7) 비용 ───────────────────────────────────────────
        ms = [float(r["research_ms"]) for r in rows
              if r["researched"] and r.get("research_ms") is not None]
        n_flip, n_resc = transitions["n_flip"], transitions["rescued"]["n"]
        total_ms = float(np.sum(ms)) if ms else None
        cost = {
            "n_researched": n_res,
            "research_ms_mean": round(float(np.mean(ms)), 2) if ms else None,
            "research_ms_total": round(total_ms, 2) if total_ms is not None else None,
            "ms_per_flip":   (round(total_ms / n_flip, 2) if (total_ms and n_flip) else None),
            "ms_per_rescue": (round(total_ms / n_resc, 2) if (total_ms and n_resc) else None),
        }

        result = {
            "n": n,
            "cls_threshold": float(cls_threshold),
            "boundary": [lo, hi],
            "auroc": {"pre": _auroc(pre_arr), "post": _auroc(post_arr)},
            "by_view":     by_view,
            "transitions": transitions,
            "delta_nd":    delta_nd,
            "selection":   selection,
            "cost":        cost,
            "ci95": {
                "flip_rate":    _wilson(n_flip, n),
                "rescued_rate": _wilson(n_resc, max(n_res, 1)),
            },
        }
        if result["auroc"]["pre"] is not None and result["auroc"]["post"] is not None:
            result["auroc"]["delta"] = round(result["auroc"]["post"] - result["auroc"]["pre"], 4)
        if n_res < 30:
            result["_warning"] = (f"n_researched={n_res} — flip 1건이 "
                                  f"{round(100.0 / max(n_res, 1), 1)}%p. "
                                  f"결론은 n>=200에서 재확인 필요")

        self._summary["research_net_gain"]  = transitions["net_gain"]
        self._summary["research_flip_rate"] = transitions["flip_rate"]
        self._summary["research_dir_acc"]   = delta_nd["directional_accuracy"]
        logger.info("research_effect_computed", n=n, n_researched=n_res,
                    net_gain=transitions["net_gain"],
                    dir_acc=delta_nd["directional_accuracy"])
        return result

    def get_summary(self) -> Dict:
        return {"experiment_case": self.experiment_case,
                "coreset_percentage": self.coreset_percentage,
                **self._summary}


# ============================================================
# ★ [V3] 선택 지표 — _nd_global 컬럼이 있으면 별도 AUROC 병기
#   (legacy 리포트와 혼합 금지: score_column을 명시해 반환)
# ============================================================
def compute_nd_global_auroc(predictions):
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None
    ys, ss = [], []
    for p in predictions:
        g = p.get("_nd_global")
        gt = p.get("gt_label", p.get("label"))
        if g is None or gt is None:
            continue
        ys.append(1 if str(gt).upper() in ("1", "ABNORMAL") else 0)
        ss.append(float(g))
    if len(set(ys)) < 2:
        return None
    return {"score_column": "_nd_global", "n": len(ys),
            "auroc": round(float(roc_auc_score(ys, ss)), 4)}