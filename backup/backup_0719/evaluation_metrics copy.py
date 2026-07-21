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
            
            cluster_dists = pred.get("cluster_distances", {})
            if not cluster_dists:
                continue
            
            # 거리순 정렬 (오름차순)
            sorted_clusters = sorted(cluster_dists.items(), key=lambda x: x[1])
            
            # gt_cluster_id의 rank 찾기 (1-indexed)
            rank = None
            for idx, (cid, _) in enumerate(sorted_clusters, start=1):
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
    ) -> Dict:
        """
        라우팅된 클러스터의 거리가 실제 최단 거리 d* 대비
        (1+ε) 배 이내인지 확인.
        
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
            
            cluster_dists = pred.get("cluster_distances", {})
            if gt_c not in cluster_dists:
                continue
            
            # 실제 라우팅된 클러스터의 거리
            routed_dist = cluster_dists[gt_c]
            
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

    def get_summary(self) -> Dict:
        return {"experiment_case": self.experiment_case,
                "coreset_percentage": self.coreset_percentage,
                **self._summary}
