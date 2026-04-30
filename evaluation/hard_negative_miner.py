"""
Hard Negative Miner - V1.0
============================

Pipeline-Aware Hard Negative Mining for Analyzer SFT.

Type 1: Router Misrouting Hard Negative
  - gt_cluster_id != predicted_cluster (Router가 잘못 배정)
  - Analyzer가 틀린 cluster persona + 틀린 cluster pool의 neighbors를 받는 케이스
  - 발생률: ~28.5% (구조적 k-means/decision-tree 기하학적 불일치)

Type 2: Cross-Cluster Retrieval Hard Negative
  - Router는 올바르게 배정 (gt_cluster == predicted_cluster)
  - Retriever top-k 중 ≥1개가 다른 cluster 소속 (HNSW Voronoi 경계 샘플)
  - coreset_df의 cluster_id 컬럼으로 판별

사용법:
    # batch_inference.py에서 자동 호출 (--mine-hard-negatives 플래그)
    python batch_inference.py \\
        --pipeline-mode router_retriever \\
        --experiment-case pca_64_k100 \\
        --percolate-version v1 \\
        --mine-hard-negatives

    # standalone
    from hard_negative_miner import HardNegativeMiner

출력 parquet 스키마:
    test_idx                   : int   전체 test set 기준 위치 (0-based)
    pred_pos                   : int   10% 서브셋 내 위치
    gt_label                   : int   0=정상 / 1=이상
    gt_cluster_id              : int   GT cluster id
    predicted_cluster          : int   Router 실제 배정 cluster (type1에서 불일치)
    top5_indices               : str   JSON — GT top-5 original_index (SFT 프롬프트용)
    retrieved_original_indices : str   JSON — Retriever 실제 반환 original_index
    neighbor_cluster_ids       : str   JSON — 각 이웃의 cluster id (type2 전용)
    cross_cluster_ids          : str   JSON — GT cluster와 다른 이웃 cluster id 목록 (type2 전용)
    n_cross_cluster_neighbors  : int   다른 cluster 이웃 수 (type2 전용)
    top1_distance              : float
    feat_*                     : 원본 거래 피처 (purchase_value, age, sex, ...)
    mining_type                : str   "type1_misrouted" | "type2_cross_cluster"
    experiment_case            : str
    percolate_version          : str
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import structlog
from datetime import datetime

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

logger = structlog.get_logger()

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

GCS_BUCKET_NAME = "fraudecom"

# GCS 저장 경로 템플릿
GCS_MINING_PATH_TEMPLATE = (
    "tree-search/analyzer_ft/{experiment_case}/"
    "hard_negatives/"
    "query_{percolate_version}_pct_{coreset_pct}/"
    "hard_negatives_{mining_date}.parquet"
)

# SFT 프롬프트 구성에 사용할 원본 피처 컬럼
FEATURE_COLS = [
    "purchase_value", "age", "sex", "source", "browser",
    "weekday_purchase", "month_purchase", "IP_country",
]


# ──────────────────────────────────────────────────────────────
# HardNegativeMiner
# ──────────────────────────────────────────────────────────────

class HardNegativeMiner:
    """
    Batch inference 결과에서 Type 1 / Type 2 Hard Negative를 추출한다.

    Parameters
    ----------
    predictions : List[Dict]
        BatchInference.predictions (router_retriever mode 결과 필수)
        각 원소에 primary_cluster_id, retrieved_original_indices 필드 필요.
    test_data : pd.DataFrame
        GTLoader.load_all() 반환 test_data (원본 피처 컬럼 포함)
    gt_data : Dict
        GTLoader.load_all() 반환 gt_data
        필수 키: label, gt_cluster_id, gt_top5_indices
    coreset_df : pd.DataFrame
        GTLoader.load_coreset_df() 반환값
        필수 컬럼: original_index, cluster_id  ← Type 2 판별에 사용
    experiment_case : str
    percolate_version : str
    coreset_percentage : str
    sample_indices : List[int]
        batch_inference에서 실제 실행한 test set 위치 목록.
        batch_inference.py가 self.sample_indices로 자동 전달.
        None이면 0-based 연속 인덱스로 가정.
    """

    def __init__(
        self,
        predictions:        List[Dict],
        test_data:          pd.DataFrame,
        gt_data:            Dict,
        coreset_df:         pd.DataFrame,
        experiment_case:    str  = "pca_64_k100",
        percolate_version:  str  = "v1",
        coreset_percentage: str  = "100",
        sample_indices:     Optional[List[int]] = None,
        # ★ Type 3, Type 4 추가 파라미터
        test_emb:           Optional[np.ndarray] = None,        # ← 추가
        cluster_centroids:  Optional[Dict[int, np.ndarray]] = None,  # ← 추가
        distance_percentiles: Optional[Dict[str, float]] = None,     # ← 추가
    ):
        self.predictions        = predictions
        self.test_data          = test_data
        self.gt_data            = gt_data
        self.coreset_df         = coreset_df
        self.experiment_case    = experiment_case
        self.percolate_version  = percolate_version
        self.coreset_percentage = coreset_percentage

        n_preds = len(predictions)

        # sample_indices 설정 및 검증
        if sample_indices is not None:
            if len(sample_indices) != n_preds:
                raise ValueError(
                    f"sample_indices 길이 불일치: "
                    f"sample_indices={len(sample_indices)}, predictions={n_preds}. "
                    f"batch_inference.py의 self.sample_indices가 올바르게 저장됐는지 확인."
                )
            self.sample_indices = sample_indices
        else:
            # fallback: 0-based 연속 인덱스 가정
            logger.warning(
                "sample_indices_not_provided",
                note="predictions[j]가 test_data[j]에 대응한다고 가정합니다.",
            )
            self.sample_indices = list(range(n_preds))

        # ── original_index → cluster_id 매핑 구축 (Type 2용) ──
        self._orig_idx_to_cluster: Dict[int, int] = {}
        missing_cols = [
            c for c in ("original_index", "cluster_id")
            if c not in coreset_df.columns
        ]
        if missing_cols:
            logger.warning(
                "coreset_df_missing_columns",
                missing=missing_cols,
                available=coreset_df.columns.tolist(),
                note="Type 2 mining은 cluster_id, original_index 컬럼이 필요합니다.",
            )
        else:
            self._orig_idx_to_cluster = dict(
                zip(
                    coreset_df["original_index"].astype(int),
                    coreset_df["cluster_id"].astype(int),
                )
            )

        logger.info(
            "hard_negative_miner_init_v1",
            experiment_case=experiment_case,
            percolate_version=percolate_version,
            coreset_percentage=coreset_percentage,
            n_predictions=n_preds,
            n_sample_indices=len(self.sample_indices),
            n_coreset_cluster_map=len(self._orig_idx_to_cluster),
            sample_indices_head=self.sample_indices[:5],
        )

        # ★ Type 3, Type 4 추가 초기화
        self.test_emb = test_emb
        self.cluster_centroids = cluster_centroids

        # distance_percentiles가 없으면 predictions에서 자동 계산
        if distance_percentiles is None and predictions:
            dists = [
                float(p.get("top1_distance", p.get("top_1_distance", -1.0)))
                for p in predictions
                if not p.get("error") and p.get("top1_distance", p.get("top_1_distance", -1.0)) > 0
            ]
            if dists:
                self.dist_p75 = float(np.percentile(dists, 75))
                self.dist_p90 = float(np.percentile(dists, 90))
                logger.info(
                    "distance_percentiles_computed",
                    p75=round(self.dist_p75, 4),
                    p90=round(self.dist_p90, 4),
                )
            else:
                self.dist_p75 = float('inf')
                self.dist_p90 = float('inf')
        else:
            self.dist_p75 = distance_percentiles.get("p75", float('inf'))
            self.dist_p90 = distance_percentiles.get("p90", float('inf'))

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    def _extract_primary_cluster_id(self, pred: Dict) -> int:
        """Router가 실제 배정한 cluster_id 추출 (batch_inference._extract_primary_cluster_id와 동일)."""
        if "primary_cluster_id" in pred:
            v = pred["primary_cluster_id"]
            return int(v) if v is not None else -1
        for key in ("top_k_clusters", "top_5_clusters"):
            clusters = pred.get(key, [])
            if clusters:
                return int(clusters[0]["cluster_id"])
        return -1

    def _get_feature_row(self, test_idx: int) -> Dict:
        """test_data에서 원본 피처 dict 추출 (SFT 프롬프트 구성용)."""
        row = self.test_data.iloc[test_idx]
        feat = {}
        for col in FEATURE_COLS:
            if col in self.test_data.columns:
                val = row[col]
                feat[col] = None if (isinstance(val, float) and pd.isna(val)) else val
        return feat

    def _is_cross_cluster(
        self,
        retrieved_original_indices: List[int],
        gt_cluster: int,
    ) -> Tuple[bool, List[int]]:
        """
        retrieved neighbors 중 gt_cluster와 다른 cluster 소속 이웃 존재 여부 확인.

        Returns
        -------
        (is_cross_cluster: bool, cross_cluster_ids: List[int])
            cross_cluster_ids: 다른 cluster에 속하는 이웃들의 cluster id 목록
        """
        if not self._orig_idx_to_cluster:
            return False, []

        cross_ids = []
        for oi in retrieved_original_indices:
            nb_cluster = self._orig_idx_to_cluster.get(int(oi), -1)
            if nb_cluster != -1 and nb_cluster != gt_cluster:
                cross_ids.append(nb_cluster)

        return len(cross_ids) > 0, cross_ids

    # ──────────────────────────────────────────────────────────
    # Type 1: Router Misrouting
    # ──────────────────────────────────────────────────────────

    def mine_type1_misrouting(self) -> List[Dict]:
        """
        Type 1: Router가 잘못 배정한 샘플 추출.

        조건:
            gt_cluster_id != predicted_cluster  (Router가 배정한 cluster)

        Analyzer 관점:
            - 틀린 cluster의 statistical persona를 받음
            - 틀린 cluster pool에서 가져온 neighbors를 받음
            → feature 값 자체와 L2 distance signal만으로 올바른 판단을 학습시키는 케이스

        Returns
        -------
        List[Dict] : type1 hard negative 레코드 목록
        """
        gt_clusters = self.gt_data["gt_cluster_id"]
        gt_labels   = self.gt_data["label"]
        top5_gt     = self.gt_data["gt_top5_indices"]

        records   = []
        n_checked = 0
        n_errors  = 0

        for pred_pos, (test_idx, pred) in enumerate(
            zip(self.sample_indices, self.predictions)
        ):
            if pred.get("error"):
                n_errors += 1
                continue

            n_checked += 1
            gt_cluster        = int(gt_clusters[test_idx])
            predicted_cluster = self._extract_primary_cluster_id(pred)
            gt_label          = int(gt_labels[test_idx])

            # ── Type 1 조건 ──────────────────────────────────
            if gt_cluster == predicted_cluster or predicted_cluster == -1:
                continue  # routing 정확 → 제외

            feat         = self._get_feature_row(test_idx)
            retrieved_oi = pred.get("retrieved_original_indices", [])

            records.append({
                # 인덱스
                "test_idx":                   test_idx,
                "pred_pos":                   pred_pos,
                # GT
                "gt_label":                   gt_label,
                "gt_cluster_id":              gt_cluster,
                "top5_indices":               top5_gt[test_idx].tolist(),
                # Router 결과
                "predicted_cluster":          predicted_cluster,
                "top_k_clusters":             pred.get("top_k_clusters", []),
                "match_type":                 pred.get("match_type", "unknown"),
                # Retriever 결과
                "retrieved_original_indices": retrieved_oi,
                "top1_distance":              float(pred.get(
                    "top1_distance", pred.get("top_1_distance", -1.0)
                )),
                # 원본 피처 (feat_ 접두어로 저장 — 기존 test_df 컬럼명과 충돌 방지)
                **{f"feat_{k}": v for k, v in feat.items()},
                # Mining 메타데이터
                "mining_type":                "type1_misrouted",
                "experiment_case":            self.experiment_case,
                "percolate_version":          self.percolate_version,
            })

        misrouting_rate = len(records) / max(n_checked, 1)
        logger.info(
            "type1_mining_complete",
            n_checked=n_checked,
            n_errors=n_errors,
            n_type1=len(records),
            misrouting_rate=round(misrouting_rate, 4),
        )
        print(
            f"[Type 1] n_checked={n_checked}  misrouted={len(records)}  "
            f"rate={misrouting_rate:.1%}"
        )
        return records

    # ──────────────────────────────────────────────────────────
    # Type 2: Cross-Cluster Retrieval
    # ──────────────────────────────────────────────────────────

    def mine_type2_cross_cluster(self) -> List[Dict]:
        """
        Type 2: Router는 정확하지만 Retriever top-k에 타 cluster 이웃 포함.

        조건:
            - gt_cluster_id == predicted_cluster  (Router 정확)
            - retrieved_original_indices 중 ≥1개가 다른 cluster 소속
              (HNSW Voronoi 경계 근처 샘플에서 발생)

        Analyzer 관점:
            - 올바른 cluster persona를 받지만
            - heterogeneous neighbors를 받음 (일부 neighbor가 다른 분포에서 옴)
            → 각 neighbor의 feature 값 자체를 해석해 판단하는 패턴 학습

        Returns
        -------
        List[Dict] : type2 hard negative 레코드 목록
        """
        if not self._orig_idx_to_cluster:
            logger.warning(
                "type2_mining_skipped",
                reason=(
                    "coreset_df에 cluster_id 컬럼이 없어 Type 2 mining을 건너뜁니다. "
                    "GTLoader.load_coreset_df()가 cluster_id 컬럼을 포함하는지 확인하세요."
                ),
            )
            print("[Type 2] SKIPPED — coreset_df에 cluster_id 컬럼 없음")
            return []

        gt_clusters = self.gt_data["gt_cluster_id"]
        gt_labels   = self.gt_data["label"]
        top5_gt     = self.gt_data["gt_top5_indices"]

        records      = []
        n_checked    = 0
        n_routing_ok = 0
        n_errors     = 0

        for pred_pos, (test_idx, pred) in enumerate(
            zip(self.sample_indices, self.predictions)
        ):
            if pred.get("error"):
                n_errors += 1
                continue

            n_checked += 1
            gt_cluster        = int(gt_clusters[test_idx])
            predicted_cluster = self._extract_primary_cluster_id(pred)
            gt_label          = int(gt_labels[test_idx])

            # ── Type 2 전제: routing이 올바른 케이스만 ──────────
            if gt_cluster != predicted_cluster or predicted_cluster == -1:
                continue
            n_routing_ok += 1

            retrieved_oi = pred.get("retrieved_original_indices", [])
            if not retrieved_oi:
                continue

            is_cross, cross_ids = self._is_cross_cluster(retrieved_oi, gt_cluster)
            if not is_cross:
                continue  # 순수 동일-cluster 이웃 → 제외

            feat = self._get_feature_row(test_idx)

            # 각 retrieved neighbor의 cluster 정보 기록 (SFT notebook에서 활용 가능)
            neighbor_cluster_ids = [
                self._orig_idx_to_cluster.get(int(oi), -1)
                for oi in retrieved_oi
            ]

            records.append({
                # 인덱스
                "test_idx":                   test_idx,
                "pred_pos":                   pred_pos,
                # GT
                "gt_label":                   gt_label,
                "gt_cluster_id":            gt_cluster,
                "top5_indices":               top5_gt[test_idx].tolist(),
                # Router 결과 (routing 정확)
                "predicted_cluster":          predicted_cluster,
                "top_k_clusters":             pred.get("top_k_clusters", []),
                "match_type":                 pred.get("match_type", "unknown"),
                # Retriever 결과
                "retrieved_original_indices": retrieved_oi,
                "neighbor_cluster_ids":       neighbor_cluster_ids,
                "cross_cluster_ids":          cross_ids,
                "n_cross_cluster_neighbors":  len(cross_ids),
                "top1_distance":              float(pred.get(
                    "top1_distance", pred.get("top_1_distance", -1.0)
                )),
                # 원본 피처
                **{f"feat_{k}": v for k, v in feat.items()},
                # Mining 메타데이터
                "mining_type":                "type2_cross_cluster",
                "experiment_case":            self.experiment_case,
                "percolate_version":          self.percolate_version,
            })

        cross_rate = len(records) / max(n_routing_ok, 1)
        logger.info(
            "type2_mining_complete",
            n_checked=n_checked,
            n_routing_ok=n_routing_ok,
            n_errors=n_errors,
            n_type2=len(records),
            cross_cluster_rate=round(cross_rate, 4),
        )
        print(
            f"[Type 2] n_routing_ok={n_routing_ok}  cross_cluster={len(records)}  "
            f"rate={cross_rate:.1%}"
        )
        return records

    # ──────────────────────────────────────────────────────────
    # Type 3: Extended Distance Band
    # ──────────────────────────────────────────────────────────

    def mine_type3_distance_band(self) -> List[Dict]:
        """
        Type 3: Top-1 neighbor와의 거리가 모호한 중간 구간에 속하는 경우.

        조건:
            - top1_distance ∈ [P75, P90]
            - "moderate match"와 "distant match" 경계
            - LLM이 가장 헷갈리는 구간

        Analyzer 관점:
            - 거리 신호가 명확하지 않은 경계 케이스
            → feature 값 자체를 정밀하게 해석하는 능력 학습

        Returns
        -------
        List[Dict] : type3 hard negative 레코드 목록
        """
        if self.dist_p75 == float('inf') or self.dist_p90 == float('inf'):
            logger.warning(
                "type3_mining_skipped",
                reason="distance percentiles 계산 실패로 Type 4 mining을 건너뜁니다.",
            )
            print("[Type 4] SKIPPED — distance percentiles 없음")
            return []

        gt_clusters = self.gt_data["gt_cluster_id"]
        gt_labels   = self.gt_data["label"]
        top5_gt     = self.gt_data["gt_top5_indices"]

        records   = []
        n_checked = 0
        n_errors  = 0

        for pred_pos, (test_idx, pred) in enumerate(
            zip(self.sample_indices, self.predictions)
        ):
            if pred.get("error"):
                n_errors += 1
                continue

            n_checked += 1
            gt_cluster        = int(gt_clusters[test_idx])
            predicted_cluster = self._extract_primary_cluster_id(pred)
            gt_label          = int(gt_labels[test_idx])

            top1_dist = float(pred.get("top1_distance", pred.get("top_1_distance", -1.0)))
            
            # ── Type 4 조건: P75 ≤ top1_distance ≤ P90 ──────────
            if top1_dist < self.dist_p75 or top1_dist > self.dist_p90:
                continue

            feat         = self._get_feature_row(test_idx)
            retrieved_oi = pred.get("retrieved_original_indices", [])

            records.append({
                # 인덱스
                "test_idx":                   test_idx,
                "pred_pos":                   pred_pos,
                # GT
                "gt_label":                   gt_label,
                "gt_cluster_id":            gt_cluster,
                "top5_indices":               top5_gt[test_idx].tolist(),
                # Router 결과
                "predicted_cluster":          predicted_cluster,
                "top_k_clusters":             pred.get("top_k_clusters", []),
                "match_type":                 pred.get("match_type", "unknown"),
                # Retriever 결과
                "retrieved_original_indices": retrieved_oi,
                "top1_distance":              top1_dist,
                # Type 4 전용
                "dist_p75":                   self.dist_p75,
                "dist_p90":                   self.dist_p90,
                # 원본 피처
                **{f"feat_{k}": v for k, v in feat.items()},
                # Mining 메타데이터
                "mining_type":                "type3_distance_band",
                "experiment_case":            self.experiment_case,
                "percolate_version":          self.percolate_version,
            })

        band_rate = len(records) / max(n_checked, 1)
        logger.info(
            "type3_mining_complete",
            n_checked=n_checked,
            n_errors=n_errors,
            n_type3=len(records),
            distance_band_rate=round(band_rate, 4),
            p75=round(self.dist_p75, 4),
            p90=round(self.dist_p90, 4),
        )
        print(
            f"[Type 4] n_checked={n_checked}  distance_band={len(records)}  "
            f"rate={band_rate:.1%}  (P75={self.dist_p75:.3f}, P90={self.dist_p90:.3f})"
        )
        return records


    # ──────────────────────────────────────────────────────────
    # Type 4: Geometric Mismatch Zone
    # ──────────────────────────────────────────────────────────

    # def mine_type4_geometric_mismatch(self) -> List[Dict]:
    #     """
    #     Type 4: Router는 정확하지만 test sample이 클러스터 중심에서 매우 먼 경우.

    #     조건:
    #         - gt_cluster_id == predicted_cluster (Router 정확)
    #         - distance(test_emb, cluster_centroid) > P90 threshold

    #     Analyzer 관점:
    #         - 올바른 cluster persona를 받지만
    #         - 클러스터 내부에서 이질적인 샘플 (intra-cluster outlier)
    #         → 클러스터 통계와 다른 패턴을 어떻게 해석할지 학습

    #     Returns
    #     -------
    #     List[Dict] : type4 hard negative 레코드 목록
    #     """
    #     if self.test_emb is None or self.cluster_centroids is None:
    #         logger.warning(
    #             "type4_mining_skipped",
    #             reason="test_emb 또는 cluster_centroids가 없어 Type 3 mining을 건너뜁니다.",
    #         )
    #         print("[Type 3] SKIPPED — test_emb or cluster_centroids 없음")
    #         return []

    #     gt_clusters = self.gt_data["gt_cluster_id"]
    #     gt_labels   = self.gt_data["label"]
    #     top5_gt     = self.gt_data["gt_top5_indices"]

    #     records      = []
    #     n_checked    = 0
    #     n_routing_ok = 0
    #     n_errors     = 0

    #     for pred_pos, (test_idx, pred) in enumerate(
    #         zip(self.sample_indices, self.predictions)
    #     ):
    #         if pred.get("error"):
    #             n_errors += 1
    #             continue

    #         n_checked += 1
    #         gt_cluster        = int(gt_clusters[test_idx])
    #            #             "mining_type":                "type4_geometric_mismatch",
    #             "experiment_case":            self.experiment_case,
    #             "percolate_version":          self.percolate_version,
    #         })

    #     mismatch_rate = len(records) / max(n_routing_ok, 1)
    #     logger.info(
    #         "type4_mining_complete",
    #         n_checked=n_checked,
    #         n_routing_ok=n_routing_ok,
    #         n_errors=n_errors,
    #         n_type4=len(records),
    #         geometric_mismatch_rate=round(mismatch_rate, 4),
    #     )
    #     print(
    #         f"[Type 4] n_routing_ok={n_routing_ok}  geometric_mismatch={len(records)}  "
    #         f"rate={mismatch_rate:.1%}"
    #     )
    #     return recordspredicted_cluster = self._extract_primary_cluster_id(pred)
    #         gt_label          = int(gt_labels[test_idx])

    #         # ── Type 4 전제: routing이 올바른 케이스만 ──────────
    #         if gt_cluster != predicted_cluster or predicted_cluster == -1:
    #             continue
    #         n_routing_ok += 1

    #         # 클러스터 중심 벡터 확인
    #         centroid = self.cluster_centroids.get(gt_cluster)
    #         if centroid is None:
    #             continue

    #         # test embedding과 클러스터 중심 간 거리 계산
    #         test_vec = self.test_emb[test_idx]
    #         dist_to_centroid = float(np.linalg.norm(test_vec - centroid))

    #         # P90 이상인 경우만 Type 3로 분류
    #         if dist_to_centroid <= self.dist_p90:
    #             continue

    #         feat         = self._get_feature_row(test_idx)
    #         retrieved_oi = pred.get("retrieved_original_indices", [])

    #         records.append({
    #             # 인덱스
    #             "test_idx":                   test_idx,
    #             "pred_pos":                   pred_pos,
    #             # GT
    #             "gt_label":                   gt_label,
    #             "gt_cluster_id":            gt_cluster,
    #             "top5_indices":               top5_gt[test_idx].tolist(),
    #             # Router 결과 (routing 정확)
    #             "predicted_cluster":          predicted_cluster,
    #             "top_k_clusters":             pred.get("top_k_clusters", []),
    #             "match_type":                 pred.get("match_type", "unknown"),
    #             # Retriever 결과
    #             "retrieved_original_indices": retrieved_oi,
    #             "top1_distance":              float(pred.get(
    #                 "top1_distance", pred.get("top_1_distance", -1.0)
    #             )),
    #             # Type 4 전용
    #             "dist_to_centroid":           dist_to_centroid,
    #             "centroid_threshold_p90":     self.dist_p90,
    #             # 원본 피처
    #             **{f"feat_{k}": v for k, v in feat.items()},
    #             # Mining 메타데이터
 
    # ──────────────────────────────────────────────────────────
    # Public: run
    # ──────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Type 1 + Type 2 mining 실행 → DataFrame 반환.

        list 타입 컬럼(top5_indices 등)은 JSON string으로 직렬화하여
        parquet 저장 호환성 확보. SFT notebook에서 json.loads()로 복원 가능.

        Returns
        -------
        pd.DataFrame
        """
        print("\n" + "=" * 60)
        print(f"[HardNegativeMiner] {self.experiment_case} / query_{self.percolate_version}")
        print(f"  n_sampled = {len(self.sample_indices)}")
        print("=" * 60)

        # 수정 후
        type1_records = self.mine_type1_misrouting()
        type2_records = self.mine_type2_cross_cluster()
        # type3_records = self.mine_type3_geometric_mismatch()  # ← 추가
        type3_records = self.mine_type3_distance_band()       # ← 추가

        all_records = type1_records + type2_records + type3_records  # ← 수정

        if not all_records:
            logger.warning("no_hard_negatives_found")
            print("[HardNegativeMiner] ⚠️  Hard negative를 찾지 못했습니다.")
            return pd.DataFrame()

        df = pd.DataFrame(all_records)

        # list/dict 컬럼 → JSON string (parquet 저장 및 GCS 업로드 안정성)
        LIST_COLS = (
            "top5_indices", "retrieved_original_indices",
            "top_k_clusters", "neighbor_cluster_ids", "cross_cluster_ids",
        )
        for col in LIST_COLS:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
                )
        n_t1      = int((df["mining_type"] == "type1_misrouted").sum())
        n_t2      = int((df["mining_type"] == "type2_cross_cluster").sum())
        n_t3      = int((df["mining_type"] == "type3_distance_band").sum())  # ← 추가
        # n_t4      = int((df["mining_type"] == "type4_distance_band").sum())       # ← 추가
        n_normal  = int((df["gt_label"] == 0).sum())
        n_fraud   = int((df["gt_label"] == 1).sum())

        # 2. 중복 통계 계산 (추가된 로직)
        set_t1 = set(r['test_idx'] for r in type1_records)
        set_t2 = set(r['test_idx'] for r in type2_records)
        set_t3 = set(r['test_idx'] for r in type3_records)

        # 멤버 변수에 통계 저장 (save 메서드에서 활용)
        self.overlap_stats = {
            "1&2": len(set_t1 & set_t2),
            "1&3": len(set_t1 & set_t3),
            "2&3": len(set_t2 & set_t3),
            "all": len(set_t1 & set_t2 & set_t3),
            "total_unique": len(set_t1 | set_t2 | set_t3)
        }

        # print(f"\n[Mining 결과]  total={len(df)}  type1={n_t1}  type2={n_t2}  type3={n_t3}  type4={n_t4}")  # ← 수정
        print(f"\n[Mining 결과]  total={len(df)}  type1={n_t1}  type2={n_t2}  type3={n_t3}")  # ← 수정

        print(f"  label: normal={n_normal}  fraud={n_fraud}")

        logger.info(
            "mining_complete",
            total=len(df),
            type1=n_t1,
            type2=n_t2,
            type3=n_t3,  # ← 추가
            # type4=n_t4,  # ← 추가
            n_normal=n_normal,
            n_fraud=n_fraud,
        )
        return df

    # ──────────────────────────────────────────────────────────
    # Save: local + GCS
    # ──────────────────────────────────────────────────────────

    def save(
        self,
        df:         pd.DataFrame,
        output_dir: str,
        upload_gcs: bool = True,
    ) -> str:
        """
        Mining 결과를 로컬 parquet + summary.txt 로 저장하고 GCS 업로드.

        파일명 형식:
            hard_negatives_{experiment_case}_query_{percolate_version}_pct_{coreset_pct}_{MMDD}.parquet

        Parameters
        ----------
        df         : run() 반환값
        output_dir : 로컬 저장 디렉토리
        upload_gcs : GCS 업로드 여부

        Returns
        -------
        str : 로컬 parquet 경로
        """
        if df.empty:
            print("[HardNegativeMiner] 저장할 데이터가 없습니다.")
            return ""

        out      = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%m%d")

        fname = (
            f"hard_negatives_"
            f"{self.experiment_case}_"
            f"query_{self.percolate_version}_"
            f"pct_{self.coreset_percentage}_"
            f"{date_str}.parquet"
        )
        local_parquet = out / fname
        local_summary = out / fname.replace(".parquet", "_summary.txt")

        # parquet 저장
        df.to_parquet(str(local_parquet), index=False)
        print(f"\n✅ [save] parquet: {local_parquet}")

        # summary 저장
        self._write_summary(df, local_summary)

        # GCS 업로드
        if upload_gcs and GCS_AVAILABLE:
            gcs_base = GCS_MINING_PATH_TEMPLATE.format(
                experiment_case=self.experiment_case,
                percolate_version=self.percolate_version,
                coreset_pct=self.coreset_percentage,
                mining_date=date_str,
            )
            self._upload_to_gcs(str(local_parquet), gcs_base)
            self._upload_to_gcs(
                str(local_summary),
                gcs_base.replace(".parquet", "_summary.txt"),
            )

        return str(local_parquet)

    # ──────────────────────────────────────────────────────────
    # Private: summary / upload
    # ──────────────────────────────────────────────────────────

    def _write_summary(self, df: pd.DataFrame, path: Path):
        """Mining 통계 요약 텍스트 저장."""
        n_t1     = int((df["mining_type"] == "type1_misrouted").sum())
        n_t2     = int((df["mining_type"] == "type2_cross_cluster").sum())
        n_t3      = int((df["mining_type"] == "type3_distance_band").sum())  # ← 추가
        n_sample = len(self.sample_indices)

        with open(path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("HARD NEGATIVE MINING SUMMARY V1.0\n")
            f.write("=" * 70 + "\n")
            f.write(f"experiment_case   : {self.experiment_case}\n")
            f.write(f"percolate_version : {self.percolate_version}\n")
            f.write(f"coreset_percentage: {self.coreset_percentage}\n")
            f.write(f"n_sample_indices  : {n_sample}  (전체 test set의 앞 10%)\n")
            f.write(f"\n총 hard negatives : {len(df)}\n")

            f.write(f"\n── Type 1 (Router Misrouting) ──────────────────────────\n")
            f.write(f"  n = {n_t1}  ({n_t1/max(n_sample,1):.1%} of sampled)\n")
            f.write(f"  조건: gt_cluster_id != predicted_cluster\n")
            f.write(f"  Analyzer: 틀린 persona + 틀린 neighbors 수신\n")

            f.write(f"\n── Type 2 (Cross-Cluster Retrieval) ────────────────────\n")
            f.write(f"  n = {n_t2}\n")
            f.write(f"  조건: routing 정확 + retrieved neighbors 중 타 cluster 존재\n")
            if n_t2 > 0 and "n_cross_cluster_neighbors" in df.columns:
                t2_df = df[df["mining_type"] == "type2_cross_cluster"]
                f.write(
                    f"  avg_cross_neighbors = "
                    f"{t2_df['n_cross_cluster_neighbors'].mean():.2f}\n"
                )

            f.write(f"\n── Type 3 (type3_distance_band) ──────────────────────────\n")
            f.write(f"  n = {n_t3}\n")
            f.write(f"  조건:- top1_distance ∈ [P75, P90]\n")
            f.write(f"  moderate match와 distant match 경계\n")

            f.write(f"\n── Label 분포 ──────────────────────────────────────────\n")
            f.write(f"  normal (label=0): {int((df['gt_label']==0).sum())}\n")
            f.write(f"  fraud  (label=1): {int((df['gt_label']==1).sum())}\n")

            if "top1_distance" in df.columns:
                f.write(f"\n── Top-1 L2 Distance ───────────────────────────────────\n")
                valid = df["top1_distance"][df["top1_distance"] > 0]
                if len(valid) > 0:
                    f.write(
                        f"  mean={valid.mean():.4f}  "
                        f"min={valid.min():.4f}  "
                        f"max={valid.max():.4f}\n"
                    )
            # save 메서드 내부 요약 파일 작성 부분
            f.write(f"\n── Type Overlap Analysis ────────────────────────────────\n")
            unique_cnt = self.overlap_stats["total_unique"]
            f.write(f"  Total Unique Indices: {unique_cnt}\n")
            f.write(f"  Overlap Type 1 & 2: {self.overlap_stats['1&2']} samples\n")
            f.write(f"  Overlap Type 1 & 3: {self.overlap_stats['1&3']} samples\n")
            f.write(f"  Overlap Type 2 & 3: {self.overlap_stats['2&3']} samples\n")
            f.write(f"  Overlap All Types : {self.overlap_stats['all']} samples\n")

            f.write("=" * 70 + "\n")

        print(f"✅ [save] summary: {path}")

    def _upload_to_gcs(self, local_path: str, gcs_path: str):
        """GCS 업로드 (batch_inference.upload_output_to_gcs와 동일 버킷)."""
        try:
            client = storage.Client()
            bucket = client.bucket(GCS_BUCKET_NAME)
            blob   = bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            print(f"✅ Uploaded: gs://{GCS_BUCKET_NAME}/{gcs_path}")
        except Exception as e:
            print(f"❌ GCS upload failed ({gcs_path}): {e}")
