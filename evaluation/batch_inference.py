"""
Batch Inference & Evaluation - V5.0
=====================================

V4.1 → V5.0 변경사항:

  ★ Router v5 response 반영
    - top_k_clusters: [{cluster_id, rank, score}]  (purity/support/leaf_id 제거)
    - vec_index: router가 결정 → retriever에 직접 전달
    - match_type 유지

  ★ Retriever v5 interface 반영
    - 엔드포인트: /search → /retrieve  (/search-batch → /retrieve-batch)
    - request: test_vector → embedding, vec_index 추가
    - response: top_k_results → results
    - SearchResult: {es_doc_id, original_index, cluster_id, distance, score}
    - top_1_distance → top1_distance (필드명 통일)
    - page_fault_delta: 신규 (retriever 페이지폴트 추적)

  ★ Recall@5 original_index 기반으로 수정 (핵심 변경)
    - ES _id(es_doc_id) 대신 original_index 기반 비교
    - ES _id=[1,2,3,4,5], original_index=[100,200,300,400,500]이면
      inference 결과 = [100,200,300,400,500]
    - predictions 딕셔너리에 "retrieved_original_indices" 명시 저장

  ★ Orchestrator v5 response 반영
    - vec_index, page_fault_delta, skip_analyzer 필드 추가
    - top_1_distance → top1_distance
    - purity/support/leaf_id 제거

  ★ page_fault_delta 수집
    - latencies["page_fault_delta"] 추가
    - 통계(mean/p95/p99/n_nonzero) 별도 보고

  ★ SKIP_ANALYZER 지원
    - --skip-analyzer 플래그 추가
    - summary에 skip_analyzer 여부 표기

Usage:
    # Router + Retriever (recall@5 포함, 권장)
    python batch_inference.py \\
        --pipeline-mode router_retriever \\
        --experiment-case pca_64_k100 \\
        --percolate-version v2 \\
        --coreset-percentage 100 \\
        --batch-size 20 --n-samples 1000

    # Full pipeline
    python batch_inference.py \\
        --pipeline-mode full \\
        --experiment-case pca_64_k100 \\
        --batch-size 10 --n-samples 1000

    # SKIP_ANALYZER latency 격리 실험
    python batch_inference.py \\
        --pipeline-mode full \\
        --skip-analyzer \\
        --experiment-case pca_64_k100 \\
        --batch-size 20 --n-samples 1000
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import httpx
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
import structlog

from gt_loader import GTLoader, VALID_EXPERIMENT_CASES
from evaluation_metrics import MetricCalculator

from hard_negative_miner import HardNegativeMiner
MINER_AVAILABLE = True
# ★ Hard Negative Mining — V1.0 추가
try:
    from hard_negative_miner import HardNegativeMiner
    MINER_AVAILABLE = True
except ImportError:
    MINER_AVAILABLE = False

logger = structlog.get_logger()

class BatchInference:
    SERVICE_URLS = {
        "router":       os.getenv("ROUTER_URL",       "http://anomaly-router:80"),
        "retriever":    os.getenv("RETRIEVER_URL",    "http://anomaly-retriever:80"),
        "orchestrator": os.getenv("ORCHESTRATOR_URL", "http://anomaly-orchestrator:80"),
    }
    MAX_BATCH_SIZE = {"router": 100, "retriever": 100, "analyzer": 20}

    def __init__(
        self,
        pipeline_mode:         str  = "full",
        timeout:               int  = 600,
        experiment_case:       str  = "pca_64",
        percolate_version:     str  = "v2",
        coreset_percentage:    str  = "100",
        enable_internal_batch: bool = False,
        skip_analyzer:         bool = False,
    ):
        valid_modes = ["router_only", "router_retriever", "full"]
        if pipeline_mode not in valid_modes:
            raise ValueError(f"pipeline_mode must be one of {valid_modes}")

        self.pipeline_mode         = pipeline_mode
        self.timeout               = timeout
        self.experiment_case       = experiment_case
        self.percolate_version     = percolate_version
        self.coreset_percentage    = coreset_percentage
        self.enable_internal_batch = enable_internal_batch
        self.skip_analyzer         = skip_analyzer

        self.gt_loader = GTLoader(
            experiment_case=experiment_case,
            coreset_percentage=coreset_percentage,
        )
        self.metric_calculator = MetricCalculator()
        self.metric_calculator.set_experiment_case(experiment_case)
        self.metric_calculator.set_coreset_percentage(coreset_percentage)

        self.predictions: List[Dict] = []
        self.latencies = {
            "total":            [],
            "router":           [],
            "retriever":        [],
            "analyzer":         [],
            "page_fault_delta": [],   # ★ V5
        }

        self.diversity_list:    List[float] = []
        self.match_type_count:  Dict[str, int] = {
            "strict": 0, "centroid": 0, "skip_analyzer": 0, "error": 0
        }

        self.test_data  = None
        self.embeddings = None
        self.gt_data:   Dict = {}

        # ★ Hard Negative Mining — batch_inference()가 실제 실행한 test 인덱스 기록
        self.sample_indices: List[int] = []

        logger.info(
            "batch_inference_initialized_v5",
            pipeline_mode=pipeline_mode,
            experiment_case=experiment_case,
            percolate_version=percolate_version,
            coreset_percentage=coreset_percentage,
            skip_analyzer=skip_analyzer,
        )

    # ──────────────────────────────────────────────────────────
    # 데이터 로드
    # ──────────────────────────────────────────────────────────

    def load_data(self):
        """
        GTLoader V5.0 load_all() 호출.
        gt_data["top5_indices"] / ["top5_indices_coreset_adjusted"]:
          ★ V5: original_index 기반 (ES _id 아님)
        """
        self.test_data, self.embeddings, self.gt_data = self.gt_loader.load_all()
        logger.info(
            "data_loaded_v5",
            n_samples=len(self.test_data),
            embedding_dim=self.embeddings.shape[1],
            n_fraud=int((self.gt_data["label"] == 1).sum()),
            n_coreset_train=self.gt_data.get("n_coreset_train", 0),
            coreset_coverage_rate=self.gt_data.get("coreset_coverage_rate", 1.0),
        )

    # ──────────────────────────────────────────────────────────
    # Router 헤더
    # ──────────────────────────────────────────────────────────

    def _router_headers(self) -> Dict[str, str]:
        return {
            "Content-Type":     "application/json",
            "X-Router-Case":    self.experiment_case,
            "X-Router-Version": self.percolate_version,
        }

    # ──────────────────────────────────────────────────────────
    # Router response 공통 파싱 헬퍼
    # ──────────────────────────────────────────────────────────

    # @staticmethod
    # def _parse_router_response(r_result: Dict) -> Dict:
    #     """
    #     Router v5 response → 공통 파싱.
    #     top_k_clusters: [{cluster_id, rank, score}]  (purity/support/leaf_id 없음)
    #     """
    #     top_k_clusters     = r_result.get("top_k_clusters", [])
    #     cluster_ids        = [c["cluster_id"] for c in top_k_clusters]
    #     primary_cluster_id = (
    #         r_result.get("primary_cluster_id")
    #         or (cluster_ids[0] if cluster_ids else None)
    #     )
    #     return {
    #         "primary_cluster_id": primary_cluster_id,
    #         "cluster_ids":        cluster_ids,
    #         "top_k_clusters":     top_k_clusters,
    #         "vec_index":          r_result.get("vec_index", ""),
    #         "match_type":         r_result.get("match_type", "unknown"),
    #     }

    @staticmethod
    def _parse_router_response(r_result: Dict) -> Dict:
        """
        Router v5 response → 공통 파싱.
        top_k_clusters: [{cluster_id, rank, score}]  (purity/support/leaf_id 없음)
        ★ V5.1: cluster_distances 추가 (MRR, ε-Recall 계산용)
        """
        top_k_clusters     = r_result.get("top_k_clusters", [])
        cluster_ids        = [c["cluster_id"] for c in top_k_clusters]
        primary_cluster_id = (
            r_result.get("primary_cluster_id")
            or (cluster_ids[0] if cluster_ids else None)
        )
        
        # ★ 추가: 클러스터별 거리 정보 추출
        cluster_distances = {}
        for c in top_k_clusters:
            cid = c.get("cluster_id")
            # distance, centroid_distance, score 등 다양한 필드명 지원
            dist = c.get("score")
            if cid is not None and dist is not None:
                cluster_distances[cid] = float(dist)
        
        return {
            "primary_cluster_id": primary_cluster_id,
            "cluster_ids":        cluster_ids,
            "top_k_clusters":     top_k_clusters,
            "cluster_distances":  cluster_distances,  # ★ 추가
            "vec_index":          r_result.get("vec_index", ""),
            "match_type":         r_result.get("match_type", "unknown"),
        }

    @staticmethod
    def _parse_retriever_response(ret_result: Dict) -> Dict:
        """
        Retriever v5 response → 공통 파싱.
        ★ original_index 추출 (recall@5의 inference key)

        SearchResult: {es_doc_id, original_index, cluster_id, distance, score}
        ES _id(es_doc_id) 아닌 original_index를 recalled index로 사용.
        """
        results    = ret_result.get("results", [])
        pf_delta   = ret_result.get("page_fault_delta", 0)
        top1_dist  = ret_result.get("top1_distance", float("inf"))
        # ★ recall@5 핵심: original_index 추출
        retrieved_original_indices = [r["original_index"] for r in results]
        return {
            "results":                    results,
            "top1_distance":              top1_dist,
            "page_fault_delta":           pf_delta,
            "retrieved_original_indices": retrieved_original_indices,
        }

    # ──────────────────────────────────────────────────────────
    # Single Request Helpers
    # ──────────────────────────────────────────────────────────

    async def call_router_only(self, client, embedding) -> Optional[Dict]:
        """Router v5 단건 호출."""
        try:
            t = time.perf_counter()
            resp = await client.post(
                f"{self.SERVICE_URLS['router']}/route",
                json={"embedding": embedding},
                headers=self._router_headers(),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            r_result = resp.json()
            latency  = (time.perf_counter() - t) * 1000

            parsed = self._parse_router_response(r_result)
            return {
                **parsed,
                "latency_ms": r_result.get("latency_ms", latency),
            }
        except httpx.TimeoutException:
            logger.warning("router_timeout")
            return None
        except Exception as e:
            logger.error("router_call_failed", error=str(e))
            return None

    async def call_router_retriever(self, client, embedding) -> Optional[Dict]:
        """
        Router v5 → Retriever v5 단건 sequential 호출.

        ★ V5 변경:
          - Retriever endpoint: /retrieve
          - Request: {embedding, cluster_ids, vec_index, top_k}
          - Response: results (SearchResult), top1_distance, page_fault_delta
          - retrieved_original_indices: recall@5용 original_index 리스트
        """
        try:
            # ── Router ──────────────────────────────────────────
            t1 = time.perf_counter()
            r_resp = await client.post(
                f"{self.SERVICE_URLS['router']}/route",
                json={"embedding": embedding},
                headers=self._router_headers(),
                timeout=self.timeout,
            )
            r_resp.raise_for_status()
            router_parsed  = self._parse_router_response(r_resp.json())
            router_latency = (time.perf_counter() - t1) * 1000

            if router_parsed["primary_cluster_id"] is None:
                logger.warning("router_no_cluster_id")
                return None

            # ── Retriever ────────────────────────────────────────
            # ★ V5: embedding (not test_vector), vec_index 추가
            t2 = time.perf_counter()
            ret_resp = await client.post(
                f"{self.SERVICE_URLS['retriever']}/retrieve",
                json={
                    "embedding":   embedding,
                    "cluster_ids": router_parsed["cluster_ids"],
                    "vec_index":   router_parsed["vec_index"],
                    "top_k":       5,
                },
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            ret_resp.raise_for_status()
            ret_parsed        = self._parse_retriever_response(ret_resp.json())
            retriever_latency = (time.perf_counter() - t2) * 1000

            return {
                **router_parsed,
                **ret_parsed,
                "latency_ms": {
                    "router":    router_latency,
                    "retriever": retriever_latency,
                    "total":     router_latency + retriever_latency,
                },
            }

        except httpx.TimeoutException:
            logger.warning("router_retriever_timeout")
            return None
        except Exception as e:
            logger.error("router_retriever_failed", error=str(e))
            return None

    async def call_full_pipeline(self, client, test_case) -> Optional[Dict]:
        """
        Orchestrator v5 단건 호출.

        ★ V5 response: vec_index, page_fault_delta, skip_analyzer,
                        top_1_distance (orchestrator 필드명 유지), top_5_clusters
        ★ NOTE: orchestrator는 results(retrieved docs)를 response에 포함하지 않으므로
                full mode에서는 recall@5 계산 불가.
                recall@5가 필요하면 router_retriever mode 사용 권장.
        """
        try:
            resp = await client.post(
                f"{self.SERVICE_URLS['orchestrator']}/detect",
                json=test_case,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            result = resp.json()

            # ★ orchestrator v5 필드명 통일: top_1_distance → top1_distance
            if "top_1_distance" in result and "top1_distance" not in result:
                result["top1_distance"] = result["top_1_distance"]

            # primary_cluster_id 정규화
            if "primary_cluster_id" not in result:
                top5 = result.get("top_5_clusters", [])
                result["primary_cluster_id"] = top5[0]["cluster_id"] if top5 else -1

            return result

        except httpx.TimeoutException:
            logger.warning("orchestrator_timeout")
            return None
        except Exception as e:
            logger.error("orchestrator_failed", error=str(e))
            return None

    # ──────────────────────────────────────────────────────────
    # Batch Request Helpers
    # ──────────────────────────────────────────────────────────

    async def call_router_batch(self, client, embeddings) -> List[Optional[Dict]]:
        try:
            resp = await client.post(
                f"{self.SERVICE_URLS['router']}/route-batch",
                json=[{"embedding": emb} for emb in embeddings],
                headers=self._router_headers(),
                timeout=self.timeout * 2,
            )
            resp.raise_for_status()
            raw_results = resp.json()
            return [
                self._parse_router_response(r) if (r and not r.get("error")) else None
                for r in raw_results
            ]
        except Exception as e:
            logger.error("router_batch_failed", error=str(e))
            return [None] * len(embeddings)

    async def call_retriever_batch(self, client, requests) -> List[Optional[Dict]]:
        """
        ★ V5: /retrieve-batch  각 request: {embedding, cluster_ids, vec_index, top_k}
        """
        try:
            resp = await client.post(
                f"{self.SERVICE_URLS['retriever']}/retrieve-batch",
                json=requests,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout * 2,
            )
            resp.raise_for_status()
            raw_results = resp.json()
            return [
                self._parse_retriever_response(r) if (r and not r.get("error")) else None
                for r in raw_results
            ]
        except Exception as e:
            logger.error("retriever_batch_failed", error=str(e))
            return [None] * len(requests)

    async def call_router_retriever_batch(self, client, embeddings) -> List[Dict]:
        """Router batch → Retriever batch 연결 (v5 interface)."""
        router_results = await self.call_router_batch(client, embeddings)

        retriever_reqs, valid_indices = [], []
        for i, (emb, rr) in enumerate(zip(embeddings, router_results)):
            if rr and rr["primary_cluster_id"] is not None:
                retriever_reqs.append({
                    "embedding":   emb,
                    "cluster_ids": rr["cluster_ids"],
                    "vec_index":   rr["vec_index"],
                    "top_k":       5,
                })
                valid_indices.append(i)

        ret_results = (
            await self.call_retriever_batch(client, retriever_reqs)
            if retriever_reqs else []
        )

        combined, ret_idx = [], 0
        for i, rr in enumerate(router_results):
            if i in valid_indices and ret_idx < len(ret_results):
                ret = ret_results[ret_idx]; ret_idx += 1
                if ret:
                    combined.append({
                        **rr,
                        **ret,
                        "latency_ms": {
                            "router":    0,  # batch: 개별 측정 불가
                            "retriever": 0,
                            "total":     0,
                        },
                    })
                else:
                    combined.append({"error": True, "reason": "retriever_failed"})
            else:
                combined.append({"error": True, "reason": "router_failed"})
        return combined

    # ──────────────────────────────────────────────────────────
    # Main Batch Inference
    # ──────────────────────────────────────────────────────────

    async def batch_inference(self, n_samples=None, batch_size=10, log_interval_pct: float = 5.0):
        if self.test_data is None:
            raise ValueError("load_data() 를 먼저 호출하세요.")

        n_total   = len(self.test_data)
        n_samples = min(n_samples, n_total) if n_samples else n_total
        predictions: List[Dict] = []

        n_batches       = (n_samples + batch_size - 1) // batch_size
        log_every_n     = max(1, int(n_batches * log_interval_pct / 100))
        t_start         = time.perf_counter()
        n_errors_so_far = 0

        mode_label = (
            f"{self.pipeline_mode}"
            f"{'/skip_analyzer' if self.skip_analyzer else ''}"
            f"/{'batch' if self.enable_internal_batch else 'async'}"
        )
        print(f"[inference] START  mode={mode_label}  n_samples={n_samples:,}  "
              f"batch_size={batch_size}  n_batches={n_batches}", flush=True)

        async with httpx.AsyncClient() as client:
            for batch_idx, i in enumerate(range(0, n_samples, batch_size)):
                end     = min(i + batch_size, n_samples)
                indices = list(range(i, end))

                batch_embeddings = [self.embeddings[idx].tolist() for idx in indices]
                test_cases = [
                    self.gt_loader.get_test_case(
                        idx, self.test_data, self.embeddings,
                        percolate_version=self.percolate_version,
                        experiment_case=self.experiment_case,
                    )
                    for idx in indices
                ]

                if self.enable_internal_batch:
                    if self.pipeline_mode == "router_only":
                        batch_results = await self.call_router_batch(client, batch_embeddings)
                    elif self.pipeline_mode == "router_retriever":
                        batch_results = await self.call_router_retriever_batch(client, batch_embeddings)
                    else:
                        logger.warning("full_mode_no_batch_fallback_async")
                        batch_results = await asyncio.gather(
                            *[self.call_full_pipeline(client, tc) for tc in test_cases]
                        )
                else:
                    if self.pipeline_mode == "router_only":
                        tasks = [self.call_router_only(client, emb) for emb in batch_embeddings]
                    elif self.pipeline_mode == "router_retriever":
                        tasks = [self.call_router_retriever(client, emb) for emb in batch_embeddings]
                    else:
                        tasks = [self.call_full_pipeline(client, tc) for tc in test_cases]
                    batch_results = await asyncio.gather(*tasks)

                for result in batch_results:
                    if result and not result.get("error"):
                        predictions.append(result)

                        # ★ 추가: top5 diversity
                        top5_ids  = [c["cluster_id"] for c in result.get("top_k_clusters", [])]
                        diversity = len(set(top5_ids)) / len(top5_ids) if top5_ids else 0.0
                        self.diversity_list.append(diversity)

                        # ★ 추가: match_type 카운트
                        mt = result.get("match_type", "unknown")
                        if mt in self.match_type_count:
                            self.match_type_count[mt] += 1

                        lat = result.get("latency_ms")
                        if isinstance(lat, dict):
                            self.latencies["total"].append(lat.get("total", 0))
                            self.latencies["router"].append(lat.get("router", 0))
                            self.latencies["retriever"].append(lat.get("retriever", 0))
                            self.latencies["analyzer"].append(lat.get("analyzer", 0))
                        elif isinstance(lat, (int, float)):
                            self.latencies["router"].append(lat)
                            self.latencies["total"].append(lat)
                        # ★ V5: page_fault_delta 수집
                        self.latencies["page_fault_delta"].append(
                            result.get("page_fault_delta", 0)
                        )
                    else:
                        n_errors_so_far += 1
                        predictions.append({"error": True, "pipeline_mode": self.pipeline_mode})

                # ── 진행률 로그 ──────────────────────────────────
                done_samples = end
                pct          = done_samples / n_samples * 100
                if (batch_idx % log_every_n == 0) or (done_samples >= n_samples):
                    elapsed     = time.perf_counter() - t_start
                    rps         = done_samples / elapsed if elapsed > 0 else 0
                    remaining_s = (n_samples - done_samples) / rps if rps > 0 else float("inf")
                    lat_str = ""
                    if self.latencies["total"]:
                        recent = self.latencies["total"][-min(50, len(self.latencies["total"])):]
                        lat_str = f"  lat_avg={sum(recent)/len(recent):.0f}ms"
                    pf_str = ""
                    if self.latencies["page_fault_delta"]:
                        recent_pf = self.latencies["page_fault_delta"][-min(50, len(self.latencies["page_fault_delta"])):]
                        pf_str = f"  pf_avg={sum(recent_pf)/len(recent_pf):.1f}"
                    eta_str = (
                        f"{int(remaining_s//60)}m{int(remaining_s%60)}s"
                        if remaining_s != float("inf") else "?"
                    )
                    print(
                        f"[inference] {pct:5.1f}%  done={done_samples:>6,}/{n_samples:,}  "
                        f"batch={batch_idx+1:>4}/{n_batches}  errors={n_errors_so_far:>4}  "
                        f"elapsed={elapsed:.0f}s  ETA={eta_str}{lat_str}{pf_str}",
                        flush=True
                    )

        self.predictions = predictions
        # ★ Hard Negative Mining: 이번 inference에서 사용한 test 인덱스 기록
        self.sample_indices = list(range(n_samples))
        elapsed_total = time.perf_counter() - t_start
        print(
            f"[inference] DONE   n={len(predictions):,}  errors={n_errors_so_far:,}  "
            f"total={elapsed_total:.1f}s  avg_rps={n_samples/elapsed_total:.2f}",
            flush=True
        )
        return predictions

    # ──────────────────────────────────────────────────────────
    # Metric 추출 헬퍼
    # ──────────────────────────────────────────────────────────

    def _extract_primary_cluster_id(self, pred: Dict) -> int:
        if "primary_cluster_id" in pred:
            return pred["primary_cluster_id"]
        for key in ("top_k_clusters", "top_5_clusters"):
            clusters = pred.get(key, [])
            if clusters:
                return clusters[0]["cluster_id"]
        return -1

    def _extract_top1_distance(self, pred: Dict) -> float:
        """top1_distance 필드명 통일 (v5: top1_distance, orchestrator: top_1_distance)."""
        return pred.get("top1_distance", pred.get("top_1_distance", float("inf")))

    # ──────────────────────────────────────────────────────────
    # Metrics 계산
    # ──────────────────────────────────────────────────────────

    def calculate_metrics(self) -> Dict:
        n_valid     = len(self.predictions)
        gt_labels   = self.gt_data["label"][:n_valid]
        gt_clusters = self.gt_data["gt_cluster_id"][:n_valid]

        gt_top5_raw = self.gt_data["gt_top5_indices"][:n_valid]                    # top5_raw → gt_top5_indices
        gt_top5_adj = self.gt_data["gt_top5_indices_coreset_adjusted"][:n_valid]   # top5_adj → gt_top5_indices_coreset_adjusted

        gt_top5_cluster_ids = self.gt_data["gt_top5_cluster_ids"][:n_valid]    # PCA 공간 Top-5 클러스터 ID
        gt_top5_cluster_dist = self.gt_data["gt_top5_cluster_dist"][:n_valid]  # PCA 공간 Top-5 거리
        gt_top5_dist = self.gt_data["gt_top5_dist"][:n_valid]

        metrics = {
            "pipeline_mode":         self.pipeline_mode,
            "skip_analyzer":         self.skip_analyzer,
            "enable_internal_batch": self.enable_internal_batch,
            "n_samples":             n_valid,
            "n_errors":              len([p for p in self.predictions if p.get("error", False)]),
            "experiment_case":       self.experiment_case,
            "percolate_version":     self.percolate_version,
            "coreset_percentage":    self.coreset_percentage,
            "n_coreset_train":       self.gt_data.get("n_coreset_train", 0),
            "coreset_coverage_rate": self.gt_data.get("coreset_coverage_rate", 1.0),
        }

        # ★ V5: page_fault_delta 통계
        pf_deltas = self.latencies["page_fault_delta"]
        if pf_deltas:
            pf_arr = np.array(pf_deltas)
            metrics["page_fault_stats"] = {
                "mean":      float(np.mean(pf_arr)),
                "p95":       float(np.percentile(pf_arr, 95)),
                "p99":       float(np.percentile(pf_arr, 99)),
                "max":       float(np.max(pf_arr)),
                "n_nonzero": int(np.sum(pf_arr > 0)),
            }

        if self.pipeline_mode == "router_only":
            pred_clusters = np.array([self._extract_primary_cluster_id(p) for p in self.predictions])
            metrics["routing_accuracy"] = self.metric_calculator.calculate_cluster_assignment_accuracy(
                gt_clusters, pred_clusters)
            metrics["router_metrics"] = self.metric_calculator.calculate_router_metrics(
                self.predictions, gt_clusters)
            if self.latencies["router"]:
                lat = np.array(self.latencies["router"])
                metrics["latency"] = {
                    "mean": float(np.mean(lat)),
                    "p95":  float(np.percentile(lat, 95)),
                    "p99":  float(np.percentile(lat, 99)),
                }
            # ★ 추가: top5 diversity + match_type 통계
            if self.diversity_list:
                metrics["top5_diversity"] = {
                    "mean": round(float(np.mean(self.diversity_list)), 4),
                    "min":  round(float(np.min(self.diversity_list)),  4),
                }
            total_mt = sum(self.match_type_count.values())
            metrics["match_type_stats"] = {
                "counts": self.match_type_count,
                "strict_ratio":   round(self.match_type_count["strict"]   / total_mt, 3) if total_mt else 0,
                "centroid_ratio": round(self.match_type_count["centroid"] / total_mt, 3) if total_mt else 0,
            }

        elif self.pipeline_mode == "router_retriever":
            pred_clusters = np.array([self._extract_primary_cluster_id(p) for p in self.predictions])
            metrics["routing_accuracy"] = self.metric_calculator.calculate_cluster_assignment_accuracy(
                gt_clusters, pred_clusters)
            metrics["router_metrics"] = self.metric_calculator.calculate_router_metrics(
                self.predictions, gt_clusters)
            metrics["distance_based_anomaly"] = self.metric_calculator.calculate_distance_based_auroc(
                self.predictions, gt_labels)

            # ★ V5 Dual Recall@5 (original_index 기반)
            gt_top5_raw = self.gt_data["gt_top5_indices"][:n_valid]              # original_index 기반
            gt_top5_adj = self.gt_data["gt_top5_indices_coreset_adjusted"][:n_valid]
            metrics["retrieval_metrics"] = self.metric_calculator.calculate_retrieval_recall_dual(
                predictions=self.predictions,
                gt_top5_raw=gt_top5_raw,
                gt_top5_coreset_adjusted=gt_top5_adj,
                coreset_coverage_rate=self.gt_data.get("coreset_coverage_rate", 1.0),
                n_coreset_train=self.gt_data.get("n_coreset_train", 0),
            )

            # Router MRR
            if gt_clusters is not None:
                mrr_result = self.metric_calculator.calculate_router_mrr(
                    self.predictions, gt_clusters
                )
                metrics["router_mrr"] = mrr_result
                logger.info("router_mrr_computed", **mrr_result)
            
            # Candidate Recall (Top-5 Cluster IDs)
            if gt_top5_cluster_ids is not None:
                candidate_result = self.metric_calculator.calculate_candidate_recall(
                    self.predictions, gt_top5_cluster_ids[:n_valid]
                )
                metrics["candidate_recall"] = candidate_result
                logger.info("candidate_recall_computed", **candidate_result)
            
            # Routing ε-Recall (10%)
            if gt_clusters is not None and gt_top5_cluster_dist is not None:
                routing_eps_result = self.metric_calculator.calculate_routing_epsilon_recall(
                    self.predictions, gt_clusters, gt_top5_cluster_dist[:n_valid], epsilon=0.1
                )
                metrics["routing_epsilon_recall"] = routing_eps_result
                logger.info("routing_epsilon_recall_computed", **routing_eps_result)
            
            # Coreset ε-Recall (10%)
            if gt_top5_dist is not None:
                coreset_eps_result = self.metric_calculator.calculate_coreset_epsilon_recall(
                    self.predictions, gt_top5_dist[:n_valid], epsilon=0.1
                )
                metrics["coreset_epsilon_recall"] = coreset_eps_result
                logger.info("coreset_epsilon_recall_computed", **coreset_eps_result)

            if self.latencies["total"]:
                total_lat = np.array(self.latencies["total"])
                r_lat     = np.array(self.latencies["router"])
                ret_lat   = np.array(self.latencies["retriever"])
                metrics["latency"] = {
                    "total_p95":      float(np.percentile(total_lat, 95)),
                    "total_p99":      float(np.percentile(total_lat, 99)),
                    "router_mean":    float(np.mean(r_lat)),
                    "retriever_mean": float(np.mean(ret_lat)),
                    "retriever_p95":  float(np.percentile(ret_lat, 95)),
                }
             # ★ 추가: top5 diversity + match_type 통계
            if self.diversity_list:
                metrics["top5_diversity"] = {
                    "mean": round(float(np.mean(self.diversity_list)), 4),
                    "min":  round(float(np.min(self.diversity_list)),  4),
                }
            total_mt = sum(self.match_type_count.values())
            metrics["match_type_stats"] = {
                "counts": self.match_type_count,
                "strict_ratio":   round(self.match_type_count["strict"]   / total_mt, 3) if total_mt else 0,
                "centroid_ratio": round(self.match_type_count["centroid"] / total_mt, 3) if total_mt else 0,
            }

        else:  # full
            pred_clusters = np.array([self._extract_primary_cluster_id(p) for p in self.predictions])
            pred_confs    = np.array([p.get("confidence", 50) for p in self.predictions])
            pred_labels   = np.array([
                1 if p.get("classification") == "ABNORMAL" else 0
                for p in self.predictions
            ])
            metrics["anomaly_detection"] = self.metric_calculator.calculate_anomaly_auroc(
                gt_labels, pred_confs, pred_labels)
            metrics["routing_accuracy"] = self.metric_calculator.calculate_cluster_assignment_accuracy(
                gt_clusters, pred_clusters)
            metrics["router_metrics"] = self.metric_calculator.calculate_router_metrics(
                self.predictions, gt_clusters)
            metrics["distance_based_anomaly"] = self.metric_calculator.calculate_distance_based_auroc(
                self.predictions, gt_labels)

            # ★ full mode: orchestrator가 results를 전달하지 않으므로 recall 불가
            #   "retrieved_original_indices"가 있는 경우만 계산 (future extension용)
            if any("retrieved_original_indices" in p for p in self.predictions):
                gt_top5_raw = self.gt_data["gt_top5_indices"][:n_valid]
                gt_top5_adj = self.gt_data["gt_top5_indices_coreset_adjusted"][:n_valid]
                metrics["retrieval_metrics"] = self.metric_calculator.calculate_retrieval_recall_dual(
                    predictions=self.predictions,
                    gt_top5_raw=gt_top5_raw,
                    gt_top5_coreset_adjusted=gt_top5_adj,
                    coreset_coverage_rate=self.gt_data.get("coreset_coverage_rate", 1.0),
                    n_coreset_train=self.gt_data.get("n_coreset_train", 0),
                )
            else:
                metrics["retrieval_metrics"] = {
                    "note": "recall@5 not available in full pipeline mode "
                            "(orchestrator does not return retrieved results). "
                            "Use --pipeline-mode router_retriever for recall evaluation."
                }

            if self.latencies["total"]:
                metrics["latency"] = self.metric_calculator.calculate_latency_metrics(
                    np.array(self.latencies["total"]))
                metrics["latency_breakdown"] = self.metric_calculator.calculate_latency_breakdown(
                    np.array(self.latencies["router"]),
                    np.array(self.latencies["retriever"]),
                    np.array(self.latencies["analyzer"]),
                )

        metrics["summary"] = self.metric_calculator.get_summary()
        metrics["goals"]   = self.metric_calculator.check_goals()
        return metrics

    # ──────────────────────────────────────────────────────────
    # Save Results
    # ──────────────────────────────────────────────────────────

    def save_results(self, output_dir: str):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "predictions.json", "w") as f:
            json.dump(self.predictions, f, indent=2)

        if any(self.latencies[k] for k in self.latencies):
            with open(out / "latencies.json", "w") as f:
                json.dump(
                    {k: [float(v) for v in vs] for k, vs in self.latencies.items()},
                    f, indent=2
                )

        metrics = self.calculate_metrics()
        with open(out / "metrics.json", "w") as f:
            def _serialize(obj):
                if isinstance(obj, set):
                    return sorted(list(obj))[:100]
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                raise TypeError(f"Not serializable: {type(obj)}")
            json.dump(metrics, f, indent=2, default=_serialize)

        ret = metrics.get("retrieval_metrics", {})
        pf  = metrics.get("page_fault_stats", {})
        with open(out / "summary.txt", "w") as f:
            f.write("=" * 80 + "\n")
            f.write(f"EVALUATION SUMMARY V5.0 — {self.pipeline_mode.upper()}"
                    f"{' [SKIP_ANALYZER]' if self.skip_analyzer else ''}\n")
            f.write("=" * 80 + "\n")
            f.write(f"experiment_case    : {self.experiment_case}\n")
            f.write(f"percolate_version  : {self.percolate_version}\n")
            f.write(f"coreset_percentage : {self.coreset_percentage}\n")
            f.write(f"n_coreset_train    : {metrics.get('n_coreset_train', 0)}\n")
            f.write(f"coverage_rate      : {metrics.get('coreset_coverage_rate', 1.0):.4f}\n")
            f.write(f"pipeline_mode      : {self.pipeline_mode}\n")
            f.write(f"skip_analyzer      : {self.skip_analyzer}\n")
            f.write(f"n_samples          : {metrics['n_samples']}\n")
            f.write(f"n_errors           : {metrics['n_errors']}\n")

            if pf:
                f.write(f"\n── Page Fault Delta (Retriever) ──\n")
                f.write(f"  mean      : {pf.get('mean', 0):.2f}\n")
                f.write(f"  p95       : {pf.get('p95', 0):.1f}\n")
                f.write(f"  p99       : {pf.get('p99', 0):.1f}\n")
                f.write(f"  n_nonzero : {pf.get('n_nonzero', 0)}\n")

            if self.pipeline_mode in ("router_only", "router_retriever", "full"):
                r  = metrics.get("routing_accuracy", {})
                rv = metrics.get("router_metrics", {})
                f.write(f"\nCluster Assignment Acc  : {r.get('cluster_assignment_accuracy', 'N/A')}\n")
                f.write(f"Router Recall@5         : {rv.get('recall_at_5', 'N/A')}\n")

            if self.pipeline_mode in ("router_retriever", "full"):
                f.write("\n── Retrieval Recall@5 (original_index 기반, Dual) ──\n")
                f.write(f"  recall@5 (raw)     : {ret.get('recall@5_raw', 'N/A')}\n")
                f.write(f"  recall@5 (adjusted): {ret.get('recall@5_coreset_adjusted', 'N/A')}\n")
                f.write(f"  delta (adj - raw)  : {ret.get('recall@5_delta', 'N/A')}\n")
                f.write(f"  n_evaluated_raw    : {ret.get('n_evaluated_raw', 0)}\n")
                f.write(f"  n_evaluated_adj    : {ret.get('n_evaluated_adjusted', 0)}\n")
                f.write(f"  n_empty_gt(adj)    : {ret.get('n_empty_gt_after_filtering', 0)}\n")
                note = ret.get("note")
                if note:
                    f.write(f"  note               : {note}\n")
                dist = metrics.get("distance_based_anomaly", {})
                f.write(f"\nDistance AUROC          : {dist.get('distance_auroc', 'N/A')}\n")
                f.write(f"Distance Ratio (F/N)    : {dist.get('distance_ratio', 'N/A')}\n")
                if "latency" in metrics:
                    lat = metrics["latency"]
                    f.write(f"Total Latency P99       : {lat.get('total_p99', lat.get('latency_p99_ms', 'N/A'))}\n")

                _METRIC_INNER_KEY = {
                    "router_mrr":              "router_mrr",
                    "candidate_recall":        "candidate_recall_top5",
                    "routing_epsilon_recall":  "routing_epsilon_recall_10pct",
                    "coreset_epsilon_recall":  "coreset_epsilon_recall_10pct",
                }

                def extract_metric(metrics_dict, key):
                    obj = metrics_dict.get(key, 0.0)
                    if isinstance(obj, dict):
                        inner_key = _METRIC_INNER_KEY.get(key, key)
                        return float(obj.get(inner_key, 0.0))
                    return float(obj) if obj else 0.0

                # # 신규 메트릭 추가
                f.write(f"  Router MRR         : {extract_metric(metrics,'router_mrr'):.4f}\n")
                f.write(f"  Candidate Recall   : {extract_metric(metrics,'candidate_recall'):.4f}\n")
                f.write(f"  Routing EPS-Recall : {extract_metric(metrics,'routing_epsilon_recall'):.4f}\n")
                f.write(f"  Coreset EPS-Recall : {extract_metric(metrics,'coreset_epsilon_recall'):.4f}\n")

                f.write(f"n_coreset_train    : {metrics.get('n_coreset_train', 0)}\n")
                f.write(f"coverage_rate      : {metrics.get('coreset_coverage_rate', 1.0):.4f}\n")

            if self.pipeline_mode == "full":
                anom = metrics.get("anomaly_detection", {})
                f.write(f"\nAnomaly AUROC           : {anom.get('auroc', 'N/A')}\n")

            if "goals" in metrics:
                passed = sum(metrics["goals"].values())
                total  = len(metrics["goals"])
                f.write(f"\n목표 달성: {passed}/{total}\n")
                for goal, ok in metrics["goals"].items():
                    f.write(f"  {'✅' if ok else '❌'} {goal}\n")
            f.write("=" * 80 + "\n")

        logger.info("results_saved", output_dir=str(out))


from google.cloud import storage
from datetime import datetime

def upload_output_to_gcs(args, local_output_path):
    """실험 결과를 지정된 파일명 형식으로 GCS에 업로드"""
    try:
        # 1. 파일명 구성을 위한 변수 준비
        # YAML 파일 이름은 환경변수나 실행 인자에서 직접 가져오기 어려우므로 
        # 보통 실행 시점의 목적을 나타내는 접두어를 사용하거나 
        # 아래처럼 환경변수 설정을 통해 처리하는 것이 정확합니다.
        date_str = datetime.now().strftime("%m%d") # 월일 (예: 0311)
        
        # 요청하신 형식: yaml_EXPERIMENT_CASE_VERSION_PCT_MMDD_output.txt
        gcs_filename = (
            f"router_retriever_batch_case_{args.experiment_case}_"
            f"query_{args.percolate_version}_"
            f"pct_{args.coreset_percentage}_{date_str}_output.txt"
        )
        
        bucket_name = "fraudecom"
        destination_blob_name = f"tree-search/exp_result/{gcs_filename}"

        # 2. GCS 업로드 실행
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(local_output_path)
        
        print(f"\n✅ Results uploaded to GCS: gs://{bucket_name}/{destination_blob_name}")
        
    except Exception as e:
        print(f"\n❌ Failed to upload to GCS: {e}")

# metrics 내부에서 결과값(float)만 안전하게 꺼내오는 로직
def extract_metric(metrics_dict, key):
    obj = metrics_dict.get(key, 0.0)
    if isinstance(obj, dict):
        return obj.get(key, 0.0) # 딕셔너리 내부의 실제 값 추출
    return obj


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="Batch Inference V5.0 — original_index Recall / Router+Retriever v5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pipeline-mode", type=str, default="full",
                        choices=["router_only", "router_retriever", "full"])
    parser.add_argument("--enable-internal-batch", action="store_true")
    parser.add_argument("--skip-analyzer", action="store_true",
                        help="SKIP_ANALYZER=true 실험: Analyzer 스킵, top1_distance 기반 score")
    parser.add_argument("--experiment-case", type=str, default="pca_64",
                        choices=VALID_EXPERIMENT_CASES)
    parser.add_argument("--percolate-version", type=str, default="v2",
                        choices=["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v11a","v11b","v11c","v14"])
    parser.add_argument("--coreset-percentage", type=str, default="100",
                        choices=["100", "10", "1"])
    parser.add_argument("--n-samples",  type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--timeout",    type=int, default=600)
    parser.add_argument("--output-dir", type=str, default="./results")
    # ★ Hard Negative Mining — V1.0 추가
    parser.add_argument(
        "--mine-hard-negatives", action="store_true",
        help=(
            "batch inference 완료 후 Type1(Router Misrouting) + "
            "Type2(Cross-Cluster Retrieval) Hard Negative를 추출한다. "
            "--pipeline-mode router_retriever 와 함께 사용 권장. "
            "n_samples는 자동으로 전체 test set의 앞 10%%로 고정됨."
        ),
    )
    parser.add_argument(
        "--mining-output-dir", type=str, default=None,
        help=(
            "Hard Negative 저장 디렉토리. "
            "기본값: {output_dir}/hard_negatives/{experiment_case}/query_{percolate_version}"
        ),
    )
    args = parser.parse_args()

    bi = BatchInference(
        pipeline_mode=args.pipeline_mode,
        timeout=args.timeout,
        experiment_case=args.experiment_case,
        percolate_version=args.percolate_version,
        coreset_percentage=args.coreset_percentage,
        enable_internal_batch=args.enable_internal_batch,
        skip_analyzer=args.skip_analyzer,
    )
    bi.load_data()

    # ★ Hard Negative Mining: n_samples를 전체 test set의 앞 10%로 고정
    effective_n_samples = args.n_samples
    if args.mine_hard_negatives:
        n_total_test = len(bi.test_data)
        if args.n_samples is None:
            # --n-samples 미지정 시에만 10% 자동 고정 (기존 동작 유지)
            effective_n_samples = n_total_test // 10
            print(
                f"[mining] n_samples 미지정 → 전체 {n_total_test}의 앞 10% "
                f"({effective_n_samples}) 자동 적용"
            )
        else:
            # --n-samples 명시 시 그대로 사용 (디버깅용)
            effective_n_samples = args.n_samples
            print(
                f"[mining] --n-samples={args.n_samples} 명시 → "
                f"디버그 모드 ({args.n_samples}/{n_total_test})"
            )

    start       = time.time()
    predictions = await bi.batch_inference(n_samples=effective_n_samples, batch_size=args.batch_size)
    elapsed     = time.time() - start

    metrics = bi.calculate_metrics()
    bi.save_results(args.output_dir)

    ret = metrics.get("retrieval_metrics", {})
    pf  = metrics.get("page_fault_stats", {})

    print("\n" + "=" * 80)
    print(f"EVALUATION V5.0 — {args.pipeline_mode.upper()}"
          f"{' [SKIP_ANALYZER]' if args.skip_analyzer else ''}")
    print("=" * 80)
    print(f"  experiment_case   : {args.experiment_case}")
    print(f"  coreset_percentage: {args.coreset_percentage}%")
    print(f"  n_coreset_train   : {metrics.get('n_coreset_train', 0)}")
    print(f"  coverage_rate     : {metrics.get('coreset_coverage_rate', 1.0):.4f}")
    print(f"  n_samples         : {metrics['n_samples']}  errors: {metrics['n_errors']}")
    print(f"  throughput        : {len(predictions)/elapsed:.2f} RPS")

    if pf:
        print(f"\n  ── Page Fault Delta (Retriever) ──")
        print(f"  mean={pf.get('mean',0):.2f}  p95={pf.get('p95',0):.1f}  "
              f"p99={pf.get('p99',0):.1f}  n_nonzero={pf.get('n_nonzero',0)}")

    if args.pipeline_mode == "router_only":
        rv = metrics.get("router_metrics", {})
        print(f"\n  Cluster Acc  : {metrics.get('routing_accuracy',{}).get('cluster_assignment_accuracy','N/A')}")
        print(f"  Recall@5     : {rv.get('recall_at_5','N/A')}")

    elif args.pipeline_mode in ("router_retriever", "full"):
        dist = metrics.get("distance_based_anomaly", {})
        rv   = metrics.get("router_metrics", {})
        print(f"\n  ── Retrieval Recall@5 (original_index 기반, Dual) ──")
        print(f"  recall@5 (raw)     : {ret.get('recall@5_raw','N/A')}")
        print(f"  recall@5 (adjusted): {ret.get('recall@5_coreset_adjusted','N/A')}")
        print(f"  delta (adj - raw)  : {ret.get('recall@5_delta','N/A')}")
        print(f"  n_empty_gt(adj)    : {ret.get('n_empty_gt_after_filtering',0)}")
        note = ret.get("note")
        if note:
            print(f"  note               : {note}")
        print(f"\n  Distance AUROC     : {dist.get('distance_auroc','N/A')}")
        print(f"  Router Recall@5    : {rv.get('recall_at_5','N/A')}")

        if args.pipeline_mode == "full":
            anom = metrics.get("anomaly_detection", {})
            print(f"  Anomaly AUROC      : {anom.get('auroc','N/A')}")

        print(f"  Router MRR         : {extract_metric(metrics,'router_mrr'):.4f}")
        print(f"  Candidate Recall   : {extract_metric(metrics,'candidate_recall'):.4f}")
        print(f"  Routing EPS-Recall : {extract_metric(metrics,'routing_epsilon_recall'):.4f}")
        print(f"  Coreset EPS-Recall : {extract_metric(metrics,'coreset_epsilon_recall'):.4f}")

    # ★ 여기에 추가 (router_only / router_retriever 모두 출력)
    if args.pipeline_mode in ("router_only", "router_retriever"):
        div = metrics.get("top5_diversity", {})
        mt  = metrics.get("match_type_stats", {})
        if div:
            print(f"  Top5 Diversity     : mean={div.get('mean','N/A')}  min={div.get('min','N/A')}")
        if mt:
            print(f"  Match Type         : strict={mt.get('strict_ratio','N/A')}  "
                  f"centroid={mt.get('centroid_ratio','N/A')}  counts={mt.get('counts',{})}")
            
    if "goals" in metrics:
        p = sum(metrics["goals"].values())
        t = len(metrics["goals"])
        print(f"\n  Goals: {p}/{t}")
        for goal, ok in metrics["goals"].items():
            print(f"    {'✅' if ok else '❌'} {goal}")

    local_path = os.path.join(args.output_dir, "summary.txt")
    if os.path.exists(local_path):
            upload_output_to_gcs(args, local_path)

    # ★ Hard Negative Mining — V1.0 추가
    if args.mine_hard_negatives:
        if not MINER_AVAILABLE:
            print(
                "❌ hard_negative_miner.py를 찾을 수 없습니다. "
                "batch_inference.py와 같은 디렉토리에 있는지 확인하세요."
            )
        else:
            print("\n" + "=" * 60)
            print("[HardNegativeMiner] Mining 시작...")
            print("=" * 60)

            # ★ centroid 로드 (Type 3: geometric mismatch)          # ← 추가
            centroid_matrix, cluster_ids = bi.gt_loader.load_centroids()  # ← 추가

            # centroid_matrix를 cluster_id → vector 매핑 딕셔너리로 변환  # ← 추가
            cluster_centroids = None                                      # ← 추가
            if centroid_matrix is not None and cluster_ids is not None:   # ← 추가
                cluster_centroids = {                                     # ← 추가
                    cluster_id: centroid_matrix[i]                        # ← 추가
                    for i, cluster_id in enumerate(cluster_ids)           # ← 추가
                }                                                         # ← 추가
                print(f"[HardNegativeMiner] Centroids loaded: {len(cluster_centroids)} clusters")  # ← 추가


            # coreset_df 로드 (Type 2: original_index → cluster_id 매핑)
            coreset_df = bi.gt_loader.load_coreset_df()

            # mining 출력 디렉토리 결정
            mining_out = args.mining_output_dir or os.path.join(
                args.output_dir,
                "hard_negatives",
                args.experiment_case,
                f"query_{args.percolate_version}",
            )

            miner = HardNegativeMiner(
                predictions=bi.predictions,
                test_data=bi.test_data,
                gt_data=bi.gt_data,
                coreset_df=coreset_df,
                experiment_case=args.experiment_case,
                percolate_version=args.percolate_version,
                coreset_percentage=args.coreset_percentage,
                sample_indices=bi.sample_indices,
                test_emb=bi.embeddings,              # ← 기존 추가
                cluster_centroids=cluster_centroids, # ← 수정 (None → dict)
                distance_percentiles=None,           # ← 기존 추가
            )
            mining_df    = miner.run()
            local_parquet = miner.save(
                df=mining_df,
                output_dir=mining_out,
                upload_gcs=True,
            )
            print(f"\n[HardNegativeMiner] 완료: {local_parquet}")

    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
