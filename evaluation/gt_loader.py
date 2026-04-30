"""
GT Loader - V5.1
=================

역할:
  batch_inference.py 에서 호출되어 테스트 데이터·임베딩·GT 라벨을 로드한다.

[V5.0 → V5.1 변경사항]

  ★ GCS 기반 경로 시스템으로 변경
    - 로컬 파일 시스템 대신 GCS에서 직접 다운로드
    - google.cloud.storage.Client 사용

  ★ top5_indices를 test parquet에서 직접 로드
    - 기존: _compute_top5_gt()로 실시간 계산
    - 변경: test_with_gt.parquet의 top5_indices 컬럼 사용 (각 experiment_case별 사전 계산됨)

  ★ Gaussian augmentation 지원
    - coreset_percentage가 "mult_20_pct_10" 같은 패턴일 때
    - 20x 증강 데이터 대신 원본 percentage만 사용 (예: pct_10)
    - coreset valid index는 원본 x1 데이터에서만 계산

[설계 원칙]

  ★ top5_indices: original_index 공간
    - test_with_gt.parquet에 사전 계산된 top5_indices 컬럼 사용
    - 인제스트 v4.4 기준 ES _id = original_index → retriever SearchResult.original_index 와 동일 공간
    - batch_inference 가 저장하는 retrieved_original_indices 와 직접 set-intersection 비교 가능

  ★ top5_indices_coreset_adjusted:
    - coreset에 없는 original_index → -1 마스킹

  ★ test embedding: 고정 경로 로드 (coreset 비율과 무관)

[gt_data 반환 스키마]
  {
    "label":                         np.ndarray (N,)   0=정상 / 1=이상
    "gt_cluster_tree":               np.ndarray (N,)   GT cluster id (-1=unavailable)
    "top5_indices":                  np.ndarray (N, 5) original_index 기반 (raw kNN)
    "top5_indices_coreset_adjusted": np.ndarray (N, 5) original_index 기반 (coreset 미포함 → -1)
    "coreset_valid_index_set":       set[int]          coreset 정상 샘플 original_index 집합
    "n_coreset_train":               int               coreset 정상 샘플 수
    "coreset_coverage_rate":         float             raw GT 중 coreset 포함 비율
  }
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import structlog
from google.cloud import storage

logger = structlog.get_logger()

# ===========================
# Constants
# ===========================

VALID_EXPERIMENT_CASES: List[str] = [
    "emb_vectors",
    "pca_32", "pca_64", "pca_128", "pca_256",
    "k50", "k100", "k200",
    "pca_64_k100", "pca_64_k200",
]

# GCS 경로 상수
GCS_BUCKET_NAME = "fraudecom"

# test_with_gt parquet (experiment_case별)
TEST_PARQUET_PATH_TEMPLATE = "tree-search/tree_fraudecom/{experiment_case}/test_with_gt.parquet"

# test embedding: coreset와 무관한 고정 경로
TEST_EMBEDDING_PATH = "tree-search/embeddings_fraudecom/anollm_lr5e-05_standard_smolLM_test_embeddings_merged.npy"

# coreset 정상 데이터 parquet (percentage별, experiment_case)
# CORESET_PARQUET_PATH_TEMPLATE = "tree-search/tree_fraudecom/raw_plus_cluster_tree_percentage_{pct}.parquet"
CORESET_PARQUET_PATH_TEMPLATE = "tree-search/tree_fraudecom/{experiment_case}/raw_plus_cluster_tree_percentage_{pct}.parquet"

# ★ centroid 행렬 경로 (experiment_case별)
CENTROID_PATH_TEMPLATE = "tree-search/tree_fraudecom/{experiment_case}/cluster_centroids.npy"  # ← 추가

# ===========================
# GTLoader
# ===========================

class GTLoader:
    """
    Ground Truth 로더 V5.1.

    사용법:
        loader = GTLoader(
            experiment_case="pca_64_k100", 
            coreset_percentage="100"  # or "mult_20_pct_10" for Gaussian aug
        )
        test_data, test_emb, gt_data = loader.load_all()
        test_case_dict = loader.get_test_case(
            idx, test_data, test_emb,
            percolate_version="v2",
            experiment_case="pca_64_k100"
        )
    """

    def __init__(
        self,
        experiment_case:    str = "pca_64",
        coreset_percentage: str = "100",
        bucket_name:        str = GCS_BUCKET_NAME,
        local_cache_dir:    str = "/tmp/gt_cache",
        project:            Optional[str] = None,
    ):
        """
        Parameters
        ----------
        experiment_case : str
            실험 케이스 (e.g., "pca_64", "pca_64_k100")
        coreset_percentage : str
            coreset 비율
            - "100", "10", "1" : 일반 케이스
            - "mult_20_pct_10" : Gaussian augmentation 케이스 (20x 증강, 원본 10%)
        bucket_name : str
            GCS 버킷 이름
        local_cache_dir : str
            로컬 캐시 디렉토리
        project : Optional[str]
            GCP 프로젝트 ID (None이면 환경변수에서 가져옴)
        """
        if experiment_case not in VALID_EXPERIMENT_CASES:
            raise ValueError(
                f"Invalid experiment_case='{experiment_case}'. "
                f"Valid: {VALID_EXPERIMENT_CASES}"
            )

        self.experiment_case    = experiment_case
        self.coreset_percentage = coreset_percentage
        self.bucket_name        = bucket_name
        self.local_cache_dir    = Path(local_cache_dir)
        self.project            = project or os.getenv("GOOGLE_CLOUD_PROJECT")

        # ★ Gaussian augmentation 패턴 파싱
        # "mult_20_pct_10" → original_pct = "10"
        self.original_pct = self._parse_original_percentage(coreset_percentage)
        
        # GCS 경로 설정
        self.test_parquet_path = TEST_PARQUET_PATH_TEMPLATE.format(
            experiment_case=experiment_case
        )
        self.test_emb_path = TEST_EMBEDDING_PATH
        
        # ★ centroid 경로 추가
        self.centroid_path = CENTROID_PATH_TEMPLATE.format(         # ← 추가
            experiment_case=experiment_case                          # ← 추가
        )                                                            # ← 

        # ★ coreset parquet은 원본 percentage만 사용
        self.coreset_parquet_path = CORESET_PARQUET_PATH_TEMPLATE.format(
            experiment_case=experiment_case,
            pct=self.original_pct
        )

        # 로컬 캐시 디렉토리 생성
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)

        # GCS 클라이언트 초기화
        self.storage_client = storage.Client(project=self.project)
        self.bucket = self.storage_client.bucket(bucket_name)

        logger.info(
            "gt_loader_init_v5_1",
            experiment_case=experiment_case,
            coreset_percentage=coreset_percentage,
            original_pct=self.original_pct,
            test_parquet_path=self.test_parquet_path,
            test_emb_path=self.test_emb_path,
            centroid_path=self.centroid_path,                        # ← 추가
            coreset_parquet_path=self.coreset_parquet_path,
        )

    # ──────────────────────────────────────────────────────────
    # Private: Gaussian augmentation 패턴 파싱
    # ──────────────────────────────────────────────────────────

    def _parse_original_percentage(self, coreset_percentage: str) -> str:
        """
        Gaussian augmentation 패턴에서 원본 percentage 추출.

        Examples
        --------
        "100"            → "100"
        "10"             → "10"
        "mult_20_pct_10" → "10"   (20x 증강, 원본 10%)
        "mult_5_pct_1"   → "1"    (5x 증강, 원본 1%)

        Returns
        -------
        str : 원본 percentage
        """
        # mult_XX_pct_YY 패턴 매칭
        match = re.match(r"mult_\d+_pct_(\d+)", coreset_percentage)
        if match:
            original_pct = match.group(1)
            logger.info(
                "gaussian_aug_detected",
                input_percentage=coreset_percentage,
                original_pct=original_pct,
                note="coreset valid index는 원본 x1 데이터에서만 계산",
            )
            return original_pct
        else:
            # 일반 케이스
            return coreset_percentage

    # ──────────────────────────────────────────────────────────
    # Private: GCS 다운로드
    # ──────────────────────────────────────────────────────────

    def _download_from_gcs(
        self,
        gcs_path:   str,
        local_path: Path,
        use_cache:  bool = True,
    ) -> Path:
        """
        GCS에서 파일 다운로드.

        Parameters
        ----------
        gcs_path : str
            GCS 경로 (예: "tree-search/tree_fraudecom/...")
        local_path : Path
            로컬 저장 경로
        use_cache : bool
            캐시 사용 여부

        Returns
        -------
        Path : 다운로드된 로컬 파일 경로
        """
        if use_cache and local_path.exists():
            logger.info("using_cached_file", file=str(local_path))
            return local_path

        full_path = f"gs://{self.bucket_name}/{gcs_path}"
        logger.info("downloading_from_gcs", path=full_path)

        try:
            blob = self.bucket.blob(gcs_path)
            blob.download_to_filename(str(local_path))
            logger.info("download_success", local_path=str(local_path))
            return local_path
        except Exception as e:
            logger.error("download_failed", path=full_path, error=str(e))
            raise FileNotFoundError(f"Failed to download {full_path}: {e}")

    # ──────────────────────────────────────────────────────────
    # Private: 데이터 로드
    # ──────────────────────────────────────────────────────────

    def _load_test_data(self, use_cache: bool = True) -> pd.DataFrame:
        """
        test_with_gt.parquet 로드.

        컬럼 (예상):
            거래 피처: purchase_value, age, sex, source, browser,
                       weekday_purchase, month_purchase, IP_country
            label (0=Normal, 1=Fraud)
            gt_cluster_tree  — 결정 트리 기반 클러스터 ID
            top5_indices     — ★ 사전 계산된 brute-force KNN top-5 정상 문서 ID (original_index)
        
        Returns
        -------
        pd.DataFrame
        """
        local_path = self.local_cache_dir / f"test_with_gt_{self.experiment_case}.parquet"
        self._download_from_gcs(self.test_parquet_path, local_path, use_cache)

        df = pd.read_parquet(local_path)

        # label 필수 체크
        if "label" not in df.columns:
            raise ValueError(f"test parquet must have 'label' column. found: {list(df.columns)}")

        # ★ top5_indices 필수 체크
        if "gt_knn_indices_576d" not in df.columns:
            raise ValueError(
                f"test parquet must have 'gt_knn_indices_576d' column. "
                f"found: {list(df.columns)}. "
                f"This GT should be pre-computed for each experiment_case."
            )
        # ★ top5_dist 필수 체크
        if "gt_knn_dists_576d" not in df.columns:
            raise ValueError(
                f"test parquet must have 'gt_knn_dists_576d' column. "
                f"found: {list(df.columns)}. "
                f"This GT should be pre-computed for each experiment_case."
            )
        
        # ★ top5_cluster_ids 필수 체크
        if "gt_top5_cluster_ids" not in df.columns:
            raise ValueError(
                f"test parquet must have 'gt_top5_cluster_ids' column. "
                f"found: {list(df.columns)}. "
                f"This GT should be pre-computed for each experiment_case."
            )
        # ★ top5_cluster_dists_eval_space 필수 체크
        if "gt_top5_cluster_dists_eval_space" not in df.columns:
            raise ValueError(
                f"test parquet must have 'gt_top5_cluster_dists_eval_space' column. "
                f"found: {list(df.columns)}. "
                f"This GT should be pre-computed for each experiment_case."
            )

        logger.info(
            "test_data_loaded",
            experiment_case=self.experiment_case,
            shape=df.shape,
            columns=df.columns.tolist(),
            n_fraud=int((df["label"] == 1).sum()),
            n_normal=int((df["label"] == 0).sum()),
        )
        return df

    def _load_test_embeddings(self, use_cache: bool = True) -> np.ndarray:
        """
        AnoLLM Test 576-dim 임베딩 로드 — coreset와 무관한 고정 경로.
        test set은 coreset sampling 대상이 아님.

        Returns
        -------
        np.ndarray : (N_test, D)
        """
        local_path = self.local_cache_dir / "test_embeddings_fixed.npy"
        self._download_from_gcs(self.test_emb_path, local_path, use_cache)

        emb = np.load(local_path)
        logger.info(
            "test_embeddings_loaded",
            shape=emb.shape,
            dtype=str(emb.dtype),
        )
        return emb

    def _load_coreset_parquet(self, use_cache: bool = True) -> pd.DataFrame:
        """
        ★ V5.1 핵심: Coreset 정상 데이터 parquet 로드.

        ingest.py의 process_raw_vec_for_percentage()가 사용하는 동일 parquet.
        ES 인덱스에 실제로 ingest된 정상 문서들의 original_index를 포함.

        ★ Gaussian augmentation 케이스:
            - coreset_percentage = "mult_20_pct_10"
            - 실제 다운로드: raw_plus_cluster_tree_percentage_10.parquet (원본 x1)
            - 20x 증강 데이터는 무시 (원본 데이터만 valid index로 사용)

        GCS 경로:
            tree-search/tree_fraudecom/{experiment_case}/raw_plus_cluster_tree_percentage_{pct}.parquet
            예) experiment_case=pca_64_k100, original_pct=10
            → pca_64_k100/raw_plus_cluster_tree_percentage_10.parquet
            
        컬럼:
            purchase_value, source, browser, sex, age, IP_country,
            month_purchase, weekday_purchase,
            label, original_index, cluster_id

        Returns
        -------
        pd.DataFrame (original_index 컬럼 포함)
        """
        # local_path = self.local_cache_dir / f"coreset_pct_{self.original_pct}.parquet"
        local_path = self.local_cache_dir / f"coreset_{self.experiment_case}_pct_{self.original_pct}.parquet"
        self._download_from_gcs(self.coreset_parquet_path, local_path, use_cache)

        df = pd.read_parquet(local_path)

        # original_index 필수 체크
        if "original_index" not in df.columns:
            raise ValueError(
                f"coreset parquet must have 'original_index' column. "
                f"found: {list(df.columns)}"
            )

        logger.info(
            "coreset_parquet_loaded",
            coreset_percentage=self.coreset_percentage,
            original_pct=self.original_pct,
            shape=df.shape,
            columns=df.columns.tolist(),
            original_index_range=[
                int(df["original_index"].min()),
                int(df["original_index"].max()),
            ],
        )
        return df

    # ──────────────────────────────────────────────────────────
    # Private: GT 처리
    # ──────────────────────────────────────────────────────────

    def _build_coreset_valid_index_set(
        self,
        coreset_df: pd.DataFrame,
    ) -> Set[int]:
        """
        ★ V5.1: Coreset parquet의 original_index → valid ES doc ID 집합 생성.

        이유:
          - ingest.py는 parquet의 original_index를 ES _id로 사용
          - coreset parquet은 전체 중 sampling된 행만 포함
          - sampling된 행의 original_index != 0~(n-1) 연속 정수

        Args
        ----
        coreset_df : pd.DataFrame
            _load_coreset_parquet()의 반환값

        Returns
        -------
        set[int] : ES에 실제 존재하는 문서 ID (original_index 값들)
        """
        # label이 있으면 정상(0)만, 없으면 전체
        if "label" in coreset_df.columns:
            normal_mask = coreset_df["label"] == 0
            valid_set = set(
                coreset_df.loc[normal_mask, "original_index"].astype(int).tolist()
            )
        else:
            valid_set = set(coreset_df["original_index"].astype(int).tolist())

        logger.info(
            "coreset_valid_index_set_built",
            coreset_pct=self.coreset_percentage,
            original_pct=self.original_pct,
            n_valid_indices=len(valid_set),
            sample_indices=sorted(list(valid_set))[:10],  # 처음 10개만 로깅
        )
        return valid_set

    def _build_top5_coreset_adjusted(
        self,
        top5_indices:            np.ndarray,
        valid_coreset_index_set: Set[int],
    ) -> np.ndarray:
        """
        top5_indices에서 coreset에 존재하지 않는 original_index를 -1로 마스킹.

        test_with_gt의 top5_indices 값 = 전체 학습셋 기준 original_index.
        coreset sampling 후에는 해당 original_index가 ES에 존재하지 않을 수 있음.
        존재하지 않는 인덱스는 -1로 마스킹.

        Args
        ----
        top5_indices : np.ndarray
            (N_test, 5) 원본 GT top-5 (original_index 값)
        valid_coreset_index_set : set[int]
            ES에 실제 존재하는 original_index 집합

        Returns
        -------
        np.ndarray : (N_test, 5) coreset 미포함 → -1
        """
        valid_arr = np.array(sorted(list(valid_coreset_index_set)))
        in_coreset = np.isin(top5_indices, valid_arr)
        gt_top5_indices_adjusted = np.where(in_coreset, top5_indices, -1)

        total_gt    = top5_indices.size
        total_valid = int(in_coreset.sum())
        coverage    = total_valid / total_gt if total_gt > 0 else 0.0
        n_all_invalid = int((gt_top5_indices_adjusted == -1).all(axis=1).sum())

        logger.info(
            "coreset_adjusted_top5_built",
            coreset_pct=self.coreset_percentage,
            original_pct=self.original_pct,
            total_gt_indices=total_gt,
            valid_gt_indices=total_valid,
            coverage_rate=round(coverage, 4),
            n_samples_with_all_invalid_gt=n_all_invalid,
            n_total_samples=len(gt_top5_indices_adjusted),
        )
        return gt_top5_indices_adjusted

    # ──────────────────────────────────────────────────────────
    # Public: Centroid 로드
    # ──────────────────────────────────────────────────────────

    def load_centroids(
        self,
        use_cache: bool = True,
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
        """
        클러스터 중심(centroid) 벡터 행렬 로드.

        GCS 경로:
            tree-search/tree_fraudecom/{experiment_case}/cluster_centroids.npy
            예) pca_64_k100 → tree-search/tree_fraudecom/pca_64_k100/cluster_centroids.npy

        Shape:
            (K, D)  K=클러스터 수, D=experiment_case의 vector_dim
            예) pca_64_k100 → (100, 64)

        Centroid row 순서:
            cluster_id 오름차순 정렬 기준 (npy 생성 규칙)
            centroid[i] ↔ cluster_ids[i]

        Parameters
        ----------
        use_cache : bool
            로컬 캐시 사용 여부

        Returns
        -------
        centroid_matrix : np.ndarray or None
            (K, D) 형태의 centroid 행렬. 파일이 없으면 None.
        cluster_ids : List[int] or None
            cluster_id 목록 (오름차순 정렬). centroid 행이 없으면 None.

        Notes
        -----
        - PCA 케이스(pca_64_k100): D = PCA 차원 (64)
        - 비PCA 케이스(k100): D = 원본 embedding 차원 (576)
        - cluster_ids는 0-indexed fallback 사용 (실제 cluster_id는 별도 로드 필요 시 구현)
        """
        local_path = self.local_cache_dir / f"centroids_{self.experiment_case}.npy"

        try:
            self._download_from_gcs(self.centroid_path, local_path, use_cache)
        except FileNotFoundError:
            logger.warning(
                "centroid_not_found",
                experiment_case=self.experiment_case,
                gcs_path=self.centroid_path,
                note="Type 3 Hard Negative Mining 불가 — centroid 파일 없음",
            )
            return None, None

        try:
            centroid_matrix = np.load(local_path).astype(np.float32)
            n_clusters, vector_dim = centroid_matrix.shape

            logger.info(
                "centroid_loaded",
                experiment_case=self.experiment_case,
                shape=centroid_matrix.shape,
                n_clusters=n_clusters,
                vector_dim=vector_dim,
                gcs_path=self.centroid_path,
            )

            # ★ cluster_ids: 0-indexed fallback
            # 실제 cluster_id는 percolator 인덱스나 coreset parquet에서 로드 가능
            # 여기서는 간단하게 0, 1, 2, ..., K-1 사용
            cluster_ids = list(range(n_clusters))

            logger.info(
                "centroid_cluster_ids_fallback",
                n_clusters=n_clusters,
                cluster_ids_sample=cluster_ids[:10],
                note="0-indexed fallback 사용. 실제 cluster_id는 별도 확인 필요.",
            )

            return centroid_matrix, cluster_ids
        
        except Exception as e:
            logger.error(
                "centroid_load_failed",
                experiment_case=self.experiment_case,
                local_path=str(local_path),
                error=str(e),
            )
            return None, None
    # ──────────────────────────────────────────────────────────
    # Public: load_coreset_df  ★ V5.1 추가 — Hard Negative Mining용
    # ──────────────────────────────────────────────────────────

    def load_coreset_df(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Coreset parquet을 DataFrame으로 직접 반환.

        Hard Negative Mining (Type 2) 에서
        original_index → cluster_id 매핑 구축에 사용된다.

        Parameters
        ----------
        use_cache : bool
            로컬 캐시 사용 여부

        Returns
        -------
        pd.DataFrame
            컬럼 포함: original_index, cluster_id, label, 피처 컬럼들
            (ingest_tree_pipeline.py가 생성하는 raw_plus_cluster_tree_percentage_{pct}.parquet)
        """
        return self._load_coreset_parquet(use_cache)

    # ──────────────────────────────────────────────────────────
    # Public: load_all
    # ──────────────────────────────────────────────────────────

    def load_all(
        self,
        use_cache: bool = True,
    ) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
        """
        전체 GT 데이터 로드.

        ★ V5.1 변경:
          - top5_indices는 test parquet에서 직접 로드 (사전 계산됨)
          - coreset valid index는 원본 x1 데이터에서만 계산 (Gaussian aug 지원)

        Parameters
        ----------
        use_cache : bool
            로컬 캐시 사용 여부

        Returns
        -------
        test_data : pd.DataFrame
        test_emb  : np.ndarray  (N_test, D)
        gt_data   : dict — 아래 스키마 참조

        gt_data 스키마
        --------------
        label                         : np.ndarray (N,)   0=정상 / 1=이상

        top5_indices_coreset_adjusted : np.ndarray (N, 5) ★ original_index 기반 (coreset 미포함 → -1)
        coreset_valid_index_set       : set[int]          ★ 원본 x1 데이터에서 계산
        n_coreset_train               : int
        coreset_coverage_rate         : float
        top5_dist                     : np.ndarray (N, 5) ★ original_index 기반 (test parquet에서 로드)
        

        gt_cluster_id               : np.ndarray (N,)   GT cluster id
        gt_top5_cluster_id            : np.ndarray (N, 5) ★ original_index 기반 (test parquet에서 로드)
        gt_top5_cluster_dist            : np.ndarray (N, 5) ★ original_index 기반 (test parquet에서 로드)
        gt_top5_indices                  : np.ndarray (N, 5) ★ original_index 기반 (test parquet에서 로드)
        gt_top5_dist  : dices                  : np.ndarray (N, 5) ★ original_index 기반 (test parquet에서 로드)
        
        """
        # ── 파일 로드 ────────────────────────────────────────────
        test_data   = self._load_test_data(use_cache)
        test_emb    = self._load_test_embeddings(use_cache)
        coreset_df  = self._load_coreset_parquet(use_cache)

        # 길이 검증
        if len(test_emb) != len(test_data):
            raise ValueError(
                f"test emb/data length mismatch: "
                f"emb={len(test_emb)}, data={len(test_data)}"
            )

        # ── 핵심 배열 ────────────────────────────────────────────
        labels = test_data["label"].values.astype(int)

        # ★ GT cluster 할당
        if "gt_cluster_id" in test_data.columns:
            gt_clusters = test_data["gt_cluster_id"].values.astype(int)
        elif "cluster_id" in test_data.columns:
            gt_clusters = test_data["cluster_id"].values.astype(int)
        else:
            logger.warning("gt_cluster_unavailable", fallback="-1")
            gt_clusters = np.full(len(test_data), -1, dtype=int)

        # # ★ GT top5 cluster 할당
        # if "gt_top5_cluster_ids" in test_data.columns:
        #     gt_top5_cluster_ids = test_data["gt_top5_cluster_ids"].values.astype(int)
        # else:
        #     raise ValueError("test parquet must have 'gt_top5_cluster_ids' column")

        # # ★ GT top5 cluster 할당
        if "gt_top5_cluster_ids" in test_data.columns:
            raw_cluster_ids = test_data["gt_top5_cluster_ids"].values
            # 리스트/배열 형태의 데이터를 (N, 5) 형태의 numpy array로 변환
            gt_top5_cluster_ids = np.array([np.array(row, dtype=int) for row in raw_cluster_ids])
        else:
            raise ValueError("test parquet must have 'gt_top5_cluster_ids' column")


        # ★ gt_top5_cluster_dist 
        if "gt_top5_cluster_dists_eval_space" not in test_data.columns:
            raise ValueError("test parquet must have 'gt_top5_cluster_dists_eval_space' column")
        
        gt_top5_cluster_dist = test_data["gt_top5_cluster_dists_eval_space"].values
        
        # ★ top5_indices (test parquet에서 직접 로드!)
        if "gt_knn_dists_576d" not in test_data.columns:
            raise ValueError("test parquet must have 'gt_knn_dists_576d' column")
        
        gt_top5_dist = test_data["gt_knn_dists_576d"].values
        

        # ★ top5_indices (test parquet에서 직접 로드!)
        if "gt_knn_indices_576d" not in test_data.columns:
            raise ValueError("test parquet must have 'gt_knn_indices_576d' column")
        
        gt_top5_indices = test_data["gt_knn_indices_576d"].values
        
        # top5_indices가 리스트로 저장되어 있을 경우 numpy array로 변환
        if isinstance(gt_top5_indices[0], (list, np.ndarray)):
            gt_top5_indices = np.array([np.array(row, dtype=int) for row in gt_top5_indices])
        
        if gt_top5_indices.shape[1] != 5:
            raise ValueError(f"top5_indices must have 5 columns, got {gt_top5_indices.shape[1]}")

        logger.info(
            "top5_indices_loaded_from_parquet",
            shape=gt_top5_indices.shape,
            dtype=str(gt_top5_indices.dtype),
            sample=gt_top5_indices[:2].tolist(),
            note="Pre-computed GT from test_with_gt.parquet",
        )

        # ── coreset valid index set (원본 x1 데이터) ───────────
        coreset_valid_index_set = self._build_coreset_valid_index_set(coreset_df)
        
        # label 컬럼이 있으면 정상 샘플 수 계산
        if "label" in coreset_df.columns:
            n_coreset_normal = int((coreset_df["label"] == 0).sum())
        else:
            n_coreset_normal = len(coreset_df)

        # ── coreset_adjusted ─────────────────────────────────────
        gt_top5_indices_adjusted = self._build_top5_coreset_adjusted(
            gt_top5_indices,
            coreset_valid_index_set,
        )

        # ── coreset_coverage_rate ─────────────────────────────────
        # raw GT 중 coreset 정상 original_index 에 포함된 비율
        raw_flat = gt_top5_indices.flatten()
        n_in     = int(np.isin(raw_flat, list(coreset_valid_index_set)).sum())
        coverage = float(n_in / max(len(raw_flat), 1))

        gt_data: Dict = {
            "label":                         labels,
            "gt_cluster_id":               gt_clusters,
            "gt_top5_indices":                  gt_top5_indices,       # ★ original_index 기반 (test parquet)
            "gt_top5_indices_coreset_adjusted": gt_top5_indices_adjusted,  # ★ original_index 기반 (coreset 미포함 → -1)
            "coreset_valid_index_set":       coreset_valid_index_set,
            "n_coreset_train":               n_coreset_normal,
            "coreset_coverage_rate":         coverage,

            "gt_top5_cluster_ids":               gt_top5_cluster_ids,
            "gt_top5_cluster_dist":               gt_top5_cluster_dist,
            "gt_top5_dist":               gt_top5_dist,
        }

        logger.info(
            "gt_data_ready_v5_1",
            n_test=len(test_data),
            n_fraud=int((labels == 1).sum()),
            n_coreset_normal=n_coreset_normal,
            coreset_coverage_rate=round(coverage, 4),
            gt_top5_indices_dtype=str(gt_top5_indices.dtype),
            gt_top5_indices_sample=gt_top5_indices[:2].tolist(),
            note="top5_indices loaded from test parquet (pre-computed)",
        )
        return test_data, test_emb, gt_data

    # ──────────────────────────────────────────────────────────
    # Public: get_test_case
    # ──────────────────────────────────────────────────────────

    def get_test_case(
        self,
        idx:               int,
        test_data:         pd.DataFrame,
        test_emb:          np.ndarray,
        percolate_version: str = "v2",
        experiment_case:   str = "pca_64",
    ) -> Dict:
        """
        단건 TestCase dict 생성 (Orchestrator v5 /detect 요청 포맷).

        Parameters
        ----------
        idx               : test_data 인덱스 (0-based)
        test_data         : load_all() 반환 DataFrame
        test_emb          : load_all() 반환 embedding 배열
        percolate_version : 사용할 percolate query 버전 (e.g. "v2")
        experiment_case   : 실험 케이스 (e.g. "pca_64_k100")

        Returns
        -------
        dict — Orchestrator /detect JSON body
        """
        row = test_data.iloc[idx]

        def _safe(col: str, default):
            val = row.get(col) if col in test_data.columns else None
            return default if (val is None or (isinstance(val, float) and np.isnan(val))) else val

        tc: Dict = {
            "embedding":         test_emb[idx].tolist(),
            "purchase_value":    float(_safe("purchase_value", 0.0)),
            "age":               float(_safe("age", 0.0)),
            "sex":               str(_safe("sex",     "M")),
            "source":            str(_safe("source",  "Direct")),
            "browser":           str(_safe("browser", "Chrome")),
            "experiment_case":   experiment_case,
            "percolate_version": percolate_version,
        }

        # 선택적 필드
        for opt in ("weekday_purchase", "month_purchase", "IP_country"):
            v = _safe(opt, None)
            if v is not None:
                tc[opt] = str(v)

        return tc