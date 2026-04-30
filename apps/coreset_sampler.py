"""
Coreset Sampler - PatchCore 방식의 대표 벡터 선택

주요 기능:
1. Greedy selection (Max-min distance)
2. Hot cluster에만 적용
3. 검색 후보 벡터 수 감소 → 검색 속도 향상

참고:
- PatchCore 논문의 coreset selection 알고리즘
- O(N*k) 복잡도
"""

import numpy as np
from typing import List, Tuple
import structlog

logger = structlog.get_logger()


class CoresetSampler:
    """
    PatchCore 방식의 Coreset Sampling
    
    주어진 벡터 집합에서 대표성을 유지하면서 부분 집합 선택
    """
    
    def __init__(self, sample_size: int = 50):
        """
        Args:
            sample_size: 선택할 대표 벡터 개수 (기본값: 50)
        """
        self.sample_size = sample_size
        logger.info("coreset_sampler_initialized", sample_size=sample_size)
    
    def sample(
        self,
        embeddings: np.ndarray,
        indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Greedy coreset selection
        
        Args:
            embeddings: (N, D) 형태의 임베딩 벡터
            indices: (N,) 형태의 원본 인덱스 (Elasticsearch document ID 매핑용)
        
        Returns:
            selected_indices: (k,) 선택된 원본 인덱스
            selected_embeddings: (k, D) 선택된 임베딩 벡터
        
        Algorithm:
            1. 첫 번째 벡터를 랜덤 선택
            2. 이후 각 단계에서, 기존 선택된 벡터들과 가장 먼 벡터 선택
            3. k개 선택될 때까지 반복
        """
        N, D = embeddings.shape
        
        # 샘플 크기가 전체보다 크면 전부 반환
        if N <= self.sample_size:
            logger.info(
                "coreset_skip",
                reason="sample_size >= total_vectors",
                total_vectors=N,
                sample_size=self.sample_size
            )
            return indices, embeddings
        
        logger.info(
            "coreset_sampling_start",
            total_vectors=N,
            target_sample_size=self.sample_size,
            embedding_dim=D
        )
        
        # 선택된 인덱스 저장
        selected_idx = []
        
        # 1. 첫 번째 벡터 랜덤 선택
        first_idx = np.random.randint(0, N)
        selected_idx.append(first_idx)
        
        # 2. Greedy selection
        for _ in range(self.sample_size - 1):
            # 현재까지 선택된 벡터들
            selected_embeddings = embeddings[selected_idx]  # (k, D)
            
            # 모든 벡터와 선택된 벡터들 간의 거리 계산
            # (N, D) vs (k, D) → (N, k)
            distances = self._compute_distances(embeddings, selected_embeddings)
            
            # 각 벡터에서 가장 가까운 선택된 벡터까지의 거리
            min_distances = np.min(distances, axis=1)  # (N,)
            
            # 이미 선택된 벡터는 제외
            min_distances[selected_idx] = -np.inf
            
            # 가장 먼 벡터 선택
            next_idx = np.argmax(min_distances)
            selected_idx.append(next_idx)
        
        # 선택된 인덱스로 필터링
        selected_idx = np.array(selected_idx)
        selected_indices = indices[selected_idx]
        selected_embeddings = embeddings[selected_idx]
        
        logger.info(
            "coreset_sampling_complete",
            selected_count=len(selected_indices),
            reduction_ratio=f"{len(selected_indices)/N:.2%}"
        )
        
        return selected_indices, selected_embeddings
    
    def _compute_distances(
        self,
        vectors: np.ndarray,
        selected: np.ndarray
    ) -> np.ndarray:
        """
        L2 거리 계산 (효율적인 행렬 연산)
        
        Args:
            vectors: (N, D)
            selected: (k, D)
        
        Returns:
            distances: (N, k)
        """
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        
        # (N, 1)
        vectors_norm = np.sum(vectors ** 2, axis=1, keepdims=True)
        
        # (1, k)
        selected_norm = np.sum(selected ** 2, axis=1, keepdims=True).T
        
        # (N, k)
        dot_product = np.dot(vectors, selected.T)
        
        # (N, k)
        distances = vectors_norm + selected_norm - 2 * dot_product
        
        # Numerical stability
        distances = np.maximum(distances, 0.0)
        
        return np.sqrt(distances)


# Singleton instance
_sampler_instance = None


def get_coreset_sampler(sample_size: int = 50) -> CoresetSampler:
    """
    Coreset Sampler 싱글톤 인스턴스 반환
    
    Args:
        sample_size: 대표 벡터 개수
    
    Returns:
        CoresetSampler 인스턴스
    """
    global _sampler_instance
    
    if _sampler_instance is None or _sampler_instance.sample_size != sample_size:
        _sampler_instance = CoresetSampler(sample_size=sample_size)
    
    return _sampler_instance
