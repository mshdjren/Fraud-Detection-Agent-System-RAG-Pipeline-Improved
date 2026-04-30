"""
Percolate Query Builder - Embedding Vector (v1~vN) Based
==========================================================

변경사항:
- use_contrastive 완전 제거
- 원본 categorical 컬럼 처리 제거 (purchase_value, age, sex 등)
- Embedding vector (v1~vN) range query 기반으로 전면 교체
- PCA 차원 및 K값에 따른 실험 케이스 대응

Experiment Cases (이미지 기준):
  - emb_vectors : PCA 없음, 원본 576-dim
  - pca_32      : PCA 32-dim  (v1~v32)
  - pca_64      : PCA 64-dim  (v1~v64)
  - pca_128     : PCA 128-dim (v1~v128)
  - pca_256     : PCA 256-dim (v1~v256)
  - k50         : K=50  클러스터, 원본 dim
  - k100        : K=100 클러스터, 원본 dim
  - k200        : K=200 클러스터, 원본 dim
  - pca_64_k100 : PCA 64-dim + K=100
  - pca_64_k200 : PCA 64-dim + K=200

Percolate Query 예시 (v1~vN range):
  {"cluster_id": 63, "query": {"bool": {"filter": [
    {"range": {"v3": {"lte": -1.43}}},
    {"range": {"v1": {"lte": 0.06}}},
    {"range": {"v42": {"lte": -0.05}}}
  ]}}}

Matching Strategies:
  - V1: Strict AND  (filter, 모든 조건 필수)
  - V2: Flexible    (should + minimum_should_match) [DEFAULT]
  - V3: Top-K Must  (상위 중요 차원 must + 나머지 should)
  - V4: Progressive (tier별 relaxation)
  - V5: Importance Weighted (분산 기반 boost)

Usage:
    builder = get_query_builder(version="v2", experiment_case="pca_64")
    query = builder.build_query(rule_doc)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple


# ===========================
# Experiment Case Configuration
# ===========================

EXPERIMENT_CONFIG = {
    "emb_vectors": {"vector_dim": 576, "description": "Original embeddings, no PCA"},
    "pca_32":      {"vector_dim": 32,  "description": "PCA 32-dim"},
    "pca_64":      {"vector_dim": 64,  "description": "PCA 64-dim"},
    "pca_128":     {"vector_dim": 128, "description": "PCA 128-dim"},
    "pca_256":     {"vector_dim": 256, "description": "PCA 256-dim"},
    "k50":         {"vector_dim": 576, "description": "K=50 clusters"},
    "k100":        {"vector_dim": 576, "description": "K=100 clusters"},
    "k200":        {"vector_dim": 576, "description": "K=200 clusters"},
    "pca_64_k100": {"vector_dim": 64,  "description": "PCA 64-dim + K=100"},
    "pca_64_k200": {"vector_dim": 64,  "description": "PCA 64-dim + K=200"},
}

# 실험 케이스별 min_should_match 기본 비율
# 차원이 높을수록 조건이 많아지므로 비율을 낮춤
MIN_MATCH_RATIO = {
    "emb_vectors": 0.05,  # 576-dim: 5% (매우 완화)
    "pca_32":      0.40,  # 32-dim:  40%
    "pca_64":      0.25,  # 64-dim:  25%
    "pca_128":     0.15,  # 128-dim: 15%
    "pca_256":     0.10,  # 256-dim: 10%
    "k50":         0.05,
    "k100":        0.05,
    "k200":        0.05,
    "pca_64_k100": 0.25,
    "pca_64_k200": 0.25,
}

# 실험 케이스별 상위 tier 비율 (root-side must 조건의 비율)
# 차원이 높을수록 전체 조건 수가 많으므로 상위 비율을 낮춤
UPPER_TIER_RATIO = {
    "emb_vectors": 0.10,  # 576-dim: 조건 많음 → 상위 10%만 must
    "pca_32":      0.50,  # 32-dim:  조건 적음 → 상위 50%를 must
    "pca_64":      0.35,  # 64-dim:  상위 35%
    "pca_128":     0.20,  # 128-dim: 상위 20%
    "pca_256":     0.15,  # 256-dim: 상위 15%
    "k50":         0.10,
    "k100":        0.10,
    "k200":        0.10,
    "pca_64_k100": 0.35,
    "pca_64_k200": 0.35,
}

# coarse routing에서 사용할 절대 조건 수 (V7 Root-Only용)
# 전체 tree depth와 무관하게 상위 N개만 사용
COARSE_TOP_N = {
    "emb_vectors": 2,
    "pca_32":      3,
    "pca_64":      3,   # 직전 논의: depth 1~2 수준
    "pca_128":     2,
    "pca_256":     2,
    "k50":         2,
    "k100":        2,
    "k200":        2,
    "pca_64_k100": 3,
    "pca_64_k200": 3,
}


def vector_to_doc(vector: List[float]) -> Dict[str, float]:
    """
    임베딩 벡터를 ES percolate document로 변환

    Args:
        vector: [0.12, -0.45, 0.33, ...]  (길이 = 실험 케이스의 vector_dim)

    Returns:
        {"v1": 0.12, "v2": -0.45, "v3": 0.33, ...}
    """
    return {f"v{i+1}": float(val) for i, val in enumerate(vector)}


# ===========================
# Base Strategy
# ===========================

class PercolateQueryStrategy(ABC):
    """v1~vN range query 기반 percolate query 빌더"""

    def __init__(self, experiment_case: str = "pca_64"):
        if experiment_case not in EXPERIMENT_CONFIG:
            raise ValueError(
                f"Unknown experiment_case '{experiment_case}'. "
                f"Available: {list(EXPERIMENT_CONFIG.keys())}"
            )
        self.experiment_case = experiment_case
        self.config = EXPERIMENT_CONFIG[experiment_case]
        self.vector_dim = self.config["vector_dim"]

    @abstractmethod
    def build_query(self, rule_doc: Dict, **kwargs) -> Dict:
        """
        rule_doc의 query.bool.filter (v1~vN range 조건들)로부터
        percolate query 생성

        Args:
            rule_doc: {
                "cluster_id": 63,
                "query": {"bool": {"filter": [
                    {"range": {"v3": {"lte": -1.43}}},
                    ...
                ]}}
            }
        Returns:
            ES percolate query dict
        """
        pass

    def _extract_range_conditions(self, rule_doc: Dict) -> List[Dict]:
        """
        rule_doc에서 range 조건 리스트 추출

        Returns:
            [{"range": {"v3": {"lte": -1.43}}}, ...]
        """
        filters = (
            rule_doc.get("query", {})
                    .get("bool", {})
                    .get("filter", [])
        )
        # range 조건만 반환 (v1~vN 필드 기준)
        return [f for f in filters if "range" in f]

    def _calc_min_should_match(self, n_conditions: int) -> int:
        """
        실험 케이스 기반으로 minimum_should_match 계산
        """
        ratio = MIN_MATCH_RATIO.get(self.experiment_case, 0.15)
        return max(1, min(int(n_conditions * ratio), n_conditions))

    def _split_by_depth(
        self,
        conditions: List[Dict],
        upper_ratio: Optional[float] = None
    ):
        """
        조건 리스트를 depth 기준으로 상위/하위로 분리.

        Args:
            conditions: root → leaf 순서의 range 조건 리스트
            upper_ratio: 상위 tier 비율 (None이면 실험 케이스 기본값 사용)

        Returns:
            (upper_conditions, lower_conditions)
            upper = root-side (신뢰도 높음, must 후보)
            lower = leaf-side (신뢰도 낮음, should 후보)
        """
        ratio = upper_ratio or UPPER_TIER_RATIO.get(self.experiment_case, 0.35)
        n = len(conditions)
        split_idx = max(1, min(int(n * ratio), n - 1))
        return conditions[:split_idx], conditions[split_idx:]



# ===========================
# V1: Strict AND
# ===========================

class V1_StrictAND(PercolateQueryStrategy):
    """
    모든 range 조건을 filter (AND)로 처리.
    차원이 낮은 케이스(pca_32)에 적합.

    Query:
    {"bool": {"filter": [
        {"range": {"v1": {"lte": 0.06}}},
        {"range": {"v3": {"lte": -1.43}}},
        ...
    ]}}
    """

    def build_query(self, rule_doc: Dict, **kwargs) -> Dict:
        conditions = self._extract_range_conditions(rule_doc)
        return {"bool": {"filter": conditions}}


# ===========================
# V2: Flexible (DEFAULT)
# ===========================

class V2_FlexibleMatching(PercolateQueryStrategy):
    """
    should + minimum_should_match.
    tree가 선택한 분기 조건 중 일부만 만족해도 매칭.

    Query:
    {"bool": {
        "should": [...range conditions...],
        "minimum_should_match": N
    }}
    """

    def build_query(
        self,
        rule_doc: Dict,
        min_should_match: Optional[int] = None,
        **kwargs
    ) -> Dict:
        conditions = self._extract_range_conditions(rule_doc)
        if not conditions:
            return {"match_all": {}}

        msm = min_should_match or self._calc_min_should_match(len(conditions))
        return {
            "bool": {
                "should": conditions,
                "minimum_should_match": msm,
            }
        }


# ===========================
# V3: Top-K Must + 나머지 Should
# ===========================

class V3_TopKMust(PercolateQueryStrategy):
    """
    트리 분기에서 처음 등장하는 K개 조건을 must(filter)로,
    나머지를 should로 처리.
    → 트리 상단(중요도 높은) 조건을 필수화.

    Query:
    {"bool": {
        "filter": [top_k conditions],
        "should": [remaining conditions],
        "minimum_should_match": N
    }}
    """

    def __init__(self, experiment_case: str = "pca_64", top_k: int = 3):
        super().__init__(experiment_case)
        self.top_k = top_k

    def build_query(
        self,
        rule_doc: Dict,
        top_k: Optional[int] = None,
        **kwargs
    ) -> Dict:
        conditions = self._extract_range_conditions(rule_doc)
        if not conditions:
            return {"match_all": {}}

        k = top_k or self.top_k
        must_conds   = conditions[:k]
        should_conds = conditions[k:]

        query: Dict[str, Any] = {"bool": {"filter": must_conds}}

        if should_conds:
            msm = self._calc_min_should_match(len(should_conds))
            query["bool"]["should"] = should_conds
            query["bool"]["minimum_should_match"] = msm

        return query


# ===========================
# V4: Progressive Relaxation (Tiered)
# ===========================

class V4_ProgressiveRelaxation(PercolateQueryStrategy):
    """
    조건을 3단계 tier로 분리하여 should로 묶음.
    - Tier1 (상위 1/3): boost=3.0
    - Tier2 (중위 1/3): boost=2.0
    - Tier3 (하위 1/3): boost=1.0
    어느 tier든 1개만 만족해도 매칭 (minimum_should_match=1).

    Query:
    {"bool": {
        "should": [
            {"bool": {"filter": [tier1], "boost": 3.0}},
            {"bool": {"filter": [tier2], "boost": 2.0}},
            {"bool": {"filter": [tier3], "boost": 1.0}},
        ],
        "minimum_should_match": 1
    }}
    """

    def build_query(self, rule_doc: Dict, **kwargs) -> Dict:
        conditions = self._extract_range_conditions(rule_doc)
        if not conditions:
            return {"match_all": {}}

        n = len(conditions)
        t1 = n // 3 or 1
        t2 = 2 * n // 3 or t1 + 1

        tiers = [
            (conditions[:t1],   3.0),
            (conditions[t1:t2], 2.0),
            (conditions[t2:],   1.0),
        ]

        should_clauses = [
            {"bool": {"filter": conds, "boost": boost}}
            for conds, boost in tiers if conds
        ]

        return {
            "bool": {
                "should": should_clauses,
                "minimum_should_match": 1,
            }
        }


# ===========================
# V5: Variance-Based Importance Weighted
# ===========================

class V5_ImportanceWeighted(PercolateQueryStrategy):
    """
    range threshold 절대값이 클수록 해당 차원의 분산이 크다 → 중요.
    절대값 기준으로 상위 조건에 boost 부여.

    Query:
    {"bool": {
        "should": [
            {"constant_score": {"filter": {"range": {"v3": ...}}, "boost": 3.0}},
            {"constant_score": {"filter": {"range": {"v1": ...}}, "boost": 1.0}},
            ...
        ],
        "minimum_should_match": N
    }}
    """

    def build_query(self, rule_doc: Dict, **kwargs) -> Dict:
        conditions = self._extract_range_conditions(rule_doc)
        if not conditions:
            return {"match_all": {}}

        # threshold 절대값으로 정렬 (높을수록 분산이 크다고 가정)
        def _threshold_abs(cond: Dict) -> float:
            field = list(cond["range"].keys())[0]
            vals = list(cond["range"][field].values())
            return abs(vals[0]) if vals else 0.0

        sorted_conds = sorted(conditions, key=_threshold_abs, reverse=True)
        n = len(sorted_conds)
        top_n = max(1, n // 3)

        should_clauses = []
        for i, cond in enumerate(sorted_conds):
            boost = 3.0 if i < top_n else 1.0
            should_clauses.append({
                "constant_score": {"filter": cond, "boost": boost}
            })

        msm = self._calc_min_should_match(n)
        return {
            "bool": {
                "should": should_clauses,
                "minimum_should_match": msm,
            }
        }

# ===========================
# V6: Depth-Aware Hierarchical
# ===========================

class V6_DepthAwareHierarchical(PercolateQueryStrategy):
    """
    상위 branch (root-side) → must(filter)
    하위 branch (leaf-side) → should (minimum_should_match=1)

    Configuration:
        UPPER_TIER_RATIO[experiment_case] 로 상위/하위 분리 비율 결정
        (kwargs 무시, config 값 고정 사용)
    """

    def build_query(self, rule_doc: Dict, **kwargs) -> Dict:
        conditions = self._extract_range_conditions(rule_doc)
        if not conditions:
            return {"match_all": {}}

        # UPPER_TIER_RATIO에서 직접 읽음 (_split_by_depth 내부에서 처리)
        upper, lower = self._split_by_depth(conditions)

        query: Dict[str, Any] = {"bool": {"filter": upper}}

        if lower:
            query["bool"]["should"] = lower
            query["bool"]["minimum_should_match"] = 1

        return query


# ===========================
# V7: Root-Only Coarse Routing
# ===========================

class V7_RootOnlyCoarse(PercolateQueryStrategy):
    """
    상위 N개 조건만 filter, 하위 조건 완전 제거.

    Configuration:
        COARSE_TOP_N[experiment_case] 로 N 결정
        (kwargs 무시, config 값 고정 사용)
    """

    def build_query(self, rule_doc: Dict, **kwargs) -> Dict:
        conditions = self._extract_range_conditions(rule_doc)
        if not conditions:
            return {"match_all": {}}

        # COARSE_TOP_N에서 직접 읽음
        n = COARSE_TOP_N.get(self.experiment_case, 3)
        coarse_conditions = conditions[:n]

        return {"bool": {"filter": coarse_conditions}}


# ===========================
# V8: Depth-Decayed Boost
# ===========================

# V8 전용 configuration (V1~V5의 MIN_MATCH_RATIO 패턴과 동일한 방식)
V8_CONFIG = {
    "emb_vectors": {"max_boost": 3.0, "decay_factor": 0.7, "min_boost": 0.1, "msm_ratio": 0.05},
    "pca_32":      {"max_boost": 3.0, "decay_factor": 0.7, "min_boost": 0.1, "msm_ratio": 0.40},
    "pca_64":      {"max_boost": 3.0, "decay_factor": 0.7, "min_boost": 0.1, "msm_ratio": 0.25},
    "pca_128":     {"max_boost": 3.0, "decay_factor": 0.7, "min_boost": 0.1, "msm_ratio": 0.15},
    "pca_256":     {"max_boost": 3.0, "decay_factor": 0.7, "min_boost": 0.1, "msm_ratio": 0.10},
    "k50":         {"max_boost": 3.0, "decay_factor": 0.7, "min_boost": 0.1, "msm_ratio": 0.05},
    "k100":        {"max_boost": 3.0, "decay_factor": 0.7, "min_boost": 0.1, "msm_ratio": 0.05},
    "k200":        {"max_boost": 3.0, "decay_factor": 0.7, "min_boost": 0.1, "msm_ratio": 0.05},
    "pca_64_k100": {"max_boost": 3.0, "decay_factor": 0.7, "min_boost": 0.1, "msm_ratio": 0.25},
    "pca_64_k200": {"max_boost": 3.0, "decay_factor": 0.7, "min_boost": 0.1, "msm_ratio": 0.25},
}

class V8_DepthDecayedBoost(PercolateQueryStrategy):
    """
    depth 순서로 지수 감쇠 boost 부여.

    Configuration:
        V8_CONFIG[experiment_case] 로 boost/decay/msm 결정
        (kwargs 무시, config 값 고정 사용)

    Boost 계산:
        boost_i = max(min_boost, max_boost * decay_factor^i)
    """

    def build_query(self, rule_doc: Dict, **kwargs) -> Dict:
        conditions = self._extract_range_conditions(rule_doc)
        if not conditions:
            return {"match_all": {}}

        # V8_CONFIG에서 직접 읽음
        cfg = V8_CONFIG.get(self.experiment_case, V8_CONFIG["pca_64"])
        max_boost    = cfg["max_boost"]
        decay_factor = cfg["decay_factor"]
        min_boost    = cfg["min_boost"]
        msm_ratio    = cfg["msm_ratio"]

        should_clauses = []
        for i, cond in enumerate(conditions):
            boost = max(min_boost, max_boost * (decay_factor ** i))
            should_clauses.append({
                "constant_score": {
                    "filter": cond,
                    "boost": round(boost, 4),
                }
            })

        msm = max(1, int(len(conditions) * msm_ratio))

        return {
            "bool": {
                "should": should_clauses,
                "minimum_should_match": msm,
            }
        }

class V9_AdaptiveFilterChain(PercolateQueryStrategy):
    UPPER_RATIO = {"pca_64_k100": 0.40, "pca_64": 0.40}
    MID_RATIO   = {"pca_64_k100": 0.30, "pca_64": 0.30}

    def build_query(self, rule_doc: Dict, **kwargs) -> Dict:
        conditions = self._extract_range_conditions(rule_doc)
        if not conditions:
            return {"match_all": {}}

        n = len(conditions)
        u = max(1, int(n * self.UPPER_RATIO.get(self.experiment_case, 0.40)))
        m = max(1, int(n * self.MID_RATIO.get(self.experiment_case, 0.30)))

        upper = conditions[:u]
        mid   = conditions[u:u+m]

        filters = [{"bool": {"filter": upper}}]
        if mid:
            filters.append({"bool": {"filter": mid}})

        return {"bool": {"filter": filters}}
    

V10_CONFIG = {
    "pca_64_k100": {"coarse_n": 3, "fine_msm_ratio": 0.60},
    "pca_64":      {"coarse_n": 3, "fine_msm_ratio": 0.60},
}

class V10_TwoStageCoarseFine(PercolateQueryStrategy):
    def build_query(self, rule_doc: Dict, **kwargs) -> Dict:
        conditions = self._extract_range_conditions(rule_doc)
        if not conditions:
            return {"match_all": {}}

        cfg = V10_CONFIG.get(self.experiment_case,
                             {"coarse_n": 3, "fine_msm_ratio": 0.60})
        n       = cfg["coarse_n"]
        coarse  = conditions[:n]
        fine    = conditions[n:]

        query = {"bool": {"filter": coarse}}
        if fine:
            msm = max(1, int(len(fine) * cfg["fine_msm_ratio"]))
            query["bool"]["should"] = fine
            query["bool"]["minimum_should_match"] = msm

        return query

# ===========================
# V11: Gated Nested Bool
# ===========================

V11_CONFIG = {
    "pca_64_k100": {"gate_ratio": 0.40, "msm_ratio": 0.40},
    "pca_64":      {"gate_ratio": 0.40, "msm_ratio": 0.40},
}

# 수정: 튜닝 variant를 별도 버전으로 등록
V11A_CONFIG = {"pca_64_k100": {"gate_ratio": 0.50, "msm_ratio": 0.50}}
V11B_CONFIG = {"pca_64_k100": {"gate_ratio": 0.50, "msm_ratio": 0.60}}
V11C_CONFIG = {"pca_64_k100": {"gate_ratio": 0.40, "msm_ratio": 0.60}}


class V11_GatedNestedBool(PercolateQueryStrategy):
    """
    상위 조건 filter(gate) + 하위 조건 should(score) 구조.

    gate 통과 못하면 즉시 제외 → early termination 작동
    gate 통과 후 should score로 순위 결정 → 다양성 확보

    구조:
      {"bool": {
        "filter": [{"bool": {"filter": upper_conditions}}],
        "should": lower_conditions,
        "minimum_should_match": "40%"
      }}
    """

    def build_query(self, rule_doc: Dict, **kwargs) -> Dict:
        conditions = self._extract_range_conditions(rule_doc)
        if not conditions:
            return {"match_all": {}}

        cfg = V11_CONFIG.get(self.experiment_case,
                             {"gate_ratio": 0.40, "msm_ratio": 0.40})
        n     = len(conditions)
        g     = max(1, int(n * cfg["gate_ratio"]))
        upper = conditions[:g]
        lower = conditions[g:]

        query: Dict[str, Any] = {
            "bool": {
                "filter": [{"bool": {"filter": upper}}]
            }
        }

        if lower:
            msm = max(1, int(len(lower) * cfg["msm_ratio"]))
            query["bool"]["should"] = lower
            query["bool"]["minimum_should_match"] = msm

        return query


class V11A_GatedNestedBool(V11_GatedNestedBool):
    def build_query(self, rule_doc, **kwargs):
        self.__class__.__dict__  # config만 교체
        cfg = V11A_CONFIG.get(self.experiment_case, {"gate_ratio": 0.50, "msm_ratio": 0.50})
        # V11 build_query 로직 그대로, cfg만 다름
        conditions = self._extract_range_conditions(rule_doc)
        if not conditions:
            return {"match_all": {}}
        n     = len(conditions)
        g     = max(1, int(n * cfg["gate_ratio"]))
        upper = conditions[:g]
        lower = conditions[g:]
        query: Dict[str, Any] = {"bool": {"filter": [{"bool": {"filter": upper}}]}}
        if lower:
            msm = max(1, int(len(lower) * cfg["msm_ratio"]))
            query["bool"]["should"] = lower
            query["bool"]["minimum_should_match"] = msm
        return query

class V11B_GatedNestedBool(V11A_GatedNestedBool):
    def build_query(self, rule_doc, **kwargs):
        cfg = V11B_CONFIG.get(self.experiment_case, {"gate_ratio": 0.50, "msm_ratio": 0.60})
        conditions = self._extract_range_conditions(rule_doc)
        if not conditions:
            return {"match_all": {}}
        n = len(conditions)
        g = max(1, int(n * cfg["gate_ratio"]))
        upper, lower = conditions[:g], conditions[g:]
        query: Dict[str, Any] = {"bool": {"filter": [{"bool": {"filter": upper}}]}}
        if lower:
            msm = max(1, int(len(lower) * cfg["msm_ratio"]))
            query["bool"]["should"] = lower
            query["bool"]["minimum_should_match"] = msm
        return query

class V11C_GatedNestedBool(V11B_GatedNestedBool):
    def build_query(self, rule_doc, **kwargs):
        cfg = V11C_CONFIG.get(self.experiment_case, {"gate_ratio": 0.40, "msm_ratio": 0.60})
        conditions = self._extract_range_conditions(rule_doc)
        if not conditions:
            return {"match_all": {}}
        n = len(conditions)
        g = max(1, int(n * cfg["gate_ratio"]))
        upper, lower = conditions[:g], conditions[g:]
        query: Dict[str, Any] = {"bool": {"filter": [{"bool": {"filter": upper}}]}}
        if lower:
            msm = max(1, int(len(lower) * cfg["msm_ratio"]))
            query["bool"]["should"] = lower
            query["bool"]["minimum_should_match"] = msm
        return query


# ===========================
# V12: Fuzzy Gate + Decay Score
# ===========================

V12_CONFIG = {
    "pca_64_k100": {
        "gate_ratio":  0.30,
        "mid_ratio":   0.40,
        "decay_scale": 0.05,   # threshold ± scale 범위에서 decay
        "decay_decay": 0.5,    # scale 거리에서의 점수 비율
    },
    "pca_64": {
        "gate_ratio":  0.30,
        "mid_ratio":   0.40,
        "decay_scale": 0.05,
        "decay_decay": 0.5,
    },
}

class V12_FuzzyGateDecay(PercolateQueryStrategy):
    """
    상위 hard gate + 중위 range should + 하위 linear decay 구조.

    하위 조건에 decay 적용 → 경계 근처 샘플 recall 개선.
    function_score는 하위 조건에만 적용 → overhead 최소화.

    구조:
      {"function_score": {
        "query": {
          "bool": {
            "filter": upper_conditions,
            "should": mid_conditions,
            "minimum_should_match": 1
          }
        },
        "functions": [linear decay per lower_condition],
        "score_mode": "sum",
        "boost_mode": "sum"
      }}
    """

    def _get_threshold(self, cond: Dict) -> Optional[float]:
        """range 조건에서 threshold 값 추출"""
        field = list(cond["range"].keys())[0]
        vals  = cond["range"][field]
        return float(list(vals.values())[0]) if vals else None

    def _get_field(self, cond: Dict) -> Optional[str]:
        return list(cond["range"].keys())[0] if "range" in cond else None

    def build_query(self, rule_doc: Dict, **kwargs) -> Dict:
        conditions = self._extract_range_conditions(rule_doc)
        if not conditions:
            return {"match_all": {}}

        cfg   = V12_CONFIG.get(self.experiment_case, V12_CONFIG["pca_64"])
        n     = len(conditions)
        g     = max(1, int(n * cfg["gate_ratio"]))
        m     = max(1, int(n * cfg["mid_ratio"]))
        upper = conditions[:g]
        mid   = conditions[g:g+m]
        lower = conditions[g+m:]

        inner_query: Dict[str, Any] = {
            "bool": {"filter": upper}
        }
        if mid:
            inner_query["bool"]["should"] = mid
            inner_query["bool"]["minimum_should_match"] = 1

        if not lower:
            return inner_query

        # 하위 조건 → linear decay function
        functions = []
        for cond in lower:
            field     = self._get_field(cond)
            threshold = self._get_threshold(cond)
            if field and threshold is not None:
                functions.append({
                    "linear": {
                        field: {
                            "origin": threshold,
                            "scale":  cfg["decay_scale"],
                            "decay":  cfg["decay_decay"],
                        }
                    }
                })

        if not functions:
            return inner_query

        return {
            "function_score": {
                "query":      inner_query,
                "functions":  functions,
                "score_mode": "sum",
                "boost_mode": "sum",
            }
        }


# ===========================
# V13: Adaptive Gate (purity 기반)
# ===========================

V13_CONFIG = {
    "pca_64_k100": {
        "high_purity_threshold": 0.8,
        "high_gate_ratio":       0.50,   # purity >= 0.8: 상위 50% filter
        "low_gate_ratio":        0.20,   # purity < 0.8:  상위 20% filter
        "msm_ratio":             0.40,
    },
    "pca_64": {
        "high_purity_threshold": 0.8,
        "high_gate_ratio":       0.50,
        "low_gate_ratio":        0.20,
        "msm_ratio":             0.40,
    },
}

class V13_AdaptiveGate(PercolateQueryStrategy):
    """
    purity 기반 동적 gate 크기 결정.

    purity >= threshold: gate 조건 수 많음 (엄격, 빠름)
    purity <  threshold: gate 조건 수 적음 (완화, recall 보존)

    rule_doc에서 purity를 읽어 gate 크기 결정.
    purity 미존재 시 low_gate_ratio 적용 (안전 폴백).
    """

    def build_query(self, rule_doc: Dict, **kwargs) -> Dict:
        conditions = self._extract_range_conditions(rule_doc)
        if not conditions:
            return {"match_all": {}}

        cfg     = V13_CONFIG.get(self.experiment_case, V13_CONFIG["pca_64"])
        purity  = float(rule_doc.get("purity", 0.0))
        n       = len(conditions)

        gate_ratio = (
            cfg["high_gate_ratio"]
            if purity >= cfg["high_purity_threshold"]
            else cfg["low_gate_ratio"]
        )

        g     = max(1, int(n * gate_ratio))
        upper = conditions[:g]
        lower = conditions[g:]

        query: Dict[str, Any] = {
            "bool": {
                "filter": [{"bool": {"filter": upper}}]
            }
        }

        if lower:
            msm = max(1, int(len(lower) * cfg["msm_ratio"]))
            query["bool"]["should"] = lower
            query["bool"]["minimum_should_match"] = msm

        return query

# ===========================
# Query Builder Registry
# ===========================

QUERY_BUILDERS = {
    "v1": V1_StrictAND,
    "v2": V2_FlexibleMatching,
    "v3": V3_TopKMust,
    "v4": V4_ProgressiveRelaxation,
    "v5": V5_ImportanceWeighted,
    "v6": V6_DepthAwareHierarchical,
    "v7": V7_RootOnlyCoarse,
    "v8": V8_DepthDecayedBoost,
    "v9": V9_AdaptiveFilterChain,
    "v10": V10_TwoStageCoarseFine,
    "v11": V11_GatedNestedBool,
    "v12": V12_FuzzyGateDecay,
    "v13": V13_AdaptiveGate,
    "v14": V1_StrictAND,   # ← 추가
    "v11a": V11A_GatedNestedBool,
    "v11b": V11B_GatedNestedBool,
    "v11c": V11C_GatedNestedBool,
}


def get_query_builder(
    version: str = "v2",
    experiment_case: str = "pca_64"
) -> PercolateQueryStrategy:
    """
    Query builder 인스턴스 반환

    Args:
        version: "v1" ~ "v5"
        experiment_case: "pca_32" | "pca_64" | "pca_128" | "pca_256" |
                         "k50" | "k100" | "k200" |
                         "pca_64_k100" | "pca_64_k200" | "emb_vectors"

    Returns:
        PercolateQueryStrategy instance
    """
    version = version.lower()
    if version not in QUERY_BUILDERS:
        raise ValueError(
            f"Unknown version '{version}'. Available: {list(QUERY_BUILDERS.keys())}"
        )
    if experiment_case not in EXPERIMENT_CONFIG:
        raise ValueError(
            f"Unknown experiment_case '{experiment_case}'. "
            f"Available: {list(EXPERIMENT_CONFIG.keys())}"
        )
    return QUERY_BUILDERS[version](experiment_case=experiment_case)


def get_vector_dim(experiment_case: str) -> int:
    """실험 케이스에 따른 벡터 차원 반환"""
    return EXPERIMENT_CONFIG.get(experiment_case, {}).get("vector_dim", 576)


if __name__ == "__main__":
    import json

    # pca_64 케이스 rule 예시
    rule_doc = {
        "cluster_id": 63,
        "tree_id": 0,
        "leaf_id": 0,
        "support": 12,
        "purity": 0.5,
        "query": {"bool": {"filter": [
            {"range": {"v3":  {"lte": -1.4347915649414062}}},
            {"range": {"v1":  {"lte":  0.06692790985107422}}},
            {"range": {"v2":  {"lte": -0.5605550408363342}}},
            {"range": {"v4":  {"lte":  0.321927547454834}}},
            {"range": {"v5":  {"lte": -0.17081022262573242}}},
            {"range": {"v42": {"lte": -0.0519564151763916}}},
        ]}}
    }

    for ver in ["v1", "v2", "v3", "v4", "v5"]:
        builder = get_query_builder(ver, "pca_64")
        q = builder.build_query(rule_doc)
        print(f"\n{'='*60}")
        print(f"{ver.upper()} — pca_64")
        print(json.dumps(q, indent=2))
