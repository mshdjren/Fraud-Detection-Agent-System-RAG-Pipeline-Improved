"""
oracle_extractor.py — GRPO Planner Oracle Rollout 수집
=========================================================
목적:
  controller.py의 신규 POST /rollout 엔드포인트를 test set 앞 10%(hard_
  negative_miner.py와 동일 서브셋 관례)에 대해 반복 호출해, state별
  action(stop/global_knn/contextual_knn×feature후보) 전수 rollout 결과를
  parquet로 모은다. 인수인계 §5 "1. rollout 추출"에 해당하는 단계.

  V1~V3 정밀 검증(oracle 분포/상관/state별 분기)과 baseline A/B/C 학습은
  다음 단계 — 이 스크립트는 원재료(raw rollout row) 수집까지만 담당한다.
  summary.txt에는 참고용 V1 스냅샷(oracle action 분포)에 더해, "왜 stop이
  압도적인가"를 진단하는 섹션(_write_diagnostics)을 포함한다:
    (1) evaluator wm_type 분포 — accept만 나오는지, 다양하게 나오는지
    (2) sufficiency(nd_before 재계산) × wm_type 유무 교차표 — SKIP_ANALYZER_
        ON_CONFIDENT가 stop 압도의 원인인지 확인
    (2b) derived_cls(nd_before 기준) vs y 일치율 참고치
    (3) action별 실행 오류율 + contextual_knn 후보 0건 test_idx 비율 —
        직전 turn의 retriever should_boost/ES 매핑 버그가 이번 결과에 얼마나
        영향을 줬는지 정량화
    (4) action별 Δmargin/reward 분포 — REWARD_LAMBDA 재보정 참고

설계 원칙(hard_negative_miner.py / agent_batch_inference.py와 동일하게 유지):
  - GTLoader / loader.get_test_case()를 무변경 재사용 (agent_batch_inference.py
    run_batch()의 로딩 패턴을 그대로 가져옴). ★ [수정] get_test_case()가 주는
    tc는 router.py의 TestCase와 동일한 flat 스키마({embedding, purchase_value,
    age, sex, source, browser, IP_country, ...})라, controller.py의 /rollout
    (nested transaction dict + base_* 필드)에 그대로 넣을 수 없다 — 이 스크립트가
    직접 router → retriever를 1회 호출(_base_call)해 base_*를 채우고, tc의
    피처들을 transaction dict로 감싼다. orchestrator를 거치지 않고 controller를
    직접 호출하기 때문에 필요한 단계다(orchestrator였다면 이 base call을
    내부적으로 대신 해줬을 것).
  - sample_indices 기본값 = 전체 test set의 앞 10% (hard_negative_miner.py
    summary 문구 "n_sample_indices: 전체 test set의 앞 10%"와 동일 관례 —
    두 잡이 같은 서브셋을 보게 해 mining/oracle 결과를 나란히 비교할 수 있다).
  - 저장: 로컬 parquet + summary.txt → GCS 업로드
    gs://fraud_ecommm/oracle/{experiment_case}/query_{percolate_version}_
    pct_{coreset_pct}/oracle_rollout_{MMDD}.parquet
    (버킷 루트의 oracle/ 폴더 — tree-search/analyzer_ft/... 하위가 아니라
    스크린샷 기준 최상위 폴더에 직접 export)

★ [확인 필요 — 이번 turn에 검증 불가한 가정]
  - CONTROLLER_URL 기본값 "http://anomaly-controller:80"은 다른 서비스
    (anomaly-retriever/-router/-analyzer/-orchestrator) 네이밍 관례에서
    추정한 값이다. 실제 controller.py(Anomaly Planner) K8s Service 이름이
    다르면 env var로 덮어쓸 것. ROUTER_URL/RETRIEVER_URL은 agent_tools_mcp_
    server.py와 동일한 기본값을 그대로 재사용했다(검증됨).
  - GTLoader.load_all()/get_test_case()/router.py TestCase/RouteResponse,
    retriever.py RetrieveRequest/RetrieveResponse는 이번 turn에 업로드된
    실제 코드에서 시그니처를 직접 확인해 반영했다(이전 turn의 추정을 대체).

Usage:
    python oracle_extractor.py \\
        --experiment-case pca_64_k100 \\
        --percolate-version v14 \\
        --n-samples 500 \\
        --output-dir /results/oracle
"""

from __future__ import annotations

import os
import json
import time
import argparse
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import httpx
import structlog

from gt_loader import GTLoader, VALID_EXPERIMENT_CASES

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

logger = structlog.get_logger()

# ============================================================
# Config
# ============================================================
# ★ [진단용] controller.py/local_analyzer.py와 동일한 threshold 상수.
#   오라클 데이터에는 sufficiency/cls가 직접 실려있지 않으므로(RolloutResponse
#   에 없음), nd_before로부터 같은 수식으로 재계산해 진단에 쓴다 — controller.py
#   변경 없이 오라클 코드만으로 "왜 stop이 압도적인가"를 확인하기 위함.
BOUNDARY_LOW      = float(os.getenv("SUFFICIENCY_BOUNDARY_LOW", "0.3"))
BOUNDARY_HIGH     = float(os.getenv("SUFFICIENCY_BOUNDARY_HIGH", "0.55"))
CLS_ND_THRESHOLD  = float(os.getenv("CLS_ND_THRESHOLD", "0.5"))

CONTROLLER_URL = os.getenv("CONTROLLER_URL", "http://anomaly-controller:80")
ROUTER_URL     = os.getenv("ROUTER_URL", "http://anomaly-router:80")
RETRIEVER_URL  = os.getenv("RETRIEVER_URL", "http://anomaly-retriever:80")
BASE_TOP_K     = int(os.getenv("BASE_TOP_K", "3"))   # controller.py BASE_TOP_K와 동일 기본값
GCS_BUCKET_NAME = "fraud_ecommm"

# ★ [버그 수정] 실행 중 "All connection attempts failed"(httpx.ConnectError)로
#   test_idx가 통째로 유실되는 사례가 다수 관측됨(재현: 4연속 test_idx 179/187/
#   188/189, 짧은 시간 창에 몰림) — 라우터/리트리버 파드의 순간적 재시작·HPA
#   스케일링·k8s 서비스 네트워킹 hiccup 등 일시적 원인일 가능성이 높다. 재시도
#   로직이 아예 없어서 blip 1회 = 해당 test_idx 영구 유실이었다. exponential
#   backoff로 감싼다 — HTTP 4xx(요청 자체가 잘못됨, 예: 422/400)는 재시도해도
#   무의미하므로 재시도 대상에서 제외하고 ConnectError/TimeoutException/5xx만
#   재시도한다.
ORACLE_MAX_RETRIES = int(os.getenv("ORACLE_MAX_RETRIES", "3"))
ORACLE_RETRY_BASE_DELAY = float(os.getenv("ORACLE_RETRY_BASE_DELAY", "0.5"))


async def _with_retry(coro_fn, *, test_idx: int, what: str):
    """coro_fn: 인자 없는 async callable(클로저로 감싸서 전달). 연결 실패/
    5xx/timeout만 재시도하고, 4xx(요청 자체 오류)는 즉시 raise한다."""
    last_exc = None
    for attempt in range(ORACLE_MAX_RETRIES + 1):
        try:
            return await coro_fn()
        except httpx.HTTPStatusError as e:
            if e.response.status_code < 500:
                raise   # 4xx는 재시도해도 같은 결과 — 즉시 전파
            last_exc = e
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as e:
            last_exc = e
        if attempt < ORACLE_MAX_RETRIES:
            delay = ORACLE_RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning("transient_error_retry", test_idx=test_idx, what=what,
                           attempt=attempt + 1, delay_s=delay, error=str(last_exc))
            await asyncio.sleep(delay)
    raise last_exc

# ★ 버킷 최상위 oracle/ 폴더(스크린샷 기준 — tree-search/analyzer_ft/...와
#   달리 analyzer_ft 하위가 아니다). 폴더 자체는 이미 존재한다고 가정.
GCS_ORACLE_PATH_TEMPLATE = (
    "oracle/{experiment_case}/"
    "query_{percolate_version}_pct_{coreset_pct}/"
    "oracle_rollout_{mining_date}.parquet"
)

# rollout 응답의 candidates[] 리스트를 펼칠 때 함께 남길 최상위 컬럼
_TOP_LEVEL_COLS = (
    "test_idx", "y", "cluster_id", "wm_type", "wm_axes",
    "agrees", "differs", "deviant", "rare", "persona_match", "spread_tag",
    "nd_before", "oracle_action",
    # ★ [신규] planner SFT/GRPO 학습 시 build_planner_user_prompt() 재구성용
    #   원재료 — candidates/agrees 같은 파생값만으로는 프롬프트를 못 되돌린다.
    "persona", "neighbors",
)
_LIST_COLS = (
    "wm_axes", "agrees", "differs", "deviant", "rare", "persona_match",
    "features", "features_used", "neighbor_oidx",
    "persona", "neighbors",   # ★ 신규 — dict/list라 동일하게 JSON 직렬화 필요
)


# ============================================================
# GT 로딩 — agent_batch_inference.py run_batch()와 동일 패턴 재사용
# ============================================================
def load_indexed_test_cases(args) -> "tuple[GTLoader, Any, Any, np.ndarray, List[int]]":
    """
    Returns: (loader, test_data, test_emb, gt_labels, indices)
    indices: 이번 job이 순회할 test set 위치 목록 (기본 = 앞 10%).

    ★ [최적화] load_all() 대신 load_minimal()을 쓴다 — oracle rollout은
    retrieval을 controller.py /rollout이 실시간으로 수행하므로 gt_loader의
    coreset parquet/GT top5 비교 데이터를 전혀 쓰지 않는다. load_all()은
    이 데이터를 위해 coreset parquet(대용량, coreset_percentage=100 기준)을
    매번 다운로드하고 GT 컬럼 존재를 엄격히 검증하는데, oracle 목적으로는
    불필요한 다운로드/전제조건이다.
    """
    loader = GTLoader(
        experiment_case=args.experiment_case,
        coreset_percentage=str(args.coreset_percentage),
    )
    test_data, test_emb, gt_data = loader.load_minimal()
    gt_labels = gt_data["label"]
    n_total = len(test_data)

    if args.n_samples is not None:
        n_samples = int(args.n_samples)
    else:
        # ★ hard_negative_miner.py 관례 — "전체 test set의 앞 10%"
        n_samples = max(1, int(0.1 * n_total))

    start = max(0, int(args.start_index))
    end = min(start + n_samples, n_total)
    indices = list(range(start, end))
    return loader, test_data, test_emb, gt_labels, indices


# ============================================================
# Base call — router + retriever (orchestrator를 거치지 않고 직접 호출)
# ============================================================
# ★ [중요 수정] GTLoader.get_test_case()가 반환하는 tc는 orchestrator/router의
#   flat TestCase 스키마({embedding, purchase_value, age, sex, source, browser,
#   IP_country, ...} — router.py TestCase와 동일)다. controller.py의 /rollout
#   (RolloutRequest)은 이와 다른 스키마(nested `transaction: Dict` + base_*
#   필드)를 쓰고, base_provided=True 경로는 "router+retriever를 이미 마쳤다"
#   고 가정한다(node_retrieve가 round0에 재검색하지 않고 base_top1_distance/
#   base_neighbors를 그대로 씀). orchestrator를 거치지 않고 controller를
#   직접 호출하는 이 스크립트는, 그래서 router→retriever를 직접 한 번 호출해
#   base_*를 채워야 한다 — 이게 없으면 vec_index가 비어 retriever가 422를
#   낸다(실제로 놓쳤던 부분).
async def _base_call(client: httpx.AsyncClient, tc: Dict, args, test_idx: int = -1) -> Dict:
    """router /route → retriever /retrieve 순서로 1회 호출해 controller
    /rollout의 base_* 필드를 채운다. 반환 dict는 RolloutRequest에 그대로
    ** 언패킹 가능하도록 base_ 접두사 키로 만든다."""
    async def _route():
        r = await client.post(
            f"{ROUTER_URL}/route", json=tc,
            headers={"X-Router-Case": args.experiment_case,
                    "X-Router-Version": args.percolate_version},
            timeout=args.timeout,
        )
        r.raise_for_status()
        return r.json()
    route = await _with_retry(_route, test_idx=test_idx, what="router")

    cluster_id = route["primary_cluster_id"]
    vec_index = route["vec_index"]

    async def _retrieve():
        r = await client.post(
            f"{RETRIEVER_URL}/retrieve",
            json={"embedding": tc["embedding"], "cluster_ids": [cluster_id],
                  "vec_index": vec_index, "top_k": BASE_TOP_K},
            timeout=args.timeout,
        )
        r.raise_for_status()
        return r.json()
    retrieved = await _with_retry(_retrieve, test_idx=test_idx, what="retriever")

    persona = route.get("persona")
    return {
        "base_cluster_id": cluster_id,
        "base_top1_distance": retrieved.get("top1_distance"),
        "base_distance_band": None,  # controller가 자체 계산(top1_distance로부터) — 필수 아님
        "base_cluster_persona_summary": (persona or {}).get("description"),
        "base_vec_index": vec_index,
        "base_match_type": route.get("match_type"),
        "base_top_k_clusters": route.get("top_k_clusters"),
        "base_persona": persona,
        "base_neighbors": retrieved.get("results"),
    }


# ============================================================
# /rollout 호출
# ============================================================
async def call_rollout(client: httpx.AsyncClient, payload: Dict, timeout: int) -> Optional[Dict]:
    async def _rollout():
        r = await client.post(f"{CONTROLLER_URL}/rollout", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    try:
        return await _with_retry(_rollout, test_idx=payload.get("test_idx", -1), what="rollout")
    except httpx.TimeoutException:
        logger.warning("rollout_timeout", test_idx=payload.get("test_idx"))
        return None
    except httpx.HTTPStatusError as e:
        logger.warning("rollout_http_error", test_idx=payload.get("test_idx"),
                       status_code=e.response.status_code, body=e.response.text[:300])
        return None
    except Exception as e:
        logger.error("rollout_failed", test_idx=payload.get("test_idx"), error=str(e))
        return None


def _flatten_rollout_response(resp: Dict) -> List[Dict]:
    """RolloutResponse(dict) → candidates[] 각각을 1행으로 펼친 row 목록."""
    top = {k: resp.get(k) for k in _TOP_LEVEL_COLS}
    rows = []
    for cand in resp.get("candidates", []):
        row = dict(top)
        row.update(cand)
        row["is_oracle"] = False  # 아래에서 reward argmax로 재확정
        rows.append(row)
    if rows:
        best_idx = max(range(len(rows)), key=lambda i: rows[i].get("reward", float("-inf")))
        for i, r in enumerate(rows):
            r["is_oracle"] = (i == best_idx)
    return rows


# ============================================================
# Batch 실행
# ============================================================
async def run_oracle(args) -> pd.DataFrame:
    loader, test_data, test_emb, gt_labels, indices = load_indexed_test_cases(args)
    n_total = len(test_data)

    print("=" * 70)
    print("ORACLE ROLLOUT EXTRACTION")
    print("=" * 70)
    print(f"  controller_url     : {CONTROLLER_URL}")
    print(f"  experiment_case    : {args.experiment_case}")
    print(f"  percolate_version  : {args.percolate_version}")
    print(f"  coreset_percentage : {args.coreset_percentage}")
    print(f"  n_samples          : {len(indices)} / {n_total}  (start={indices[0] if indices else '-'})")
    print("=" * 70)

    CONCURRENCY = int(os.getenv("ORACLE_CONCURRENCY", "4"))
    _sem = asyncio.Semaphore(CONCURRENCY)
    _done = {"n": 0, "err": 0}
    all_rows: List[Dict] = []

    async def _one(idx: int, client: httpx.AsyncClient):
        tc = loader.get_test_case(
            idx, test_data, test_emb,
            percolate_version=args.percolate_version,
            experiment_case=args.experiment_case,
        )
        # tc는 router/orchestrator의 flat TestCase 스키마 — controller의
        # RolloutRequest는 nested transaction dict를 쓰므로 변환한다.
        transaction = {k: v for k, v in tc.items()
                       if k not in ("embedding", "experiment_case", "percolate_version")}
        async with _sem:
            try:
                base = await _base_call(client, tc, args, test_idx=idx)
            except Exception as e:
                logger.warning("base_call_failed", test_idx=idx, error=str(e))
                _done["n"] += 1
                _done["err"] += 1
                return []
            payload = {
                "transaction": transaction,
                "embedding": tc["embedding"],
                "test_idx": idx,
                "y": int(gt_labels[idx]),
                **base,
            }
            result = await call_rollout(client, payload, args.timeout)
        _done["n"] += 1
        if result is None:
            _done["err"] += 1
            logger.warning("rollout_missing", test_idx=idx)
            return []
        rows = _flatten_rollout_response(result)
        for r in rows:
            r["experiment_case"] = args.experiment_case
            r["percolate_version"] = args.percolate_version
            r["coreset_percentage"] = args.coreset_percentage
        if _done["n"] % 20 == 0 or _done["n"] == 1:
            logger.info("oracle_progress", done=_done["n"], total=len(indices),
                        errors=_done["err"], concurrency=CONCURRENCY)
        return rows

    _limits = httpx.Limits(max_connections=CONCURRENCY * 2, max_keepalive_connections=CONCURRENCY * 2)
    async with httpx.AsyncClient(limits=_limits) as client:
        results = await asyncio.gather(*[_one(idx, client) for idx in indices])
    for rows in results:
        all_rows.extend(rows)

    print(f"\n[oracle_extractor] test_idx 완료 {_done['n']}/{len(indices)}  실패 {_done['err']}건")
    if not all_rows:
        logger.warning("no_rollout_rows")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    for col in _LIST_COLS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)
    return df


# ============================================================
# Save: local + GCS (hard_negative_miner.py save/_write_summary/_upload_to_gcs와 동일 패턴)
# ============================================================
def save(df: pd.DataFrame, args, upload_gcs: bool = True) -> str:
    if df.empty:
        print("[oracle_extractor] 저장할 데이터가 없습니다.")
        return ""

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%m%d")

    fname = (
        f"oracle_rollout_{args.experiment_case}_"
        f"query_{args.percolate_version}_pct_{args.coreset_percentage}_"
        f"{date_str}.parquet"
    )
    local_parquet = out / fname
    local_summary = out / fname.replace(".parquet", "_summary.txt")

    df.to_parquet(str(local_parquet), index=False)
    print(f"\n✅ [save] parquet: {local_parquet}  ({len(df)} rows)")

    _write_summary(df, local_summary, args)

    if upload_gcs and GCS_AVAILABLE:
        gcs_base = GCS_ORACLE_PATH_TEMPLATE.format(
            experiment_case=args.experiment_case,
            percolate_version=args.percolate_version,
            coreset_pct=args.coreset_percentage,
            mining_date=date_str,
        )
        _upload_to_gcs(str(local_parquet), gcs_base)
        _upload_to_gcs(str(local_summary), gcs_base.replace(".parquet", "_summary.txt"))

    return str(local_parquet)


def _sufficiency_from_nd(nd) -> str:
    """local_analyzer.py의 _suf_from_nd와 동일 수식 (controller.py BOUNDARY_LOW/HIGH와 동일 상수)."""
    if nd is None or (isinstance(nd, float) and np.isnan(nd)):
        return "unknown"
    return "ambiguous" if (BOUNDARY_LOW <= nd < BOUNDARY_HIGH) else "confident"


def _write_diagnostics(df: pd.DataFrame, f) -> None:
    """
    ★ [신규] "왜 stop이 압도적인가"의 원인 분석 섹션. agent_batch_inference.py의
    "[재검색 효과]"/"wm direction" 리포팅 스타일(라벨-값 표, 비율 병기)을
    최대한 그대로 따른다. 전부 이미 수집된 oracle_df 컬럼(wm_type/nd_before/
    action/error/reward/delta_margin)만으로 계산 — controller.py/local_
    analyzer.py는 건드리지 않는다.
    """
    base = df.drop_duplicates("test_idx").copy()
    n = len(base)

    # ── (1) wm_type(what_missing typed) 분포 — "accept만 나오는 거 아닌가?" ──
    f.write("\n── [진단 1] evaluator wm_type 분포 (test_idx 기준) ─────────\n")
    wm_counts = base["wm_type"].value_counts(dropna=False)
    for k, v in wm_counts.items():
        label = "None (analyzer 미호출)" if pd.isna(k) else str(k)
        f.write(f"  {label:<28s}: {v:5d}  ({v/n:.1%})\n")
    n_wm_notna = int(base["wm_type"].notna().sum())
    if n_wm_notna > 0:
        distinct = sorted(base.loc[base["wm_type"].notna(), "wm_type"].unique().tolist())
        f.write(f"  analyzer가 실제로 호출된 표본 중 관측된 고유 wm_type: {distinct}\n")
        if len(distinct) <= 1:
            f.write("  ⚠ 관측된 wm_type이 1종류뿐 — evaluator가 특정 유형만 출력하고 있을 가능성\n"
                    "    (SFT 데이터 분포 편향 또는 프롬프트/파싱 문제 의심, local_analyzer.py 쪽 점검 필요)\n")
    else:
        f.write("  ⚠ 표본 전체에서 analyzer가 한 번도 호출되지 않음 — 아래 진단 2와 함께 확인\n")

    # ── (2) sufficiency 분포 (nd_before 재계산) + wm_type 유무 교차표 ──────
    # 가설: SKIP_ANALYZER_ON_CONFIDENT(기본 true)가 nd_before를 band 밖으로
    # 보내는 표본 대부분에서 analyzer 호출 자체를 건너뛴다 — 그 표본들은
    # candidate_pool이 원천적으로 비어 stop이 자동 승자가 된다.
    base["_sufficiency"] = base["nd_before"].apply(_sufficiency_from_nd)
    f.write("\n── [진단 2] sufficiency 분포 (nd_before 재계산, band=[%.2f,%.2f)) ──\n"
            % (BOUNDARY_LOW, BOUNDARY_HIGH))
    suf_counts = base["_sufficiency"].value_counts()
    for k, v in suf_counts.items():
        f.write(f"  {k:<12s}: {v:5d}  ({v/n:.1%})\n")

    f.write("\n  교차표 (sufficiency × wm_type 유무) — SKIP_ANALYZER_ON_CONFIDENT 가설 검증:\n")
    base["_wm_present"] = base["wm_type"].notna()
    xtab = pd.crosstab(base["_sufficiency"], base["_wm_present"])
    for suf_val in xtab.index:
        n_no_wm  = int(xtab.loc[suf_val, False]) if False in xtab.columns else 0
        n_has_wm = int(xtab.loc[suf_val, True]) if True in xtab.columns else 0
        f.write(f"    {suf_val:<12s}: wm_type 없음={n_no_wm:5d}   wm_type 있음={n_has_wm:5d}\n")
    conf_notna = int(xtab.loc["confident", True]) if ("confident" in xtab.index and True in xtab.columns) else 0
    amb_isna   = int(xtab.loc["ambiguous", False]) if ("ambiguous" in xtab.index and False in xtab.columns) else 0
    f.write(f"  → confident인데 wm_type 있음 = {conf_notna}건  "
            f"(0에 가까워야 정상 — SKIP_ANALYZER_ON_CONFIDENT가 예상대로 작동 중)\n")
    f.write(f"  → ambiguous인데 wm_type 없음 = {amb_isna}건  "
            f"(0에 가까워야 정상 — 있으면 analyzer 호출 실패가 조용히 넘어갔을 가능성)\n")

    # ── (2b) derived_cls(nd_before>=CLS_ND_THRESHOLD) vs y 일치율 (참고) ──
    base["_derived_cls"] = np.where(base["nd_before"] >= CLS_ND_THRESHOLD, 1, 0)
    if base["y"].notna().any():
        acc = float((base["_derived_cls"] == base["y"]).mean())
        f.write(f"\n  derived_cls(nd_before≥{CLS_ND_THRESHOLD}) vs y 일치율 참고치: {acc:.1%}\n"
                f"  (★ 모델 cls 아님 — det-mode와 동일 threshold로 nd_before만 보고 근사한 값.\n"
                f"     너무 낮으면 nd 자체의 판별력을 의심, 너무 높으면 애초에 재검색이 필요한\n"
                f"     표본이 적다는 뜻이라 stop 우세가 자연스러운 결과일 수 있음)\n")

    # ── (3) action별 실행 오류/degraded 비율 (retriever 매핑 버그 영향 정량화) ──
    f.write("\n── [진단 3] action별 실행 오류(error) 비율 ──────────────────\n")
    if "error" in df.columns:
        for act in ("global_knn", "contextual_knn"):
            sub = df[df["action"] == act]
            if len(sub) == 0:
                f.write(f"  {act:<16s}: 후보 0건\n")
                continue
            n_err = int(sub["error"].notna().sum())
            f.write(f"  {act:<16s}: n={len(sub):5d}  error={n_err:4d}  ({n_err/len(sub):.1%})\n")
    else:
        f.write("  error 컬럼 없음 → skip\n")

    # contextual_knn 후보 자체가 0개였던 test_idx 비율 (agrees∪persona_match가 빈 경우)
    ctx_per_idx = df[df["action"] == "contextual_knn"].groupby("test_idx").size()
    n_zero_ctx = n - len(ctx_per_idx)
    f.write(f"\n  contextual_knn 후보가 0개였던 test_idx: {n_zero_ctx}/{n} ({n_zero_ctx/n:.1%})\n"
            f"  (agrees∪persona_match가 빈 상태 — analyzer 미호출 표본과 상당 부분 겹칠 것으로 예상)\n")

    # ── (4) Δmargin/reward 분포 (λ 재보정 참고 — 인수인계 §1 Reward Stage 1) ──
    f.write("\n── [진단 4] action별 Δmargin/reward 분포 (REWARD_LAMBDA 재보정 참고) ──\n")
    for act in ("global_knn", "contextual_knn"):
        sub = df[df["action"] == act]
        if len(sub) == 0:
            continue
        dm = sub["delta_margin"].astype(float)
        rw = sub["reward"].astype(float)
        pos_rate = float((dm > 0).mean())
        f.write(f"  {act:<16s}: n={len(sub):5d}  "
                f"Δmargin mean={dm.mean():+.4f} median={dm.median():+.4f}  "
                f"reward mean={rw.mean():+.4f}  Δmargin>0 비율={pos_rate:.1%}\n")
    f.write("  (stop이 압도적인 이유가 '재검색이 진짜 안 통해서'인지 'cost 페널티가 관대한 이득도\n"
            "   못 넘게 세서'인지 이 mean/median으로 가늠 — 인수인계 §1 REWARD_LAMBDA는 현재\n"
            "   placeholder 0.1, 이 표를 보고 역산 권장)\n")


def _write_summary(df: pd.DataFrame, path: Path, args) -> None:
    """
    참고용 V1 스냅샷(oracle action 분포) + 원인 분석 진단 섹션(_write_diagnostics).
    정식 V1~V3 판정 및 baseline A/B/C 비교는 다음 단계의 집계 스크립트 몫이다
    (인수인계 §4).
    """
    n_test_idx = df["test_idx"].nunique()
    oracle_rows = df[df.get("is_oracle", False) == True]  # noqa: E712
    oracle_dist = (oracle_rows["action"].value_counts(normalize=True)
                   if not oracle_rows.empty else pd.Series(dtype=float))

    with open(path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("ORACLE ROLLOUT EXTRACTION SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"experiment_case    : {args.experiment_case}\n")
        f.write(f"percolate_version  : {args.percolate_version}\n")
        f.write(f"coreset_percentage : {args.coreset_percentage}\n")
        f.write(f"n_test_idx         : {n_test_idx}  (전체 test set의 앞 10% 관례)\n")
        f.write(f"n_rollout_rows     : {len(df)}  (test_idx × action 전수)\n")

        f.write("\n── [참고 V1 스냅샷] oracle action 분포 ──────────────────\n")
        f.write("  (정식 판정 기준: global_knn 비율 < 80% — 인수인계 §4.\n")
        f.write("   여기 수치는 참고용이며, 정식 검증은 다음 단계 집계 스크립트에서.)\n")
        for act, p in oracle_dist.items():
            f.write(f"  {act:<16s} : {p:.1%}\n")
        if not oracle_dist.empty:
            global_share = float(oracle_dist.get("global_knn", 0.0))
            verdict = "PASS (참고)" if global_share < 0.8 else "주의 — global_knn 비중 높음"
            f.write(f"  V1 참고 판정      : {verdict}\n")

        if "y" in df.columns:
            labels = df.drop_duplicates("test_idx")["y"]
            f.write("\n── Label 분포 ────────────────────────────────────────\n")
            f.write(f"  normal (y=0): {int((labels == 0).sum())}\n")
            f.write(f"  fraud  (y=1): {int((labels == 1).sum())}\n")

        if "error" in df.columns:
            n_err = int(df["error"].notna().sum())
            f.write(f"\n── candidate 실행 오류 (전체) ────────────────────────\n")
            f.write(f"  n_error_candidates: {n_err} / {len(df)}\n")

        # ★ [신규] 원인 분석 진단 섹션
        _write_diagnostics(df, f)

        f.write("\n" + "=" * 70 + "\n")

    print(f"✅ [save] summary: {path}")


def _upload_to_gcs(local_path: str, gcs_path: str) -> None:
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        print(f"✅ Uploaded: gs://{GCS_BUCKET_NAME}/{gcs_path}")
    except Exception as e:
        print(f"❌ GCS upload failed ({gcs_path}): {e}")


# ============================================================
# Entrypoint
# ============================================================
async def main():
    parser = argparse.ArgumentParser(
        description="GRPO Planner oracle rollout extraction (controller.py /rollout)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiment-case", type=str, default="pca_64_k100",
                        choices=VALID_EXPERIMENT_CASES)
    parser.add_argument("--percolate-version", type=str, default="v14")
    parser.add_argument("--coreset-percentage", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=None,
                        help="비우면 전체 test set의 앞 10% 자동 (hard_negative_miner.py와 동일 관례)")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=300,
                        help="/rollout 1건 timeout(초) — action 전수(최대 ~12개) 병렬 실행 포함")
    parser.add_argument("--output-dir", type=str, default="/results/oracle")
    parser.add_argument("--no-gcs-upload", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    df = await run_oracle(args)
    save(df, args, upload_gcs=not args.no_gcs_upload)
    print(f"\n[oracle_extractor] 총 소요: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
