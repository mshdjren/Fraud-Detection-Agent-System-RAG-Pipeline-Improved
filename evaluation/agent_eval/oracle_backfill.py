"""
oracle_backfill.py — persona/neighbors 컬럼 backfill
=========================================================
목적: 기존 oracle parquet(action/reward/delta_margin/cost는 이미 유효 —
     RolloutResponse에 persona/neighbors가 추가되기 *전*에 뽑힌 것)에
     persona/neighbors만 채워 넣는다.

     스키마 사전 점검(03 노트북 마지막 셀)에서 확인된 문제는 정확히 이
     2개 컬럼 부재였고, 그 외 다른 결측/이상은 없었다 — 그래서 이 스크립트도
     그 2개만 backfill하고 나머지 컬럼(action/reward/...)은 원본 그대로
     보존한다. action/reward는 test_idx당 최대 ~12개 candidate를 전수
     실행해야 나오는 값(4시간 걸린 이유)이지만, persona/neighbors는
     oracle_extractor.py의 _base_call()(round0 — router /route + retriever
     /retrieve, exhaustive 아님) 딱 한 번만 있으면 되므로 test_idx당 API
     호출이 12개→2개로 줄어든다.

     oracle_extractor.py를 import해서 GTLoader/_base_call/GCS 상수를 그대로
     재사용한다 — router/retriever URL, GCS 버킷/경로 관례, 재시도 로직 등을
     중복 구현하지 않기 위함(같은 디렉터리/ConfigMap에 있어야 import된다).

Usage:
    python oracle_backfill.py \\
        --input /results/oracle/oracle_rollout_pca_64_k100_query_v1_pct_100_0811.parquet \\
        --experiment-case pca_64_k100 --percolate-version v1 --coreset-percentage 100 \\
        --output-dir /results/oracle
"""

from __future__ import annotations

import os
import json
import argparse
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd
import httpx
import structlog

# ★ oracle_extractor.py와 같은 디렉터리(ConfigMap)에 있어야 import된다.
from oracle_extractor import (
    GTLoader, _base_call, ROUTER_URL, RETRIEVER_URL, BASE_TOP_K,
    GCS_BUCKET_NAME, GCS_ORACLE_PATH_TEMPLATE, _upload_to_gcs,
)

logger = structlog.get_logger()


# ============================================================
# Precheck — oracle_backfill.py 자체도 03 노트북과 동일한 점검을 먼저 한다
# (이미 backfill된 parquet을 실수로 다시 돌리는 것 방지 + 입력 검증)
# ============================================================
def _needs_backfill(df: pd.DataFrame) -> bool:
    if "persona" not in df.columns or "neighbors" not in df.columns:
        return True
    return bool(df["persona"].isna().any() or df["neighbors"].isna().any())


# ============================================================
# Backfill 본체 — test_idx당 _base_call() 1회 (router+retriever, exhaustive
# rollout 아님)
# ============================================================
async def _backfill_one(
    client: httpx.AsyncClient, loader: GTLoader, test_data, test_emb,
    test_idx: int, args,
) -> Tuple[int, Optional[Dict], Optional[list], Optional[int]]:
    tc = loader.get_test_case(
        test_idx, test_data, test_emb,
        percolate_version=args.percolate_version,
        experiment_case=args.experiment_case,
    )
    base = await _base_call(client, tc, args, test_idx=test_idx)
    return test_idx, base.get("base_persona"), base.get("base_neighbors"), base.get("base_cluster_id")


async def run_backfill(args) -> pd.DataFrame:
    df = pd.read_parquet(args.input)
    print(f"■ 입력 parquet: {args.input}  (rows={len(df)}, test_idx={df['test_idx'].nunique()})")

    if not _needs_backfill(df):
        print("✅ 이미 persona/neighbors가 채워져 있음 — backfill 불필요, 원본 그대로 반환")
        return df

    loader = GTLoader(
        experiment_case=args.experiment_case,
        coreset_percentage=str(args.coreset_percentage),
    )
    # ★ load_minimal() — coreset parquet/GT top5 컬럼 불필요(oracle_extractor.py와 동일 이유)
    test_data, test_emb, _ = loader.load_minimal()

    unique_idx = sorted(df["test_idx"].unique().tolist())
    print(f"■ backfill 대상 test_idx: {len(unique_idx)}개 "
          f"(rollout 전수 재실행 대비 API 호출 ~1/6 수준 — router+retriever만)")

    concurrency = int(os.getenv("BACKFILL_CONCURRENCY", "8"))
    sem = asyncio.Semaphore(concurrency)
    results: Dict[int, Tuple[Optional[Dict], Optional[list], Optional[int]]] = {}
    done = {"n": 0, "err": 0}

    async def _one(idx: int, client: httpx.AsyncClient):
        async with sem:
            try:
                _, persona, neighbors, cluster_id = await _backfill_one(
                    client, loader, test_data, test_emb, idx, args)
                results[idx] = (persona, neighbors, cluster_id)
            except Exception as e:
                logger.warning("backfill_failed", test_idx=idx, error=str(e))
                results[idx] = (None, None, None)
                done["err"] += 1
        done["n"] += 1
        if done["n"] % 200 == 0 or done["n"] == 1:
            print(f"  진행 {done['n']}/{len(unique_idx)}  실패 {done['err']}")

    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency * 2)
    async with httpx.AsyncClient(limits=limits) as client:
        await asyncio.gather(*[_one(idx, client) for idx in unique_idx])

    print(f"\n■ backfill 완료: {done['n']}/{len(unique_idx)}  실패 {done['err']}")

    df = df.copy()
    df["persona"] = df["test_idx"].map(lambda i: results.get(i, (None, None, None))[0])
    df["neighbors"] = df["test_idx"].map(lambda i: results.get(i, (None, None, None))[1])

    n_missing = int(df["persona"].isna().sum())
    if n_missing:
        print(f"⚠ persona 여전히 결측: {n_missing}/{len(df)}행 (router/retriever 실패한 test_idx) "
              f"— 이 test_idx들은 NB2b 학습 시 자동 제외되거나 재시도 필요")

    # ★ [sanity check] 지금 다시 라우팅한 cluster_id가 원본 오라클의 cluster_id와
    #   일치하는지 확인 — router 설정/percolate_version이 원본 수집 때와
    #   달라졌으면(=인덱스 재구성, 버전 변경 등) 여기서 어긋난다. 어긋나면
    #   persona/neighbors가 원본 rollout 당시의 state와 다른 것이므로, prompt가
    #   action/reward 계산 당시와 안 맞을 위험이 있다.
    _fresh_cluster = df["test_idx"].map(lambda i: results.get(i, (None, None, None))[2])
    _cmp = df[["test_idx", "cluster_id"]].drop_duplicates("test_idx").set_index("test_idx")["cluster_id"]
    _fresh_by_idx = pd.Series({i: v[2] for i, v in results.items() if v[2] is not None})
    _common = _cmp.index.intersection(_fresh_by_idx.index)
    if len(_common) > 0:
        _mismatch = (_cmp.loc[_common] != _fresh_by_idx.loc[_common]).mean()
        print(f"\n■ [sanity check] 원본 cluster_id vs 재라우팅 cluster_id 불일치율: {_mismatch:.1%}")
        if _mismatch > 0.02:
            print("  ⚠ 2% 넘게 어긋남 — router 설정/인덱스가 원본 오라클 수집 시점과 달라졌을 "
                  "수 있습니다. persona/neighbors가 원본 action/reward 계산 당시의 state와 "
                  "다를 수 있으니 결과 해석 시 주의하세요.")
        else:
            print("  ✅ 거의 일치 — 재라우팅된 persona/neighbors를 원본 action/reward와 "
                  "같은 state로 신뢰해도 됩니다.")

    return df


# ============================================================
# Save: local + GCS (oracle_extractor.py save()와 동일 패턴, 파일명만 _backfilled 접미)
# ============================================================
def save(df: pd.DataFrame, args) -> str:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = Path(args.input).name.replace(".parquet", "_backfilled.parquet")
    local_path = out / fname

    df_save = df.copy()
    for col in ("persona", "neighbors"):
        if col in df_save.columns:
            df_save[col] = df_save[col].apply(
                lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)
    df_save.to_parquet(str(local_path), index=False)
    print(f"\n✅ [save] parquet: {local_path}  ({len(df_save)} rows)")

    if not args.no_gcs_upload:
        date_str = datetime.now().strftime("%m%d")
        gcs_base = GCS_ORACLE_PATH_TEMPLATE.format(
            experiment_case=args.experiment_case,
            percolate_version=args.percolate_version,
            coreset_pct=args.coreset_percentage,
            mining_date=date_str,
        ).replace(".parquet", "_backfilled.parquet")
        _upload_to_gcs(str(local_path), gcs_base)

    return str(local_path)


# ============================================================
# Entrypoint
# ============================================================
async def main():
    parser = argparse.ArgumentParser(
        description="Oracle parquet에 persona/neighbors만 backfill (전체 재수집 없이)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True,
                        help="backfill 대상 oracle parquet 경로 (로컬 파일)")
    parser.add_argument("--experiment-case", type=str, default="pca_64_k100")
    parser.add_argument("--percolate-version", type=str, default="v1")
    parser.add_argument("--coreset-percentage", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=60,
                        help="router/retriever 개별 호출 timeout(초) — 전수 rollout(300초) 대비 짧음")
    parser.add_argument("--output-dir", type=str, default="/results/oracle")
    parser.add_argument("--no-gcs-upload", action="store_true")
    args = parser.parse_args()

    df = await run_backfill(args)
    save(df, args)


if __name__ == "__main__":
    asyncio.run(main())
