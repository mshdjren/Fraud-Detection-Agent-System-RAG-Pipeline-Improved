"""
oracle_rollout_batch.py — 2-step 오라클 수집 (oracle_extractor + oracle_backfill 통합)
=====================================================================================
설계 원칙
 1. backfill 통합 — persona/neighbors를 _base_call() 결과에서 그 자리에서 붙인다.
    2차 패스 Job이 불필요하고 추가 API 호출도 0이다.
 2. reward 미저장 — nd/이웃 identity/이웃 feature/이웃 cluster_id/spread/evidence만
    남긴다. R_margin·R_flip·R_resolve·R_auroc·R_diag는 전부 노트북에서 파생시킨다
    (λ 사후계산과 동일 원칙). reward 축이 바뀌어도 재수집이 필요 없다.
 3. vec_index 하드 검증 — 오라클의 정상 target은 augmentation(368만건)이 아니라
    원본 coreset100이어야 한다. 첫 base_call에서 검증하고 불일치면 즉시 종료한다.
 4. analyzer는 state당 1회(round0)뿐. 후보 실행 경로에는 analyzer가 없고,
    2-step evidence는 prompts.compute_evidence_axes로 결정론적으로 재계산된다.

★ v3 재발 방지 (14시간분 소실 사고 대응)
 · state별 타임아웃  — MCP 무한 대기 1건이 as_completed 전체를 막던 문제
 · 예외 흡수         — 예외가 run_oracle을 죽여 Job이 재시작되며 진행분이 소실되던 문제
 · 증분 체크포인트   — 로컬 + GCS. 죽어도 여기까지는 남는다
 · --resume          — 체크포인트에서 이어받기 (완료된 test_idx 건너뜀)
 · 경로 정리         — 기존 관례 oracle/{case}/query_{ver}_pct_{pct}/ 로 통일

Usage:
    python oracle_rollout_batch.py \
        --experiment-case pca_64_k100 --percolate-version v1 \
        --coreset-percentage 100 --split head \
        --require-index-substring percentage_100_cluster_tree \
        --output-dir /results/oracle [--resume --date-tag 0816]
"""

from __future__ import annotations

import os
import json
import time
import argparse
import asyncio
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import httpx
import structlog

from gt_loader import GTLoader

logger = structlog.get_logger()

# ============================================================
# 설정
# ============================================================
CONTROLLER_URL = os.getenv("CONTROLLER_URL", "http://anomaly-controller:80")
ROUTER_URL     = os.getenv("ROUTER_URL", "http://anomaly-router:80")
RETRIEVER_URL  = os.getenv("RETRIEVER_URL", "http://anomaly-retriever:80")
BASE_TOP_K     = int(os.getenv("BASE_TOP_K", "3"))
CONCURRENCY    = int(os.getenv("ORACLE_CONCURRENCY", "16"))
MAX_RETRY      = int(os.getenv("ORACLE_MAX_RETRY", "3"))

# ★ state 1개 상한. MCP 한 건이 멈춰 전체가 대기하던 문제의 방어선.
STATE_TIMEOUT  = float(os.getenv("STATE_TIMEOUT", "900"))
CKPT_EVERY     = int(os.getenv("CKPT_EVERY", "100"))
CKPT_TO_GCS    = os.getenv("CKPT_TO_GCS", "true").lower() == "true"

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "fraud_ecommm")
# ★ 기존 관례 유지: oracle/{case}/query_{ver}_pct_{pct}/
GCS_ORACLE_DIR_TEMPLATE = "oracle/{experiment_case}/query_{percolate_version}_pct_{coreset_pct}"
ORACLE_FNAME_TEMPLATE   = "oracle_rollout2_{split}_{mining_date}.parquet"
CKPT_FNAME_TEMPLATE     = "_ckpt/oracle_rollout2_{split}_{mining_date}_partial.pkl"

_INDEX_SEEN: Dict[str, int] = {}


# ============================================================
# 경로 헬퍼
# ============================================================
def _gcs_dir(args) -> str:
    return GCS_ORACLE_DIR_TEMPLATE.format(
        experiment_case=args.experiment_case,
        percolate_version=args.percolate_version,
        coreset_pct=args.coreset_percentage)


def _date_str(args) -> str:
    return args.date_tag or datetime.now().strftime("%m%d")


def _ckpt_rel(args) -> str:
    return CKPT_FNAME_TEMPLATE.format(split=args.split, mining_date=_date_str(args))


def _local_ckpt(args) -> Path:
    p = Path(args.output_dir) / _ckpt_rel(args)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ============================================================
# 재시도
# ============================================================
async def _with_retry(coro_fn, *, test_idx: int, what: str):
    last = None
    for attempt in range(MAX_RETRY):
        try:
            return await coro_fn()
        except Exception as e:
            last = e
            if attempt < MAX_RETRY - 1:
                await asyncio.sleep(1.5 * (2 ** attempt))
    logger.warning("retry_exhausted", what=what, test_idx=test_idx, error=str(last))
    raise last


# ============================================================
# 데이터 로드 / split
# ============================================================
def load_indexed_test_cases(args):
    """
    split:
      head — test set 앞 10% (negative mining / evaluator SFT와 동일 구간. 학습+EVPI)
      tail — 뒤 10% (서빙 성능 검증 구간. EVPI 전이성 확인)
    """
    loader = GTLoader(experiment_case=args.experiment_case,
                      coreset_percentage=str(args.coreset_percentage))
    test_data, test_emb, gt_data = loader.load_minimal()
    gt_labels = gt_data["label"]
    n_total = len(test_data)

    n_samples = int(args.n_samples) if args.n_samples else max(1, int(0.1 * n_total))
    if args.split == "head":
        start = max(0, int(args.start_index))
    else:
        start = max(0, n_total - n_samples - int(args.start_index))
    end = min(start + n_samples, n_total)
    indices = list(range(start, end))

    print(f"■ split={args.split}  n_total={n_total}  범위=[{start}, {end})  n={len(indices)}")
    _y = np.asarray(gt_labels)[indices]
    print(f"  y=1 비율 = {float(np.mean(_y)):.3f}   "
          f"(전체 test set = {float(np.mean(np.asarray(gt_labels))):.3f})")
    return loader, test_data, test_emb, gt_labels, indices


# ============================================================
# Base call — router + retriever 1쌍 (backfill 통합 소스)
# ============================================================
def _assert_vec_index(vec_index: str, args) -> None:
    """★ 오라클은 원본 coreset100에서 수집되어야 한다. 설정 실수로 aug가
    반환되면 수 시간을 낭비하므로 첫 호출에서 즉시 종료한다."""
    _INDEX_SEEN[vec_index] = _INDEX_SEEN.get(vec_index, 0) + 1
    if len(_INDEX_SEEN) == 1 and _INDEX_SEEN[vec_index] == 1:
        print(f"■ [index] router가 반환한 vec_index = {vec_index}")
    if args.forbid_index_substring and args.forbid_index_substring in vec_index:
        raise RuntimeError(
            f"[index-guard] 금지 인덱스 반환: '{vec_index}' "
            f"(forbid='{args.forbid_index_substring}'). anomaly-config의 router 설정 확인")
    if args.require_index_substring and args.require_index_substring not in vec_index:
        raise RuntimeError(
            f"[index-guard] 기대 인덱스 아님: '{vec_index}' "
            f"(require='{args.require_index_substring}')")


async def _base_call(client: httpx.AsyncClient, tc: Dict, args, test_idx: int = -1) -> Dict:
    async def _route():
        r = await client.post(
            f"{ROUTER_URL}/route", json=tc,
            headers={"X-Router-Case": args.experiment_case,
                     "X-Router-Version": args.percolate_version},
            timeout=args.timeout)
        r.raise_for_status()
        return r.json()
    route = await _with_retry(_route, test_idx=test_idx, what="router")

    cluster_id = route["primary_cluster_id"]
    vec_index  = route["vec_index"]
    _assert_vec_index(vec_index, args)

    async def _retrieve():
        r = await client.post(
            f"{RETRIEVER_URL}/retrieve",
            json={"embedding": tc["embedding"], "cluster_ids": [cluster_id],
                  "vec_index": vec_index, "top_k": BASE_TOP_K},
            timeout=args.timeout)
        r.raise_for_status()
        return r.json()
    retrieved = await _with_retry(_retrieve, test_idx=test_idx, what="retriever")

    persona = route.get("persona")
    return {
        "base_cluster_id": cluster_id,
        "base_top1_distance": retrieved.get("top1_distance"),
        "base_distance_band": None,
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
    except httpx.HTTPStatusError as e:
        logger.warning("rollout_http_error", test_idx=payload.get("test_idx"),
                       status=e.response.status_code, body=e.response.text[:300])
    except Exception as e:
        logger.error("rollout_failed", test_idx=payload.get("test_idx"), error=str(e))
    return None


# ============================================================
# Flatten
# ============================================================
_STATE_FIELDS = (
    "request_id", "test_idx", "y", "cluster_id", "wm_type", "wm_axes",
    "agrees", "differs", "deviant", "rare", "persona_match", "spread_tag",
    "nd_before", "base_spread", "base_neighbor_cluster_ids",
    "candidate_pool", "oracle_action", "n_steps", "latency_ms", "sufficiency",
)

def _flatten_rollout_response(resp: Dict, base: Dict) -> List[Dict]:
    """candidate 1개 = row 1개. state-level 필드를 각 row에 복제한다.
    ★ persona/neighbors(round0)는 base_call 결과에서 그 자리에서 붙인다
       — oracle_backfill.py의 2차 패스를 대체하는 부분."""
    head = {k: resp.get(k) for k in _STATE_FIELDS}
    head["persona"]          = resp.get("persona")   or base.get("base_persona")
    head["neighbors"]        = resp.get("neighbors") or base.get("base_neighbors")
    head["vec_index"]        = base.get("base_vec_index")     # 사후 감사용
    head["base_match_type"]  = base.get("base_match_type")
    head["state_latency_ms"] = resp.get("latency_ms")

    rows: List[Dict] = []
    for cand in (resp.get("candidates") or []):
        row = dict(head)
        row.update(cand)
        row["test_idx"] = resp.get("test_idx")
        rows.append(row)
    return rows


# ============================================================
# 본체
# ============================================================
# ============================================================
# 본체
# ============================================================
async def run_oracle(args) -> pd.DataFrame:
    loader, test_data, test_emb, gt_labels, indices = load_indexed_test_cases(args)
    sem = asyncio.Semaphore(CONCURRENCY)
    all_rows: List[Dict] = []
    done = {"n": 0, "err": 0, "cand": 0}
    stat = {"wm": Counter(), "act": Counter(), "pool": Counter(), "y": Counter(),
            "suf": Counter(), "nd_same": Counter(), "nd_tot": Counter(),
            "degraded": 0, "cand_err": 0, "nd": []}
    t_start = time.time()
    _ckpt = _local_ckpt(args)

    # ── resume: 체크포인트에서 이어받기 ──────────────────────
    _done_ti: set = set()
    if args.resume and _ckpt.exists():
        try:
            _prev = pd.read_pickle(str(_ckpt))
            all_rows = _prev.to_dict("records")
            _done_ti = set(int(t) for t in _prev["test_idx"].dropna().unique())
            done["n"] = len(_done_ti)
            done["cand"] = len(all_rows)
            print(f"■ [resume] {len(all_rows)} rows / {len(_done_ti)} test_idx 이어받음 "
                  f"← {_ckpt}")
        except Exception as e:
            print(f"⚠ [resume] 체크포인트 로드 실패({e}) — 처음부터 시작")
    _todo = [i for i in indices if i not in _done_ti]
    if len(_todo) != len(indices):
        print(f"■ 남은 대상: {len(_todo)}/{len(indices)}")

    def _save_ckpt(tag: str = "ckpt") -> None:
        if not all_rows:
            return
        try:
            pd.DataFrame(all_rows).to_pickle(str(_ckpt))
            print(f"  [{tag}] {len(all_rows)} rows → {_ckpt}", flush=True)
            if CKPT_TO_GCS:
                _upload_to_gcs(str(_ckpt), f"{_gcs_dir(args)}/{_ckpt_rel(args)}", quiet=True)
        except Exception as e:
            logger.warning("ckpt_failed", error=str(e))

    async def _one(idx: int, client: httpx.AsyncClient) -> List[Dict]:
        """★ 세마포어를 여기서 잡지 않는다 — _guarded가 wait_for 바깥에서 잡는다.
        (안쪽에 두면 타임아웃이 '실행 시간'이 아니라 '대기 시간'을 재게 된다)"""
        tc = loader.get_test_case(idx, test_data, test_emb,
                                  percolate_version=args.percolate_version,
                                  experiment_case=args.experiment_case)
        # tc는 flat TestCase 스키마 — RolloutRequest는 nested transaction을 쓴다.
        transaction = {str(k).replace(" ", "_"): v for k, v in tc.items()
                       if k not in ("embedding", "experiment_case", "percolate_version")}
        # ★ [방어] gt_loader가 공백 컬럼명 때문에 필드를 누락/기본값 대체했을 때
        #   test_data에서 직접 보충한다(candidate pool이 3개로 잘리던 실측 원인).
        _row = test_data.iloc[idx]
        _cmap = {c.replace(" ", "_"): c for c in test_data.columns}
        for _c in ("purchase_value", "age", "sex", "source", "browser",
                   "IP_country", "month_purchase", "weekday_purchase"):
            _src = _cmap.get(_c)
            if _src is None:
                continue
            _v = _row[_src]
            if pd.isna(_v):
                continue
            transaction[_c] = (float(_v) if isinstance(_v, (int, float, np.floating))
                               else str(_v))

        base = await _base_call(client, tc, args, test_idx=idx)
        payload = {
            "transaction": transaction,
            "embedding": tc["embedding"],
            "test_idx": idx,
            "y": int(gt_labels[idx]),
            "max_steps": args.max_steps,
            "step2_topn": args.step2_topn,
            **base,
        }
        result = await call_rollout(client, payload, args.timeout)

        done["n"] += 1
        if result is None:
            done["err"] += 1
            return []
        rows = _flatten_rollout_response(result, base)

        # ── 수집 도중 상태 분포 누적 ──
        stat["wm"][result.get("wm_type") or "(confident)"] += 1
        stat["suf"][result.get("sufficiency") or "?"] += 1
        stat["y"][int(gt_labels[idx])] += 1
        stat["pool"][len(result.get("candidate_pool") or [])] += 1
        if result.get("nd_before") is not None:
            stat["nd"].append(float(result["nd_before"]))
        for _c in (result.get("candidates") or []):
            stat["act"][_c.get("action")] += 1
            if _c.get("degraded"):
                stat["degraded"] += 1
            if _c.get("error"):
                stat["cand_err"] += 1
            if _c.get("action") != "stop":
                _nb, _na = _c.get("nd_before"), _c.get("nd_after")
                if _nb is not None and _na is not None:
                    stat["nd_same"][_c["action"]] += int(abs(float(_na) - float(_nb)) < 1e-9)
                    stat["nd_tot"][_c["action"]] += 1

        for r in rows:
            r["experiment_case"]    = args.experiment_case
            r["percolate_version"]  = args.percolate_version
            r["coreset_percentage"] = args.coreset_percentage
            r["split"]              = args.split
        done["cand"] += len(rows)

        _every = 10 if len(indices) <= 200 else 50
        if done["n"] % _every == 0 or done["n"] == 1 or done["n"] == len(indices):
            _el = time.time() - t_start
            _eta = _el / max(done["n"], 1) * (len(indices) - done["n"])
            print(f"  진행 {done['n']}/{len(indices)} ({done['n']/len(indices)*100:.1f}%)  "
                  f"실패 {done['err']}  후보 {done['cand']} "
                  f"(state당 {done['cand']/max(done['n'],1):.1f})  "
                  f"경과 {_el/60:.1f}분  ETA {_eta/60:.1f}분", flush=True)
        return rows

    async def _guarded(i: int, client: httpx.AsyncClient) -> List[Dict]:
        """★ state 하나가 멈추거나 실패해도 전체가 죽지 않게 한다.

        ★ [수정] 세마포어를 wait_for **바깥**에서 잡는다.
          이전엔 세마포어가 _one() 안에 있고 타임아웃이 밖에 있어서,
          create_task로 2,830개를 한 번에 만든 순간 모든 타이머가 동시에
          시작됐다. 뒤쪽 태스크는 세마포어를 기다리다 일도 못 해보고
          900초를 소진했다(실측: 2782/2830 실패, 동일 시각 대량 타임아웃).
          타임아웃은 '실행 시간'만 재야 한다.

        index-guard(RuntimeError)만 전파해 즉시 중단시킨다."""
        async with sem:                                   # ← 먼저 자리를 얻고
            try:
                return await asyncio.wait_for(            # ← 그다음 타이머 시작
                    _one(i, client), timeout=STATE_TIMEOUT)
            except RuntimeError as e:
                if "[index-guard]" in str(e):
                    raise
                logger.warning("state_failed", test_idx=i, error=str(e)[:200])
            except asyncio.TimeoutError:
                logger.warning("state_timeout", test_idx=i, sec=STATE_TIMEOUT)
            except Exception as e:
                logger.warning("state_failed", test_idx=i,
                               error=f"{type(e).__name__}: {str(e)[:200]}")
            done["n"] += 1
            done["err"] += 1
            return []

    async def _heartbeat():
        """장시간 무출력 방지 — as_completed는 완료 전까지 조용하다."""
        while done["n"] < len(indices):
            await asyncio.sleep(30)
            _el = time.time() - t_start
            _rate = done["n"] / max(_el, 1e-9)
            _eta = (len(indices) - done["n"]) / _rate if done["n"] > 0 else float("nan")
            print(f"  [heartbeat] {done['n']}/{len(indices)} "
                  f"({done['n']/len(indices)*100:.1f}%)  실패 {done['err']}  "
                  f"후보 {done['cand']}  속도 {_rate*60:.1f} state/분  "
                  f"경과 {_el/60:.1f}분  ETA {_eta/60:.1f}분", flush=True)
            if done["n"] > 0:
                _nd = np.asarray(stat["nd"]) if stat["nd"] else np.array([0.0])
                _ndsame = {a: "%d/%d" % (stat["nd_same"][a], stat["nd_tot"][a])
                           for a in stat["nd_tot"]}
                print(f"     wm_type={dict(stat['wm'])}  "
                      f"y={dict(stat['y'])}", flush=True)
                print(f"     sufficiency={dict(stat['suf'])}", flush=True)
                print(f"     pool크기={dict(stat['pool'])}  action={dict(stat['act'])}  "
                      f"degraded={stat['degraded']}  후보에러={stat['cand_err']}", flush=True)
                print(f"     nd 불변(=stop과 동일) {_ndsame}", flush=True)
                print(f"     nd_before p10={np.percentile(_nd,10):.3f} "
                      f"p50={np.percentile(_nd,50):.3f} p90={np.percentile(_nd,90):.3f}  "
                      f"band[0.30,0.55) 내 "
                      f"{float(((_nd>=0.30)&(_nd<0.55)).mean())*100:.0f}%", flush=True)

    limits = httpx.Limits(max_connections=CONCURRENCY * 2,
                          max_keepalive_connections=CONCURRENCY * 2)
    async with httpx.AsyncClient(limits=limits) as client:
        hb = asyncio.create_task(_heartbeat())
        tasks = [asyncio.create_task(_guarded(i, client)) for i in _todo]
        _nd_done = 0
        _last_ckpt_n = len(all_rows)      # ★ 동일 내용 반복 저장 방지
        try:
            for fut in asyncio.as_completed(tasks):
                all_rows.extend(await fut)
                _nd_done += 1
                if _nd_done % CKPT_EVERY == 0 and len(all_rows) > _last_ckpt_n:
                    _last_ckpt_n = len(all_rows)
                    _save_ckpt("ckpt")
        except BaseException as e:
            # ★ 어떤 이유로 중단되든 지금까지 모은 것은 반드시 저장한다
            print(f"\n⚠ 중단됨({type(e).__name__}: {str(e)[:200]}) — "
                  f"{len(all_rows)} rows 저장 시도", flush=True)
            _save_ckpt("ckpt-final")
            raise
        finally:
            hb.cancel()
            for t in tasks:
                if not t.done():
                    t.cancel()

    _save_ckpt("ckpt-done")
    df = pd.DataFrame(all_rows)
    print(f"\n■ 수집 완료: state {done['n']}/{len(indices)}  실패 {done['err']}  "
          f"rows {len(df)}  소요 {(time.time()-t_start)/60:.1f}분")
    return df


# ============================================================
# 자체 검증
# ============================================================
def self_check(df: pd.DataFrame, args) -> None:
    print("\n" + "=" * 70)
    print("[self-check] 수집 결과 검증")
    if df.empty:
        print("  ❌ 빈 DataFrame")
        return

    for c in ("persona", "neighbors"):
        if c not in df.columns:
            print(f"  ❌ '{c}' 컬럼 없음")
        else:
            print(f"  {'✅' if not df[c].isna().any() else '⚠'} {c} 결측 "
                  f"{int(df[c].isna().sum())}/{len(df)}")

    if "vec_index" in df.columns:
        _idx = df["vec_index"].value_counts().to_dict()
        print(f"  ■ vec_index 분포: {_idx}")
        if len(_idx) > 1:
            print("  ⚠ 인덱스가 여러 개 — router가 케이스별로 다른 인덱스를 반환")

    if "candidate_pool" in df.columns:
        _pool = set()
        for p in df["candidate_pool"].dropna():
            _pool |= set(p if isinstance(p, list) else [])
        print(f"  ■ candidate feature 합집합({len(_pool)}개): {sorted(_pool)}")
        if len(_pool) <= 3:
            print("  ⚠ 3개 이하 — TestCase 스키마/컬럼명 정규화 미적용 의심")

    for c in ("neighbor_features", "neighbor_cluster_ids", "spread_after", "nd_topk"):
        print(f"  {'✅' if c in df.columns else '❌'} {c}")

    if "step" in df.columns:
        print(f"  ■ step 분포: {df['step'].value_counts().to_dict()}")
    if "action" in df.columns:
        print(f"  ■ action 분포: "
              f"{df['action'].value_counts(normalize=True).round(3).to_dict()}")

    _ns = df[df["action"] != "stop"] if "action" in df.columns else df
    if len(_ns) and "support" in _ns.columns:
        print(f"  ■ 검색 action support=0 비율: "
              f"{float((_ns['support'].fillna(0) == 0).mean()):.2%}  (n={len(_ns)})")
    if "error" in df.columns:
        print(f"  ■ 후보 실행 실패율: {float(df['error'].notna().mean()):.2%}")
        _e = df[df["error"].notna()]
        if len(_e):
            print(f"  ■ error top5: "
                  f"{_e['error'].astype(str).str.slice(0,80).value_counts().head(5).to_dict()}")
            print(f"  ■ error x action: {_e.groupby('action').size().to_dict()}")

    # ── tool 이질성 ──────────────────────────────────────────
    if {"step", "action", "neighbor_oidx", "features"} <= set(df.columns):
        _s1 = df[df["step"] == 1].copy()
        _s1["_fkey"] = _s1["features"].apply(
            lambda x: ",".join(sorted(x)) if isinstance(x, list) else str(x))

        def _jac_pair(a1, a2):
            """★ 같은 (test_idx, features) 쌍끼리 비교한다. test_idx당 첫 row만
            보면 support가 가장 큰 단일 feature로 편향돼 '안 갈린다'가 나온다.
            한쪽 집합이 비면 제외한다 — 공집합 Jaccard 0을 '이질 축 확보'로
            오독하던 문제 방어."""
            _x = _s1[_s1.action == a1].set_index(["test_idx", "_fkey"])["neighbor_oidx"]
            _y = _s1[_s1.action == a2].set_index(["test_idx", "_fkey"])["neighbor_oidx"]
            _x = _x[~_x.index.duplicated()]
            _y = _y[~_y.index.duplicated()]
            _c = _x.index.intersection(_y.index)
            if len(_c) < 5:
                return None, 0, 0
            _pairs = [(set(_x[i] or []), set(_y[i] or [])) for i in _c]
            _valid = [(a, b) for a, b in _pairs if a and b]
            _ne = len(_pairs) - len(_valid)
            if not _valid:
                return None, 0, _ne
            return (float(np.mean([len(a & b) / len(a | b) for a, b in _valid])),
                    len(_valid), _ne)

        print("\n  ★ tool 쌍별 이웃 Jaccard (step1, 같은 feature 조합끼리)")
        for _a, _b in (("contextual_knn", "attribute_knn"),):
            _v, _n, _ne = _jac_pair(_a, _b)
            print(f"    {_a:<15s} vs {_b:<15s} = "
                  f"{'계산불가' if _v is None else format(_v, '.3f')}  "
                  f"(n={_n}, 한쪽공집합={_ne})")
            print("       < 0.3 → 이질 축 확보 | > 0.7 → 같은 축(도입 논리 재검토)")

        _g1 = _s1[_s1.action == "global_knn"].groupby("test_idx")["neighbor_oidx"].first()
        _a1 = _s1[_s1.action == "attribute_knn"].copy()
        if len(_a1) and "support" in _a1.columns:
            def _j2(r):
                _x = set(r["neighbor_oidx"] or [])
                _y = set(_g1.get(r["test_idx"]) or [])
                return len(_x & _y) / max(len(_x | _y), 1) if (_x and _y) else np.nan
            _a1["jac_vs_global"] = _a1.apply(_j2, axis=1)
            _a1["nd_same"] = (_a1.nd_after - _a1.nd_before).abs() < 1e-9
            _a1["sup_bin"] = pd.cut(_a1["support"].astype(float),
                                    [0, 100, 1000, 10000, 25000, 50000, 1e9],
                                    labels=["<100", "100-1k", "1k-10k",
                                            "10k-25k", "25k-50k", ">50k"])
            print("\n  ★ support 구간별 attribute vs global Jaccard")
            print(_a1.groupby("sup_bin", observed=True)
                     .agg(n=("jac_vs_global", "size"),
                          jaccard=("jac_vs_global", "mean"),
                          nd_unchanged=("nd_same", "mean")).round(3).to_string())
            print("     저선택도 구간에서 Jaccard가 낮으면 attribute는 살아있다")

        _nz = _s1[_s1.action != "stop"].copy()
        _nz["nd_same"] = (_nz.nd_after - _nz.nd_before).abs() < 1e-9
        print("\n  ★ nd 불변 비율 (=stop과 동일한 결과)")
        print(_nz.groupby("action")["nd_same"].agg(["mean", "count"]).round(3).to_string())

        if "wm_type" in _s1.columns and "sufficiency" in _s1.columns:
            _amb = _s1[_s1.sufficiency == "ambiguous"]
            if len(_amb) and _amb["wm_type"].notna().any():
                _best = _amb.loc[_amb.groupby("test_idx")["nd_after"].idxmax()]
                print("\n  ★ wm_type x action (ambiguous, 참고용)")
                print(pd.crosstab(_best["wm_type"], _best["action"],
                                  normalize="index").round(3).to_string())

    # ── 노트북 전달 스키마 ───────────────────────────────────
    print("\n  [스키마] 노트북(02b)이 소비하는 컬럼")
    _need = {
        "키":     ["test_idx", "y", "step", "cand_id", "parent_id", "path",
                   "action", "features"],
        "관측":   ["nd_before", "nd_after", "nd_topk", "spread_after",
                   "support", "n_calls"],
        "이웃":   ["neighbor_oidx", "neighbor_cluster_ids", "neighbor_features",
                   "cross_cluster_escape"],
        "진단후": ["agrees_after", "differs_after", "deviant_after", "rare_after",
                   "persona_match_after", "spread_tag_after"],
        "state":  ["wm_type", "sufficiency", "candidate_pool", "persona", "neighbors",
                   "base_spread", "vec_index", "cluster_id"],
        "참고":   ["delta_margin", "cost", "latency_ms", "degraded", "error"],
    }
    for _g, _cols in _need.items():
        _miss = [c for c in _cols if c not in df.columns]
        _empty = [c for c in _cols
                  if c in df.columns and df[c].isna().all() and c != "error"]
        print(f"    {'✅' if not _miss and not _empty else '❌'} {_g:<6s} "
              f"누락={_miss or '-'}  전체결측={_empty or '-'}")

    try:
        _nsd = df[(df.action != "stop") & df.nd_after.notna()]
        _dm = (2 * _nsd["y"].astype(int) - 1) * (_nsd["nd_after"] - _nsd["nd_before"])
        _fl = (((_nsd.nd_after >= 0.42).astype(int) == _nsd.y.astype(int)).astype(int)
               - ((_nsd.nd_before >= 0.42).astype(int) == _nsd.y.astype(int)).astype(int))
        print(f"\n  [파생검증] R_margin mean={_dm.mean():+.5f}  "
              f"부호+ {float((_dm > 0).mean()):.1%}")
        print(f"             R_flip(tau=0.42) rescued={int((_fl > 0).sum())} "
              f"degraded={int((_fl < 0).sum())}  (n={len(_nsd)})")
        print(f"             spread_after 유효 "
              f"{int(df['spread_after'].notna().sum())}/{len(df)}  "
              f"neighbor_features 유효 "
              f"{int(df['neighbor_features'].apply(bool).sum())}/{len(df)}")
    except Exception as e:
        print(f"  ⚠ [파생검증] 실패: {type(e).__name__}: {e}")

    print("\n  [샘플 1행]")
    _s = df[df.action != "stop"].head(1)
    if len(_s):
        for _c in ["test_idx", "y", "step", "action", "features", "nd_before", "nd_after",
                   "spread_after", "support", "neighbor_cluster_ids", "agrees_after",
                   "cross_cluster_escape", "path", "n_calls"]:
            if _c in _s.columns:
                _v = str(_s.iloc[0][_c])
                print(f"    {_c:<22s} = {_v[:110]}{'...' if len(_v) > 110 else ''}")
    print("=" * 70)


# ============================================================
# 저장
# ============================================================
_JSON_COLS = ("persona", "neighbors", "features", "features_used", "nd_topk",
              "neighbor_oidx", "neighbor_cluster_ids", "neighbor_features",
              "base_neighbor_cluster_ids", "candidate_pool", "wm_axes",
              "agrees", "differs", "deviant", "rare", "persona_match",
              "agrees_after", "differs_after", "deviant_after", "rare_after",
              "persona_match_after")

def save(df: pd.DataFrame, args) -> str:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = ORACLE_FNAME_TEMPLATE.format(split=args.split, mining_date=_date_str(args))
    local_path = out / fname

    dfs = df.copy()
    for c in _JSON_COLS:
        if c in dfs.columns:
            dfs[c] = dfs[c].apply(
                lambda x: json.dumps(x, ensure_ascii=False)
                if isinstance(x, (list, dict, tuple, np.ndarray)) else x)
    dfs.to_parquet(str(local_path), index=False)
    print(f"\n✅ [save] {local_path}  ({len(dfs)} rows, "
          f"{local_path.stat().st_size/1e6:.1f} MB)")

    if not args.no_gcs_upload:
        _upload_to_gcs(str(local_path), f"{_gcs_dir(args)}/{fname}")
    return str(local_path)


def _upload_to_gcs(local_path: str, gcs_path: str, quiet: bool = False) -> None:
    try:
        from google.cloud import storage
        bucket = storage.Client().bucket(GCS_BUCKET_NAME)
        bucket.blob(gcs_path).upload_from_filename(local_path)
        if not quiet:
            print(f"✅ [gcs] gs://{GCS_BUCKET_NAME}/{gcs_path}")
    except Exception as e:
        logger.warning("gcs_upload_failed", path=gcs_path, error=str(e))
        if not quiet:
            print(f"⚠ GCS 업로드 실패(로컬 파일은 보존됨): {e}")


# ============================================================
# Entrypoint
# ============================================================
async def main():
    p = argparse.ArgumentParser(
        description="2-step 오라클 rollout 수집 (backfill 통합)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--experiment-case", type=str, default="pca_64_k100")
    p.add_argument("--percolate-version", type=str, default="v1")
    p.add_argument("--coreset-percentage", type=int, default=100)
    p.add_argument("--split", type=str, default="head", choices=["head", "tail"])
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=2)
    p.add_argument("--step2-topn", type=int, default=3)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--date-tag", type=str, default=None,
                   help="파일명 날짜(MMDD). resume 시 이전 실행과 같게 줘야 한다")
    p.add_argument("--resume", action="store_true",
                   help="체크포인트에서 이어받기 (완료된 test_idx 건너뜀)")
    p.add_argument("--require-index-substring", type=str,
                   default="percentage_100_cluster_tree")
    p.add_argument("--forbid-index-substring", type=str, default="aug_")
    p.add_argument("--output-dir", type=str, default="/results/oracle")
    p.add_argument("--no-gcs-upload", action="store_true")
    args = p.parse_args()

    print(f"■ 저장 경로 : gs://{GCS_BUCKET_NAME}/{_gcs_dir(args)}/"
          + ORACLE_FNAME_TEMPLATE.format(split=args.split, mining_date=_date_str(args)))
    print(f"■ 체크포인트: {_local_ckpt(args)}  (GCS 동기화={CKPT_TO_GCS})")
    print(f"■ 동시성={CONCURRENCY}  state_timeout={STATE_TIMEOUT}s  ckpt_every={CKPT_EVERY}")

    df = await run_oracle(args)
    self_check(df, args)
    save(df, args)


if __name__ == "__main__":
    asyncio.run(main())