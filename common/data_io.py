"""
data_io.py — 두 노트북 공통 데이터 로드 + 인덱스 매핑 + distance 백분위수.

원본 Section 3-4(cell 11~21) + Section 5 distance 분포 계산(cell 23 앞부분)을 외부화.
load_all() 한 번으로:
  1) persona / coreset / train_emb / test / test_emb 로드 + 컬럼 정규화
  2) coreset_feature_map / coreset_emb_map 구축
  3) top1 L2 distance 백분위수(DIST_*) 계산
  4) prompts.bind_runtime(...) 로 prompts 모듈에 런타임 상태 주입
  5) 모든 public 객체를 dict 로 반환 → 노트북에서 globals().update(...) 로 펼침
"""
import json
import numpy as np
import pandas as pd
from typing import Dict

import common.config as config
import prompts


def load_all(mode: str = "agent", verbose: bool = True) -> Dict:
    """전체 데이터 로드 + 런타임 바인딩. 반환 dict를 globals().update()로 펼쳐 쓴다."""
    prompts.set_mode(mode)

    # ── 3-1. Persona ──────────────────────────────────────
    with open(config.PERSONA_PATH, "r", encoding="utf-8") as f:
        _persona_raw = json.load(f)
    CLUSTER_PERSONAS: Dict[int, Dict] = {int(k): v for k, v in _persona_raw.items()}

    # ── 3-2. Coreset 100% ────────────────────────────────
    coreset_df = pd.read_parquet(config.CORESET_100_PATH)

    # ── 3-3. Train Embeddings ────────────────────────────
    train_emb = np.load(config.TRAIN_EMB_PATH)

    # ── 3-4. Test Parquet ────────────────────────────────
    test_df = pd.read_parquet(config.TEST_PARQUET_PATH)

    # ── 컬럼명 정규화 ─────────────────────────────────────
    _rename = {
        "purchase value": "purchase_value",
        "IP country": "IP_country",
        "month purchase": "month_purchase",
        "weekday purchase": "weekday_purchase",
    }
    test_df = test_df.rename(columns={k: v for k, v in _rename.items() if k in test_df.columns})
    coreset_df = coreset_df.rename(columns={k: v for k, v in _rename.items() if k in coreset_df.columns})

    # ── 3-5. Test Embeddings ─────────────────────────────
    test_emb = np.load(config.TEST_EMB_PATH)
    assert len(test_emb) == len(test_df), f"test_emb({len(test_emb)}) vs test_df({len(test_df)})"

    # ── Section 4. 인덱스 매핑 ────────────────────────────
    coreset_feature_map: Dict[int, Dict] = {}
    coreset_emb_map: Dict[int, np.ndarray] = {}
    for row_pos, row in coreset_df.iterrows():
        oi = int(row["original_index"]) if "original_index" in coreset_df.columns else int(row_pos)
        coreset_feature_map[oi] = {
            col: (None if pd.isna(row[col]) else row[col])
            for col in config.FEATURE_COLS if col in coreset_df.columns
        }
        coreset_emb_map[oi] = train_emb[row_pos]

    labels       = test_df["label"].values.astype(int)
    gt_clusters  = test_df["gt_cluster_id"].values.astype(int)
    top5_indices = test_df["gt_knn_indices_576d"].values

    # ── Section 5. top1 L2 distance 백분위수 + top-K spread ──
    K_SPREAD = 3
    all_top1, topk_spread = [], np.zeros(len(test_df))
    for i in range(len(test_df)):
        oi_list = [int(x) for x in top5_indices[i]][:K_SPREAD]
        dds = []
        for oi in oi_list:
            ref = coreset_emb_map.get(oi)
            if ref is not None:
                dds.append(float(np.linalg.norm(
                    test_emb[i].astype(np.float32) - ref.astype(np.float32))))
        all_top1.append(dds[0] if dds else float("nan"))
        topk_spread[i] = float(np.std(dds)) if len(dds) > 1 else 0.0
    dist_arr = np.array([d for d in all_top1 if not np.isnan(d)])

    DIST_P10 = float(np.percentile(dist_arr, 10))
    DIST_P25 = float(np.percentile(dist_arr, 25))
    DIST_P50 = float(np.percentile(dist_arr, 50))
    DIST_P75 = float(np.percentile(dist_arr, 75))
    DIST_P90 = float(np.percentile(dist_arr, 90))
    DIST_IQR = DIST_P75 - DIST_P25
    NORM_CLOSE_THRESHOLD    = DIST_P25
    NORM_MODERATE_THRESHOLD = DIST_P75

    runtime = dict(
        CLUSTER_PERSONAS=CLUSTER_PERSONAS,
        coreset_df=coreset_df, train_emb=train_emb,
        test_df=test_df, test_emb=test_emb,
        coreset_feature_map=coreset_feature_map, coreset_emb_map=coreset_emb_map,
        labels=labels, gt_clusters=gt_clusters, top5_indices=top5_indices,
        topk_spread=topk_spread,
        DIST_P10=DIST_P10, DIST_P25=DIST_P25, DIST_P50=DIST_P50,
        DIST_P75=DIST_P75, DIST_P90=DIST_P90, DIST_IQR=DIST_IQR,
        NORM_CLOSE_THRESHOLD=NORM_CLOSE_THRESHOLD,
        NORM_MODERATE_THRESHOLD=NORM_MODERATE_THRESHOLD,
        test_df_original=test_df.copy(),
        test_emb_original=test_emb.copy(),
        labels_original=labels.copy(),
        gt_clusters_original=gt_clusters.copy(),
    )

    # ── prompts 모듈에 런타임 주입 (함수 본문이 bare-name으로 참조) ──
    prompts.bind_runtime(**runtime)

    if verbose:
        print(f"✅ data_io.load_all(mode={mode})")
        print(f"   personas={len(CLUSTER_PERSONAS)}  coreset_map={len(coreset_feature_map):,}")
        print(f"   test={len(test_df):,}  normal/abnormal={(labels==0).sum()}/{(labels==1).sum()}")
        print(f"   dist p25/p50/p75={DIST_P25:.3f}/{DIST_P50:.3f}/{DIST_P75:.3f}  IQR={DIST_IQR:.3f}")

    return runtime