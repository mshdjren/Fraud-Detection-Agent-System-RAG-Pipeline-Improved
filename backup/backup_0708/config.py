"""
config.py — 두 노트북(agent / forwardRAG) 공통 실험 설정.

원본 노트북 Section 2(cell 9)를 그대로 외부화한 모듈.
PHASE / USE_GRPO 등 런(run)마다 바뀌는 값은 import 후 노트북에서 재할당해도 된다.
    예) NB1(data gen):   import config; config.PHASE = "generate"
        NB2(train):      config.PHASE = "train"; config.USE_GRPO = True
API key는 요청대로 하드코딩 유지 (운영 전 env 분리 권장).
"""
import os, json, time, random
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

__all__ = [n for n in dir() if not n.startswith("_")]

# ============================================================
# PHASE — 실행 목적
# ============================================================
PHASE = "train"                      # "generate" | "train"

# ============================================================
# Teacher Provider (PHASE="generate" 시에만 사용)
# ============================================================
TEACHER_PROVIDER = "huggingface"     # "groq" | "gemini" | "huggingface"

# ── Groq ──────────────────────────────────────────────────
GROQ_API_KEY       = "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"
GROQ_TEACHER_MODEL = "llama-3.1-8b-instant"
GROQ_FALLBACK_MODELS: List[str] = []

# ── Gemini ────────────────────────────────────────────────
GEMINI_API_KEY         = "AIzaSyDIUje2dcNbjrr2clI4xAl_LAQQfnKVP4c"
GEMINI_TEACHER_MODEL   = "gemini-2.5-flash-lite"
GEMINI_FALLBACK_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash-8b"]

# ── HuggingFace (로컬 GPU) ────────────────────────────────
HF_TEACHER_MODEL   = "Qwen/Qwen2.5-14B-Instruct"   # 7B→14B: z-score 산술비교 개선
# HF_TEACHER_MODEL = "Qwen/QwQ-32B"
# HF_TEACHER_MODEL = "Qwen/Qwen3-4B-Thinking-2507"
HF_FALLBACK_MODELS = ["Qwen/Qwen2.5-7B-Instruct"]
HF_TOKEN           = "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"
HF_USE_4BIT        = True
HF_BATCH_SIZE      = 50

max_new_tokens     = 64
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ============================================================
# 데이터 생성 설정 (PHASE="generate")
# ============================================================
GENERATION_MODE           = "mining_only"   # "distance" | "mined" | "both" | "mining_only"
BOUNDARY_ND_LOW           = 0.3
BOUNDARY_ND_HIGH          = 0.5
INCLUDE_BOUNDARY_DISTANCE = True
TRAIN_ON                  = "mining_only"

# toy 단계: toy1=200(형식검증) / toy2=500(grounding 증류측정) / full=1000
TOY_STAGE = "full"
SAMPLE_N  = {"toy1": 200, "toy2": 500, "full": 1000}[TOY_STAGE]
TRAIN_RATIO = 0.1
EVAL_RATIO  = 0.1

EVAL_MAX_NUM               = 50
MEASURE_TEACHER_COMPLIANCE = True
EVAL_SAMPLE_N              = 1000

# 1000개를 세 갈래로 명시 배분 (합=1.0)
MINING_FRAC   = 0.5     # hard negative (mining type1/2)  → 500
BOUNDARY_FRAC = 0.2     # boundary nd 0.3~0.5             → 200
EASY_FRAC     = 0.3     # easy nd 양끝(<0.2, >0.6)        → 300
# (EASY_RATIO 등 기존 비율 변수는 이걸로 대체)

# EASY_RATIO=0.3
# MINING_RATIO               = 0.2
# SAMPLE_N    = 12

EVIDENCE_TOPK_CHOICES = [1, 3]
EVIDENCE_TOPK_DEFAULT = 1

# ============================================================
# 학습 방법론 (비교 실험용)
# ============================================================
USE_CURRICULUM = False
USE_GRPO       = False
USE_MINING     = True

CLS_ONLY = False

GRPO_NUM_GENERATIONS = 2
GRPO_TEMPERATURE     = 0.9
GRPO_REWARD_WEIGHTS  = {"accuracy": 0.40, "calibration": 0.30,
                        "coherence": 0.20, "factuality": 0.10}
GRPO_EPOCHS     = 1
GRPO_LR         = 5e-6
GRPO_BATCH_SIZE = 4
GRPO_KL_COEF    = 0.1

CURRICULUM_STAGE1_EPOCHS = 3
CURRICULUM_STAGE2_EPOCHS = 3

# ============================================================
# Feature columns (Section 4) — data_io / prompts 공통 사용
# ============================================================
NUMERIC_COLS     = ["purchase_value", "age"]
CATEGORICAL_COLS = ["sex", "source", "browser", "IP_country",
                    "month_purchase", "weekday_purchase"]
FEATURE_COLS     = NUMERIC_COLS + CATEGORICAL_COLS

# ============================================================
# 실험 / 경로 설정 (두 PHASE 공통)
# ============================================================
EXPERIMENT_CASE = "pca_64_k100"

GCS_BUCKET  = "fraud_ecomm"
GCP_PROJECT = "kakaoo-501107"
MOUNT_POINT = "/content/gcs_mount"
GCS_ROOT    = MOUNT_POINT

PERSONA_PATH      = f"{GCS_ROOT}/tree-search/tree_fraudecom/{EXPERIMENT_CASE}/cluster_personas.json"
CORESET_100_PATH  = f"{GCS_ROOT}/tree-search/tree_fraudecom/{EXPERIMENT_CASE}/raw_plus_cluster_tree_percentage_100.parquet"
TRAIN_EMB_PATH    = f"{GCS_ROOT}/tree-search/embeddings_fraudecom/anollm_lr5e-05_standard_smolLM_train_embeddings_percentage_100_merged.npy"
TEST_PARQUET_PATH = f"{GCS_ROOT}/tree-search/tree_fraudecom/{EXPERIMENT_CASE}/test_with_gt.parquet"
TEST_EMB_PATH     = f"{GCS_ROOT}/tree-search/embeddings_fraudecom/anollm_lr5e-05_standard_smolLM_test_embeddings_merged.npy"

DATE = "0629"
MINED_PARQUET_PATH = (
    f"{GCS_ROOT}/tree-search/analyzer_ft/{EXPERIMENT_CASE}/"
    "hard_negatives/query_v14_pct_100/hard_negatives_0414.parquet"
)

_ptag = {"groq": "groq", "gemini": "gemini", "huggingface": "hf"}[TEACHER_PROVIDER]
OUT_BASE           = f"{GCS_ROOT}/tree-search/analyzer_ft/{EXPERIMENT_CASE}/{_ptag}_{DATE}"
OUT_JSONL_DISTANCE = f"{OUT_BASE}/training_data_distance.jsonl"
OUT_JSONL_MINED    = f"{OUT_BASE}/training_data_mined.jsonl"
OUT_ADAPTER_PATH   = f"{OUT_BASE}/adapter"

TRAINING_JSONL_PATH = (OUT_JSONL_MINED
                       if TRAIN_ON in ("mined", "mining_only")
                       else OUT_JSONL_DISTANCE)
LOCAL_ADAPTER = f"/content/adapter-{EXPERIMENT_CASE}"

if TEACHER_PROVIDER == "groq":
    TEACHER_MODEL, TEACHER_RPM = GROQ_TEACHER_MODEL, 25
elif TEACHER_PROVIDER == "gemini":
    TEACHER_MODEL, TEACHER_RPM = GEMINI_TEACHER_MODEL, 25
else:
    TEACHER_MODEL, TEACHER_RPM = HF_TEACHER_MODEL, None


def ensure_dirs():
    """GCS 마운트 이후 호출 — 출력 디렉터리 생성."""
    os.makedirs(OUT_BASE, exist_ok=True)


def print_config():
    sep = "=" * 62
    print(sep)
    print(f"PHASE            : {PHASE}")
    print(f"EXPERIMENT_CASE  : {EXPERIMENT_CASE}")
    if PHASE == "generate":
        print(f"TEACHER_PROVIDER : {TEACHER_PROVIDER}")
        print(f"TEACHER_MODEL    : {TEACHER_MODEL}")
        print(f"GENERATION_MODE  : {GENERATION_MODE}")
        print(f"TRAIN_ON         : {TRAIN_ON}")
    print(f"TRAINING_JSONL   : {TRAINING_JSONL_PATH}")
    print(f"OUT_ADAPTER_PATH : {OUT_ADAPTER_PATH}")
    print(sep)