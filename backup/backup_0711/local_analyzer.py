"""
local_analyzer.py — Local LLM Analyzer (Qwen2.5 + LoRA, HF 또는 GGUF)
==============================================================

controller.py(/detect, LangGraph analyze node)와 호환되는 /analyze
엔드포인트. analyzer_finetune_v9 노트북(Section 9 3-key 변환) 및
NB02(SFT/GRPO 학습)과 동일한 출력 schema를 사용한다.

Input  (controller → analyzer):
  test_transaction, cluster_id, top_k_normal_results,
  top1_distance, persona (Optional), tree_features (Optional)

Output (analyzer → controller):
  cls, sufficiency, what_missing (+ persona_used/tree_rules_used 메타데이터)

★ 이 schema가 또 바뀌어도, 영향 범위는 이 파일과 analyzer_schema.py로
한정된다. controller.py/orchestrator_agentic.py는 analyzer_schema.py의
정규화 어댑터를 거치므로 무수정으로 동작한다.

★ MODEL_BACKEND 스위치 (GGUF 서빙 도입):
  - "hf"   : 기존 방식. AutoModelForCausalLM + PeftModel merge (float32/bf16).
             0.5B처럼 메모리 여유가 있을 때.
  - "gguf" : llama-cpp-python + Q4_K_M (NB02 Section 14 산출물).
             3B를 GKE free-tier(e2-standard-2, 6Gi 미만)에 올리기 위한 경로.
  두 백엔드 모두 build_user_prompt()가 만든 동일한 raw 프롬프트 문자열을
  입력으로 쓰므로, 프롬프트 조립 로직(prompts.py 연동 포함)은 백엔드와
  무관하게 공유된다 — 갈라지는 곳은 "그 문자열을 누가 생성하느냐" 뿐이다.

★ prompts.py 연동:
  format_transaction / format_persona_zscore / normalize_dist / dist_tag는
  NB02 학습이 실제로 쓰는 함수와 동일한 것을 그대로 가져와 쓴다(공통 모듈).
  단, top5 neighbor 포맷터(prompts.format_top5_neighbors)는 학습 노트북의
  전역 coreset_feature_map을 참조하는데 서빙 환경엔 그게 없으므로(그리고
  요청마다 전역을 mutate하면 동시성 문제가 생기므로), 이 파일 안에
  _format_top5_neighbors()를 별도로 두고 retriever가 내려준 neighbor
  dict에서 직접 feature를 뽑는다 — 출력 포맷은 prompts.py와 동일하게 유지.
"""

import os
import re
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
import asyncio
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import structlog

import prompts   # ConfigMap mount: /app/prompts.py, /app/config.py (prompts.py가 import함)

# ===========================
# Config 로드
# ===========================

EXPERIMENT_CASE  = os.getenv("EXPERIMENT_CASE",  "pca_64_k100")
TEACHER_PROVIDER = os.getenv("TEACHER_PROVIDER", "hf")
ADAPTER_TYPE     = os.getenv("ADAPTER_TYPE",     "mined")
BASE_MODEL_ID    = os.getenv("BASE_MODEL_ID",    "Qwen/Qwen2.5-0.5B-Instruct")
MAX_NEW_TOKENS   = int(os.getenv("MAX_NEW_TOKENS", "512"))
DEVICE           = os.getenv("DEVICE",           "cpu")
TORCH_DTYPE_STR  = os.getenv("TORCH_DTYPE",      "float32")

# ── 서빙 백엔드 스위치 ────────────────────────────────────────
MODEL_BACKEND  = os.getenv("MODEL_BACKEND", "hf").lower()   # "hf" | "gguf"
GGUF_FILENAME  = os.getenv("GGUF_FILENAME", "model_q4.gguf")
N_CTX          = int(os.getenv("N_CTX", "2048"))
assert MODEL_BACKEND in ("hf", "gguf"), f"unknown MODEL_BACKEND: {MODEL_BACKEND}"

DIST_P25 = float(os.getenv("DIST_P25", "1.264"))
DIST_P50 = float(os.getenv("DIST_P50", "1.448"))
DIST_P75 = float(os.getenv("DIST_P75", "1.651"))
DIST_IQR = float(os.getenv("DIST_IQR", "0.387"))

PERSONA_ENABLED    = os.getenv("PERSONA_ENABLED",    "true").lower()  == "true"
TREE_RULES_ENABLED = os.getenv("TREE_RULES_ENABLED", "false").lower() == "true"
# /analyze가 즉시 단건 추론(_run_single, run_inference 1회)을 쓸지, 동시에
# 들어온 요청들을 모아 run_inference_batch로 한 번에 처리하는 마이크로배처
# (DynamicBatcher)를 쓸지 선택. hf 백엔드에서만 진짜 배치 forward 이득이
# 있음 — gguf는 run_inference_batch가 내부적으로 순차 루프이므로(아래
# 주석 참고), 켜도 안전하지만 처리량 이득은 없다.
ENABLE_CONTINUOUS_BATCHING = os.getenv("ENABLE_CONTINUOUS_BATCHING", "false").lower() == "true"

ADAPTER_PATH = os.getenv("ADAPTER_LOCAL_DIR", "/data/adapter")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))

logger = structlog.get_logger()

# ===========================
# Prometheus
# ===========================

analyze_requests  = Counter("analyzer_requests_total",  "Total /analyze requests",       ["status", "mode"])
analyze_latency   = Histogram("analyzer_latency_seconds", "Inference latency",
                               buckets=[1, 2, 5, 10, 30, 60, 120, 300])
parse_errors      = Counter("analyzer_parse_errors_total", "JSON parse errors", ["mode"])
# Continuous batching 실측 — hf 백엔드에서만 의미 있는 지표. gguf에서는
# 항상 1로 관측된다(순차 처리이므로) — 백엔드 전환 시 이 값의 분포 변화로
# "진짜 배치가 걸리고 있는지"를 바로 확인할 수 있다.
batch_inference_size = Histogram(
    "analyzer_batch_inference_size", "Number of requests packed into one model.generate() call",
    buckets=[1, 2, 3, 4, 5, 6, 7, 8],
)

# ===========================
# 모델 로드 (startup)
# ===========================

_model     = None       # hf: AutoModelForCausalLM merged / gguf: llama_cpp.Llama
_tokenizer = None       # hf: AutoTokenizer / gguf: 사용 안 함 (None 유지)


def load_model():
    global _model, _tokenizer
    logger.info("loading_model", backend=MODEL_BACKEND, base=BASE_MODEL_ID,
                adapter=ADAPTER_PATH, device=DEVICE)
    t0 = time.time()

    if MODEL_BACKEND == "gguf":
        _load_model_gguf(t0)
    else:
        _load_model_hf(t0)


def _load_model_gguf(t0: float):
    """llama-cpp-python으로 GGUF Q4_K_M 로드. tokenizer 별도 로드 불필요
    (프롬프트를 raw 문자열로 그대로 넣는다 — run_inference 참고)."""
    global _model
    from llama_cpp import Llama

    n_threads = int(os.getenv("OMP_NUM_THREADS", "2"))
    gguf_path = os.path.join(ADAPTER_PATH, GGUF_FILENAME)
    logger.info("loading_gguf", path=gguf_path, n_ctx=N_CTX, n_threads=n_threads)

    _model = Llama(
        model_path=gguf_path,
        n_ctx=N_CTX,
        n_threads=n_threads,
        verbose=False,
    )
    logger.info("gguf_model_loaded", elapsed_s=f"{time.time()-t0:.1f}", path=gguf_path)


def _load_model_hf(t0: float):
    """기존 HF transformers + LoRA merge 경로 (0.5B 등 메모리 여유 있을 때)."""
    global _model, _tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    torch_dtype = torch.bfloat16 if TORCH_DTYPE_STR == "bfloat16" else torch.float32

    if DEVICE == "cpu":
        n_threads = int(os.getenv("OMP_NUM_THREADS", "2"))
        torch.set_num_threads(n_threads)
        logger.info("cpu_threads_set", n_threads=n_threads)

    logger.info("loading_tokenizer", adapter_path=ADAPTER_PATH)
    _tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, padding_side="left")
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    logger.info("tokenizer_loaded", vocab_size=_tokenizer.vocab_size,
                pad_token=_tokenizer.pad_token, elapsed_s=f"{time.time()-t0:.1f}")

    logger.info("loading_base_model", base_model_id=BASE_MODEL_ID, torch_dtype=str(torch_dtype))
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
    )
    if DEVICE != "cpu":
        base = base.to(DEVICE)
    logger.info("base_model_loaded",
                params_M=f"{sum(p.numel() for p in base.parameters())/1e6:.0f}M",
                elapsed_s=f"{time.time()-t0:.1f}")

    logger.info("loading_lora_adapter", adapter_path=ADAPTER_PATH)
    model_with_lora = PeftModel.from_pretrained(base, ADAPTER_PATH)
    logger.info("lora_adapter_loaded", elapsed_s=f"{time.time()-t0:.1f}")

    logger.info("merging_adapter")
    _model = model_with_lora.merge_and_unload()
    _model.eval()

    elapsed = time.time() - t0
    logger.info(
        "model_loaded", elapsed_s=f"{elapsed:.1f}", device=DEVICE, dtype=str(torch_dtype),
        model_param=f"{sum(p.numel() for p in _model.parameters()) / 1e6:.0f}M",
    )


# ===========================
# Prompt 구성 — prompts.py(NB02와 공유하는 공통 모듈) 재사용
# ===========================

# ── neighbor feature 표시에서 제외할 non-feature 키 (retriever SearchResult 필드) ──
_NON_FEATURE_KEYS = {"es_doc_id", "original_index", "cluster_id", "distance", "score"}


def _format_top5_neighbors(top_k_results: List[Dict]) -> str:
    """prompts.format_top5_neighbors와 동일한 출력 포맷.
    학습 노트북의 coreset_feature_map(전역, GCS 전체 coreset 상주) 대신,
    retriever가 이미 내려준 neighbor dict에서 feature를 직접 사용한다.
    → 요청마다 prompts.py의 전역 상태를 mutate하지 않으므로 동시 요청에
    안전하다. retriever가 feature 컬럼을 안 내려주는 동안은 각 neighbor의
    "Neighbor i: ..." 라인이 생략되고 distance/spread 정보만 남는다.
    """
    if not top_k_results:
        return "No retrieved results."

    distances = [float(r.get("distance", float("inf"))) for r in top_k_results]
    top1_raw  = distances[0]
    top1_norm = prompts.normalize_dist(top1_raw)
    tag       = prompts.dist_tag(top1_raw)

    finite = [d for d in distances if d != float("inf")]
    if len(finite) > 1:
        cv = float(np.std(finite)) / float(np.mean(finite)) if np.mean(finite) > 1e-9 else 0.0
        spread_line = f"Neighbor spread: {'neighbors spread out ⚠' if cv > 0.15 else 'neighbors tightly agree ✓'}"
    else:
        spread_line = "Neighbor spread: n/a (only top-1)"

    lines = [f"Top-1 distance(norm): {top1_norm:.2f}  ({tag})", spread_line, ""]
    for i, r in enumerate(top_k_results, 1):
        feat_parts = [
            f"{k}={round(v, 1) if isinstance(v, float) else v}"
            for k, v in r.items() if k not in _NON_FEATURE_KEYS and v is not None
        ]
        if feat_parts:
            lines.append(f"Neighbor {i}: {', '.join(feat_parts)}")
    return "\n".join(lines)


# ── sufficiency deterministic 계산 (NB02 to_chatml의 _suf_from_nd와 동일) ──
# WM_FOCUS=True로 학습됐으므로 sufficiency는 모델이 예측하는 게 아니라,
# 여기서 계산해 프롬프트 입력에 주입하고(모델은 그 값을 보고 what_missing만
# 채움), 그대로 AnalyzeResponse.sufficiency로도 반환한다 (model output 아님).
# ★ [비교실험 스위치] ANALYZER_DECISION_MODE
#   "det" (기본, NB02 WM_FOCUS 정합): sufficiency=nd-band + cls=nd-threshold
#         모두 deterministic — LLM은 what_missing 서술만 담당.
#         (모델이 뱉은 cls는 model_cls 메타로 보존해 비교 가능)
#   "llm": [SYSTEM-COMPUTED] 주입 없이 모델이 cls/sufficiency/what_missing
#          모두 판단 — "LLM이 전부 판단" baseline.
ANALYZER_DECISION_MODE = os.getenv("ANALYZER_DECISION_MODE", "det").lower()
CLS_ND_THRESHOLD       = float(os.getenv("CLS_ND_THRESHOLD", "0.5"))   # nd>=0.5 → ABNORMAL

SUFFICIENCY_BOUNDARY_LOW  = float(os.getenv("SUFFICIENCY_BOUNDARY_LOW",  "0.3"))
SUFFICIENCY_BOUNDARY_HIGH = float(os.getenv("SUFFICIENCY_BOUNDARY_HIGH", "0.55"))


def _suf_from_nd(nd: float) -> str:
    """NB02 to_chatml의 _suf_from_nd와 동일 규칙.
    ★ 반드시 정규화된 nd(0~1, prompts.normalize_dist 결과)에 적용할 것 —
    raw L2 distance에 적용하면 항상 False가 되어 늘 confident로 떨어진다
    (실제로 controller.py/orchestrator_agentic.py에 있었던 버그)."""
    return "ambiguous" if (SUFFICIENCY_BOUNDARY_LOW <= nd < SUFFICIENCY_BOUNDARY_HIGH) else "confident"


def build_user_prompt(
    test_transaction:     Dict,
    cluster_id:           int,
    top_k_normal_results: List[Dict],
    top1_distance:        float,
    persona:              Optional[Dict] = None,
) -> Tuple[str, str]:
    """prompts.build_analyzer_user_prompt_agent(NB02가 실제로 학습에 쓴
    프롬프트)를 최대한 그대로 재현. coreset_feature_map 전역이 서빙엔
    없으므로 top5 포맷터만 로컬 어댑터(_format_top5_neighbors)로 대체하고,
    나머지 섹션 헤더/instruction 문구는 원본과 100% 동일하게 유지한다.

    ★ instruction이 5키(cls/sufficiency/next_action/what_missing/reason)를
    요구하는 것도 원본 그대로 둔다 — 이상해 보이지만 의도적이다. NB02는
    WM_FOCUS로 실제 학습 target은 {cls, what_missing} 2키만 남기면서도
    이 instruction 문구는 그대로 뒀기 때문에, 모델이 학습 때 "본" 입력
    형태를 서빙에서도 똑같이 재현하는 게 정합성 유지의 핵심이다. 문구를
    "깔끔하게" 2키로 고치면 오히려 새로운 train-serve mismatch가 생긴다.

    반환값: (user_prompt, sufficiency) — sufficiency는 모델 출력이 아니라
    여기서 계산한 deterministic 값 (AnalyzeResponse.sufficiency로 그대로 사용).
    """
    top_k = max(1, min(len(top_k_normal_results) or 1, 3))
    neighbors = top_k_normal_results[:top_k]

    test_fmt    = prompts.format_transaction(test_transaction)
    top_fmt     = _format_top5_neighbors(neighbors)
    persona_fmt = prompts.format_persona_zscore(test_transaction, persona) if persona else None

    sections = [f"## TEST TRANSACTION\n{test_fmt}"]
    cluster_block = [f"## CLUSTER #{cluster_id}"]
    cluster_block.append(persona_fmt if persona_fmt else f"Cluster ID: {cluster_id} (no profile)")
    sections.append("\n".join(cluster_block))
    sections.append(f"## RETRIEVED NORMAL NEIGHBORS (top-{top_k})\n{top_fmt}")
    sections.append(
        """You decide the verdict AND explain what is uncertain.

- cls: use the normalized distance (nd) as the PRIMARY basis — low nd → NORMAL,
  high nd → ABNORMAL. Restating nd for cls is fine; nd is the right signal for it.
- sufficiency: "ambiguous" if nd is mid-range OR the neighbors are spread out;
  otherwise "confident".
- next_action (only if ambiguous): "expand" if neighbors are spread / too few;
  "reroute" if the sample does not fit this cluster's persona; else "none".
- what_missing (the part only you can do): when ambiguous, explain in PLAIN
  FEATURE / PERSONA language WHICH concrete attributes make this case uncertain
  (e.g. "purchase_value far above cluster range and IP_country mismatch"), so a
  re-search can target them. Name features, not the distance number. Null if confident.

Output ONLY compact JSON (no markdown):
{"cls":"NORMAL|ABNORMAL","sufficiency":"confident|ambiguous","next_action":"none|expand|reroute","what_missing":"<=15 words naming concrete features, or null","reason":"<=20 words"}"""
    )
    user_prompt = "\n\n".join(sections)

    # ── WM_FOCUS 학습과 동일: deterministic sufficiency를 입력에 주입 ──────
    # ★ llm 모드에서는 주입하지 않는다 — 모델이 sufficiency까지 스스로 판단하는
    #   baseline (단, WM_FOCUS 학습 입력과 달라지는 train-serve gap이 비교실험의
    #   측정 대상 그 자체다).
    nd  = prompts.normalize_dist(top1_distance)   # raw L2 -> 0~1 정규화 (필수)
    suf = _suf_from_nd(nd)
    if ANALYZER_DECISION_MODE == "det":
        user_prompt += f"\n\n[SYSTEM-COMPUTED] retrieval_sufficiency = {suf.upper()}"
    return user_prompt, suf, nd


# ===========================
# 추론 — 백엔드 분기
# ===========================

def _extract_json(pred_txt: str) -> str:
    """마크다운 래핑 방어 (hf/gguf 공통)."""
    if "```json" in pred_txt:
        return pred_txt.split("```json")[-1].split("```")[0].strip()
    elif pred_txt.startswith("```"):
        return pred_txt.split("```")[1].strip()
    return pred_txt


def run_inference(user_prompt: str) -> Dict:
    """단건 추론. 백엔드에 따라 HF generate() 또는 llama_cpp completion을 탄다."""
    prompt = (
        f"<|im_start|>system\n{prompts.SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    if MODEL_BACKEND == "gguf":
        out = _model(
            prompt, max_tokens=MAX_NEW_TOKENS, temperature=0.0,
            stop=["<|im_end|>"],
        )
        pred_txt = out["choices"][0]["text"].strip()
    else:
        inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
        with torch.no_grad():
            out = _model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                pad_token_id=_tokenizer.pad_token_id, eos_token_id=_tokenizer.eos_token_id,
            )
        decoded = _tokenizer.decode(out[0], skip_special_tokens=False)
        pred_txt = decoded.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()

    logger.info("model_raw_output", backend=MODEL_BACKEND,
                pred_txt_len=len(pred_txt), pred_txt_preview=pred_txt[:300])

    pred_txt = _extract_json(pred_txt)
    try:
        return json.loads(pred_txt)
    except json.JSONDecodeError as e:
        logger.error("model_output_json_parse_failed", error=str(e), pred_txt_full=pred_txt)
        raise


def run_inference_batch(user_prompts: List[str]) -> List[Dict]:
    """
    hf 백엔드: [B, Seq_Len] 텐서로 패킹해 forward pass 1번만 태우는 진짜 배치.
    gguf 백엔드: llama.cpp가 단일 시퀀스 continuous batching을 지원하지
    않으므로, run_inference()를 순차 반복하는 것으로 대체한다 — 배치
    처리량 이득은 없지만 상위 인터페이스(DynamicBatcher)는 동일하게 동작.
    """
    if not user_prompts:
        return []

    if MODEL_BACKEND == "gguf":
        results: List[Dict] = []
        for p in user_prompts:
            try:
                results.append(run_inference(p))
            except json.JSONDecodeError as e:
                results.append({"_parse_error": str(e), "_raw": ""})
        return results

    prompts_fmt = [
        f"<|im_start|>system\n{prompts.SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{p}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        for p in user_prompts
    ]
    inputs = _tokenizer(
        prompts_fmt, return_tensors="pt", truncation=True, max_length=2048, padding=True,
    ).to(DEVICE)

    with torch.no_grad():
        out = _model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            temperature=None, top_p=None, top_k=None,
            pad_token_id=_tokenizer.pad_token_id, eos_token_id=_tokenizer.eos_token_id,
        )

    results: List[Dict] = []
    for i in range(out.shape[0]):
        decoded  = _tokenizer.decode(out[i], skip_special_tokens=False)
        pred_txt = _extract_json(
            decoded.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
        )
        try:
            results.append(json.loads(pred_txt))
        except json.JSONDecodeError as e:
            results.append({"_parse_error": str(e), "_raw": pred_txt})
    return results


# ===========================
# Pydantic Models
# ===========================

class SearchResult(BaseModel):
    es_doc_id:      str
    original_index: int
    cluster_id:     int
    distance:       float
    score:          float


class AnalyzeRequest(BaseModel):
    test_transaction:     Dict[str, Any]
    cluster_id:           int
    top_k_normal_results: List[Dict[str, Any]]
    top1_distance:        float
    persona:              Optional[Dict[str, Any]] = None
    tree_features:        Optional[Dict[str, Any]] = None  # v5에서 미사용, 호환용


class AnalyzeResponse(BaseModel):
    """
    노트북(analyzer_finetune_v9, NB02)과 동일한 3키 출력 schema.
    controller.py는 analyzer_schema.py의 normalize_analyzer_output()으로
    자동 인식하므로(_detect_3key가 "sufficiency" 키 존재로 판별), 이 파일을
    또 다른 schema로 바꿔도 controller.py/orchestrator_agentic.py는
    무수정으로 동작한다 — 바뀐 schema를 인식하는 _detect_xxx/_parse_xxx
    쌍만 analyzer_schema.py에 추가하면 된다.
    """
    cls:          str                  # "NORMAL" | "ABNORMAL"
    sufficiency:  str                  # deterministic (top1_distance 기반) — 모델 출력 아님
    what_missing: Optional[str] = None # WM_FOCUS 학습 target — 유일한 모델 자유 출력 필드

    # ── 디버깅/운영 메타데이터 (controller.py는 보지 않음, 모니터링용으로만 유지) ──
    persona_used:    bool = False
    tree_rules_used: bool = False
    decision_mode:   str  = "det"          # ★ det | llm (비교실험 추적용)
    model_cls:       Optional[str] = None  # ★ det 모드에서 모델이 낸 cls (참고용)


# ===========================
# FastAPI
# ===========================

app = FastAPI(
    title="Local Analyzer (Qwen2.5 + LoRA, HF/GGUF)",
    version="2.0.0",
    description="Gemini analyzer 대체 — 로컬 SFT 모델 기반 이상 탐지 분석",
)


@app.on_event("startup")
async def startup():
    load_model()

    # prompts.py 런타임 바인딩 — 정적 값(dist 통계, boundary threshold)은
    # 요청마다 바뀌지 않으므로 startup 시점 1회면 충분 (동시 요청 안전).
    prompts.set_mode("agent")
    prompts.bind_runtime(
        DIST_P50=DIST_P50, DIST_IQR=DIST_IQR,
        NORM_CLOSE_THRESHOLD=DIST_P25,
        NORM_MODERATE_THRESHOLD=DIST_P75,
    )

    # continuous batching 배처 백그라운드 루프 기동. load_model() 이후에
    # 시작해야 한다 — 모델이 준비되기 전에 첫 요청이 배치로 묶이면 안 됨.
    _batcher.start()


# ===========================
# 공통 추론 로직 (단일 요청 단위)
# ===========================

def _parse_inference_result(
    result: Dict, *, mode: str, t0: float, sufficiency: str, nd: float,
    persona_used: bool, tree_rules_used: bool,
) -> AnalyzeResponse:
    """
    run_inference()/run_inference_batch()가 반환한 raw dict 1개를 검증하고
    AnalyzeResponse로 조립한다. _run_single(단건)과 마이크로배처(N건)
    양쪽이 이 로직을 공유하므로, 검증 규칙이 바뀌어도 한 곳만 고치면 된다.

    sufficiency는 build_user_prompt()가 이미 top1_distance 기준으로
    deterministic 계산한 값을 파라미터로 받아 그대로 쓴다 — 모델 출력이
    아니다 (WM_FOCUS 학습: 모델은 cls/what_missing만 예측하도록 학습됨).
    """
    model_cls = str(result.get("cls", "")).upper()
    if ANALYZER_DECISION_MODE == "det":
        # ★ det: cls도 nd threshold로 deterministic 결정 — 6회 독립 측정에서
        #   L2 거리만이 유일한 분류 신호였던 결론과 정합. 모델 cls는 무시하되
        #   model_cls 메타로 보존해 det vs llm 비교가 가능하게 한다.
        cls = "ABNORMAL" if nd >= CLS_ND_THRESHOLD else "NORMAL"
    else:
        # llm: 모델 판단 그대로 (기존 검증 유지)
        cls = model_cls
        if cls not in ("NORMAL", "ABNORMAL"):
            raise ValueError(f"Invalid cls: '{cls}'")
        # sufficiency도 모델 출력 사용 — 유효하지 않으면 deterministic fallback
        _m_suf = str(result.get("sufficiency", "")).lower()
        if _m_suf in ("confident", "ambiguous"):
            sufficiency = _m_suf

    # what_missing: WM_FOCUS 학습 target에 있던 유일한 자유 출력 필드.
    # sufficiency 조건부 게이트를 걸지 않는다 — NB02 to_chatml도 teacher가
    # 생성한 값을 그대로 target으로 썼다(추가 null 처리 없음). sufficiency는
    # 이미 입력에 주어졌으므로, 모델이 그 맥락에 맞게 null 여부를 스스로
    # 판단한다는 전제다. (이 게이트가 예전엔 "항상 None" 버그의 원인이었음)
    what_missing = result.get("what_missing")

    # instruction 문구가 5키를 요구하다 보니 모델이 sufficiency/next_action/
    # reason도 같이 뱉을 수 있음 — WM_FOCUS 학습 target엔 없던 키라 신뢰하지
    # 않고 로그만 남긴다. 나중에 모델이 실제로 이 키들도 학습하게 되면,
    # 이 로그로 관측하고 필요 시 반영 여부를 결정하면 된다.
    untrained_keys = [k for k in result if k not in ("cls", "what_missing")]
    if untrained_keys:
        logger.info(
            "model_emitted_untrained_keys", mode=mode, keys=untrained_keys,
            values={k: result.get(k) for k in untrained_keys},
        )

    logger.info(
        "inference_parsed", mode=mode, cls=cls,
        sufficiency=sufficiency, what_missing=what_missing, raw_keys=list(result.keys()),
    )

    elapsed = time.time() - t0
    analyze_latency.observe(elapsed)
    analyze_requests.labels(status="success", mode=mode).inc()

    logger.info(
        "analyze_success", mode=mode, cls=cls, sufficiency=sufficiency,
        latency_s=f"{elapsed:.2f}", persona_used=persona_used,
    )

    return AnalyzeResponse(
        cls=cls, sufficiency=sufficiency, what_missing=what_missing,
        persona_used=persona_used, tree_rules_used=tree_rules_used,
        decision_mode=ANALYZER_DECISION_MODE,
        model_cls=model_cls if ANALYZER_DECISION_MODE == "det" else None,
    )


def _run_single(
    req:              AnalyzeRequest,
    persona_enabled:  bool,
    tree_rules_used:  bool,
    mode:             str = "single",
) -> AnalyzeResponse:
    t0 = time.time()
    effective_persona = req.persona if (persona_enabled and req.persona is not None) else None

    logger.info("run_single_building_prompt", mode=mode, cluster_id=req.cluster_id,
                top_k=len(req.top_k_normal_results))
    user_prompt, suf, nd = build_user_prompt(
        test_transaction     = req.test_transaction,
        cluster_id           = req.cluster_id,
        top_k_normal_results = req.top_k_normal_results,
        top1_distance        = req.top1_distance,
        persona              = effective_persona,
    )
    logger.info("run_single_prompt_built", prompt_len=len(user_prompt), sufficiency=suf,
                elapsed_s=f"{time.time()-t0:.2f}")

    logger.info("run_single_inference_start", backend=MODEL_BACKEND,
                model_loaded=(_model is not None))
    result = run_inference(user_prompt)

    logger.info("run_single_inference_done", elapsed_s=f"{time.time()-t0:.2f}",
                result_keys=list(result.keys()) if isinstance(result, dict) else type(result).__name__)

    return _parse_inference_result(
        result, mode=mode, t0=t0, sufficiency=suf, nd=nd,
        persona_used=effective_persona is not None, tree_rules_used=tree_rules_used,
    )


# ===========================
# Continuous Batching — 동적 마이크로배처
# ===========================
# hf 백엔드: N개의 동시 요청을 짧은 시간 창 동안 모아 run_inference_batch()로
# 한 번의 forward pass에 태운다 (vLLM/TGI의 continuous batching과 같은
# 아이디어를 작은 스케일로 구현).
# gguf 백엔드: run_inference_batch가 내부적으로 순차 루프이므로, 배처를
# 켜도 정확성엔 문제없지만 처리량 이득은 없다 — 필요하면 gguf 배포에서는
# ENABLE_CONTINUOUS_BATCHING=false로 두는 걸 권장.
BATCH_WAIT_MS   = int(os.getenv("BATCH_WAIT_MS", "30"))
MAX_BATCH_SIZE  = int(os.getenv("MAX_BATCH_SIZE", "8"))


@dataclass
class _BatchItem:
    user_prompt:      str
    sufficiency:      str    # build_user_prompt()가 계산한 deterministic 값
    nd:               float  # ★ 정규화 거리 — det 모드 cls 결정에 필요
    persona_used:     bool
    tree_rules_used:  bool
    t0:               float
    future:           "asyncio.Future" = field(default=None)


class DynamicBatcher:
    """
    싱글톤 배처. FastAPI startup에서 백그라운드 루프 1개를 띄우고,
    각 /analyze 요청은 submit()으로 큐에 넣은 뒤 자신의 future만 await한다.
    """
    def __init__(self) -> None:
        self._queue: "asyncio.Queue[_BatchItem]" = asyncio.Queue()
        self._loop_task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._loop_task is None:
            self._loop_task = asyncio.create_task(self._run_loop())
            logger.info("dynamic_batcher_started", batch_wait_ms=BATCH_WAIT_MS, max_batch_size=MAX_BATCH_SIZE)

    async def submit(self, user_prompt: str, *, sufficiency: str, nd: float, persona_used: bool, tree_rules_used: bool) -> AnalyzeResponse:
        item = _BatchItem(
            user_prompt=user_prompt, sufficiency=sufficiency, nd=nd, persona_used=persona_used,
            tree_rules_used=tree_rules_used, t0=time.time(),
        )
        item.future = asyncio.get_event_loop().create_future()
        await self._queue.put(item)
        return await item.future

    async def _run_loop(self) -> None:
        while True:
            batch: List[_BatchItem] = []
            try:
                first = await self._queue.get()
                batch.append(first)
                deadline = time.time() + (BATCH_WAIT_MS / 1000.0)
                while len(batch) < MAX_BATCH_SIZE:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
            except Exception as e:
                logger.error("batcher_loop_collect_failed", error=str(e))
                continue

            try:
                results = await asyncio.to_thread(run_inference_batch, [b.user_prompt for b in batch])
                batch_inference_size.observe(len(batch))
                for item, raw in zip(batch, results):
                    try:
                        resp = _parse_inference_result(
                            raw, mode="batched", t0=item.t0, sufficiency=item.sufficiency, nd=item.nd,
                            persona_used=item.persona_used, tree_rules_used=item.tree_rules_used,
                        )
                        if not item.future.done():
                            item.future.set_result(resp)
                    except Exception as e:
                        analyze_requests.labels(status="parse_error", mode="batched").inc()
                        if not item.future.done():
                            item.future.set_exception(e)
            except Exception as e:
                logger.error("batch_inference_failed", error=str(e), n=len(batch))
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(e)


_batcher = DynamicBatcher()


# ===========================
# /analyze  (단일)
# ===========================

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_transaction(
    request:              AnalyzeRequest,
    x_experiment_case:    Optional[str] = Header(None),
    x_persona_enabled:    Optional[str] = Header(None),
    x_tree_rules_enabled: Optional[str] = Header(None),
):
    effective_persona_enabled = (
        (x_persona_enabled.lower() == "true")
        if x_persona_enabled is not None else PERSONA_ENABLED
    )
    effective_tree_rules_enabled = (
        (x_tree_rules_enabled.lower() == "true")
        if x_tree_rules_enabled is not None else TREE_RULES_ENABLED
    )

    first_neighbor = request.top_k_normal_results[0] if request.top_k_normal_results else None
    logger.info(
        "analyze_request_received",
        cluster_id=request.cluster_id, top_k=len(request.top_k_normal_results),
        top1_distance=request.top1_distance, persona_present=request.persona is not None,
        first_neighbor_keys=list(first_neighbor.keys()) if first_neighbor else None,
        test_transaction_keys=list(request.test_transaction.keys()) if request.test_transaction else None,
    )

    try:
        if ENABLE_CONTINUOUS_BATCHING:
            effective_persona = request.persona if (effective_persona_enabled and request.persona is not None) else None
            user_prompt, suf, nd = build_user_prompt(
                test_transaction=request.test_transaction, cluster_id=request.cluster_id,
                top_k_normal_results=request.top_k_normal_results, top1_distance=request.top1_distance,
                persona=effective_persona,
            )
            return await _batcher.submit(
                user_prompt, sufficiency=suf, nd=nd, persona_used=effective_persona is not None,
                tree_rules_used=effective_tree_rules_enabled,
            )
        logger.info("analyze_dispatching_to_thread", backend=MODEL_BACKEND,
                    model_loaded=(_model is not None))
        result = await asyncio.to_thread(
            _run_single, request, effective_persona_enabled, effective_tree_rules_enabled, "single",
        )
        logger.info("analyze_thread_returned", cluster_id=request.cluster_id)
        return result
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        parse_errors.labels(mode="single").inc()
        analyze_requests.labels(status="parse_error", mode="single").inc()
        logger.error("analyze_parse_error", mode="single", error=str(e))
        raise HTTPException(status_code=422, detail=f"LLM output parse failed: {e}")
    except Exception as e:
        import traceback
        analyze_requests.labels(status="error", mode="single").inc()
        tb = traceback.format_exc()
        logger.error("analyze_error", mode="single", error_type=type(e).__name__,
                     error=str(e), traceback=tb[:2000])
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


# ===========================
# /analyze-batch  (배치)
# ===========================

@app.post("/analyze-batch", response_model=List[AnalyzeResponse])
async def analyze_batch(
    requests:             List[AnalyzeRequest],
    x_experiment_case:    Optional[str] = Header(None),
    x_persona_enabled:    Optional[str] = Header(None),
    x_tree_rules_enabled: Optional[str] = Header(None),
):
    """복수 거래를 순차 처리. CPU 추론 특성상 asyncio.to_thread로 블로킹 방지."""
    if not requests:
        return []

    effective_persona_enabled    = (x_persona_enabled.lower()    == "true") \
                                    if x_persona_enabled    is not None else PERSONA_ENABLED
    effective_tree_rules_enabled = (x_tree_rules_enabled.lower() == "true") \
                                    if x_tree_rules_enabled is not None else TREE_RULES_ENABLED

    logger.info("analyze_batch_start", n=len(requests),
                persona_enabled=effective_persona_enabled, tree_rules_enabled=effective_tree_rules_enabled)

    results: List[AnalyzeResponse] = []
    for i, req in enumerate(requests):
        try:
            resp = await asyncio.to_thread(
                _run_single, req, effective_persona_enabled, effective_tree_rules_enabled, "batch",
            )
            results.append(resp)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            parse_errors.labels(mode="batch").inc()
            analyze_requests.labels(status="parse_error", mode="batch").inc()
            logger.error("batch_item_parse_error", idx=i, error=str(e))
            results.append(AnalyzeResponse(
                cls="NORMAL", sufficiency="ambiguous",
                what_missing=f"parse_error at batch index {i}: {e}",
                persona_used=False, tree_rules_used=False,
            ))
        except Exception as e:
            analyze_requests.labels(status="error", mode="batch").inc()
            logger.error("batch_item_error", idx=i, error=str(e))
            results.append(AnalyzeResponse(
                cls="NORMAL", sufficiency="ambiguous",
                what_missing=f"inference_error at batch index {i}: {e}",
                persona_used=False, tree_rules_used=False,
            ))

    logger.info("analyze_batch_done", n=len(requests),
                n_error=sum(1 for r in results if r.what_missing and "_error at batch index" in (r.what_missing or "")))
    return results


# ===========================
# Health / Metrics
# ===========================

@app.get("/health")
async def health():
    return {
        "status":              "healthy",
        "service":             "local-analyzer",
        "version":             "2.0.0",
        "backend":              MODEL_BACKEND,
        "model_loaded":        _model is not None,
        "experiment_case":     EXPERIMENT_CASE,
        "adapter_type":        ADAPTER_TYPE,
        "device":              DEVICE,
        "persona_enabled":     PERSONA_ENABLED,
        "tree_rules_enabled":  TREE_RULES_ENABLED,
    }


@app.get("/ready")
async def ready():
    """
    모델 로드 상태 + 간단한 추론 smoke test.
    - model_loaded=False → load_model()이 아직 안 끝난 것 (startup 중)
    - inference_ok=False → 모델은 떴지만 추론이 실패하는 것
    - inference_ok=True  → 실제 /analyze 요청 처리 가능한 상태
    """
    if _model is None or (MODEL_BACKEND == "hf" and _tokenizer is None):
        raise HTTPException(status_code=503, detail="model not loaded yet")

    smoke_prompt = (
        "<|im_start|>system\nYou are a classifier.<|im_end|>\n"
        "<|im_start|>user\nReply with exactly: {\"ok\": true}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    try:
        t0 = time.time()
        if MODEL_BACKEND == "gguf":
            out = _model(smoke_prompt, max_tokens=16, temperature=0.0, stop=["<|im_end|>"])
            decoded = out["choices"][0]["text"]
        else:
            inputs = _tokenizer(smoke_prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = _model.generate(
                    **inputs, max_new_tokens=16, do_sample=False,
                    pad_token_id=_tokenizer.pad_token_id, eos_token_id=_tokenizer.eos_token_id,
                )
            decoded = _tokenizer.decode(out[0], skip_special_tokens=True)
        elapsed = time.time() - t0
        logger.info("ready_smoke_test_ok", elapsed_s=f"{elapsed:.2f}", decoded_preview=decoded[-50:])
        return {
            "status": "ready", "model_loaded": True, "inference_ok": True,
            "backend": MODEL_BACKEND, "smoke_elapsed_s": f"{elapsed:.2f}",
            "smoke_output": decoded[-100:], "device": DEVICE, "max_new_tokens": MAX_NEW_TOKENS,
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error("ready_smoke_test_failed", error_type=type(e).__name__,
                     error=str(e), traceback=tb[:1000])
        raise HTTPException(status_code=503,
                            detail=f"model loaded but inference failed: {type(e).__name__}: {e}")


@app.get("/config")
async def config():
    return {
        "backend":             MODEL_BACKEND,
        "experiment_case":     EXPERIMENT_CASE,
        "teacher_provider":    TEACHER_PROVIDER,
        "adapter_type":        ADAPTER_TYPE,
        "adapter_path":        ADAPTER_PATH,
        "gguf_filename":       GGUF_FILENAME if MODEL_BACKEND == "gguf" else None,
        "base_model_id":       BASE_MODEL_ID,
        "max_new_tokens":      MAX_NEW_TOKENS,
        "device":              DEVICE,
        "torch_dtype":         TORCH_DTYPE_STR,
        "persona_enabled":     PERSONA_ENABLED,
        "tree_rules_enabled":  TREE_RULES_ENABLED,
        "dist_stats": {
            "p25": DIST_P25, "p50": DIST_P50, "p75": DIST_P75, "iqr": DIST_IQR,
        },
    }


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)