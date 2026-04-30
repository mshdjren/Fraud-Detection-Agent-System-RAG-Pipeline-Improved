"""
local_analyzer.py — Local LLM Analyzer (Qwen2.5-0.5B + LoRA)
==============================================================

orchestrator.py V5.0의 call_analyzer()와 호환되는 /analyze 엔드포인트.
Gemini 기반 analyzer.py를 local SFT 모델로 대체.

Input  (orchestrator → analyzer):
  test_transaction, cluster_id, top_k_normal_results,
  top1_distance, persona (Optional), tree_features (Optional)

Output (analyzer → orchestrator):
  classification, confidence, reasoning, key_evidence,
  final_verdict, persona_used, tree_rules_used
"""

import os
import json
import time
import math
import logging
from typing import Dict, List, Optional, Any

import yaml
import torch
import asyncio
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import Response
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import structlog

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

DIST_P25 = float(os.getenv("DIST_P25", "1.264"))
DIST_P50 = float(os.getenv("DIST_P50", "1.448"))
DIST_P75 = float(os.getenv("DIST_P75", "1.651"))
DIST_IQR = float(os.getenv("DIST_IQR", "0.387"))

PERSONA_ENABLED    = os.getenv("PERSONA_ENABLED",    "true").lower()  == "true"
TREE_RULES_ENABLED = os.getenv("TREE_RULES_ENABLED", "false").lower() == "true"

ADAPTER_PATH = os.getenv("ADAPTER_LOCAL_DIR", "/data/adapter")

TORCH_DTYPE = (
    torch.bfloat16 if TORCH_DTYPE_STR == "bfloat16"
    else torch.float32
)

logger = structlog.get_logger()

# ===========================
# Prometheus
# ===========================

analyze_requests  = Counter("analyzer_requests_total",  "Total /analyze requests",       ["status", "mode"])
analyze_latency   = Histogram("analyzer_latency_seconds", "Inference latency",
                               buckets=[1, 2, 5, 10, 30, 60, 120, 300])
parse_errors      = Counter("analyzer_parse_errors_total", "JSON parse errors", ["mode"])

# ===========================
# 모델 로드 (startup)
# ===========================

_model     = None
_tokenizer = None


def load_model():
    global _model, _tokenizer
    logger.info("loading_model", base=BASE_MODEL_ID, adapter=ADAPTER_PATH, device=DEVICE)
    t0 = time.time()

    # CPU 스레드 수 명시 (e2-standard-2: 2 vCPU 전체 활용)
    if DEVICE == "cpu":
        n_threads = int(os.getenv("OMP_NUM_THREADS", "2"))
        torch.set_num_threads(n_threads)
        logger.info("cpu_threads_set", n_threads=n_threads)

    _tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_PATH,           # tokenizer가 adapter 폴더에 저장됨
        padding_side="left",
    )
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    # ── CPU 보장 로딩 ───────────────────────────────────────────
    # device_map 미사용: accelerate의 multi-device 로직 완전 배제.
    # CPU 환경에서는 from_pretrained 기본값이 CPU이므로 직접 지정.
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        low_cpu_mem_usage=True,
        # device_map 의도적으로 제거 → CPU 단일 디바이스 강제
    )
    if DEVICE != "cpu":
        base = base.to(DEVICE)   # GPU 사용 시에만 명시적 이동

    # LoRA 어댑터 로드 후 merge → 추론 시 adapter forward pass 오버헤드 제거
    model_with_lora = PeftModel.from_pretrained(base, ADAPTER_PATH)
    _model = model_with_lora.merge_and_unload()
    _model.eval()

    elapsed = time.time() - t0
    logger.info(
        "model_loaded",
        elapsed_s   = f"{elapsed:.1f}",
        device      = DEVICE,
        dtype       = str(TORCH_DTYPE),
        model_param = f"{sum(p.numel() for p in _model.parameters()) / 1e6:.0f}M",
    )


# ===========================
# Prompt 구성 (colab과 동일)
# ===========================

SYSTEM_PROMPT = (
    "You are an expert fraud detection analyst. "
    "Analyze the TEST TRANSACTION by comparing it with the cluster profile "
    "and retrieved similar normal patterns."
)


def _normalize_dist(d: float) -> float:
    if DIST_IQR == 0:
        return 0.5
    raw = (d - DIST_P50) / DIST_IQR
    return float(1 / (1 + math.exp(-raw)))


def _dist_tag(d: float) -> str:
    if d < DIST_P25:
        return "close match ✓"
    elif d < DIST_P75:
        return "moderate match"
    return "distant match ⚠"


def _format_transaction(data: Dict) -> str:
    sections = {
        "Numeric Features":     ["purchase_value", "age"],
        "Categorical Features": ["sex", "source", "browser"],
        "Additional Context":   ["weekday_purchase", "month_purchase", "IP_country"],
    }
    out = []
    for sec_name, cols in sections.items():
        vals = [(c, data.get(c)) for c in cols if data.get(c) is not None]
        if not vals:
            continue
        out.append(f"[{sec_name}]")
        for col, val in vals:
            fmt = f"{val:.4f}" if isinstance(val, float) else str(val)
            out.append(f"  - {col}: {fmt}")
        out.append("")
    return "\n".join(out).strip()


def _format_persona(persona: Dict) -> str:
    if not persona:
        return ""
    lines = []
    if "description" in persona:
        lines.append(f"**Summary**: {persona['description']}")
    if "numeric_stats" in persona:
        lines.append("\n**Numeric Statistics:**")
        for feat, stats in persona["numeric_stats"].items():
            lines.append(
                f"  - {feat}: mean={stats.get('mean', 0):.3f}, "
                f"std={stats.get('std', 0):.3f}, "
                f"min={stats.get('min', 0):.3f}, max={stats.get('max', 0):.3f}"
            )
    if "categorical_dist" in persona:
        lines.append("\n**Categorical Distribution:**")
        for feat, dist in persona["categorical_dist"].items():
            dominant = max(dist, key=dist.get) if dist else "?"
            pct = int(dist.get(dominant, 0) / max(sum(dist.values()), 1) * 100)
            details = ", ".join(f"{k}:{v}" for k, v in list(dist.items())[:4])
            lines.append(f"  - {feat}: dominant={dominant} ({pct}%)  [{details}]")
    return "\n".join(lines)


def _format_top5_neighbors(top_k_results: List[Dict]) -> str:
    """
    orchestrator → retriever 결과:
      [{es_doc_id, original_index, cluster_id, distance, score}, ...]
    colab의 format_top5_neighbors와 동일한 구조로 변환.
    """
    if not top_k_results:
        return "No retrieved results."

    top1_raw  = float(top_k_results[0].get("distance", float("inf")))
    top1_norm = _normalize_dist(top1_raw)
    tag       = _dist_tag(top1_raw)

    lines = [
        f"Top-1 L2 distance: {top1_raw:.4f}  [normalized: {top1_norm:.3f}]  ({tag})",
        f"  (dataset p25={DIST_P25:.3f}, p50={DIST_P50:.3f}, p75={DIST_P75:.3f})",
        "",
    ]
    for i, r in enumerate(top_k_results, 1):
        dist   = float(r.get("distance", float("inf")))
        oi     = r.get("original_index", "?")
        norm_d = _normalize_dist(dist)
        sig    = "✓" if dist < DIST_P25 else ("~" if dist < DIST_P75 else "⚠")
        lines.append(
            f"Neighbor {i}: original_index={oi}  "
            f"distance={dist:.4f}  norm={norm_d:.3f}  {sig}"
        )
        # retriever result에 feature 필드가 있으면 출력 (없으면 생략)
        feat_parts = []
        for col in ["purchase_value", "age", "sex", "source", "browser"]:
            val = r.get(col)
            if val is not None:
                feat_parts.append(
                    f"{col}={val:.4f}" if isinstance(val, float) else f"{col}={val}"
                )
        if feat_parts:
            lines.append(f"  Features: {', '.join(feat_parts)}")
    return "\n".join(lines)


def build_user_prompt(
    test_transaction:     Dict,
    cluster_id:           int,
    top_k_normal_results: List[Dict],
    top1_distance:        float,
    persona:              Optional[Dict] = None,
) -> str:
    """colab build_analyzer_user_prompt과 동일한 구조."""
    test_fmt    = _format_transaction(test_transaction)
    top5_fmt    = _format_top5_neighbors(top_k_normal_results)
    persona_fmt = _format_persona(persona) if persona else None

    sections = [f"## TEST TRANSACTION (TO ANALYZE)\n{test_fmt}"]

    cluster_block = [f"## CLUSTER #{cluster_id} PROFILE"]
    if persona_fmt:
        cluster_block.extend(["\n**Cluster Statistical Profile:**", persona_fmt])
    else:
        cluster_block.append(f"\nCluster ID: {cluster_id} (no profile available)")
    sections.append("\n".join(cluster_block))

    sections.append(f"## RETRIEVED SIMILAR NORMAL PATTERNS\n{top5_fmt}")
    sections.append(
        """## ANALYSIS TASK
Determine if the test transaction is **NORMAL** or **ABNORMAL**.

**Analysis Framework:**
1. **Statistical Deviation**: Compare test values against the cluster's numeric/categorical profile.
2. **Pattern Similarity**: Interpret the L2 distances — low distance = similar to known normal patterns, high distance = potentially anomalous.

**Output Format (JSON):**
```json
{
  "classification": "NORMAL" or "ABNORMAL",
  "confidence": 0-100,
  "reasoning": "Brief analysis following the framework above",
  "key_evidence": ["Point 1", "Point 2", "Point 3"],
  "final_verdict": "One concise sentence summary"
}
```

**Guidelines:**
- ABNORMAL if test deviates significantly from cluster profile or retrieved patterns
- NORMAL if test aligns with known cluster patterns
- High top-1 L2 distance (> 1.0) is a strong anomaly signal
- Confidence: 0=uncertain, 100=certain
- Reference specific feature values in reasoning"""
    )
    return "\n\n---\n\n".join(sections)


# ===========================
# 추론
# ===========================

def run_inference(user_prompt: str) -> Dict:
    """colab Cell 47 추론 코드와 동일한 구조."""
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(DEVICE)

    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=_tokenizer.pad_token_id,
            eos_token_id=_tokenizer.eos_token_id,
        )

    decoded  = _tokenizer.decode(out[0], skip_special_tokens=False)
    pred_txt = decoded.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()

    # 마크다운 래핑 방어
    if "```json" in pred_txt:
        pred_txt = pred_txt.split("```json")[-1].split("```")[0].strip()
    elif pred_txt.startswith("```"):
        pred_txt = pred_txt.split("```")[1].strip()

    return json.loads(pred_txt)


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
    classification:  str
    confidence:      int
    reasoning:       str
    key_evidence:    List[str]
    final_verdict:   str
    persona_used:    bool
    tree_rules_used: bool


# ===========================
# FastAPI
# ===========================

app = FastAPI(
    title="Local Analyzer (Qwen2.5-0.5B + LoRA)",
    version="1.0.0",
    description="Gemini analyzer 대체 — 로컬 SFT 모델 기반 이상 탐지 분석",
)


@app.on_event("startup")
async def startup():
    load_model()


# ===========================
# 공통 추론 로직 (단일 요청 단위)
# ===========================

def _run_single(
    req:              AnalyzeRequest,
    persona_enabled:  bool,
    tree_rules_used:  bool,
    mode:             str = "single",
) -> AnalyzeResponse:
    """
    단일 AnalyzeRequest → AnalyzeResponse.
    /analyze, /analyze-batch 양쪽에서 호출.

    persona_enabled:
      True  → req.persona를 프롬프트에 포함 (Header 또는 ConfigMap 값 반영)
      False → 프롬프트에서 persona 섹션 제거

    tree_rules_used:
      현재 local LLM은 tree_features를 사용하지 않으므로 항상 False로 고정.
      04-analyzer.yaml과의 인터페이스 호환성을 위해 필드만 유지.
    """
    t0 = time.time()

    # persona_enabled=False 면 persona를 None으로 덮어씀
    effective_persona = req.persona if (persona_enabled and req.persona is not None) else None

    user_prompt = build_user_prompt(
        test_transaction     = req.test_transaction,
        cluster_id           = req.cluster_id,
        top_k_normal_results = req.top_k_normal_results,
        top1_distance        = req.top1_distance,
        persona              = effective_persona,
    )

    result = run_inference(user_prompt)

    classification = result.get("classification", "").upper()
    if classification not in ("NORMAL", "ABNORMAL"):
        raise ValueError(f"Invalid classification: '{classification}'")

    key_evidence = result.get("key_evidence", [])
    if not isinstance(key_evidence, list):
        key_evidence = [str(key_evidence)]

    elapsed = time.time() - t0
    analyze_latency.observe(elapsed)
    analyze_requests.labels(status="success", mode=mode).inc()

    logger.info(
        "analyze_success",
        mode           = mode,
        classification = classification,
        confidence     = result.get("confidence"),
        latency_s      = f"{elapsed:.2f}",
        persona_used   = effective_persona is not None,
    )

    return AnalyzeResponse(
        classification  = classification,
        confidence      = int(result.get("confidence", 50)),
        reasoning       = result.get("reasoning", ""),
        key_evidence    = key_evidence[:3],
        final_verdict   = result.get("final_verdict", ""),
        persona_used    = effective_persona is not None,
        tree_rules_used = tree_rules_used,
    )


# ===========================
# /analyze  (단일)
# ===========================

# /analyze 엔드포인트 — asyncio.to_thread 추가
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

    # ★ 요청 수신 즉시 로그 (디버깅용)
    logger.info(
        "analyze_request_received",
        cluster_id=request.cluster_id,
        top_k=len(request.top_k_normal_results),
        persona_present=request.persona is not None,
    )

    try:
        # ★ asyncio.to_thread로 blocking inference를 별도 스레드에서 실행
        return await asyncio.to_thread(
            _run_single,
            request,
            effective_persona_enabled,
            effective_tree_rules_enabled,
            "single",
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        parse_errors.labels(mode="single").inc()
        analyze_requests.labels(status="parse_error", mode="single").inc()
        logger.error("analyze_parse_error", mode="single", error=str(e))
        raise HTTPException(status_code=422, detail=f"LLM output parse failed: {e}")
    except Exception as e:
        analyze_requests.labels(status="error", mode="single").inc()
        logger.error("analyze_error", mode="single", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


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
    """
    복수 거래를 순차 처리.

    CPU 추론 특성상 asyncio.to_thread로 블로킹 방지.
    각 요청은 독립적으로 _run_single() 호출 → 개별 실패 시 error 응답으로 대체.
    """
    if not requests:
        return []

    effective_persona_enabled    = (x_persona_enabled.lower()    == "true") \
                                    if x_persona_enabled    is not None else PERSONA_ENABLED
    effective_tree_rules_enabled = (x_tree_rules_enabled.lower() == "true") \
                                    if x_tree_rules_enabled is not None else TREE_RULES_ENABLED

    logger.info(
        "analyze_batch_start",
        n                    = len(requests),
        persona_enabled      = effective_persona_enabled,
        tree_rules_enabled   = effective_tree_rules_enabled,
    )

    results: List[AnalyzeResponse] = []
    for i, req in enumerate(requests):
        try:
            resp = await asyncio.to_thread(
                _run_single,
                req,
                effective_persona_enabled,
                effective_tree_rules_enabled,
                "batch",
            )
            results.append(resp)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            parse_errors.labels(mode="batch").inc()
            analyze_requests.labels(status="parse_error", mode="batch").inc()
            logger.error("batch_item_parse_error", idx=i, error=str(e))
            # 개별 실패 → error sentinel 삽입 (배치 전체 중단 방지)
            results.append(AnalyzeResponse(
                classification  = "ERROR",
                confidence      = 0,
                reasoning       = f"Parse error at batch index {i}: {e}",
                key_evidence    = [],
                final_verdict   = "Parse error",
                persona_used    = False,
                tree_rules_used = False,
            ))
        except Exception as e:
            analyze_requests.labels(status="error", mode="batch").inc()
            logger.error("batch_item_error", idx=i, error=str(e))
            results.append(AnalyzeResponse(
                classification  = "ERROR",
                confidence      = 0,
                reasoning       = f"Inference error at batch index {i}: {e}",
                key_evidence    = [],
                final_verdict   = "Inference error",
                persona_used    = False,
                tree_rules_used = False,
            ))

    logger.info("analyze_batch_done", n=len(requests), n_error=sum(1 for r in results if r.classification == "ERROR"))
    return results


# ===========================
# Health / Metrics
# ===========================

@app.get("/health")
async def health():
    return {
        "status":              "healthy",
        "service":             "local-analyzer",
        "version":             "1.0.0",
        "model_loaded":        _model is not None,
        "experiment_case":     EXPERIMENT_CASE,
        "adapter_type":        ADAPTER_TYPE,
        "device":              DEVICE,
        "persona_enabled":     PERSONA_ENABLED,
        "tree_rules_enabled":  TREE_RULES_ENABLED,
    }


@app.get("/config")
async def config():
    """04-analyzer.yaml과 동일하게 /config 엔드포인트 제공."""
    return {
        "experiment_case":     EXPERIMENT_CASE,
        "teacher_provider":    TEACHER_PROVIDER,
        "adapter_type":        ADAPTER_TYPE,
        "adapter_path":        ADAPTER_PATH,
        "base_model_id":       BASE_MODEL_ID,
        "max_new_tokens":      MAX_NEW_TOKENS,
        "device":              DEVICE,
        "torch_dtype":         TORCH_DTYPE_STR,
        "persona_enabled":     PERSONA_ENABLED,
        "tree_rules_enabled":  TREE_RULES_ENABLED,
        "dist_stats": {
            "p25": DIST_P25, "p50": DIST_P50,
            "p75": DIST_P75, "iqr": DIST_IQR,
        },
    }


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=CFG["host"], port=CFG["port"])