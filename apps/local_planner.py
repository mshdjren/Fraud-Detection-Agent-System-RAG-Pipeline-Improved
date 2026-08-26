"""
local_planner.py — GRPO Planner serving (Qwen2.5-0.5B, HF/GGUF, 제약 디코딩)
================================================================================
역할:
  02b 노트북(GRPO_ONLY + 제약 디코딩)에서 학습한 planner adapter를 서빙한다.
  state(round0 retrieval 결과) → retrieval action 1개.

      {"action":"stop"}
      {"action":"contextual_knn","features":["sex","source"]}
      {"action":"attribute_knn","features":["IP_country"]}
      {"action":"global_knn"}                        # 기본 비활성 (config로 on)

설계 원칙:
  ★ 재선언 금지 — action space / cost / feature 후보 규칙 / tool payload는
    전부 agent_tools.py의 registry에서 import한다. 이 파일에는 planner
    고유의 "서빙 로직"만 있고, "무엇이 유효한 action인가"에 대한 정의는
    한 줄도 없다. agent_tools.py는 controller / agent-tools-mcp와 같은
    파일이 mount되므로 세 서비스의 action space가 구조적으로 일치한다.
  ★ 프롬프트도 prompts.py의 planner 네임스페이스만 사용한다
    (PLANNER_SYSTEM_PROMPT / build_planner_user_prompt[_v2|_v3]).
  ★ 제약 디코딩이 포맷을 보장한다 — 학습(GRPO_ONLY)이 SFT 없이 성립한
    전제이므로 서빙에서도 반드시 켠다. 끄면 train-serve mismatch다.
      hf   : prefix_allowed_tokens_fn(trie)   — 노트북 학습 경로와 동일
      gguf : GBNF grammar                     — 같은 유효 출력 집합에서 생성
  ★ local_analyzer.py와 동형(同型): 같은 base model, 같은 서빙 이미지,
    같은 백엔드 스위치, 같은 _MODEL_LOCK 직렬화, 같은 /health /ready
    /config /metrics 계약. 그래서 Dockerfile을 새로 만들지 않는다.
  ★ 파싱 실패/미지 action은 예외를 던지지 않고 PLANNER_FALLBACK_ACTION으로
    degrade한다 — planner 장애가 파이프라인 장애가 되면 안 된다. 대신
    is_fallback/invalid_reason으로 반드시 계측된다.

nd 척도(재검색 후 보정 여부)는 이 서비스가 계산하지 않는다. PLANNER_ND_MODE를
응답에 echo만 하고, 실제 보정은 controller의 node_research가 동일 config로
수행한다 — nd는 controller의 판정 frame이므로 계산 주체를 옮기지 않는다.
"""

from __future__ import annotations

import os
import re
import json
import time
import math
import threading
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import structlog

import prompts   # ★ evaluator와 공유하는 common 모듈 (planner 네임스페이스만 사용)

# ★ [단일 출처] action space / cost / 후보 규칙 / payload — 전부 여기서 온다.
#   controller / agent-tools-mcp와 같은 agent_tools.py가 mount되므로
#   세 서비스가 같은 정의를 본다(stage 확장 시 수정 지점도 한 곳).
from agent_tools import (
    PLANNER_STAGE,
    planner_stage_actions,
    planner_action_names,
    planner_action_spec,
    resolve_planner_action,
    planner_tool_payload,
    build_planner_action_space,
    build_planner_tool_catalog,
    candidate_feature_sets,
    CONTEXTUAL_CANDIDATE_FIELDS,
    MAX_CONTEXTUAL_FEATURES,
    ACTION_COST,
)

logger = structlog.get_logger()

# ===========================
# Config (전부 ConfigMap에서 주입)
# ===========================
EXPERIMENT_CASE = os.getenv("EXPERIMENT_CASE", "pca_64_k100")

# ── 모델/어댑터 ──────────────────────────────────────────────
BASE_MODEL_ID   = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_PATH    = os.getenv("ADAPTER_LOCAL_DIR", "/data/adapter")
MODEL_BACKEND   = os.getenv("PLANNER_MODEL_BACKEND", os.getenv("MODEL_BACKEND", "hf")).lower()
GGUF_FILENAME   = os.getenv("GGUF_FILENAME", "model.gguf")
GGUF_QUANT      = os.getenv("GGUF_QUANT", "q8_0")
N_CTX           = int(os.getenv("N_CTX", "2048"))
DEVICE          = os.getenv("DEVICE", "cpu")
TORCH_DTYPE_STR = os.getenv("TORCH_DTYPE", "float32")
# 학습(02b GRPO_MAX_NEW)과 동일해야 한다 — 늘리면 JSON 뒤에 잡음이 붙는다.
MAX_NEW_TOKENS  = int(os.getenv("PLANNER_MAX_NEW_TOKENS", "24"))

# ── 프롬프트 버전 ────────────────────────────────────────────
#   v1: build_planner_user_prompt      (DECISION RULE 휴리스틱 포함)
#   v2: build_planner_user_prompt_v2   (evidence-only)
#   v3: build_planner_user_prompt_v3   (제약 디코딩 전용 — GRPO_ONLY 학습본)
#   ※ 서빙 adapter를 만든 노트북 설정과 반드시 같은 값이어야 한다.
PLANNER_PROMPT_VERSION = os.getenv("PLANNER_PROMPT_VERSION", "v3").lower()

# ── ★ [신규] 활성 action 선택 ────────────────────────────────
# registry(stage)가 정의한 action 중 실제로 쓸 것만 고른다. 정의를 새로 만드는
# 게 아니라 stage 집합의 부분집합을 취하는 것 — 단일 출처 원칙은 유지된다.
#   기본값에서 global_knn을 뺀 이유: round0의 superset이라 nd_after <= nd_before가
#   구조적으로 성립하고(실측 99.7%), Δmargin 부호가 state가 아니라 y로 결정된다.
#   실행 전 알 수 없는 정보라 학습 대상이 아니며 오라클 채택률도 0.0%였다.
#   노트북의 TRAIN_ACTIONS와 같은 값으로 두어야 train-serve가 일치한다.
_DEFAULT_ACTIONS = "stop,contextual_knn,attribute_knn"
PLANNER_ENABLED_ACTIONS: Tuple[str, ...] = tuple(
    a.strip() for a in os.getenv("PLANNER_ENABLED_ACTIONS", _DEFAULT_ACTIONS).split(",")
    if a.strip()
)

# ── ★ [신규] 제약 디코딩 ─────────────────────────────────────
# 학습(GRPO_ONLY)이 SFT 없이 성립한 전제. 끄면 포맷 붕괴 + train-serve
# mismatch가 동시에 발생한다. 진단 목적 외에는 끄지 말 것.
CONSTRAINED_DECODE = os.getenv("PLANNER_CONSTRAINED_DECODE", "true").lower() == "true"

# ── ★ [1-step 고정] ──────────────────────────────────────────
# 학습된 정책은 step1만 학습했다(노트북 [K3] oracle_df["step"]==1).
# 2-step으로 올리려면 프롬프트에 "이미 실행한 검색" 블록이 필요하고,
# 유효 출력 집합도 step1에서 쓴 action을 제외해야 한다 — 지금은 하드 가드.
# ※ controller의 PLANNER_MAX_STEPS와 같은 값이어야 한다.
PLANNER_MAX_STEPS = int(os.getenv("PLANNER_MAX_STEPS", "1"))

# ── ★ [chat template 정책] ───────────────────────────────────
# 노트북의 업로드 셀은 adapter 디렉터리에 adapter_config.json /
# adapter_model.safetensors / planner_manifest.json만 올린다(tokenizer 미포함).
# 그래서 ADAPTER_PATH의 tokenizer는 없거나 부분 파일이고, chat_template이
# 빠져 있으면 apply_chat_template이 매 요청 ValueError로 죽는다.
#
# 해결: chat_template에 의존하지 않고 ChatML 문자열을 직접 조립한다.
#   - Qwen2.5-Instruct의 chat_template 출력과 문자 단위로 동일하다
#     (학습 시 apply_chat_template이 만든 것과 같은 토큰열).
#   - gguf 백엔드가 이미 쓰던 방식이라 두 백엔드가 같은 프롬프트를 본다.
#   - local_analyzer.py도 같은 방식이다(그래서 이 문제를 안 겪었다).
# "true"로 두면 tokenizer.apply_chat_template을 쓴다(A/B 및 회귀 확인용).
PLANNER_USE_CHAT_TEMPLATE = os.getenv("PLANNER_USE_CHAT_TEMPLATE", "false").lower() == "true"

# ── nd 척도 (계산은 controller, 여기서는 선언/echo만) ─────────
PLANNER_ND_MODE = os.getenv("PLANNER_ND_MODE", "raw_top1").lower()

# ── 정책 게이트 ─────────────────────────────────────────────
PLANNER_AMBIGUOUS_ONLY  = os.getenv("PLANNER_AMBIGUOUS_ONLY", "true").lower() == "true"
PLANNER_FALLBACK_ACTION = os.getenv("PLANNER_FALLBACK_ACTION", "stop")
ENABLE_TTFT_PROBE       = os.getenv("ENABLE_TTFT_PROBE", "true").lower() == "true"

# ── 거리 정규화 상수 (analyzer와 동일 출처) ─────────────────
DIST_P25 = float(os.getenv("DIST_P25", "1.264"))
DIST_P50 = float(os.getenv("DIST_P50", "1.448"))
DIST_P75 = float(os.getenv("DIST_P75", "1.651"))
DIST_IQR = float(os.getenv("DIST_IQR", "0.387"))

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))


def _active_action_keys(stage: Optional[str] = None) -> Tuple[str, ...]:
    """stage registry ∩ PLANNER_ENABLED_ACTIONS.
    registry key는 'contextual_knn@cluster'처럼 scope 접미가 붙을 수 있으므로
    action 이름 기준으로 매칭한다."""
    out = []
    for k in planner_stage_actions(stage):
        spec = planner_action_spec(k)
        if spec and spec["action"] in PLANNER_ENABLED_ACTIONS:
            out.append(k)
    return tuple(out)


def _active_action_names(stage: Optional[str] = None) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(planner_action_spec(k)["action"]
                               for k in _active_action_keys(stage)))


# ===========================
# Metrics
# ===========================
planner_requests_total = Counter("planner_requests_total", "Planner requests", ["status"])
planner_action_total   = Counter("planner_action_total", "Planner action chosen", ["action", "source"])
planner_invalid_total  = Counter("planner_invalid_total", "Planner invalid outputs", ["reason"])
planner_latency        = Histogram("planner_latency_seconds", "Planner end-to-end latency",
                                   buckets=[0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0])
planner_ttft           = Histogram("planner_ttft_seconds", "Time to first token",
                                   buckets=[0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0])
planner_tpot           = Histogram("planner_tpot_ms", "Time per output token (ms)",
                                   buckets=[5, 10, 20, 40, 80, 160, 320])
planner_out_tokens     = Histogram("planner_output_tokens", "Output token count",
                                   buckets=[4, 8, 12, 16, 20, 24, 32])
planner_valid_outputs  = Histogram("planner_valid_output_count", "유효 출력 후보 수",
                                   buckets=[1, 3, 5, 9, 13, 21, 43])
planner_ready_gauge    = Gauge("planner_model_ready", "1 if model loaded")


# ===========================
# 모델 로드 (local_analyzer.py와 동일 패턴)
# ===========================
_model = None
_tokenizer = None
_MODEL_LOCK = threading.Lock()
_INFER_STAT = {"n": 0, "first_ms": None, "lock_wait_ms_sum": 0.0}
_PROMPT_OK = {"ok": False, "reason": "not_checked"}
# 노트북이 adapter와 함께 올린 학습 메타(planner_manifest.json)
_MANIFEST: Dict[str, Any] = {}


def load_model():
    global _model, _tokenizer
    logger.info("planner_loading_model", backend=MODEL_BACKEND, base=BASE_MODEL_ID,
                adapter=ADAPTER_PATH, device=DEVICE, stage=PLANNER_STAGE,
                actions=list(_active_action_names()), constrained=CONSTRAINED_DECODE)
    t0 = time.time()
    if MODEL_BACKEND == "gguf":
        _load_model_gguf(t0)
    else:
        _load_model_hf(t0)
    planner_ready_gauge.set(1)


def _load_model_gguf(t0: float):
    """llama-cpp-python으로 planner GGUF 로드.
    ★ Q4_K_M은 evaluator에서 format collapse(parse rate 0.00)가 실측됐다 —
      planner는 출력이 24토큰 JSON 한 줄이라 더 취약하다. Q8_0을 쓸 것.
      (제약 디코딩을 켜면 포맷은 지켜지지만, 양자화가 로짓을 흔들면
       action 선택 자체가 바뀐다 — 다른 층위의 문제라 여전히 유효한 경고)"""
    global _model
    from llama_cpp import Llama

    n_threads = int(os.getenv("OMP_NUM_THREADS", "2"))
    gguf_path = os.path.join(ADAPTER_PATH, GGUF_FILENAME)
    logger.info("planner_loading_gguf", path=gguf_path, n_ctx=N_CTX,
                n_threads=n_threads, quant=GGUF_QUANT)
    if GGUF_QUANT.lower().startswith("q4"):
        logger.warning("planner_gguf_quant_risky", quant=GGUF_QUANT,
                       note="Q4_K_M caused 0.00 parse rate on evaluator — use q8_0")

    _model = Llama(
        model_path=gguf_path,
        n_ctx=N_CTX,
        n_threads=n_threads,
        n_threads_batch=int(os.getenv("OMP_NUM_THREADS_BATCH", str(n_threads))),
        n_batch=int(os.getenv("LLAMA_N_BATCH", "512")),
        n_ubatch=int(os.getenv("LLAMA_N_UBATCH", "512")),
        use_mmap=True,
        use_mlock=os.getenv("LLAMA_MLOCK", "true").lower() == "true",
        logits_all=False,
        verbose=False,
    )
    logger.info("planner_gguf_loaded", elapsed_s=f"{time.time()-t0:.1f}", path=gguf_path)


def _load_model_hf(t0: float):
    """HF transformers + LoRA merge. 02b 노트북의 학습/평가 경로와 동일 디코딩."""
    global _model, _tokenizer
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    torch_dtype = torch.bfloat16 if TORCH_DTYPE_STR == "bfloat16" else torch.float32
    if DEVICE == "cpu":
        torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "2")))

    # ★ adapter 디렉터리에 tokenizer가 온전히 있는지 먼저 본다.
    #   노트북 업로드 셀은 adapter 가중치만 올리므로 대개 없거나 부분이다.
    _tok_files = [f for f in ("tokenizer_config.json", "tokenizer.json",
                              "vocab.json", "merges.txt", "chat_template.jinja")
                  if os.path.exists(os.path.join(ADAPTER_PATH, f))]
    try:
        _tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
        logger.info("planner_tokenizer_loaded", src=ADAPTER_PATH, files=_tok_files)
    except Exception as e:
        logger.warning("planner_tokenizer_from_adapter_failed", error=str(e),
                       files=_tok_files, fallback=BASE_MODEL_ID)
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    # ★ [FIX] chat_template 결손 복구.
    #   기본 경로(_render_chat)는 템플릿을 쓰지 않지만,
    #   PLANNER_USE_CHAT_TEMPLATE=true 회귀 확인과 startup parity probe에서
    #   필요하므로 여기서 복구해 둔다. LoRA는 vocab을 바꾸지 않으므로
    #   base 템플릿을 그대로 쓰는 것이 곧 학습 경로 재현이다.
    if not getattr(_tokenizer, "chat_template", None):
        try:
            _tokenizer.chat_template = AutoTokenizer.from_pretrained(
                BASE_MODEL_ID).chat_template
            logger.warning("planner_chat_template_restored_from_base",
                           base=BASE_MODEL_ID, adapter=ADAPTER_PATH,
                           note="adapter에 chat_template 없음 — base에서 복구")
        except Exception as _e:
            logger.warning("planner_chat_template_restore_failed", error=str(_e))
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    _tokenizer.truncation_side = "left"   # 학습(02b)과 동일

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
    if DEVICE != "cpu":
        base = base.to(DEVICE)
    merged = PeftModel.from_pretrained(base, ADAPTER_PATH).merge_and_unload()
    merged.eval()
    _model = merged
    logger.info("planner_hf_loaded", elapsed_s=f"{time.time()-t0:.1f}",
                dtype=str(torch_dtype),
                params_M=f"{sum(p.numel() for p in _model.parameters())/1e6:.0f}M")


# ============================================================
# 제약 디코딩 — 유효 출력 열거 → trie(hf) / GBNF(gguf)
# ============================================================
# ★ 유효 출력은 agent_tools.build_planner_action_space()가 만든다.
#   그 함수는 오라클 rollout이 실제로 전수 실행한 후보와 동일한 규칙
#   (candidate_feature_sets: agrees∪persona_match, ≤MAX_CONTEXTUAL_FEATURES)을
#   쓰므로, "생성 가능한 것 = 오라클이 reward를 아는 것"이 정의상 보장된다.
#   노트북의 build_valid_outputs()와 같은 집합을 만든다 — 다만 노트북은
#   REWARD_COMPONENTS(오프라인 테이블)에서, 여기서는 registry에서 유도한다.

def _canonical_json(action: str, feats: Tuple[str, ...]) -> str:
    """토큰 단위로 유일해야 하므로 공백 없는 canonical 형태로 고정.
    학습 때와 같은 직렬화여야 trie/GBNF가 같은 문자열을 받는다."""
    if action == "stop" or not feats:
        return json.dumps({"action": action}, separators=(",", ":"))
    return json.dumps({"action": action, "features": list(feats)},
                      separators=(",", ":"))


def build_valid_outputs(candidate_pool: List[str]) -> List[str]:
    """이 state에서 생성 가능한 출력 문자열 전체."""
    outs: List[str] = []
    for a in build_planner_action_space(candidate_pool):
        if a["action"] not in PLANNER_ENABLED_ACTIONS:
            continue
        if a.get("scope") == "cluster":     # stage3 variant — 별도 처리 필요
            continue
        outs.append(_canonical_json(a["action"], tuple(a.get("features") or ())))
    _stop = _canonical_json("stop", ())
    if "stop" in PLANNER_ENABLED_ACTIONS and _stop not in outs:
        outs.append(_stop)
    return sorted(set(outs))


class _TokenTrie:
    """유효 문자열들의 토큰 시퀀스 trie. 생성된 prefix에 이어질 수 있는
    토큰만 허용하고, 끝에서는 EOS만 허용한다."""
    __slots__ = ("root", "eos")

    def __init__(self, token_seqs: List[List[int]], eos_id: int):
        self.eos = eos_id
        self.root: Dict = {}
        for seq in token_seqs:
            node = self.root
            for t in seq:
                node = node.setdefault(t, {})
            node[eos_id] = {}

    def allowed(self, generated: List[int]) -> List[int]:
        node = self.root
        for t in generated:
            if t not in node:
                return [self.eos]
            node = node[t]
        return list(node.keys()) or [self.eos]


def build_gbnf(valid_outputs: List[str]) -> str:
    """열거된 유효 출력을 그대로 GBNF alternation으로 만든다.
    문법을 손으로 설계하지 않으므로 trie와 같은 언어를 받는다.
    ※ 다만 GBNF는 문자열 단위, trie는 토큰 단위라 GBNF 쪽이 더 느슨하다
      (같은 문자열의 다른 토큰화도 허용). canonical JSON(공백 제거)이라
      대안 토큰화 여지는 작지만 0은 아니다 — /config의 backend와 함께
      해석할 것."""
    alts = " | ".join('"' + o.replace('\\', '\\\\').replace('"', '\\"') + '"'
                      for o in valid_outputs)
    return f"root ::= {alts}\n"


_GRAMMAR_CACHE: Dict[str, Any] = {}


def _get_gguf_grammar(valid_outputs: List[str]):
    """llama_cpp.LlamaGrammar 캐시 (pool이 같으면 문법도 같다)."""
    key = "\x00".join(valid_outputs)
    if key in _GRAMMAR_CACHE:
        return _GRAMMAR_CACHE[key]
    from llama_cpp import LlamaGrammar
    g = LlamaGrammar.from_string(build_gbnf(valid_outputs), verbose=False)
    if len(_GRAMMAR_CACHE) > 512:
        _GRAMMAR_CACHE.clear()
    _GRAMMAR_CACHE[key] = g
    return g


# ===========================
# Prompt — prompts.py의 planner 네임스페이스만 사용
# ===========================
_PROMPT_BUILDERS = {
    "v1": "build_planner_user_prompt",
    "v2": "build_planner_user_prompt_v2",
    "v3": "build_planner_user_prompt_v3",
}


def _check_prompt_module() -> None:
    """startup에서 1회. 없으면 /ready를 503으로 유지한다 — 프롬프트가 학습과
    다르면 조용히 틀린 action을 내는 게 최악이므로 인라인 degrade하지 않는다."""
    fn_name = _PROMPT_BUILDERS.get(PLANNER_PROMPT_VERSION)
    if fn_name is None:
        _PROMPT_OK.update(ok=False,
                          reason=f"unknown PLANNER_PROMPT_VERSION={PLANNER_PROMPT_VERSION}")
        return
    if not hasattr(prompts, fn_name) or not hasattr(prompts, "PLANNER_SYSTEM_PROMPT"):
        _PROMPT_OK.update(ok=False,
                          reason=f"prompts.py missing {fn_name}/PLANNER_SYSTEM_PROMPT")
        return
    _PROMPT_OK.update(ok=True, reason="ok")


def _build_user_prompt(*, test_features: Dict, cluster_id: int, distances: List[float],
                       persona: Optional[Dict], topk_feats: List[Dict],
                       candidate_features: List[str]) -> str:
    fn = getattr(prompts, _PROMPT_BUILDERS[PLANNER_PROMPT_VERSION])
    kw: Dict[str, Any] = {}
    if PLANNER_PROMPT_VERSION == "v3":
        # ★ v3는 ACTIONS 블록을 호출자가 주입한다 — cost/action 목록이
        #   registry 단일 출처를 따르도록(프롬프트가 옛 cost를 말하는 사고 방지)
        _acts = list(_active_action_names())
        kw = dict(actions=_acts,
                  action_costs={a: float(ACTION_COST.get(a, 1.0)) for a in _acts},
                  max_features=MAX_CONTEXTUAL_FEATURES)
    return fn(test_features=test_features, cluster_id=cluster_id, distances=distances,
              persona=persona, topk_feats=topk_feats,
              candidate_features=candidate_features, **kw)


# ===========================
# 추론
# ===========================
def _render_chat(system: str, user: str) -> str:
    """★ ChatML 프롬프트 문자열 — hf/gguf 두 백엔드의 단일 출처.

    Qwen2.5-Instruct의 chat_template이 생성하는 문자열과 동일하다:
        <|im_start|>system\n{sys}<|im_end|>\n
        <|im_start|>user\n{user}<|im_end|>\n
        <|im_start|>assistant\n
    system을 항상 명시하므로 템플릿의 기본 system 삽입 분기도 타지 않는다.
    → adapter에 tokenizer/chat_template이 없어도 학습 프롬프트를 재현한다.
    """
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def _extract_json(text: str) -> str:
    t = text.strip()
    if "```json" in t:
        t = t.split("```json")[-1].split("```")[0].strip()
    elif t.startswith("```"):
        parts = t.split("```")
        t = parts[1].strip() if len(parts) > 1 else t
    m = re.search(r"\{.*?\}", t, re.DOTALL)
    return m.group(0) if m else t


def run_inference(user_prompt: str, valid_outputs: List[str],
                  meta: Optional[Dict] = None) -> str:
    """raw 생성 텍스트를 반환(파싱은 호출자). analyzer와 동일하게 _MODEL_LOCK 직렬화."""
    system = prompts.PLANNER_SYSTEM_PROMPT
    _t_q = time.perf_counter()
    with _MODEL_LOCK:
        _t_i = time.perf_counter()
        wait_ms = (_t_i - _t_q) * 1000.0

        if MODEL_BACKEND == "gguf":
            # ★ hf 경로와 같은 렌더러 — 두 백엔드가 동일 프롬프트를 본다.
            prompt = _render_chat(system, user_prompt)
            _kw: Dict[str, Any] = {}
            if CONSTRAINED_DECODE and valid_outputs:
                _kw["grammar"] = _get_gguf_grammar(valid_outputs)   # ★ GBNF
            _ttft, _n_out = None, -1
            if ENABLE_TTFT_PROBE:
                chunks = []
                for ch in _model(prompt, max_tokens=MAX_NEW_TOKENS, temperature=0.0,
                                 repeat_penalty=1.0, top_k=1, top_p=1.0,
                                 stop=["<|im_end|>"], stream=True, **_kw):
                    if _ttft is None:
                        _ttft = time.perf_counter() - _t_i
                    chunks.append(ch["choices"][0]["text"])
                text = "".join(chunks).strip()
                _n_out = len(chunks)
            else:
                out = _model(prompt, max_tokens=MAX_NEW_TOKENS, temperature=0.0,
                             repeat_penalty=1.0, top_k=1, top_p=1.0,
                             stop=["<|im_end|>"], **_kw)
                text = out["choices"][0]["text"].strip()
                _n_out = int((out.get("usage") or {}).get("completion_tokens", -1))
            _infer_ms = (time.perf_counter() - _t_i) * 1000.0
            if _ttft is not None and _n_out > 0:
                _tpot = (_infer_ms - _ttft * 1000.0) / max(_n_out - 1, 1)
                planner_ttft.observe(_ttft)
                planner_tpot.observe(_tpot)
                planner_out_tokens.observe(_n_out)
                if meta is not None:
                    meta.update({"ttft_ms": round(_ttft * 1000, 2),
                                 "tpot_ms": round(_tpot, 2),
                                 "n_output_tokens": _n_out})
        else:
            import torch
            # ★ 학습/eval(02b evaluate_planner)과 동일한 프롬프트 + greedy +
            #   prefix_allowed_tokens_fn(trie).
            #   apply_chat_template 대신 _render_chat()으로 조립한다 —
            #   adapter에 chat_template이 없어도 같은 토큰열이 나온다.
            if PLANNER_USE_CHAT_TEMPLATE:
                chat = [{"role": "system", "content": system},
                        {"role": "user", "content": user_prompt}]
                inputs = _tokenizer.apply_chat_template(
                    chat, add_generation_prompt=True,
                    return_tensors="pt", return_dict=True).to(_model.device)
            else:
                inputs = _tokenizer(_render_chat(system, user_prompt),
                                    return_tensors="pt",
                                    add_special_tokens=False).to(_model.device)
            _plen = inputs["input_ids"].shape[1]

            _gkw: Dict[str, Any] = {}
            if CONSTRAINED_DECODE and valid_outputs:
                _seqs = [_tokenizer(o, add_special_tokens=False)["input_ids"]
                         for o in valid_outputs]
                _trie = _TokenTrie(_seqs, _tokenizer.eos_token_id)
                _gkw["prefix_allowed_tokens_fn"] = (
                    lambda _b, ids: _trie.allowed(ids.tolist()[_plen:]))

            with torch.no_grad():
                out = _model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                      do_sample=False,
                                      pad_token_id=_tokenizer.eos_token_id, **_gkw)
            text = _tokenizer.decode(out[0][_plen:], skip_special_tokens=True).strip()
            _infer_ms = (time.perf_counter() - _t_i) * 1000.0
            _n_out = int(out.shape[1] - _plen)
            planner_out_tokens.observe(_n_out)
            if meta is not None:
                meta["n_output_tokens"] = _n_out

    _INFER_STAT["n"] += 1
    _INFER_STAT["lock_wait_ms_sum"] += wait_ms
    if _INFER_STAT["first_ms"] is None:
        _INFER_STAT["first_ms"] = _infer_ms
    logger.info("planner_infer_timing", backend=MODEL_BACKEND, call_n=_INFER_STAT["n"],
                infer_ms=round(_infer_ms, 1), lock_wait_ms=round(wait_ms, 1),
                out_tokens=_n_out, constrained=CONSTRAINED_DECODE,
                n_valid=len(valid_outputs), raw_preview=text[:120])
    return text


# ===========================
# Pydantic Models
# ===========================
class PlanActionRequest(BaseModel):
    """
    controller의 round0 state를 그대로 받는다. 오라클 수집(/rollout)이
    RolloutResponse에 실어보낸 필드와 같은 이름/의미를 쓰므로, 학습 프롬프트
    재구성(02b build_planner_prompt_for_test_idx)과 1:1 대응된다.
    """
    request_id:   Optional[str] = None
    test_idx:     Optional[int] = None
    transaction:  Dict[str, Any] = Field(default_factory=dict)
    cluster_id:   int = -1
    persona:      Optional[Dict[str, Any]] = None
    neighbors:    List[Dict[str, Any]] = Field(default_factory=list)
    # 후보 pool — 미제공 시 agrees∪persona_match로 자체 구성
    candidate_features: Optional[List[str]] = None
    agrees:        List[str] = Field(default_factory=list)
    persona_match: List[str] = Field(default_factory=list)
    # 게이트 신호
    nd_before:    Optional[float] = None
    top1_distance: Optional[float] = None
    wm_type:      Optional[str] = None
    sufficiency:  Optional[str] = None
    round:        int = 0
    expansion_used: bool = False


class PlanActionResponse(BaseModel):
    request_id:   Optional[str] = None
    action:       str = "stop"
    features:     List[str] = Field(default_factory=list)
    scope:        Optional[str] = None
    cost:         float = 0.0
    tool_name:    Optional[str] = None
    # controller가 tool 호출에 그대로 쓰는 payload (registry가 만든 것)
    tool_payload: Optional[Dict[str, Any]] = None
    # 진단/계측
    source:       str = "model"     # "model" | "gate" | "fallback"
    is_fallback:  bool = False
    invalid_reason: Optional[str] = None
    raw_text:     Optional[str] = None
    n_valid_outputs: int = 0
    candidate_pool:  List[str] = Field(default_factory=list)
    # config echo — controller가 자기 설정과 비교해 불일치를 경고한다
    stage:            str = PLANNER_STAGE
    enabled_actions:  List[str] = Field(default_factory=list)
    prompt_version:   str = PLANNER_PROMPT_VERSION
    backend:          str = MODEL_BACKEND
    constrained:      bool = CONSTRAINED_DECODE
    nd_mode:          str = PLANNER_ND_MODE
    max_steps:        int = PLANNER_MAX_STEPS
    latency_ms:       float = 0.0
    ttft_ms:          Optional[float] = None
    tpot_ms:          Optional[float] = None
    n_output_tokens:  Optional[int] = None


# ===========================
# FastAPI
# ===========================
app = FastAPI(
    title="Local Planner (Qwen2.5-0.5B GRPO, constrained decoding)",
    version="2.0.0",
    description="Retrieval action planner — stop / contextual_knn / attribute_knn",
)


@app.on_event("startup")
async def startup():
    load_model()
    # analyzer와 동일한 런타임 바인딩 — planner 프롬프트의 z-score/[RARE] 태그가
    # 같은 임계로 계산돼야 학습 프롬프트와 문자열이 일치한다.
    if hasattr(prompts, "bind_runtime"):
        prompts.bind_runtime(DIST_P50=DIST_P50, DIST_IQR=DIST_IQR,
                             NORM_CLOSE_THRESHOLD=DIST_P25,
                             NORM_MODERATE_THRESHOLD=DIST_P75)
    for _k, _dflt in (("Z_TAU", "1.5"), ("NI_SPREAD_TAU", "0.05"),
                      ("W_CAT", "0.50"), ("P_MISS", "0.10")):
        if hasattr(prompts, _k):
            setattr(prompts, _k, float(os.getenv(_k, _dflt)))
    _check_prompt_module()

    # ★ [parity probe] _render_chat()이 tokenizer.chat_template과 같은 문자열을
    #   내는지 기동 시 1회 확인한다. 다르면 학습(apply_chat_template)과 서빙의
    #   프롬프트가 갈라진 것이므로 조용히 넘기면 안 된다.
    if MODEL_BACKEND != "gguf":
        _sys, _usr = "SYS_PROBE", "USER_PROBE"
        _manual = _render_chat(_sys, _usr)
        _tmpl = None
        try:
            _tmpl = _tokenizer.apply_chat_template(
                [{"role": "system", "content": _sys},
                 {"role": "user", "content": _usr}],
                add_generation_prompt=True, tokenize=False)
        except Exception as _e:
            logger.warning("planner_chat_template_probe_skipped", error=str(_e),
                           note="템플릿 없음 — _render_chat으로만 동작(정상)")
        if _tmpl is not None and _tmpl != _manual:
            logger.error("planner_prompt_parity_mismatch",
                         manual_tail=_manual[-60:], template_tail=_tmpl[-60:],
                         note="_render_chat과 chat_template 출력이 다르다 — "
                              "base 모델이 Qwen2.5 계열이 아니거나 템플릿이 바뀜")
        elif _tmpl is not None:
            logger.info("planner_prompt_parity_ok")

    # ★ [manifest 검증] 노트북이 adapter와 함께 올린 planner_manifest.json과
    #   서빙 config를 대조한다. train_actions / tau / lambda / kernel이 어긋나면
    #   "다른 정책을 학습한 adapter를 다른 설정으로 서빙"하는 것이라 지표가
    #   무의미해진다 — 실패시키지는 않되 반드시 로그로 남긴다.
    _mf_path = os.path.join(ADAPTER_PATH, "planner_manifest.json")
    if os.path.exists(_mf_path):
        try:
            with open(_mf_path) as _f:
                _MANIFEST.update(json.load(_f))
            _tr = set(_MANIFEST.get("train_actions") or [])
            _srv = set(_active_action_names())
            if _tr and _tr != _srv:
                logger.error("planner_action_space_mismatch",
                             trained=sorted(_tr), serving=sorted(_srv),
                             note="PLANNER_ENABLED_ACTIONS를 학습값에 맞출 것")
            if _MANIFEST.get("stage") and _MANIFEST["stage"] != PLANNER_STAGE:
                logger.error("planner_manifest_stage_mismatch",
                             trained=_MANIFEST["stage"], serving=PLANNER_STAGE)
            logger.info("planner_manifest_loaded",
                        kernel=_MANIFEST.get("kernel"), lam=_MANIFEST.get("lam"),
                        tau=_MANIFEST.get("tau"), run_tag=_MANIFEST.get("run_tag"),
                        train_actions=_MANIFEST.get("train_actions"))
        except Exception as _e:
            logger.warning("planner_manifest_read_failed", error=str(_e))
    else:
        logger.warning("planner_manifest_missing", path=_mf_path,
                       note="학습 설정 대조 불가 — RUN_TAG/ADAPTER_TYPE 확인")

    _unknown = [a for a in PLANNER_ENABLED_ACTIONS
                if a not in planner_action_names(PLANNER_STAGE)]
    if _unknown:
        logger.error("planner_enabled_actions_not_in_stage",
                     unknown=_unknown, stage=PLANNER_STAGE,
                     stage_actions=list(planner_action_names(PLANNER_STAGE)),
                     note="PLANNER_STAGE를 올리거나 PLANNER_ENABLED_ACTIONS에서 뺄 것")
    logger.info("planner_startup_done", stage=PLANNER_STAGE,
                max_steps=PLANNER_MAX_STEPS, pool_policy="evidence_only",
                enabled=list(_active_action_names()),
                prompt_version=PLANNER_PROMPT_VERSION, prompt_ok=_PROMPT_OK["ok"],
                backend=MODEL_BACKEND, constrained=CONSTRAINED_DECODE,
                nd_mode=PLANNER_ND_MODE, max_features=MAX_CONTEXTUAL_FEATURES)


def _derive_prompt_inputs(req: PlanActionRequest):
    """02b build_planner_prompt_for_test_idx와 동일한 재료 추출.

    ★ [candidate pool 정책] pool은 **local_analyzer의 evidence에서만** 나온다:
        agrees ∪ persona_match  (controller가 candidate_features로 실어보냄)
      비면 categorical 전체로 임의 확장하지 **않는다** — 그러면 "evaluator가
      근거를 못 찾았는데 아무 축으로나 검색"이 되어 planner 결정의 근거가
      사라지고, 오라클 pool과도 갈라진다. 대신 stop으로 종료한다(게이트 4).
      controller.PLANNER_POOL_FALLBACK이 true인 경우에만 controller가
      확장된 pool을 candidate_features로 실어보내므로, 정책 스위치는
      controller 한 곳에만 있다.
    """
    nbs = req.neighbors or []
    distances = [float(n["distance"]) for n in nbs
                 if isinstance(n, dict) and n.get("distance") is not None]
    topk_feats = [(n.get("features") or {}) if isinstance(n, dict) else {} for n in nbs]
    if req.candidate_features is not None:
        # controller가 계산해 보낸 값이 1순위 — 계산 주체를 하나로 모은다
        pool = list(dict.fromkeys(req.candidate_features))
    else:
        # 단독 호출(스모크 테스트 등) 대비 자체 유도
        pool = list(dict.fromkeys(list(req.agrees) + list(req.persona_match)))
    # ★ 프롬프트가 제안 가능한 feature와 tool이 실행 가능한 feature를 일치시킨다.
    #   (연속형 축은 term boost 대상이 아니라 실행 단계에서 조용히 drop된다)
    pool = [f for f in pool if f in CONTEXTUAL_CANDIDATE_FIELDS]
    return distances, topk_feats, pool


def _gate_stop(reason: str, req: PlanActionRequest, t0: float,
               pool: Optional[List[str]] = None) -> PlanActionResponse:
    planner_action_total.labels(action="stop", source="gate").inc()
    planner_requests_total.labels(status="gate").inc()
    return PlanActionResponse(
        request_id=req.request_id, action="stop", source="gate",
        invalid_reason=reason, cost=float(ACTION_COST.get("stop", 0.0)),
        candidate_pool=pool or [], enabled_actions=list(_active_action_names()),
        latency_ms=(time.time() - t0) * 1000,
    )


@app.post("/plan-action", response_model=PlanActionResponse)
async def plan_action(req: PlanActionRequest):
    import asyncio
    t0 = time.time()

    # ── 게이트 1: 학습 모집단(ambiguous-only) 밖이면 LLM 미호출 ──
    if PLANNER_AMBIGUOUS_ONLY:
        if req.sufficiency == "confident" or req.wm_type in (None, ""):
            return _gate_stop("out_of_training_population(confident/no_wm_type)", req, t0)
    # ── 게이트 2: 이미 재검색했으면 루프 가드 ──
    if req.expansion_used or req.round >= PLANNER_MAX_STEPS:
        return _gate_stop(
            f"max_steps_reached(round={req.round}, max={PLANNER_MAX_STEPS})", req, t0)
    # ── 게이트 3: 프롬프트 모듈 결손 ──
    if not _PROMPT_OK["ok"]:
        logger.error("planner_prompt_unavailable", reason=_PROMPT_OK["reason"])
        return _gate_stop(f"prompt_unavailable:{_PROMPT_OK['reason']}", req, t0)

    distances, topk_feats, pool = _derive_prompt_inputs(req)

    # ── 게이트 4: 후보 pool이 비면 검색 tool이 실행 불가 → stop 확정 ──
    valid_outputs = build_valid_outputs(pool)
    planner_valid_outputs.observe(len(valid_outputs))
    if len(valid_outputs) <= 1:
        # evidence(agrees ∪ persona_match)가 비었거나 실행 가능한 categorical
        # 축이 없다 → 검색할 근거가 없으므로 LLM 호출 없이 stop.
        logger.info("planner_empty_pool_stop",
                    request_id=req.request_id, wm_type=req.wm_type,
                    agrees=req.agrees, persona_match=req.persona_match,
                    candidate_features=req.candidate_features)
        return _gate_stop("empty_candidate_pool(evidence-only policy)", req, t0, pool)

    user_prompt = _build_user_prompt(
        test_features=req.transaction, cluster_id=req.cluster_id, distances=distances,
        persona=req.persona, topk_feats=topk_feats, candidate_features=pool,
    )

    meta: Dict[str, Any] = {}
    try:
        raw_text = await asyncio.to_thread(run_inference, user_prompt, valid_outputs, meta)
    except Exception as e:
        logger.error("planner_inference_failed", error_type=type(e).__name__, error=str(e))
        planner_invalid_total.labels(reason="inference_error").inc()
        planner_requests_total.labels(status="error").inc()
        return PlanActionResponse(
            request_id=req.request_id, action=PLANNER_FALLBACK_ACTION,
            source="fallback", is_fallback=True,
            invalid_reason=f"inference_error:{type(e).__name__}",
            candidate_pool=pool, n_valid_outputs=len(valid_outputs),
            enabled_actions=list(_active_action_names()),
            latency_ms=(time.time() - t0) * 1000)

    # ── 파싱 + stage registry 검증 ──
    action, feats, scope, invalid = None, [], None, None
    try:
        d = json.loads(_extract_json(raw_text))
        action = d.get("action") if isinstance(d.get("action"), str) else None
        raw_feats = d.get("features", [])
        feats = ([str(f) for f in raw_feats if isinstance(f, (str, int, float))]
                 if isinstance(raw_feats, list) else [])
        scope = d.get("scope") if isinstance(d.get("scope"), str) else None
        if action is None:
            invalid = "action_not_string"
    except Exception as e:
        invalid = f"json_parse_failed:{type(e).__name__}"

    resolved = None
    if invalid is None:
        if action not in _active_action_names():
            invalid = f"action_disabled:{action}"
        else:
            resolved = resolve_planner_action(action, features=feats, scope=scope,
                                              candidate_pool=pool)
            if not resolved["valid"]:
                invalid = resolved["reason"]

    if invalid is not None:
        # ★ 제약 디코딩이 켜져 있으면 여기 오는 것 자체가 이상 신호다
        #   (grammar/trie가 유효 집합 밖 생성을 막았어야 한다)
        _lvl = logger.error if CONSTRAINED_DECODE else logger.warning
        _lvl("planner_invalid_output", reason=invalid, raw=raw_text[:160],
             constrained=CONSTRAINED_DECODE, n_valid=len(valid_outputs),
             note=("제약 디코딩이 켜졌는데 무효 출력 — grammar/trie 미적용 의심"
                   if CONSTRAINED_DECODE else None))
        planner_invalid_total.labels(reason=invalid.split(":")[0]).inc()
        planner_action_total.labels(action=PLANNER_FALLBACK_ACTION, source="fallback").inc()
        planner_requests_total.labels(status="invalid").inc()
        planner_latency.observe(time.time() - t0)
        return PlanActionResponse(
            request_id=req.request_id, action=PLANNER_FALLBACK_ACTION,
            source="fallback", is_fallback=True, invalid_reason=invalid,
            raw_text=raw_text[:200], candidate_pool=pool,
            n_valid_outputs=len(valid_outputs),
            enabled_actions=list(_active_action_names()),
            latency_ms=(time.time() - t0) * 1000, **meta)

    spec = resolved["spec"]
    # ★ controller가 그대로 tool에 보낼 payload — registry가 만든다.
    #   embedding/vec_index는 controller만 아는 값이라 여기서는 비운다.
    payload = planner_tool_payload(
        spec, features=resolved["features"], transaction=req.transaction,
        cluster_id=req.cluster_id) if spec["tool_name"] else None

    planner_action_total.labels(action=spec["action"], source="model").inc()
    planner_requests_total.labels(status="success").inc()
    planner_latency.observe(time.time() - t0)
    logger.info("planner_decision", action=spec["action"], features=resolved["features"],
                scope=spec["scope"], cost=spec["cost"], stage=PLANNER_STAGE,
                nd_before=req.nd_before, wm_type=req.wm_type,
                n_valid=len(valid_outputs))
    return PlanActionResponse(
        request_id=req.request_id, action=spec["action"], features=resolved["features"],
        scope=spec["scope"], cost=float(spec["cost"]), tool_name=spec["tool_name"],
        tool_payload=payload, source="model", raw_text=raw_text[:200],
        candidate_pool=pool, n_valid_outputs=len(valid_outputs),
        enabled_actions=list(_active_action_names()),
        latency_ms=(time.time() - t0) * 1000, **meta)


@app.post("/debug/valid-outputs")
async def debug_valid_outputs(req: PlanActionRequest):
    """서빙 중인 유효 출력 집합을 그대로 노출한다.
    노트북 build_valid_outputs()와 같은 집합인지 눈으로 대조하기 위한 것 —
    train-serve action space 불일치가 가장 잡기 어려운 사고이므로."""
    _, _, pool = _derive_prompt_inputs(req)
    outs = build_valid_outputs(pool)
    return {"candidate_pool": pool, "n_valid": len(outs), "valid_outputs": outs,
            "gbnf": build_gbnf(outs) if outs else "",
            "enabled_actions": list(_active_action_names()),
            "stage": PLANNER_STAGE, "max_features": MAX_CONTEXTUAL_FEATURES}


@app.get("/health")
async def health():
    return {"status": "healthy" if _model is not None else "loading",
            "model_loaded": _model is not None, "backend": MODEL_BACKEND}


@app.get("/ready")
async def ready():
    ok = _model is not None and _PROMPT_OK["ok"] and bool(_active_action_names())
    body = {"ready": ok, "model_loaded": _model is not None,
            "prompt_ok": _PROMPT_OK["ok"], "prompt_reason": _PROMPT_OK["reason"],
            "stage": PLANNER_STAGE, "prompt_version": PLANNER_PROMPT_VERSION,
            "enabled_actions": list(_active_action_names()),
            "constrained": CONSTRAINED_DECODE}
    return Response(content=json.dumps(body), status_code=200 if ok else 503,
                    media_type="application/json")


@app.get("/config")
async def config():
    """controller/harness가 서빙 설정을 그대로 읽어 실험 메타에 기록한다."""
    return {
        "experiment_case": EXPERIMENT_CASE,
        "stage": PLANNER_STAGE,
        "stage_actions": list(planner_stage_actions()),
        "enabled_actions": list(_active_action_names()),
        "action_cost": {a: float(ACTION_COST.get(a, 1.0))
                        for a in _active_action_names()},
        "max_contextual_features": MAX_CONTEXTUAL_FEATURES,
        "candidate_fields": list(CONTEXTUAL_CANDIDATE_FIELDS),
        "action_catalog": build_planner_tool_catalog(),
        "prompt_version": PLANNER_PROMPT_VERSION,
        "constrained_decode": CONSTRAINED_DECODE,
        "constraint_mechanism": ("gbnf" if MODEL_BACKEND == "gguf" else "trie"),
        "model_backend": MODEL_BACKEND,
        "gguf_quant": GGUF_QUANT if MODEL_BACKEND == "gguf" else None,
        "base_model_id": BASE_MODEL_ID,
        "adapter_path": ADAPTER_PATH,
        "max_new_tokens": MAX_NEW_TOKENS,
        "nd_mode": PLANNER_ND_MODE,
        "max_steps": PLANNER_MAX_STEPS,
        "use_chat_template": PLANNER_USE_CHAT_TEMPLATE,
        "prompt_render": ("chat_template" if PLANNER_USE_CHAT_TEMPLATE
                          else "manual_chatml"),
        "manifest": _MANIFEST or None,
        "pool_policy": "evidence_only(agrees ∪ persona_match); empty -> stop",
        "ambiguous_only": PLANNER_AMBIGUOUS_ONLY,
        "fallback_action": PLANNER_FALLBACK_ACTION,
        "n_inferences": _INFER_STAT["n"],
    }


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)