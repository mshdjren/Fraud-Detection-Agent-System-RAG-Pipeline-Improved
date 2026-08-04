#!/usr/bin/env bash
# =====================================================================
# lib.sh — 직렬 실행 안전장치
# =====================================================================
# GKE free tier 에서 local-analyzer 는 동시 요청을 받으면 OOM/probe 실패로
# rollout 된다. 그러면 그 run 의 latency·AUROC 가 통째로 오염되는데,
# 로그만 봐서는 알아채기 어렵다. 그래서:
#
#   1. run 시작 전  : rollout 완료 + /health 200 + warmup 요청까지 확인
#   2. run 시작 전  : 감시 대상 deployment 의 RESTARTS 합계를 스냅샷
#   3. run 종료 후  : RESTARTS 재측정 → 증가했으면 그 run 을 suspect 로 표시
#   4. run 사이     : cooldown (pod 안정화 + page cache 정리)
#
# 매트릭스 전체를 중단시키지 않고 개별 run 만 suspect 로 마킹한다 —
# 재실행 대상을 나중에 plan 에서 골라낼 수 있어야 하기 때문.
# =====================================================================

set -uo pipefail

# ── Windows(Git Bash) 호환 ───────────────────────────────────────────
#  ★ MSYS_NO_PATHCONV 를 전역으로 켜면 안 된다.
#    "/c/Users/.../patch.json" 같은 로컬 경로가 변환되지 않은 채
#    python.exe / kubectl.exe 로 넘어가 전부 FileNotFound 가 된다.
#    (kubectl exec -- sh -c "curl .../health" 는 인자가 '/' 로 시작하지
#     않아 애초에 MSYS 변환 대상이 아니다 — 끌 이유가 없었다.)
#  · python3 가 없는 환경이 흔하므로 PY 로 통일 (PY=python 으로 override)
#  ★ PYTHONUTF8=1 : Windows Python 은 open() 기본 인코딩이 cp949 라
#    UTF-8 로 쓰인 manifest.json/patch.json 을 읽다 UnicodeDecodeError 가 난다.
#    UTF-8 모드(3.7+)를 켜서 open()/stdout 을 모두 UTF-8 로 고정한다.
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

#  ★ CLOUDSDK_PYTHON : gsutil/gcloud 가 PATH 의 Python 3.9 를 집으면
#    번들 urllib3 가 "TypeError: unsupported operand type(s) for |"
#    (bytes | str 는 3.10+ 문법) 로 죽는다. Cloud SDK 동봉 python 을 쓰게 한다.
if [[ -z "${CLOUDSDK_PYTHON:-}" ]] && command -v gcloud >/dev/null 2>&1; then
  _sdk_bin="$(dirname "$(command -v gcloud)")"
  for _c in "$_sdk_bin/../platform/bundledpython/python.exe" \
            "$_sdk_bin/../platform/bundledpython/python" \
            "$_sdk_bin/../platform/bundledpythonunix/bin/python3"; do
    if [[ -x "$_c" ]]; then export CLOUDSDK_PYTHON="$_c"; break; fi
  done
fi

PY="${PY:-}"
if [[ -z "$PY" ]]; then
  if   command -v python3 >/dev/null 2>&1 && python3 -c "import yaml" 2>/dev/null; then PY=python3
  elif command -v python  >/dev/null 2>&1; then PY=python
  else PY=python3; fi
fi
export PY

NS="${NAMESPACE:-elastic}"

log()  { printf "\033[1;36m[%s]\033[0m %s\n" "$(date +%H:%M:%S)" "$*"; }
warn() { printf "\033[1;33m[%s] ⚠ %s\033[0m\n" "$(date +%H:%M:%S)" "$*"; }
err()  { printf "\033[1;31m[%s] ✗ %s\033[0m\n" "$(date +%H:%M:%S)" "$*"; }
ok()   { printf "\033[1;32m[%s] ✓ %s\033[0m\n" "$(date +%H:%M:%S)" "$*"; }

# ── ConfigMap patch ──────────────────────────────────────────────────
patch_config() {   # $1=configmap  $2=patch json path
  local cm="$1" f="$2" out
  [[ -f "$f" ]] || { err "patch 파일 없음: $f"; return 1; }
  # ★ 종료코드를 반드시 확인한다. 여기서 실패를 삼키면 이전 run 의 config
  #   그대로 실험이 돌아 조용히 오염된 결과가 나온다.
  if ! out=$(kubectl patch configmap "$cm" -n "$NS" --type merge \
               --patch-file "$f" 2>&1); then
    err "configmap/$cm patch 실패: $out"; return 1
  fi
  # 반영 검증 — patch 의 첫 키가 실제로 들어갔는지 되읽는다
  local k v got
  k=$("$PY" -c "import io,json,sys;d=json.load(io.open(sys.argv[1],encoding='utf-8'))['data'];print(next(iter(d)))" "$f" 2>/dev/null || true)
  if [[ -n "$k" ]]; then
    v=$("$PY" -c "import io,json,sys;print(json.load(io.open(sys.argv[1],encoding='utf-8'))['data'][sys.argv[2]])" "$f" "$k" 2>/dev/null || true)
    got=$(kubectl get configmap "$cm" -n "$NS" -o jsonpath="{.data.$k}" 2>/dev/null || true)
    if [[ "$got" != "$v" ]]; then
      err "configmap/$cm 반영 검증 실패: $k='$got' (기대 '$v')"; return 1
    fi
  fi
  log "patched configmap/$cm ← $(basename "$f")  [$k=$v]"
}

# ── [STEP1-A] 선택적 재시작 ──────────────────────────────────────────
# 매 run 마다 5개 deployment 를 전부 재시작하면 run 당 7분이 대기다.
# 실제로 바뀐 config 키만 보고 영향받는 deployment 만 재시작한다.
# tier1/2 에서 analyzer 관련 키는 한 번도 바뀌지 않으므로 3분이 통째로 절약된다.
#
# 키 → deployment 매핑. 한 키가 여러 서비스에 영향을 줄 수 있다.
_key_targets() {   # $1=config key  → 공백구분 deployment 목록
  case "$1" in
    # 검색 경로
    EXPERIMENT_CASE|N_CLUSTERS|GCS_TREE_BASE)
        echo "anomaly-router anomaly-retriever anomaly-controller anomaly-orchestrator local-analyzer" ;;
    PERCOLATE_QUERY_VERSION|ENABLE_BUCKETIZATION|BUCKET_DEPTH_PCT|BUCKET_TOLERANCE|\
    BUCKET_MIN_MATCH|ROUTER_CANDIDATE_MODE|CORESET_HOT_PERCENTILE|PERSONA_ENABLED)
        echo "anomaly-router" ;;
    VEC_INDEX_CORESET_PCT|VEC_QUANT|HNSW_M|HNSW_EF_CONSTRUCTION|VEC_INDEX_SUFFIX|\
    AUG_MULTIPLIER|AUG_PCT|VEC_TYPE)
        echo "anomaly-router" ;;                    # 인덱스명 결정은 router
    SEARCH_MODE|KNN_NUM_CANDIDATES|RETRIEVER_OVERSAMPLE|ES_REQUEST_TIMEOUT)
        echo "anomaly-retriever" ;;
    SKIP_ANALYZER)
        echo "anomaly-router anomaly-retriever" ;;
    # agent 경로
    RESEARCH_MODE|TOOL_GATE_ENABLE|PATCHCORE_ON_RESEARCH|PATCHCORE_REWEIGHT_MODE|\
    PATCHCORE_TAU|EVALUATOR_ONCE|AGENT_MAX_ROUNDS|SKIP_ANALYZER_ON_CONFIDENT|\
    EXPAND_PLATEAU_TAU|REROUTE_ON_CENTROID|REROUTE_SCORE_RATIO|BASE_TOP_K|EXPAND_TOP_K)
        echo "anomaly-controller" ;;
    TOOL_EXECUTION_MODE|PIPELINE_MODE|DIST_TOP_K)
        echo "anomaly-controller anomaly-orchestrator" ;;
    # analyzer (tier1/2 에서는 바뀌지 않는다)
    PROMPT_VERSION|WM_SCHEMA|ANALYZER_DECISION_MODE|CLS_ND_THRESHOLD|\
    SUFFICIENCY_BOUNDARY_LOW|SUFFICIENCY_BOUNDARY_HIGH)
        echo "local-analyzer anomaly-controller" ;;
    MODEL_BACKEND|GGUF_QUANT|N_CTX|MAX_NEW_TOKENS|ENABLE_KV_CACHE|\
    ENABLE_TTFT_PROBE|LLAMA_N_BATCH|LLAMA_N_UBATCH|ADAPTER_TYPE|DATA_VERSION)
        echo "local-analyzer" ;;
    RUN_ID|EVAL_SOURCE|EVAL_JSONL_GCS|EVAL_RATIO|AGENT_EVAL_GCS_PREFIX|\
    CORESET_PERCENTAGES|SKIP_RAW_VEC|VERSIONS|INGEST_SEARCH_MODE|N_SHARDS|\
    N_SHARDS_VEC|N_REPLICAS|PERCOLATOR_SHARD_DISTRIBUTION|NODE_PINNING)
        echo "" ;;                                  # ingest/eval 전용 — 서빙 무관
    *)  echo "anomaly-router anomaly-retriever anomaly-controller anomaly-orchestrator" ;;
  esac
}

# 현재 ConfigMap 값과 patch 값을 비교해 "실제로 바뀐 키"만 뽑고,
# 그 키들이 영향을 주는 deployment 집합을 반환한다.
restart_targets() {   # $1=patch json  → 공백구분 deployment 목록 (stdout)
  local f="$1"
  if [[ "${FORCE_RESTART:-0}" == "1" ]]; then
    echo "anomaly-router anomaly-retriever anomaly-controller anomaly-orchestrator local-analyzer"
    return
  fi

  # ★ 키마다 python 을 띄우면 Windows 에서 run 당 4~5초를 버린다.
  #   한 번의 호출로 "현재 ConfigMap vs patch" 를 비교해 변경 키만 출력한다.
  #   실패 시 stderr 를 남겨야 한다 — 2>/dev/null 로 숨기면 모든 키가
  #   changed 로 잡혀 매번 5개가 재시작된다(실제로 그렇게 동작했다).
  local cur changed err
  cur=$(kubectl get configmap anomaly-config -n "$NS" -o json 2>&1)
  if [[ "$cur" != \{* ]]; then
    warn "restart_targets: configmap 조회 실패 → 전체 재시작으로 폴백" >&2
    printf '    %s\n' "$(printf '%s' "$cur" | tail -2)" >&2
    echo "anomaly-router anomaly-retriever anomaly-controller anomaly-orchestrator local-analyzer"
    return
  fi

  err=$(mktemp)
  changed=$(printf '%s' "$cur" | "$PY" -c "
import io, json, sys
live = (json.load(sys.stdin).get('data') or {})
want = json.load(io.open(sys.argv[1], encoding='utf-8'))['data']
print(' '.join(k for k, v in want.items() if live.get(k) != str(v)))
" "$f" 2>"$err")
  if [[ -s "$err" ]]; then
    warn "restart_targets: 비교 실패 → 전체 재시작으로 폴백" >&2
    printf '    %s\n' "$(tail -2 "$err")" >&2; rm -f "$err" 
    echo "anomaly-router anomaly-retriever anomaly-controller anomaly-orchestrator local-analyzer"
    return
  fi
  rm -f "$err"

  if [[ -z "$changed" ]]; then log "  [diff] config 변경 없음" >&2; echo ""; return; fi
  log "  [diff] 변경 키: $changed" >&2

  local out=" " k t
  for k in $changed; do
    for t in $(_key_targets "$k"); do
      [[ "$out" == *" $t "* ]] || out+="$t "
    done
  done
  local ordered=""
  for t in anomaly-router anomaly-retriever anomaly-controller anomaly-orchestrator local-analyzer; do
    [[ "$out" == *" $t "* ]] && ordered+="$t "
  done
  printf '%s' "${ordered% }"
}

# ── 재기동 + 헬스 게이트 ─────────────────────────────────────────────
restart_and_wait() {   # $@ = deployment names
  local to="${ROLLOUT_TIMEOUT_S:-600}" d
  kubectl rollout restart -n "$NS" $(printf "deployment/%s " "$@") >/dev/null
  for d in "$@"; do
    # ★ rollout status 를 백그라운드로 돌리면서 CrashLoopBackOff 를 감시한다.
    #   그냥 기다리면 pod 가 죽어도 progressDeadline(기본 600s)까지 붙잡혀 있다.
    ( kubectl rollout status "deployment/$d" -n "$NS" --timeout="${to}s" >/dev/null 2>&1 ) &
    local sp=$! t=0 bad
    while kill -0 "$sp" 2>/dev/null && (( t < to )); do
      bad=$(kubectl get pods -n "$NS" -l "app=${d}" \
            -o jsonpath='{range .items[*]}{.status.containerStatuses[*].state.waiting.reason}{"\n"}{end}' \
            2>/dev/null | grep -E "CrashLoopBackOff|ErrImagePull|CreateContainerError" | head -1 || true)
      if [[ -n "$bad" ]]; then
        kill "$sp" 2>/dev/null || true
        err "rollout 조기 실패: $d ($bad)"
        kubectl logs -n "$NS" -l "app=${d}" --tail=25 --all-containers=true 2>/dev/null \
          | sed 's/^/    /' | tail -25
        return 1
      fi
      sleep 10; t=$((t+10))
    done
    wait "$sp" 2>/dev/null
    if (( $? != 0 )); then err "rollout timeout: $d"; return 1; fi
  done
  ok "rollout ready: $*"
}

# ★ bucketization run 사전 검증 — percolator _meta 에 bucket_config 가 없으면
#   router 가 기동 중 raise 해서 10분 rollout timeout 을 낭비한다. 미리 막는다.
check_bucket_config() {   # $1=experiment_case  $2=percolate index version
  local case="$1" ver="$2"
  local snippet='import os,json,sys
from elasticsearch import Elasticsearch
c=Elasticsearch([os.environ["ES_URL"]],verify_certs=False,ssl_show_warn=False,
                basic_auth=("elastic",os.environ["PASSWORD"]))
idx=sys.argv[1]
try:
    m=c.indices.get_mapping(index=idx)
    meta=m[idx]["mappings"].get("_meta",{})
    print("OK" if "bucket_config" in meta else "MISSING")
except Exception as e:
    print("ERR:"+str(e)[:120])'
  local idx="fraud_ecom_${case}_${ver}_tree_rules_percolator"
  local out
  out=$(kubectl exec -n "$NS" deploy/manage-indices -- python -c "$snippet" "$idx" 2>/dev/null | tr -d '\r')
  case "$out" in
    OK) ok "bucket_config 확인: $idx"; return 0 ;;
    MISSING) err "bucket_config 없음: $idx → percolator 재적재 필요 (run_case.sh ingest f32)"; return 1 ;;
    *) warn "bucket_config 확인 실패($out) — 진행은 하되 rollout 실패 가능"; return 0 ;;
  esac
}

# ★ pod 의 Ready 조건을 권위 있는 신호로 쓴다.
#   readinessProbe 가 이미 /ready·/health 를 HTTP 로 때리고 있으므로,
#   컨테이너 안에서 curl 을 또 실행할 이유가 없다 (python:slim 계열엔
#   curl 이 없어 무한 폴링에 빠진다 — 실제로 그렇게 6분 멈췄다).
#   curl 이 있으면 보조 확인만 한 번 더 한다.
has_cmd() {   # $1=deployment  $2=command
  kubectl exec -n "$NS" "deploy/$1" -- sh -c "command -v $2 >/dev/null" 2>/dev/null
}

wait_endpoint() {   # $1=deploy  $2=port  $3=path  $4=timeout_s
  local dep="$1" port="$2" path="${3:-/health}" to="${4:-600}" t=0
  while (( t < to )); do
    if kubectl get pods -n "$NS" -l "app=${dep}" \
         -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' \
         2>/dev/null | grep -q "True"; then
      # 보조: curl 이 있으면 실제 엔드포인트도 한 번 확인
      if has_cmd "$dep" curl; then
        if kubectl exec -n "$NS" "deploy/${dep}" -- \
             sh -c "curl -sf -m 5 http://localhost:${port}${path} >/dev/null" 2>/dev/null; then
          ok "ready: ${dep} (pod Ready + ${path} 200, ${t}s)"; return 0
        fi
        # Ready 인데 curl 실패 → 조금 더 기다린다
      else
        ok "ready: ${dep} (pod Ready, ${t}s)  [curl 없음 — probe 신뢰]"; return 0
      fi
    fi
    (( t % 30 == 0 && t > 0 )) && log "  waiting ${dep} ... ${t}s/${to}s"
    sleep 10; t=$((t+10))
  done
  err "ready timeout: ${dep} (${to}s)"
  kubectl get pods -n "$NS" -l "app=${dep}" 2>/dev/null | sed 's/^/    /'
  return 1
}

# ── RESTARTS 스냅샷 (오염 감지) ──────────────────────────────────────
snapshot_restarts() {
  local total=0 n
  for d in ${WATCH_DEPLOYMENTS:-}; do
    n=$(kubectl get pods -n "$NS" -l "app=${d}" \
          -o jsonpath='{range .items[*]}{.status.containerStatuses[*].restartCount}{"\n"}{end}' \
          2>/dev/null | tr ' ' '\n' | awk '{s+=$1} END {print s+0}')
    total=$((total + n))
  done
  echo "$total"
}

assert_no_restart() {   # $1=before  → 0=clean, 1=suspect
  local before="$1" after
  after=$(snapshot_restarts)
  if (( after > before )); then
    warn "RESTARTS ${before} → ${after} — 이 run 은 pod 재시작 중 측정됨 (suspect)"
    return 1
  fi
  ok "no pod restart during run (${after})"
  return 0
}

# ── warmup — 모델 로드 / JVM / page cache ────────────────────────────
# ★ 직렬 1건씩만 보낸다. 여기서 동시 요청을 보내면 정작 그 warmup 이
#   local-analyzer 를 rollout 시키는 원인이 된다.
warmup() {
  local n="${WARMUP_REQUESTS:-2}" i
  # ★ /detect 는 flat 필드 + 576-dim embedding 을 받는다.
  #   {"transaction": {...}} 로 보내면 422 가 되어 파이프라인을 타지 않고,
  #   warmup 효과가 0 이 된다(실제로 그렇게 돌아 v1 P95 가 2.5배 부풀었다).
  local snippet='import httpx
p={"purchase_value":0.5,"age":0.1,"sex":"M","source":"Ads","browser":"Chrome",
   "weekday_purchase":"Monday","month_purchase":"January","IP_country":"KR",
   "embedding":[0.01]*576}
try:
    r=httpx.post("http://localhost:8080/detect",json=p,timeout=300.0)
    print("HTTP",r.status_code,r.text[:120] if r.status_code>=400 else "")
except Exception as e:
    print("ERR",type(e).__name__,str(e)[:100])'
  for ((i=1; i<=n; i++)); do
    local out
    out=$(kubectl exec -n "$NS" deploy/anomaly-orchestrator -- \
          python -c "$snippet" 2>/dev/null | tr -d '\r' | tail -1)
    case "$out" in
      "HTTP 200"*) log "warmup ${i}/${n} ok" ;;
      HTTP*)       warn "warmup ${i}/${n} ${out}" ;;
      *)           warn "warmup ${i}/${n} 실패(${out:-no output}) — cold cache 로 측정될 수 있음" ;;
    esac
    sleep 3
  done
}

# ── Job 실행 & 대기 (1 run = 1 job, 재시도 없음) ─────────────────────
run_job() {   # $1=yaml  $2=job name  $3=timeout_s  $4=logfile
  local yaml="$1" name="$2" to="$3" logf="${4:-/dev/null}"
  [[ -f "$yaml" ]] || { err "job yaml 없음: $yaml"; return 1; }
  kubectl delete job "$name" -n "$NS" --ignore-not-found >/dev/null 2>&1
  sleep 3
  if ! kubectl apply -f "$yaml" -n "$NS" >/dev/null 2>&1; then
    err "kubectl apply 실패: $yaml"; return 1
  fi
  log "job/${name} 시작 (timeout ${to}s)"

  # ★ apply 직후 바로 logs -f 를 걸면 initContainer 가 아직 시작 전이라
  #   "PodInitializing" 에러로 스트림이 즉시 죽는다. Running 이 될 때까지
  #   기다린 뒤 붙는다. (그래도 놓칠 수 있으므로 완료 후 전체 로그를 재캡처)
  ( local w=0 ph
    while (( w < 300 )); do
      ph=$(kubectl get pods -n "$NS" -l "job-name=${name}" \
           -o jsonpath='{.items[0].status.phase}' 2>/dev/null || true)
      [[ "$ph" == "Running" || "$ph" == "Succeeded" || "$ph" == "Failed" ]] && break
      sleep 5; w=$((w+5))
    done
    kubectl logs -f "job/${name}" -n "$NS" --all-containers=true \
      > "$logf" 2>&1 || true ) &
  local logpid=$!

  local t=0 st
  while (( t < to )); do
    st=$(kubectl get job "$name" -n "$NS" \
         -o jsonpath='{.status.succeeded}|{.status.failed}' 2>/dev/null || echo "|")
    case "$st" in
      1\|*)  _finalize_job_log "$name" "$logf" "$logpid"; ok  "job/${name} 완료 (${t}s)"; return 0 ;;
      *\|1*) _finalize_job_log "$name" "$logf" "$logpid"; err "job/${name} 실패 (${t}s)";  return 1 ;;
    esac
    sleep 15; t=$((t+15))
  done
  _finalize_job_log "$name" "$logf" "$logpid"
  err "job/${name} timeout (${to}s)"
  return 2
}

# ★ kubectl logs 는 Completed pod 에서도 동작한다(exec 와 달리).
#   스트리밍이 중간에 끊겼어도 여기서 전체 로그를 확보한다.
_finalize_job_log() {   # $1=job name  $2=logfile  $3=stream pid
  local name="$1" logf="$2" pid="${3:-}"
  [[ -n "$pid" ]] && { wait "$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true; }
  [[ "$logf" == "/dev/null" ]] && return 0
  sleep 2
  local full
  if full=$(kubectl logs "job/${name}" -n "$NS" --all-containers=true --tail=-1 2>/dev/null) \
     && [[ -n "$full" ]]; then
    printf '%s\n' "$full" > "$logf"
    log "  로그 확보: $(wc -l < "$logf") lines"
  fi
}

# ── GCS 헬퍼 ─────────────────────────────────────────────────────────
# ★ 실패 원인을 절대 숨기지 않는다. gsutil 이 Windows(Git Bash)에서
#   경로/인증 문제로 조용히 실패하는 경우가 있어, stderr 를 붙잡아 출력하고
#   gcloud storage 로 한 번 더 시도한다.
_gs() {   # $1=동작설명, $2.. = gsutil 인자
  local what="$1"; shift
  local out rc
  out=$(gsutil "$@" 2>&1); rc=$?
  if (( rc == 0 )); then return 0; fi
  # 폴백: gcloud storage (동일 인증, 다른 구현)
  local g=("${@}")
  case "${g[0]}" in
    -q|-m) unset 'g[0]' ; g=("${g[@]}") ;;
  esac
  case "${g[0]}" in
    -q|-m) unset 'g[0]' ; g=("${g[@]}") ;;
  esac
  local out2 rc2
  out2=$(gcloud storage "${g[@]}" 2>&1); rc2=$?
  if (( rc2 == 0 )); then log "  ($what: gcloud storage 폴백 성공)"; return 0; fi
  warn "$what 실패"
  printf '    gsutil : %s\n' "$(echo "$out"  | tail -3)"
  printf '    gcloud : %s\n' "$(echo "$out2" | tail -3)"
  return 1
}

# Job 산출물 회수 — kubectl cp 는 Completed pod 에서 동작하지 않으므로
# eval Job 이 스스로 올린 GCS 경로에서 받아온다.
fetch_gcs() {   # $1=gcs dir  $2=local dir  $3.. = 파일명들
  local src="${1%/}" dst="$2"; shift 2
  mkdir -p "$dst"
  local f got=0
  for f in "$@"; do
    if _gs "수집 $f" -q cp "$src/$f" "$dst/$f"; then
      ok "수집: $f"; got=$((got+1))
    fi
  done
  return $(( got == 0 ))
}

# run 산출물 업로드
upload_run() {   # $1=local dir  $2=gcs dest
  local src="${1%/}" dst="${2%/}"
  [[ -d "$src" ]] || { warn "업로드 대상 없음: $src"; return 0; }
  if _gs "업로드 $dst" -m rsync -r "$src" "$dst"; then
    ok "GCS → $dst"
  fi
}

# ★ experiments/ 전체를 GCS 와 동기화 (local == gs://.../experiments)
#   cases/ runner/ rendered/ results/ 전부. __pycache__ 등은 제외.
sync_experiments() {
  local local_dir="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
  # ★ GCS_ROOT 에는 이미 /<MMDD> 가 붙어 있다. 코드/설정은 날짜 무관이므로
  #   _code 하위로 올려 실험 결과와 섞이지 않게 한다.
  local dst="${GCS_ROOT%/}/_code"
  log "experiments/ → $dst  (rsync)"
  if _gs "experiments 동기화" -m rsync -r \
       -x '(^|.*/)__pycache__/.*|.*\.pyc$|(^|.*/)\.git/.*' \
       "$local_dir" "$dst"; then
    ok "동기화 완료: $dst"
  fi
}
