#!/usr/bin/env bash
# =====================================================================
# run_matrix.sh — plan.tsv 를 위에서 아래로 "한 번에 하나씩" 실행
# =====================================================================
#   bash experiments/runner/run_matrix.sh                 # 전체
#   bash experiments/runner/run_matrix.sh --preflight     # 사전 점검만
#   bash experiments/runner/run_matrix.sh --resume        # 완료된 건 건너뜀
#   bash experiments/runner/run_matrix.sh --only c-quant-f32,c-quant-int8os3
#   bash experiments/runner/run_matrix.sh --skip-ingest   # 인덱스 재사용
#
# ★ 병렬 실행을 절대 하지 않는다. GKE free tier 에서 동시 부하는
#   local-analyzer rollout → run 오염으로 직결된다.
#   (locust 다중 user 실험은 stage 2 의 별도 페이즈에서 다룬다 — 그건
#    설계 의도상 동시성이 필요한 유일한 케이스)
# =====================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RENDERED="${RENDERED:-$HERE/../rendered}"

[[ -f "$RENDERED/plan.tsv" ]] || {
  echo "plan.tsv 없음 — 먼저 expand_matrix.py 실행"; exit 1; }

# shellcheck disable=SC1091
source "$RENDERED/serialization.env"
# ★ 날짜별 분리 (expand_matrix.py --date 로 지정, 기본 오늘)
RUN_DATE="${RUN_DATE:-$(date +%m%d)}"
RESULTS="${RESULTS:-$HERE/../results/$RUN_DATE}"
GCS_ROOT="${GCS_ROOT%/}/$RUN_DATE"
export RUN_DATE
# shellcheck disable=SC1091
source "$HERE/lib.sh"

PREFLIGHT=false; RESUME=false; SKIP_INGEST=false; ONLY=""; SKIP_ES_CHECK=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --preflight)   PREFLIGHT=true ;;
    --resume)      RESUME=true ;;
    --skip-ingest)   SKIP_INGEST=true ;;
    --skip-es-check) SKIP_ES_CHECK=true ;;
    --only)        ONLY="$2"; shift ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
  shift
done

# =====================================================================
# 0. Preflight — 여기서 걸리는 게 3시간 뒤에 걸리는 것보다 훨씬 싸다
# =====================================================================
preflight() {
  local fail=0
  log "── preflight ──────────────────────────────────────────"

  kubectl get ns "$NS" >/dev/null 2>&1 \
    && ok "namespace $NS" || { err "namespace $NS 없음"; fail=1; }

  # ES 버전 — int8_hnsw(8.12+) / bbq_hnsw(8.16+) / rescore_vector(8.18+)
  local esv
  if $SKIP_ES_CHECK; then
    log "ES 버전 체크 생략 (--skip-es-check)"; esv="?"
  else
  esv=$(kubectl exec -n "$NS" deploy/manage-indices -- \
        sh -c 'curl -sk -u elastic:$PASSWORD https://elasticsearch-ha-es-http:9200/ 2>/dev/null' \
        2>/dev/null | "$PY" -c "import sys,json;print(json.load(sys.stdin)['version']['number'])" 2>/dev/null || echo "?")
  fi
  if [[ "$esv" == "?" ]]; then
    warn "ES 버전 미확인 (경고만 — 실행은 계속)"
  else
    ok "ES version = $esv"
    "$PY" - "$esv" <<'PYES' || true
import sys
v = tuple(int(x) for x in sys.argv[1].split(".")[:2])
if v < (8, 18):
    print("  \033[1;33m⚠ rescore_vector 는 8.18+ 필요 — RETRIEVER_OVERSAMPLE 이 무시됩니다\033[0m")
if v < (8, 16):
    print("  \033[1;33m⚠ bbq_hnsw 는 8.16+ 필요 — bbq 인덱스 적재가 실패합니다\033[0m")
PYES
  fi

  # 코드 ConfigMap 존재 여부
  for cm in es-scripts router-code retriever-code controller-code \
            local-analyzer-code evaluation-code evaluation-agent-eval-code; do
    kubectl get configmap "$cm" -n "$NS" >/dev/null 2>&1 \
      && ok "configmap/$cm" || { err "configmap/$cm 없음"; fail=1; }
  done

  # 코드 수정이 실제로 반영됐는지 — 문자열 검사 (배포 누락의 90%를 여기서 잡음)
  check_cm() {  # $1=cm  $2=key  $3=needle  $4=설명
    if kubectl get configmap "$1" -n "$NS" -o jsonpath="{.data.$2}" 2>/dev/null \
         | grep -q -- "$3"; then ok "  $4"
    else err "  $4 — 미반영 (configmap 재생성 필요)"; fail=1; fi
  }
  check_cm es-scripts       ingest_tree_pipeline\\.py  "index_options"            "ingest: quant/HNSW 파라미터화"
  # ★ 존재 검사는 약하다. 버그 패턴의 "부재"를 확인해야 실제로 고쳐진 것이다.
  #   (PERCOLATE_INDEX_VERSION 이 한 군데만 있어도 존재 검사는 통과해버린다)
  check_cm_absent() {   # $1=cm $2=key $3=버그패턴 $4=설명
    local n
    n=$(kubectl get configmap "$1" -n "$NS" -o jsonpath="{.data.$2}" 2>/dev/null \
        | grep -c -- "$3" || true)
    if [[ "${n:-0}" == "0" ]]; then ok "  $4"
    else err "  $4 — 미수정 지점 ${n}곳 남음"; fail=1; fi
  }
  check_cm        router-code router\\.py "PERCOLATE_INDEX_VERSION" "router: alias 상수 정의"
  check_cm_absent router-code router\\.py "{PERCOLATE_QUERY_VERSION}_tree_rules_percolator" \
                  "router: v14 alias 전 구간 적용(버그 패턴 0)"
  check_cm router-code      router\\.py                "_vec_index_suffix"        "router: vec index suffix"
  check_cm retriever-code   retriever\\.py             "rescore_vector"           "retriever: oversample"
  check_cm controller-code  controller\\.py            "neighbor_oidx"            "controller: Recall@5 배선"
  check_cm controller-code  controller\\.py            "_patchcore_status"        "controller: 재검색 nd 진단"
  check_cm controller-code  controller\\.py            "ttft_ms"                  "controller: TTFT pass-through"
  check_cm local-analyzer-code local_analyzer\\.py     "analyzer_ttft"            "analyzer: TTFT/TPOT 계측"
  # ★ /app/evaluation_metrics.py 는 agent-eval-batch.yaml 에서 ConfigMap
  #   'evaluation-code'(code-volume)로 마운트된다. 'evaluation-agent-eval-code'
  #   쪽만 갱신하면 컨테이너엔 옛 파일이 들어가 research_effect 가 {} 가 된다.
  check_cm evaluation-code            evaluation_metrics\\.py    "calculate_research_effect" "metrics(evaluation-code) ← 실제 마운트 대상"
  check_cm evaluation-agent-eval-code agent_batch_inference\\.py "_collect_axes"             "harness: run_id/axes"
  check_cm evaluation-agent-eval-code agent_batch_inference\\.py "research_effect"           "harness: research_effect 호출"

  # ★★ 로컬 파일 ↔ 배포된 ConfigMap 동기 검사 ★★
  #   위 마커 검사는 "배포된 ConfigMap"만 본다. 로컬 파일이 회귀했거나
  #   재배포를 잊으면 옛 코드로 실험이 돌면서 preflight 는 통과해버린다.
  #   (실제로 controller.py 의 neighbor_oidx 가 로컬에서 유실됐는데
  #    ConfigMap 엔 남아 있어 통과했고, 배포 순간 recall@5 가 0 이 될 뻔했다)
  check_cm_sync() {   # $1=cm  $2=jsonpath key  $3=로컬 파일 경로  $4=설명
    local cm="$1" key="$2" f="$3" desc="$4"
    if [[ ! -f "$f" ]]; then warn "  $desc — 로컬 파일 없음($f), 건너뜀"; return 0; fi
    local tmp; tmp=$(mktemp)
    if ! kubectl get configmap "$cm" -n "$NS" -o jsonpath="{.data.$key}" > "$tmp" 2>/dev/null; then
      err "  $desc — configmap 조회 실패"; rm -f "$tmp"; fail=1; return 1
    fi
    # CRLF/개행 차이를 무시하고 내용만 비교
    local a b
    a=$("$PY" -c "import io,sys,hashlib;print(hashlib.sha1(io.open(sys.argv[1],encoding='utf-8',errors='replace').read().replace('\r\n','\n').rstrip().encode()).hexdigest()[:10])" "$f")
    b=$("$PY" -c "import io,sys,hashlib;print(hashlib.sha1(io.open(sys.argv[1],encoding='utf-8',errors='replace').read().replace('\r\n','\n').rstrip().encode()).hexdigest()[:10])" "$tmp")
    rm -f "$tmp"
    if [[ "$a" == "$b" ]]; then ok "  $desc (sha=$a)"
    else
      err "  $desc — 로컬($a) ≠ ConfigMap($b). 재배포 필요"
      fail=1
    fi
  }
  local APPS="${APPS_DIR:-apps}"
  local EVALD="${EVAL_DIR:-evaluation/agent_eval}"
  check_cm_sync router-code            router\\.py               "$APPS/router.py"                "sync: router.py"
  check_cm_sync retriever-code         retriever\\.py            "$APPS/retriever.py"             "sync: retriever.py"
  check_cm_sync controller-code        controller\\.py           "$APPS/controller.py"            "sync: controller.py"
  check_cm_sync local-analyzer-code    local_analyzer\\.py       "$APPS/local_analyzer.py"        "sync: local_analyzer.py"
  check_cm_sync es-scripts             ingest_tree_pipeline\\.py "$APPS/ingest_tree_pipeline.py"  "sync: ingest_tree_pipeline.py"
  check_cm_sync evaluation-code        evaluation_metrics\\.py   "$EVALD/evaluation_metrics.py"   "sync: evaluation_metrics.py"
  check_cm_sync evaluation-agent-eval-code agent_batch_inference\\.py "$EVALD/agent_batch_inference.py" "sync: agent_batch_inference.py"

  # gsutil / python3 / PyYAML
  # ★ gsutil 이 PATH 에 있는 것만으로는 부족하다. Cloud SDK 가 Python 3.9 를
  #   집으면 "TypeError: ... for |: 'type' and 'type'" 로 죽는다. 실행해서 확인.
  if command -v gsutil >/dev/null; then
    if gsutil version >/dev/null 2>&1; then
      ok "gsutil 동작 (CLOUDSDK_PYTHON=${CLOUDSDK_PYTHON:-PATH default})"
    else
      err "gsutil 실행 실패 — Cloud SDK 가 잘못된 Python 을 쓰고 있음"
      echo "    $(gsutil version 2>&1 | tail -2)"
      echo "    해결: export CLOUDSDK_PYTHON=\"<SDK>/platform/bundledpython/python.exe\""
      echo "          또는 gcloud components update"
      fail=1
    fi
  else err "gsutil 없음"; fail=1; fi
  ok "python = $PY  ($("$PY" -V 2>&1))"
  "$PY" -c "import yaml" 2>/dev/null && ok "PyYAML ($PY)" \
    || { err "PyYAML 없음 — '$PY -m pip install pyyaml' 또는 PY=<경로> 로 지정"; fail=1; }

  # 렌더 결과
  local n; n=$(grep -c . "$RENDERED/plan.tsv")
  ok "plan.tsv $((n-1)) 항목"

  log "── preflight $( ((fail==0)) && echo 통과 || echo 실패 ) ──"
  return $fail
}

preflight || { err "preflight 실패 — 위 항목을 고친 뒤 재시도"; exit 1; }
$PREFLIGHT && exit 0

# =====================================================================
# 1. 순차 실행
# =====================================================================
declare -a DONE_OK=() DONE_SUSPECT=() DONE_FAIL=() SKIPPED=()
declare -A BAD_INDEX=()

should_skip() {   # $1=id
  [[ -n "$ONLY" ]] && [[ ",$ONLY," != *",$1,"* ]] && return 0
  return 1
}

# ★ [STEP1-A] 매트릭스 시작 시 1회만 전체 재시작(prime).
#   코드 ConfigMap 을 새로 배포했을 수 있으므로 여기서 한 번 확실히 반영하고,
#   이후 개별 run 은 "바뀐 config 키가 영향 주는 deployment" 만 재시작한다.
if [[ "${PRIME_RESTART:-1}" == "1" ]] && ! $PREFLIGHT; then
  log "── prime restart (1회, 코드 반영 보장) ──"
  restart_and_wait anomaly-router anomaly-retriever anomaly-controller \
                   anomaly-orchestrator local-analyzer \
    || { err "prime restart 실패"; exit 1; }
  wait_endpoint local-analyzer 8080 /health "$ROLLOUT_TIMEOUT_S" || exit 1
fi

T0=$(date +%s)
while IFS=$'\t' read -r kind id index_id evalk nsamp conc; do
  [[ "$kind" == "kind" ]] && continue           # 헤더

  # ── INGEST ─────────────────────────────────────────────────────────
  if [[ "$kind" == "INGEST" ]]; then
    $SKIP_INGEST && { SKIPPED+=("INGEST:$id"); log "skip ingest: $id"; continue; }
    if $RESUME && [[ -f "$RESULTS/index/$id/finished_at.txt" ]]; then
      SKIPPED+=("INGEST:$id"); log "resume skip: ingest $id"; continue
    fi
    if bash "$HERE/run_case.sh" ingest "$id"; then
      DONE_OK+=("INGEST:$id")
    else
      DONE_FAIL+=("INGEST:$id"); BAD_INDEX["$id"]=1
      warn "인덱스 $id 적재 실패 — 이 인덱스를 쓰는 run 은 자동 skip"
    fi
    continue
  fi

  # ── RUN ────────────────────────────────────────────────────────────
  should_skip "$id" && { SKIPPED+=("RUN:$id"); continue; }
  if [[ -n "${BAD_INDEX[$index_id]:-}" ]]; then
    SKIPPED+=("RUN:$id(bad-index)"); warn "skip $id — index $index_id 실패"; continue
  fi
  if $RESUME && [[ -f "$RESULTS/run/$id/agent_eval_report.json" ]]; then
    SKIPPED+=("RUN:$id"); log "resume skip: $id"; continue
  fi

  bash "$HERE/run_case.sh" run "$id"
  rc=$?
  st=$("$PY" -c "import io,json,sys;print(json.load(io.open(sys.argv[1],encoding='utf-8')).get('status','?'))" \
       "$RESULTS/run/$id/manifest.json" 2>/dev/null || echo "?")
  case "$st" in
    ok)      DONE_OK+=("$id") ;;
    suspect) DONE_SUSPECT+=("$id") ;;
    *)       DONE_FAIL+=("$id") ;;
  esac
done < "$RENDERED/plan.tsv"

# =====================================================================
# 2. 요약
# =====================================================================
ELAPSED=$(( $(date +%s) - T0 ))
echo
log "═══════════════ MATRIX 종료 (${ELAPSED}s) ═══════════════"
printf "  ok      (%2d) : %s\n" "${#DONE_OK[@]}"      "${DONE_OK[*]:-—}"
printf "  suspect (%2d) : %s\n" "${#DONE_SUSPECT[@]}" "${DONE_SUSPECT[*]:-—}"
printf "  failed  (%2d) : %s\n" "${#DONE_FAIL[@]}"    "${DONE_FAIL[*]:-—}"
printf "  skipped (%2d) : %s\n" "${#SKIPPED[@]}"      "${SKIPPED[*]:-—}"
echo
if (( ${#DONE_SUSPECT[@]} > 0 )); then
  warn "suspect = run 중 pod 재시작이 있었던 건. 아래로 재실행 권장:"
  echo "    bash $0 --skip-ingest --only $(IFS=,; echo "${DONE_SUSPECT[*]}")"
fi
sync_experiments      # ★ [STEP1-E] run 마다가 아니라 매트릭스 끝에서 1회
log "결과: $RESULTS/  ·  GCS: $GCS_ROOT/$EXPERIMENT_CASE/stage$STAGE/"

(( ${#DONE_FAIL[@]} == 0 )) || exit 1
