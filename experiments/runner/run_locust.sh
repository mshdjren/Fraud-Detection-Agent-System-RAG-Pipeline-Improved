#!/usr/bin/env bash
# =====================================================================
# run_locust.sh — 부하 3단계 sweep (약함 / 중간 / 높음)
# =====================================================================
# 기존 coreset_sweep.sh 의 흐름을 그대로 따른다:
#   ConfigMap 패치 → 서비스 재시작 → locust env 설정 → port-forward
#   → /swarm 으로 부하 시작 → /stats/requests 폴링 → /stop → 결과 저장
#
# ★ 이 스크립트가 하네스 전체에서 "동시성이 설계 의도인 유일한 지점"이다.
#   run_matrix.sh 는 절대 병렬 실행하지 않지만, 여기서는 동시 사용자 N 명이
#   측정 대상 그 자체다.
#
# GKE free tier 기준 3단계 (router/retriever limits 1000m, ES docs 노드 1.8코어):
#   light  =  5 users  — 큐잉 없는 baseline. 순수 서비스 latency
#   medium = 20 users  — CPU 포화 직전. 실사용 상정 구간
#   heavy  = 50 users  — 포화 이후. 큐잉·throttling·page fault 관측
#   (100+ 는 free tier 에서 orchestrator LB timeout 이 먼저 나서 제외)
#
# 사용:
#   bash experiments/runner/run_locust.sh <run_id>
#   bash experiments/runner/run_locust.sh c-quant-f32 --levels light,medium
#   bash experiments/runner/run_locust.sh d-hnsw-s1 --stage 2
#   STAGES_ONLY=1 bash ... run_locust.sh <run_id>   # 재시작 없이 부하만
# =====================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RENDERED="${RENDERED:-$HERE/../rendered}"

# shellcheck disable=SC1091
source "$RENDERED/serialization.env"
RUN_DATE="${RUN_DATE:-$(date +%m%d)}"
RESULTS="${RESULTS:-$HERE/../results/$RUN_DATE}"
GCS_ROOT="${GCS_ROOT%/}/$RUN_DATE"
# shellcheck disable=SC1091
source "$HERE/lib.sh"

RUN_ID="${1:?usage: run_locust.sh <run_id> [--levels light,medium,heavy]}"
shift || true

# ── 부하 단계 정의 ───────────────────────────────────────────────────
#   name:users:spawn_rate:duration_s
LEVEL_light="5:2:60"
LEVEL_medium="20:5:90"
LEVEL_heavy="50:10:90"
LEVELS="light,medium,heavy"

LOCUST_DEPLOY="${LOCUST_DEPLOY:-locust-master}"
LOCUST_SVC="${LOCUST_SVC:-locust}"
LOCUST_PORT="${LOCUST_PORT:-8089}"
ACTIVE_STAGE="${ACTIVE_STAGE:-2}"      # locustfile 의 Stage (2 = page fault 실험)
COOLDOWN="${LOCUST_COOLDOWN:-20}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --levels) LEVELS="$2"; shift ;;
    --stage)  ACTIVE_STAGE="$2"; shift ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
  shift
done

D="$RENDERED/run/$RUN_ID"
[[ -f "$D/config.patch.json" ]] || { err "렌더 결과 없음: $D → expand_matrix.py 실행"; exit 1; }

OUT="$RESULTS/locust/$RUN_ID"; mkdir -p "$OUT"
cp "$D/manifest.json" "$OUT/" 2>/dev/null || true

# ─────────────────────────────────────────────
# 정리 (coreset_sweep.sh 와 동일 패턴)
# ─────────────────────────────────────────────
cleanup() {
  pkill -f "port-forward.*$LOCUST_PORT" 2>/dev/null || true
  if command -v taskkill &>/dev/null; then
    taskkill //F //IM kubectl.exe //T 2>/dev/null || true
  fi
}
trap cleanup EXIT

locust_api() { curl -sf --retry 3 --retry-delay 2 "$@"; }

start_portforward() {
  pkill -f "port-forward.*$LOCUST_PORT" 2>/dev/null || true
  sleep 2
  kubectl port-forward -n "$NS" "svc/$LOCUST_SVC" \
    "${LOCUST_PORT}:${LOCUST_PORT}" > /dev/null 2>&1 &
  printf "  ⏳ port-forward"
  for _ in {1..12}; do
    if curl -sf "http://localhost:${LOCUST_PORT}/stats/requests" &>/dev/null; then
      echo " [OK]"; return 0
    fi
    printf "."; sleep 2
  done
  echo " [FAIL]"; return 1
}

# ─────────────────────────────────────────────
# 1) 대상 run 의 config 로 전환
# ─────────────────────────────────────────────
log "═══ LOCUST SWEEP  run=$RUN_ID  levels=$LEVELS  stage=$ACTIVE_STAGE ═══"

if [[ "${STAGES_ONLY:-0}" != "1" ]]; then
  TARGETS=$(restart_targets "$D/config.patch.json")
  [[ " $TARGETS " == *" anomaly-retriever "* ]] || TARGETS="$TARGETS anomaly-retriever"
  TARGETS=$(echo "$TARGETS" | tr -s ' ' | sed 's/^ //;s/ $//')
  
  patch_config anomaly-config           "$D/config.patch.json"    || exit 1
  patch_config anomaly-retriever-config "$D/retriever.patch.json" || exit 1

  if [[ -n "$TARGETS" ]]; then
    log "재시작 대상: $TARGETS"
    # shellcheck disable=SC2086
    restart_and_wait $TARGETS || exit 1
    wait_endpoint anomaly-retriever 8001 /ready "$ROLLOUT_TIMEOUT_S" || exit 1
    wait_endpoint anomaly-router    8000 /ready "$ROLLOUT_TIMEOUT_S" || exit 1
  else
    ok "재시작 불필요 — config 변경 없음"
  fi
  warmup
fi

# ─────────────────────────────────────────────
# 2) locust 설정 및 기동
# ─────────────────────────────────────────────
kubectl set env "deployment/$LOCUST_DEPLOY" -n "$NS" \
  ACTIVE_STAGE="$ACTIVE_STAGE" RUN_ID="$RUN_ID" >/dev/null 2>&1 \
  || { err "locust deployment 없음 ($LOCUST_DEPLOY) — locust-deployment.yaml 먼저 apply"; exit 1; }
kubectl rollout status "deployment/$LOCUST_DEPLOY" -n "$NS" --timeout=180s >/dev/null \
  || { err "locust rollout 실패"; exit 1; }
start_portforward || exit 1

BEFORE=$(snapshot_restarts)
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/started_at.txt"

# ─────────────────────────────────────────────
# 3) 부하 단계별 실행
# ─────────────────────────────────────────────
SUMMARY="$OUT/summary.tsv"
printf "level\tusers\tspawn\tdur_s\trps\tp50\tp95\tp99\tfail\tn_req\n" > "$SUMMARY"

IFS=',' read -ra LV <<< "$LEVELS"
for level in "${LV[@]}"; do
  level=$(echo "$level" | tr -d ' ')
  spec_var="LEVEL_${level}"
  spec="${!spec_var:-}"
  [[ -z "$spec" ]] && { warn "알 수 없는 level: $level (light|medium|heavy)"; continue; }
  IFS=':' read -r USERS SPAWN DUR <<< "$spec"

  echo
  log "▶ ${level}  users=${USERS}  spawn=${SPAWN}  dur=${DUR}s"
  locust_api "http://localhost:${LOCUST_PORT}/stop" >/dev/null 2>&1 || true
  sleep 3
  locust_api -X POST "http://localhost:${LOCUST_PORT}/swarm" \
    -d "user_count=${USERS}&spawn_rate=${SPAWN}" >/dev/null \
    || { warn "swarm 시작 실패 — 이 level 건너뜀"; continue; }

  ELAPSED=0; POLL=15
  while (( ELAPSED < DUR )); do
    sleep "$POLL"; ELAPSED=$((ELAPSED + POLL))
    LIVE=$(curl -sf --max-time 5 "http://localhost:${LOCUST_PORT}/stats/requests" 2>/dev/null || true)
    [[ -z "$LIVE" ]] && continue
    STAT=$(printf '%s' "$LIVE" | "$PY" -c "
import sys, json
try:
    d = json.load(sys.stdin)
    t = d.get('stats_total') or (d.get('stats') or [{}])[-1]
    rps  = round(d.get('total_rps') or t.get('current_rps') or 0, 1)
    p50  = int(t.get('response_time_percentile_0.5')  or t.get('median_response_time') or 0)
    p95  = int(t.get('response_time_percentile_0.95') or 0)
    p99  = int(t.get('response_time_percentile_0.99') or t.get('response_time_percentile_99') or 0)
    print(f\"{rps} {p50} {p95} {p99} {t.get('num_failures') or 0} {t.get('num_requests') or 0}\")
except Exception:
    print('0 0 0 0 0 0')
" 2>/dev/null || echo "0 0 0 0 0 0")
    read -r RPS P50 P95 P99 FAIL NREQ <<< "$STAT"
    printf "    [%3ds/%ds] rps=%-6s p50=%-6s p95=%-6s p99=%-7s fail=%-4s n=%s\n" \
      "$ELAPSED" "$DUR" "$RPS" "$P50" "$P95" "$P99" "$FAIL" "$NREQ"
  done

  # 최종 통계 저장
  locust_api "http://localhost:${LOCUST_PORT}/stats/requests" > "$OUT/${level}_users${USERS}.json" \
    && ok "저장: ${level}_users${USERS}.json" || warn "통계 저장 실패: $level"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$level" "$USERS" "$SPAWN" "$DUR" "${RPS:-0}" "${P50:-0}" "${P95:-0}" \
    "${P99:-0}" "${FAIL:-0}" "${NREQ:-0}" >> "$SUMMARY"

  locust_api "http://localhost:${LOCUST_PORT}/stop" >/dev/null 2>&1 || true
  log "  cooldown ${COOLDOWN}s"
  sleep "$COOLDOWN"
done

date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/finished_at.txt"

# ─────────────────────────────────────────────
# 4) 오염 판정 + 업로드
# ─────────────────────────────────────────────
# ★ 부하 실험에서는 pod 재시작이 "결과"일 수도 있다(포화 증거).
#   실패로 처리하지 않고 기록만 한다.
AFTER=$(snapshot_restarts)
if (( AFTER > BEFORE )); then
  warn "부하 중 pod 재시작 ${BEFORE} → ${AFTER} — 포화 신호로 기록"
fi
printf '{"run_id":"%s","restarts_before":%s,"restarts_after":%s,"active_stage":"%s","levels":"%s"}\n' \
  "$RUN_ID" "$BEFORE" "$AFTER" "$ACTIVE_STAGE" "$LEVELS" > "$OUT/locust_meta.json"

echo
log "── 결과 요약 ──"
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"

upload_run "$OUT" "$GCS_ROOT/$EXPERIMENT_CASE/stage$STAGE/_locust/$RUN_ID"
ok "═══ LOCUST 완료: $RUN_ID → $OUT ═══"
