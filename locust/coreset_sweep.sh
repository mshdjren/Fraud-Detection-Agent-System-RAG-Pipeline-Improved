#!/usr/bin/env bash
# =====================================================================
# run_coreset_sweep.sh — Coreset 비율별 Locust 자동화 (v2.2)
# =====================================================================

set -euo pipefail

NS="elastic"
LOCUST_DEPLOY="locust-master"
LOCUST_SVC="locust"
LOCUST_PORT=8089
ACTIVE_STAGE="${ACTIVE_STAGE:-2}"

# ★ 실험 파라미터
USERS_LIST=(10 20 50 100 200)
SPAWN_RATE=10
RUN_DURATION=90
COOLDOWN=15

mkdir -p results

# ─────────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────────

cleanup() {
    echo "  🧹 Cleaning up port-forwarding..."
    pkill -f "port-forward.*$LOCUST_PORT" 2>/dev/null || true
    if command -v taskkill &> /dev/null; then
        taskkill //F //IM kubectl.exe //T 2>/dev/null || true
    fi
}
trap cleanup EXIT

wait_rollout() {
    local deploy=$1
    echo "  ⏳ rollout wait: $deploy"
    kubectl rollout status deployment/"$deploy" -n "$NS" --timeout=180s
    sleep 5
}

locust_api() {
    curl -sf --retry 3 --retry-delay 2 "$@"
}

start_portforward() {
    echo "  🔍 Checking port $LOCUST_PORT..."
    pkill -f "port-forward.*$LOCUST_PORT" 2>/dev/null || true
    sleep 2

    echo "  📡 Starting new port-forward to $LOCUST_SVC..."
    kubectl port-forward -n "$NS" svc/"$LOCUST_SVC" "${LOCUST_PORT}:${LOCUST_PORT}" > /dev/null 2>&1 &
    
    echo -n "  ⏳ Waiting for port-forward readiness..."
    for i in {1..10}; do
        if curl -sf "http://localhost:${LOCUST_PORT}/stats/requests" &>/dev/null; then
            echo " [OK]"
            return 0
        fi
        echo -n "."
        sleep 2
    done
    echo " [FAIL] Could not connect to Locust API"
    return 1
}

# ─────────────────────────────────────────────
# 메인 루프
# ─────────────────────────────────────────────
# for PCT in 100 10 1; do
for PCT in 10 1 100; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [Coreset ${PCT}%]  VEC_INDEX_CORESET_PCT=${PCT}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # 1. ConfigMap 패치
    echo "  🔧 Updating config..."
    kubectl patch configmap anomaly-config -n "$NS" \
      --patch "{\"data\": {\"VEC_INDEX_CORESET_PCT\": \"${PCT}\"}}"

    # 2. 서비스 재시작 및 대기
    kubectl rollout restart deployment/anomaly-router deployment/anomaly-retriever -n "$NS"
    wait_rollout anomaly-router
    wait_rollout anomaly-retriever

    # 3. Locust 재설정
    kubectl set env deployment/"$LOCUST_DEPLOY" \
      ACTIVE_STAGE="$ACTIVE_STAGE" CORESET_PCT="$PCT" -n "$NS"
    wait_rollout "$LOCUST_DEPLOY"

    # 4. Port-forward
    start_portforward

    # 5. users 단계별 실행
    for USERS in "${USERS_LIST[@]}"; do
        echo ""
        echo "  ▶ users=${USERS}  spawn_rate=${SPAWN_RATE}  duration=${RUN_DURATION}s"

        locust_api -X POST "http://localhost:${LOCUST_PORT}/swarm" \
          -d "user_count=${USERS}&spawn_rate=${SPAWN_RATE}" > /dev/null

        echo "    ⏱  running ${RUN_DURATION}s ..."
        ELAPSED=0
        POLL_INTERVAL=15
        while [[ $ELAPSED -lt $RUN_DURATION ]]; do
            sleep "$POLL_INTERVAL"
            ELAPSED=$(( ELAPSED + POLL_INTERVAL ))

            LIVE=$(curl -sf --max-time 5 "http://localhost:${LOCUST_PORT}/stats/requests" 2>/dev/null)
            if [[ -n "$LIVE" ]]; then
                # 파이썬 파싱 로직 안정화
                STAT=$(echo "$LIVE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    total = data.get('stats_total') or (data.get('stats') or [{}])[-1]
    rps  = round(data.get('total_rps') or total.get('current_rps') or 0, 1)
    p99  = int(total.get('response_time_percentile_0.99') or total.get('response_time_percentile_99') or 0)
    fail = total.get('num_failures') or 0
    print(f'{rps} {p99} {fail}')
except:
    print('0 0 0')
" 2>/dev/null || echo "0 0 0")
                read RPS P99 FAIL <<< "$STAT"
                printf "    [%3ds/%ds]  rps=%-6s  p99=%-6s ms  fail=%s\n" \
                    "$ELAPSED" "$RUN_DURATION" "$RPS" "$P99" "$FAIL"
            fi
        done

        # 통계 저장
        OUT="results/coreset${PCT}_users${USERS}.json"
        locust_api "http://localhost:${LOCUST_PORT}/stats/requests" > "$OUT"
        echo "    💾 saved → ${OUT}"

        # 테스트 중지
        locust_api "http://localhost:${LOCUST_PORT}/stop" > /dev/null
        sleep "$COOLDOWN"
    done

    # 6. JSON 덤프 수집 (Pod 내부 파일 로컬 복사)
    LOCUST_POD=$(kubectl get pod -n "$NS" -l app=locust,role=master -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    if [[ -n "$LOCUST_POD" ]]; then
        DUMP_FILE="results/coreset${PCT}_result.json"
        kubectl cp "${NS}/${LOCUST_POD}:/tmp/locust_result_coreset${PCT}.json" "$DUMP_FILE" 2>/dev/null && echo "  💾 dump saved → $DUMP_FILE" || echo "  ⚠️ dump not found"
    fi

    cleanup
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  All runs complete. Results are in the 'results/' folder."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"