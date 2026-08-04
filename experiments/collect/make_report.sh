#!/usr/bin/env bash
# =====================================================================
# make_report.sh — 수집 → 리포트 → GCS 동기화 (한 번에)
# =====================================================================
#   bash experiments/collect/make_report.sh
#   bash experiments/collect/make_report.sh --include-suspect
#   bash experiments/collect/make_report.sh --no-sync
#
# 클러스터를 건드리지 않는다. 로컬 results/ 만 읽는다.
# =====================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
RENDERED="${RENDERED:-$ROOT/rendered}"

# lib.sh 로 PY / CLOUDSDK_PYTHON / PYTHONUTF8 / _gs 를 재사용
[[ -f "$RENDERED/serialization.env" ]] && source "$RENDERED/serialization.env"
# ★ 날짜 스코프. --date 로 과거 실험도 다시 리포트할 수 있다.
#   지정이 없으면 results/ 아래 가장 최근 날짜 폴더를 자동 선택한다.
RUN_DATE="${RUN_DATE:-}"; _prev=""
for _a in "$@"; do case "$_prev" in --date) RUN_DATE="$_a";; esac; _prev="$_a"; done
if [[ -z "$RUN_DATE" ]]; then
  RUN_DATE=$(ls -1 "$ROOT/results" 2>/dev/null | grep -E "^[0-9]{4}$" | sort | tail -1)
fi
RUN_DATE="${RUN_DATE:-$(date +%m%d)}"
RESULTS="${RESULTS:-$ROOT/results/$RUN_DATE}"
GCS_ROOT="${GCS_ROOT%/}/$RUN_DATE"
source "$ROOT/runner/lib.sh"

SYNC=1; INCLUDE_SUSPECT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-sync)         SYNC=0 ;;
    --date)            shift ;;                 # 위에서 이미 파싱됨
    --include-suspect) INCLUDE_SUSPECT=1 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
  shift
done

[[ -d "$RESULTS/run" ]] || { err "결과 없음: $RESULTS/run"; exit 1; }

log "── collect ── (date=$RUN_DATE)"
# ★ 빈 배열 "${EXTRA[@]:-}" 는 bash 에서 빈 문자열 인자를 만들어
#   argparse 가 "unrecognized arguments" 로 죽는다. 플래그로 분기한다.
if (( INCLUDE_SUSPECT )); then
  "$PY" "$HERE/collect_runs.py" --results "$RESULTS" --include-suspect || exit 1
else
  "$PY" "$HERE/collect_runs.py" --results "$RESULTS" || exit 1
fi

# ★ Prometheus 스냅샷 (접속 불가하면 건너뛴다 — 리포트는 계속 생성)
if [[ "${SKIP_PROM:-0}" != "1" ]]; then
  log "── prometheus snapshot ──"
  "$PY" "$HERE/prom_snapshot.py" --results "$RESULTS" || \
    warn "prom 스냅샷 실패 — page fault/MCP/TTFT 표가 비게 된다 (SKIP_PROM=1 로 생략 가능)"
fi

log "── report ──"
"$PY" "$HERE/sweep_report.py" --rollup "$RESULTS/_rollup" || exit 1

if (( SYNC )) && [[ -n "${GCS_ROOT:-}" ]]; then
  log "── GCS 동기화 ──"
  sync_experiments "$ROOT"
fi

ok "완료 → $RESULTS/_rollup/results.md"
echo
echo "  표      : $RESULTS/_rollup/tables/*.md   (README 에 그대로 붙여넣기)"
echo "  그래프  : $RESULTS/_rollup/plots/*.png"
echo "  raw     : $RESULTS/_rollup/runs_wide.csv  ·  runs_long.csv"
