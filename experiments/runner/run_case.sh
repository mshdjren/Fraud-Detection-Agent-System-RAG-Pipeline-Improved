#!/usr/bin/env bash
# =====================================================================
# run_case.sh — 단일 run 1건 실행 (INGEST 또는 RUN)
# =====================================================================
#   bash experiments/runner/run_case.sh ingest <index_id>
#   bash experiments/runner/run_case.sh run    <run_id>
#
# 한 번에 정확히 하나만 실행하고, 끝날 때까지 대기한다.
# run_matrix.sh 가 이 스크립트를 순차 호출하는 구조.
# =====================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RENDERED="${RENDERED:-$HERE/../rendered}"

# shellcheck disable=SC1091
source "$RENDERED/serialization.env"
# ★ 날짜별 분리 — results/<MMDD>/ 와 gs://.../experiments/<MMDD>/
#   같은 축을 여러 번(n=20 디버그 → n=200 본측정) 돌려도 섞이지 않는다.
RUN_DATE="${RUN_DATE:-$(date +%m%d)}"
RESULTS="${RESULTS:-$HERE/../results/$RUN_DATE}"
GCS_ROOT="${GCS_ROOT%/}/$RUN_DATE"
# shellcheck disable=SC1091
source "$HERE/lib.sh"

KIND="${1:?usage: run_case.sh <ingest|run> <id>}"
ID="${2:?usage: run_case.sh <ingest|run> <id>}"

# =====================================================================
# A. INGEST — 인덱스 적재
# =====================================================================
if [[ "$KIND" == "ingest" ]]; then
  D="$RENDERED/index/$ID"
  for f in manifest.json config.patch.json ingest.yaml; do
    [[ -f "$D/$f" ]] || { err "렌더 결과 없음: $D/$f  → expand_matrix.py 재실행"; exit 1; }
  done

  log "═══ INGEST  index=$ID ═══"
  patch_config anomaly-config "$D/config.patch.json" || exit 1

  OUT="$RESULTS/index/$ID"; mkdir -p "$OUT"
  cp "$D/manifest.json" "$OUT/"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/started_at.txt"

  run_job "$D/ingest.yaml" "ingest-$ID" "$INGEST_TIMEOUT_S" "$OUT/ingest.log"
  rc=$?
  date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/finished_at.txt"
  (( rc != 0 )) && { err "ingest 실패 — 이 인덱스를 쓰는 run 은 건너뛰어야 함"; exit $rc; }

  # ★ index_stats 는 Job 로그의 "[stats] <index>: disk=..MB ws=..MB docs=.. segs=.."
  #   라인에서 파싱한다. Completed pod 는 exec 불가라 kubectl cp 를 쓸 수 없다.
  "$PY" - "$OUT/ingest.log" "$OUT/index_stats.jsonl" <<'PYEOF'
import json, re, sys, pathlib
log = pathlib.Path(sys.argv[1])
if not log.exists():
    sys.exit(0)
pat = re.compile(r"\[stats\]\s+(\S+):\s+disk=([\d.]+)MB\s+ws=([\d.]+)MB\s+docs=([\d,]+)\s+segs=(\d+)")
rows = []
for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
    m = pat.search(line)
    if m:
        rows.append({"index": m.group(1),
                     "store_mb": float(m.group(2)),
                     "vector_working_set_mb": float(m.group(3)),
                     "docs": int(m.group(4).replace(",", "")),
                     "segments": int(m.group(5))})
pathlib.Path(sys.argv[2]).write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""),
    encoding="utf-8")
print(f"  index_stats: {len(rows)} rows")
PYEOF

  # ES 쪽 실측 (store size / segments) — 코드 경로와 무관한 교차 검증.
  # ★ es-tree-ingest 이미지엔 curl 이 없다. python + elasticsearch 클라이언트로 조회.
  _ES_SNIPPET='import os,json
from elasticsearch import Elasticsearch
c=Elasticsearch([os.environ["ES_URL"]],verify_certs=False,ssl_show_warn=False,
                basic_auth=("elastic",os.environ["PASSWORD"]))
# ES 8.x 클라이언트는 ListApiResponse 를 돌려준다 → json.dumps 불가.
# .body 로 꺼내고, 없으면 list() 로 감싼다.
r=c.cat.indices(index="fraud_ecom_*",format="json",bytes="b",
                h="index,docs.count,store.size,pri,rep")
print(json.dumps(list(getattr(r,"body",r))))'
  if kubectl exec -n "$NS" deploy/manage-indices -- python -c "$_ES_SNIPPET" \
       > "$OUT/cat_indices.json" 2>"$OUT/cat_indices.err"; then
    ok "cat_indices 수집 ($(wc -c < "$OUT/cat_indices.json") bytes)"
    rm -f "$OUT/cat_indices.err"
  else
    warn "cat_indices 실패: $(tail -2 "$OUT/cat_indices.err" 2>/dev/null)"
    rm -f "$OUT/cat_indices.json"
  fi

  # index_stats 행 수 = CORESET_PERCENTAGES 개수여야 정상
  _EXP_N=$("$PY" -c "import io,json,sys;print(len([x for x in json.load(io.open(sys.argv[1],encoding='utf-8'))['data'].get('CORESET_PERCENTAGES','').split(',') if x.strip()]))" "$D/config.patch.json" 2>/dev/null || echo "?")
  _GOT_N=$(wc -l < "$OUT/index_stats.jsonl" 2>/dev/null | tr -d ' ' || echo 0)
  if [[ "$_EXP_N" != "?" && "$_GOT_N" != "$_EXP_N" ]]; then
    warn "index_stats 행 수 불일치: got=$_GOT_N expected=$_EXP_N (ingest 로그 확인)"
  else
    ok "index_stats $_GOT_N rows (CORESET_PERCENTAGES 개수와 일치)"
  fi

  upload_run "$OUT" "$GCS_ROOT/$EXPERIMENT_CASE/stage$STAGE/_index/$ID"
  [[ "${SYNC_EXPERIMENTS:-0}" == "1" ]] && sync_experiments
  ok "═══ INGEST 완료: $ID ═══"
  exit 0
fi

# =====================================================================
# B. RUN — 서빙 config 전환 + eval Job
# =====================================================================
D="$RENDERED/run/$ID"
# ★ 디렉터리 존재만으로는 부족하다. 필요한 4개 파일을 모두 확인한다.
#   (bash 내장 테스트는 MSYS 경로를 이해하지만 외부 프로그램은 아니므로,
#    여기서 통과해도 뒤에서 깨지는 사고를 이 검사로 막는다.)
for f in manifest.json config.patch.json retriever.patch.json eval.yaml; do
  [[ -f "$D/$f" ]] || { err "렌더 결과 없음: $D/$f  → expand_matrix.py 재실행"; exit 1; }
done

_mf() { "$PY" -c "import io,json,sys;print(json.load(io.open(sys.argv[1],encoding='utf-8'))[sys.argv[2]])" "$1" "$2" 2>/dev/null; }
EVAL_KIND=$(_mf "$D/manifest.json" eval)
CONC=$(_mf      "$D/manifest.json" concurrency)
if [[ -z "$EVAL_KIND" || -z "$CONC" ]]; then
  err "manifest.json 파싱 실패 (PY=$PY, $("$PY" -V 2>&1))"
  err "  Windows 라면: set PY=python  +  set PYTHONUTF8=1"
  exit 1
fi

log "═══ RUN  $ID   (eval=$EVAL_KIND, concurrency=$CONC) ═══"

OUT="$RESULTS/run/$ID"; mkdir -p "$OUT"
cp "$D/manifest.json" "$OUT/"

# ── 1) config 전환 ────────────────────────────────────────────────────
# ── 1) 재시작 대상 산정 (★ patch 이전에 해야 한다) ──────────────────
#   patch 를 먼저 적용하면 "현재 == patch" 가 되어 변경 키가 항상 0 이 되고,
#   그러면 아무것도 재시작하지 않아 새 config 가 서비스에 반영되지 않는다.
TARGETS=$(restart_targets "$D/config.patch.json")
[[ " $TARGETS " == *" anomaly-retriever "* ]] || TARGETS="$TARGETS anomaly-retriever"
[[ -f "$D/controller.patch.json" ]] && TARGETS="$TARGETS anomaly-controller"
TARGETS=$(echo "$TARGETS" | tr -s ' ' | sed 's/^ //;s/ $//')

# ── 1-a) config 전환 ─────────────────────────────────────────────────
patch_config anomaly-config           "$D/config.patch.json"    || exit 1
patch_config anomaly-retriever-config "$D/retriever.patch.json" || exit 1
if [[ -f "$D/controller.patch.json" ]]; then
  patch_config anomaly-controller-config "$D/controller.patch.json" || exit 1
fi

# ── 1-b) bucketization run 사전 검증 ────────────────────────────────
# ENABLE_BUCKETIZATION=true 인데 percolator _meta 에 bucket_config 가 없으면
# router.py 가 기동 중 raise → CrashLoop → 10분 rollout timeout 낭비.
_BUCKET=$("$PY" -c "import io,json,sys;print(json.load(io.open(sys.argv[1],encoding='utf-8'))['data'].get('ENABLE_BUCKETIZATION',''))" "$D/config.patch.json" 2>/dev/null)
if [[ "$_BUCKET" == "true" ]]; then
  _PV=$("$PY" -c "import io,json,sys;v=json.load(io.open(sys.argv[1],encoding='utf-8'))['data'].get('PERCOLATE_QUERY_VERSION','v1');print({'v14':'v1','v11a':'v11','v11b':'v11','v11c':'v11'}.get(v,v))" "$D/config.patch.json")
  if ! check_bucket_config "$EXPERIMENT_CASE" "$_PV"; then
    err "═══ RUN 중단: $ID (bucket_config 미비) ═══"; exit 1
  fi
fi

# ── 2) 서빙 재기동 (바뀐 키가 영향 주는 것만) + 헬스 게이트 ─────────
# ★ [STEP1-A] 매 run 전체 재시작 → 선택적 재시작.
#   tier1/2 는 analyzer 키를 건드리지 않으므로 local-analyzer(3분)가 대부분 생략된다.
#   FORCE_RESTART=1 로 전체 재시작을 강제할 수 있다(코드 재배포 직후 등).
if [[ -z "$TARGETS" ]]; then
  ok "재시작 불필요 — config 변경 없음"
else
  log "재시작 대상: $TARGETS"
  # shellcheck disable=SC2086
  restart_and_wait $TARGETS || exit 1
  for _d in $TARGETS; do
    case "$_d" in
      local-analyzer)      wait_endpoint local-analyzer     8080 /health "$ROLLOUT_TIMEOUT_S" || exit 1 ;;
      anomaly-retriever)   wait_endpoint anomaly-retriever  8001 /ready  "$ROLLOUT_TIMEOUT_S" || exit 1 ;;
      anomaly-router)      wait_endpoint anomaly-router     8000 /ready  "$ROLLOUT_TIMEOUT_S" || exit 1 ;;
    esac
  done
fi

# ── 3) warmup (직렬 1건씩) ───────────────────────────────────────────
warmup

# ── 4) 오염 감지 스냅샷 ──────────────────────────────────────────────
BEFORE=$(snapshot_restarts)
log "restarts snapshot(before) = $BEFORE"
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/started_at.txt"

# ── 5) eval Job — 이 안에서 AGENT_CONCURRENCY 가 적용된다 ────────────
run_job "$D/eval.yaml" "$ID" "$EVAL_TIMEOUT_S" "$OUT/eval.log"
rc=$?
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/finished_at.txt"

# ── 6) 오염 판정 ─────────────────────────────────────────────────────
SUSPECT=false
assert_no_restart "$BEFORE" || SUSPECT=true
"$PY" - "$OUT/manifest.json" "$SUSPECT" "$rc" <<'PYSTAT'
import io, json, sys
p, suspect, rc = sys.argv[1], sys.argv[2] == "true", int(sys.argv[3])
m = json.load(io.open(p, encoding="utf-8"))
m["suspect_restart"] = suspect
m["exit_code"] = rc
m["status"] = "ok" if (rc == 0 and not suspect) else ("suspect" if rc == 0 else "failed")
io.open(p, "w", encoding="utf-8").write(json.dumps(m, indent=2, ensure_ascii=False))
print("  status = " + m["status"])
PYSTAT

# ── 7) 산출물 수집 ───────────────────────────────────────────────────
# ★ eval Job 이 스스로 GCS 에 업로드하므로 거기서 받아온다.
#   (kubectl cp 는 Completed pod 에서 동작하지 않음)
fetch_gcs "$EVAL_GCS_BASE/$ID" "$OUT" agent_eval_report.json agent_predictions.json \
  || warn "GCS 산출물 없음 — Job 로그($OUT/eval.log)의 '[GCS] uploaded' 확인"

# Prometheus 스냅샷용 시간창을 남겨둔다 (stage 2 의 prom_snapshot.py 입력)
"$PY" - "$OUT" <<'PYWIN'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
w = {}
for k, f in (("from", "started_at.txt"), ("to", "finished_at.txt")):
    p = d / f
    if p.exists():
        w[k] = p.read_text(encoding="utf-8").strip()
(d / "window.json").write_text(json.dumps(w, indent=2), encoding="utf-8")
PYWIN

upload_run "$OUT" "$GCS_ROOT/$EXPERIMENT_CASE/stage$STAGE/$ID"
[[ "${SYNC_EXPERIMENTS:-0}" == "1" ]] && sync_experiments

# ── 8) cooldown ──────────────────────────────────────────────────────
log "cooldown ${COOLDOWN_S}s"
sleep "$COOLDOWN_S"

ok "═══ RUN 완료: $ID (rc=$rc, suspect=$SUSPECT) ═══"
exit $rc
