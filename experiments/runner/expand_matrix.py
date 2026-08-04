#!/usr/bin/env python3
"""
expand_matrix.py — tier1.yaml → 실행 계획 + ConfigMap patch + Job YAML 렌더링
=============================================================================
출력 (모두 experiments/rendered/ 아래, gitignore 대상):

  rendered/plan.tsv                      실행 순서 (run_matrix.sh 가 읽음)
  rendered/index/<index_id>/
      config.patch.json                  ingest 용 anomaly-config patch
      ingest.yaml                        01-ingest.yaml 의 Job name 만 교체
      manifest.json
  rendered/run/<run_id>/
      config.patch.json                  serving 용 anomaly-config patch
      retriever.patch.json               EXPERIMENT_LABEL = run_id
      eval.yaml                          agent-eval-batch.yaml 의 name/env 교체
      manifest.json                      축 값 + 원본 매니페스트 해시

설계 원칙:
  · 기존 배포 YAML 을 수정하지 않는다. metadata.name 과 일부 env 만 갈아끼운다.
  · run_id 는 ConfigMap / Job name / EXPERIMENT_LABEL / GCS 경로에
    동일 문자열로 쓰여 결과 join key 가 된다.

사용:
  python experiments/runner/expand_matrix.py \
      --case experiments/cases/tier1.yaml --stage 1 --repo-root .
  python ... --stage 1 --only C_quant        # 특정 그룹만
  python ... --stage 1 --print               # 계획만 보고 렌더링 안 함
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML 필요:  pip install pyyaml")

# _AXIS_KEYS 근처에 추가
_CONTROLLER_ONLY_KEYS = {
    "TOOL_GATE_ENABLE", "PATCHCORE_ON_RESEARCH", "PATCHCORE_TAU",
    "PATCHCORE_REWEIGHT_MODE", "EVALUATOR_ONCE", "AGENT_MAX_ROUNDS",
    "SKIP_ANALYZER_ON_CONFIDENT", "EXPAND_PLATEAU_TAU",
    "REROUTE_ON_CENTROID", "REROUTE_SCORE_RATIO",
    # ★ 양쪽에 중복 존재 — 반드시 동시 patch 해야 갈라지지 않는다
    "CLS_ND_THRESHOLD", "SUFFICIENCY_BOUNDARY_LOW", "SUFFICIENCY_BOUNDARY_HIGH",
}

# ============================================================
# 1) 로드 & 검증
# ============================================================

def load_case(path: Path) -> dict:
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("defaults", "indices", "groups", "serialization"):
        if key not in spec:
            sys.exit(f"[case] 필수 키 누락: {key}")
    ids = [i["id"] for i in spec["indices"]]
    if len(ids) != len(set(ids)):
        sys.exit("[case] indices.id 중복")
    # 그룹이 참조하는 index 가 실재하는지
    for g in spec["groups"]:
        for v in g["vary"]:
            idx = v.get("index", g.get("index"))
            if idx is None:
                sys.exit(f"[case] group {g['id']} / vary {v['id']}: index 미지정")
            if idx not in ids:
                sys.exit(f"[case] group {g['id']}: 알 수 없는 index '{idx}'")
    return spec


def _sha8(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:8] if p.exists() else "missing"


# ============================================================
# 2) 계획 전개
# ============================================================

def expand(spec: dict, stage: int, only: list[str] | None) -> tuple[list[dict], list[dict]]:
    """(index 작업 목록, run 목록) 반환. 둘 다 실행 순서대로 정렬돼 있다."""
    case = spec["experiment_case"]
    defaults = spec["defaults"]
    index_by_id = {i["id"]: i for i in spec["indices"]}

    runs: list[dict] = []
    needed_index_ids: list[str] = []

    for g in spec["groups"]:
        if int(g.get("stage", 1)) > stage:
            continue
        if only and g["id"] not in only:
            continue
        for v in g["vary"]:
            index_id = v.get("index", g.get("index"))
            cfg = dict(defaults)
            # ① 인덱스가 고정하는 값 (VEC_QUANT / HNSW_* ) 을 먼저 반영해
            #    serving 이 참조하는 인덱스명 suffix 가 ingest 와 일치하게 한다.
            for k in ("VEC_QUANT", "HNSW_M", "HNSW_EF_CONSTRUCTION"):
                if k in index_by_id[index_id]["config"]:
                    cfg[k] = index_by_id[index_id]["config"][k]
            # ② 그룹/variant override
            cfg.update(g.get("config", {}))
            cfg.update(v.get("config", {}))

            eval_kind = g.get("eval", "agent")
            # ★ SKIP_ANALYZER 는 eval 라벨에서 파생하지 않는다.
            #   이 플래그는 router percolate 스킵 + retriever cluster filter
            #   제거(=전역 벡터검색)를 의미하며, LLM 호출 여부와 무관하다.
            _skip = v.get("skip_analyzer", g.get("skip_analyzer", False))
            cfg["SKIP_ANALYZER"] = "true" if _skip else "false"

            run_id = f"{g['id'].lower()}-{v['id']}".replace("_", "-")
            cfg["RUN_ID"] = run_id

            runs.append({
                "inline_index": bool(index_by_id[index_id].get("inline", False)),
                "run_id":      run_id,
                "group":       g["id"],
                "variant":     v["id"],
                "stage":       int(g.get("stage", 1)),
                "index_id":    index_id,
                "eval":        eval_kind,
                "n_samples":   int(g.get("n_samples", 60)),
                "concurrency": int(g.get("concurrency", 1)),
                "note":        g.get("note", ""),
                "experiment_case": case,
                "config":      cfg,
            })
            if index_id not in needed_index_ids:
                needed_index_ids.append(index_id)

    # 인덱스 작업 — percolator 를 만드는 인덱스를 항상 맨 앞으로.
    # ★ inline: true 인 인덱스는 "매 run 직전 재적재"가 목적이므로
    #   앞단 일괄 적재 목록에서 빼고, plan.tsv 에서 각 run 바로 앞에 끼워 넣는다.
    #   (percolator 샤드 sweep 처럼 같은 인덱스명을 설정만 바꿔 다시 만들 때 필요)
    indices = [index_by_id[i] for i in needed_index_ids]
    indices.sort(key=lambda i: (not i.get("ingest_percolator", False), i["id"]))

    idx_jobs = []
    for i in indices:
        cfg = dict(defaults)
        cfg.update(i["config"])
        cfg["RUN_ID"] = f"ingest-{i['id']}"
        # percolator 를 만들지 않는 인덱스는 percolate 적재를 건너뛴다
        if not i.get("ingest_percolator", False):
            cfg["VERSIONS"] = ""          # ingest 파이프라인에서 percolator 스킵
        idx_jobs.append({
            "index_id": i["id"],
            "inline":   bool(i.get("inline", False)),
            "stage":    int(i.get("stage", 1)),
            "ingest_percolator": bool(i.get("ingest_percolator", False)),
            "experiment_case": case,
            "config":   cfg,
        })
    return idx_jobs, runs


# ============================================================
# 3) 렌더링 — 기존 배포 YAML 의 name/env 만 교체
# ============================================================

def _load_docs(path: Path) -> list:
    return [d for d in yaml.safe_load_all(path.read_text(encoding="utf-8")) if d]


def _set_env(container: dict, name: str, value: str) -> None:
    """env 리스트에서 name 을 찾아 value 로 교체(없으면 추가). valueFrom 은 제거."""
    envs = container.setdefault("env", [])
    for e in envs:
        if e.get("name") == name:
            e.pop("valueFrom", None)
            e["value"] = str(value)
            return
    envs.append({"name": name, "value": str(value)})


def render_job(src: Path, job_name: str, env_overrides: dict, out: Path) -> None:
    docs = _load_docs(src)
    n_job = 0
    for d in docs:
        if d.get("kind") != "Job":
            continue
        n_job += 1
        d["metadata"]["name"] = job_name
        # ttl 을 넉넉히 — 실패한 run 을 나중에 조사할 수 있어야 한다
        d["spec"].setdefault("ttlSecondsAfterFinished", 604800)
        d["spec"]["backoffLimit"] = 0          # ★ 재시도 금지: 1 run = 1 실행
        for c in d["spec"]["template"]["spec"]["containers"]:
            # ★ 원본 매니페스트의 중복 env 제거 (뒤에 온 정의가 이긴다).
            #   kubectl 이 "hides previous definition of X" 경고를 내던 부분.
            seen, dedup = {}, []
            for e in c.get("env", []):
                seen[e.get("name")] = e
            for e in c.get("env", []):
                if seen.get(e.get("name")) is e:
                    dedup.append(e)
            c["env"] = dedup
            for k, v in env_overrides.items():
                _set_env(c, k, v)
    if n_job != 1:
        sys.exit(f"[render] {src}: kind=Job 문서가 {n_job}개 (1개여야 함)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump_all(docs, sort_keys=False, allow_unicode=True),
                   encoding="utf-8")


# ============================================================
# 4) main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="experiments/cases/tier1.yaml")
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--only", default="", help="쉼표구분 group id 필터")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--out", default="experiments/rendered")
    ap.add_argument("--print", dest="dry", action="store_true", help="계획만 출력")
    ap.add_argument("--date", default="",
                    help="실험 묶음 날짜 (기본: 오늘, MMDD). 결과는 "
                         "results/<date>/ 와 gs://.../experiments/<date>/ 로 분리된다")
    args = ap.parse_args()

    import datetime as _dt
    run_date = args.date or _dt.datetime.now().strftime("%m%d")

    root = Path(args.repo_root).resolve()
    spec = load_case(Path(args.case))
    only = [x.strip() for x in args.only.split(",") if x.strip()] or None
    idx_jobs, runs = expand(spec, args.stage, only)

    # ── 계획 출력 ────────────────────────────────────────────
    print(f"\n[plan] case={spec['experiment_case']}  stage<={args.stage}"
          f"  date={run_date}  indices={len(idx_jobs)}  runs={len(runs)}")
    print(f"  {'#':>2}  {'kind':<8} {'id':<24} {'index':<10} {'eval':<10} {'n':>4}  note")
    print("  " + "-" * 88)
    for n, j in enumerate(idx_jobs, 1):
        _tag = "INGEST*" if j.get("inline") else "INGEST"
        print(f"  {n:>2}  {_tag:<8} {j['index_id']:<24} {'-':<10} {'-':<10} "
              f"{'-':>4}  percolator={j['ingest_percolator']}"
              f"{'  (run 직전 재적재)' if j.get('inline') else ''}")
    for n, r in enumerate(runs, 1):
        print(f"  {n:>2}  {'RUN':<8} {r['run_id']:<24} {r['index_id']:<10} "
              f"{r['eval']:<10} {r['n_samples']:>4}  {r['note'][:34]}")
    print()
    if args.dry:
        return

    out = Path(args.out)
    ing_src  = root / spec["manifests"]["ingest"]
    eval_src = root / spec["manifests"]["agent_eval"]
    for p in (ing_src, eval_src):
        if not p.exists():
            sys.exit(f"[render] 원본 매니페스트 없음: {p}  (--repo-root 확인)")

    src_hash = {"ingest": _sha8(ing_src), "agent_eval": _sha8(eval_src)}

    # ── 인덱스 작업 렌더 ─────────────────────────────────────
    for j in idx_jobs:
        d = out / "index" / j["index_id"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.patch.json").write_text(
            json.dumps({"data": {k: str(v) for k, v in j["config"].items()}},
                       indent=2, ensure_ascii=False), encoding="utf-8")
        render_job(ing_src, f"ingest-{j['index_id']}", {}, d / "ingest.yaml")
        (d / "manifest.json").write_text(
            json.dumps({**j, "src_hash": src_hash}, indent=2, ensure_ascii=False),
            encoding="utf-8")

    # ── run 렌더 ─────────────────────────────────────────────
    for r in runs:
        d = out / "run" / r["run_id"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.patch.json").write_text(
            json.dumps({"data": {k: str(v) for k, v in r["config"].items()}},
                       indent=2, ensure_ascii=False), encoding="utf-8")
        (d / "retriever.patch.json").write_text(
            json.dumps({"data": {"EXPERIMENT_LABEL": r["run_id"]}},
                       indent=2, ensure_ascii=False), encoding="utf-8")
        (d / "retriever.patch.json").write_text(
            json.dumps({"data": {"EXPERIMENT_LABEL": r["run_id"]}},
                       indent=2, ensure_ascii=False), encoding="utf-8")

        _ctrl = {k: str(v) for k, v in r["config"].items() if k in _CONTROLLER_ONLY_KEYS}
        if _ctrl:
            (d / "controller.patch.json").write_text(
                json.dumps({"data": _ctrl}, indent=2, ensure_ascii=False), encoding="utf-8")
        render_job(eval_src, r["run_id"], {
            "RUN_NAME":          r["run_id"],
            "RUN_ID":            r["run_id"],
            "N_SAMPLES":         r["n_samples"],
            "AGENT_CONCURRENCY": r["concurrency"],   # ★ free-tier 직렬화
            "EVAL_SOURCE":       r["config"].get("EVAL_SOURCE", "smoke"),
        }, d / "eval.yaml")
        (d / "manifest.json").write_text(
            json.dumps({**r, "src_hash": src_hash}, indent=2, ensure_ascii=False),
            encoding="utf-8")

    # ── 실행 계획 TSV ────────────────────────────────────────
    lines = ["kind\tid\tindex_id\teval\tn_samples\tconcurrency"]
    _inline = {j["index_id"] for j in idx_jobs if j.get("inline")}
    for j in idx_jobs:
        if j["index_id"] in _inline:
            continue                     # run 직전에 끼워 넣는다
        lines.append(f"INGEST\t{j['index_id']}\t{j['index_id']}\t-\t-\t-")
    for r in runs:
        if r["index_id"] in _inline:      # ★ 이 run 전용 재적재
            lines.append(f"INGEST\t{r['index_id']}\t{r['index_id']}\t-\t-\t-")
        lines.append(f"RUN\t{r['run_id']}\t{r['index_id']}\t{r['eval']}"
                     f"\t{r['n_samples']}\t{r['concurrency']}")
    (out / "plan.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # serialization 설정을 shell 이 읽기 쉬운 형태로
    ser = spec["serialization"]
    (out / "serialization.env").write_text(
        "\n".join([
            f"COOLDOWN_S={ser['cooldown_s']}",
            f"ROLLOUT_TIMEOUT_S={ser['rollout_timeout_s']}",
            f"INGEST_TIMEOUT_S={ser['ingest_timeout_s']}",
            f"EVAL_TIMEOUT_S={ser['eval_timeout_s']}",
            f"WARMUP_REQUESTS={ser['warmup_requests']}",
            f"WATCH_DEPLOYMENTS=\"{' '.join(ser['watch_deployments'])}\"",
            f"NAMESPACE={spec['namespace']}",
            f"GCS_ROOT={spec['gcs_root']}",
            f"EVAL_GCS_BASE={spec.get('eval_gcs_base', 'gs://fraud_ecomm/evaluation/agent-eval')}",
            f"EXPERIMENT_CASE={spec['experiment_case']}",
            f"STAGE={args.stage}",
            f"RUN_DATE={run_date}",
        ]) + "\n", encoding="utf-8")

    print(f"[render] → {out}/  (plan.tsv, index/*, run/*)")


if __name__ == "__main__":
    main()
