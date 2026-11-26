"""Run the A1 agent on a set of tasks and log everything.

Baseline (spec Task 3): stateless client, prompt cache off, all tool schemas, no grammar.

  python -m harness.run --bfcl-root third_party/gorilla/berkeley-function-call-leaderboard \
      --split splits/tasks_TradingBot_MessageAPI.json --which test --out results/harness_baseline

Add the custom long tasks with --extra-tasks data/long_tasks.jsonl. With --split,
only the long tasks assigned to the chosen group (extra_test / extra_trace) run.

Outputs in --out: metadata.json, requests.jsonl (one line per model request),
sessions.jsonl (one line per task). Summarise with `python -m harness.summarize <out>`.

At the end, the run is scored with BFCL's official checker (harness.official_score):
official_scores.jsonl (per task: session/turn results, expected tool calls matched, made with
wrong arguments, missed, and extra calls) and bfcl_scores.json (run totals). The scores are
also written to the log. Skip with --no-score.
"""

from __future__ import annotations

import logging
from .debug_log import configure_logging, trace

import argparse
import os
import json
import platform
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from .bfcl_env import BFCL, Task
from .llm_client import LlamaServer
from .official_score import save as save_scores, score_sessions
from .pi_monitor import Monitor
from .runner import jsonl_writer, run_session
from .task_split import TaskSplitError, select_task_ids

log = logging.getLogger(__name__)

# Qwen3 model card: thinking T=0.6, top-p 0.95, top-k 20; non-thinking T=0.7, top-p 0.8, top-k 20 (min-p 0).
# max_tokens: thinking needs room to reason before the call; 512 was the old non-thinking budget.
QWEN3_SAMPLING = {
    "on": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 4096},
    "off": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_tokens": 512},
}


@trace
def load_extra_tasks(path: str) -> dict[str, Task]:
    """Our own long tasks: JSONL rows with id, question, initial_config, involved_classes, ground_truth."""
    out = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["id"] in out:
            raise TaskSplitError(f"duplicate custom task ID: {r['id']}")
        out[r["id"]] = Task(id=r["id"], turns=r["question"], initial_config=r["initial_config"],
                            involved_classes=r["involved_classes"], ground_truth=r["ground_truth"],
                            excluded_functions=r.get("excluded_function", []), source=path)
    return out


@trace
def select_ids(a, tasks: dict, extra: dict, classes: list[str]) -> list[str]:
    """Select exact BFCL/custom task IDs using the frozen split configuration.

    --ids explicitly overrides the split (for debugging only).
    With --split, every loaded custom task must have exactly one assignment.
    Without --split, preserve the existing full-subset run behavior.
    """
    if a.ids:
        ids = [i.strip() for i in a.ids.split(",")]
        if not all(ids) or len(ids) != len(set(ids)):
            raise TaskSplitError("--ids must contain unique, nonempty task IDs")
    elif a.split:
        sp = json.load(open(a.split, encoding="utf-8"))
        if sp["classes"] != classes:
            raise TaskSplitError(f"--classes {classes} does not match the split's {sp['classes']}")
        ids = select_task_ids(sp, a.which, list(extra) if a.extra_tasks else None)
    else:
        ids = sorted((t.id for t in tasks.values() if t.id not in extra and set(t.involved_classes) <= set(classes)),
                     key=lambda s: int(s.rsplit("_", 1)[1]))
        ids += list(extra)

    unknown = [i for i in ids if i not in tasks]
    if unknown:
        raise TaskSplitError(f"unknown task ids {unknown} (an extra task needs --extra-tasks)")
    return ids


@trace
def long_context_lists() -> dict:
    """Lengths of BFCL's long-context filler lists and the A1 cap, if applied (scripts/cap_long_context.py)."""
    import importlib
    L = importlib.import_module("bfcl_eval.eval_checker.multi_turn_eval.func_source_code.long_context")
    names = ["MA_5_EXTENSION", "MA_20_EXTENSION", "WATCH_LIST_EXTENSION", "TRANSACTION_HISTORY_EXTENSION"]
    out = {n: len(getattr(L, n)) for n in names}
    out["a1_list_cap"] = vars(L).get("_A1_LIST_CAP")
    out["a1_list_caps"] = vars(L).get("_A1_LIST_CAPS")
    return out


@trace
def git_rev(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


@trace
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bfcl-root", required=True)
    ap.add_argument("--classes", default="TradingBot,MessageAPI", help="the fixed tool subset")
    ap.add_argument("--split", help="file from harness.make_split")
    ap.add_argument("--which", default="test", choices=["test", "trace", "dev", "all"])
    ap.add_argument("--ids", help="comma-separated task ids (overrides --split)")
    ap.add_argument("--extra-tasks", help="JSONL of our own long tasks; with --split, only those in the chosen extra_test/extra_trace group run")
    ap.add_argument("--limit", type=int, help="run only the first N tasks")
    ap.add_argument("--max-turns", type=int, help="truncate each session to N user turns (Pi latency runs)")
    ap.add_argument("--server-url", default="http://127.0.0.1:8080")
    ap.add_argument("--launch-server", help="full llama-server command; harness starts it and measures cold start")
    ap.add_argument("--thinking", choices=["on", "off"], default="on",
                    help="Qwen3 thinking mode (baseline: on, TA guidance of Nov 16)")
    ap.add_argument("--temperature", type=float, help="default: Qwen3 preset for the chosen mode")
    ap.add_argument("--top-p", type=float, help="default: Qwen3 preset for the chosen mode")
    ap.add_argument("--top-k", type=int, help="default: Qwen3 preset for the chosen mode")
    ap.add_argument("--min-p", type=float, default=0.0, help="Qwen3 recommends 0 (llama-server's own default is 0.05)")
    ap.add_argument("--max-tokens", type=int, help="decode budget per request; default 4096 with thinking, 512 without")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed, recorded for reproducibility")
    ap.add_argument("--cache-prompt", action="store_true", help="allow prefix reuse (NOT for the baseline)")
    ap.add_argument("--system-prompt", choices=["none", "bfcl"], default="none")
    ap.add_argument("--log-level", choices=["DEBUG","INFO","WARNING","ERROR"], default=os.environ.get("A1_LOG_LEVEL","INFO"))
    ap.add_argument("--log-file", help="default: <out>/harness.log; - disables file")
    ap.add_argument("--no-score", action="store_true", help="skip BFCL scoring at the end of the run")
    ap.add_argument("--teacher-forcing", action="store_true",
                    help="gap experiment: start every turn from BFCL's correct history and app state, "
                         "and score each turn on its own (accuracy only; implies --no-score)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    # Qwen3's recommended sampling per mode (model card), used unless set explicitly
    preset = QWEN3_SAMPLING[a.thinking]
    for k in ("temperature", "top_p", "top_k", "max_tokens"):
        if getattr(a, k) is None:
            setattr(a, k, preset[k])

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    configure_logging(a.log_level, a.log_file if a.log_file is not None else out / 'harness.log')
    log.info('run_start mode=%s cache_prompt=%s', a.which, a.cache_prompt)
    classes = a.classes.split(",")
    bfcl = BFCL(a.bfcl_root)
    tasks = bfcl.load_tasks()

    extra = load_extra_tasks(a.extra_tasks) if a.extra_tasks else {}
    duplicates = set(tasks) & set(extra)
    if duplicates:
        sys.exit(f"custom task IDs collide with BFCL IDs: {sorted(duplicates)}")
    tasks.update(extra)
    log.info('tasks_loaded bfcl=%d custom=%d', len(tasks)-len(extra),len(extra))
    try:
        ids = select_ids(a, tasks, extra, classes)
    except (TaskSplitError, KeyError) as exc:
        sys.exit(f"task selection error: {exc}")
    if a.limit:
        ids = ids[: a.limit]
    log.info('tasks_selected count=%d group=%s',len(ids),a.which)
    for i in ids:
        if not set(tasks[i].involved_classes) <= set(classes):
            sys.exit(f"task {i} uses {tasks[i].involved_classes}, outside the tool subset {classes}")

    server = LlamaServer(a.server_url)
    proc, cold_start_s = None, None
    if a.launch_server:
        log.info('server_launch_start')
        t0 = time.monotonic()
        proc = subprocess.Popen(shlex.split(a.launch_server), stdout=open(out / "server.log", "w"),
                                stderr=subprocess.STDOUT)
        server.wait_ready(max_wait_s=1800)
        cold_start_s = time.monotonic() - t0
        log.info('server_ready cold_start_s=%.1f',cold_start_s)
    elif not server.health():
        log.error('server_unreachable')
        sys.exit(f"llama-server not reachable at {a.server_url} (start it, or use --launch-server)")

    try:
        with urllib.request.urlopen(a.server_url.rstrip("/") + "/props", timeout=10) as r:
            props = json.loads(r.read())
    except Exception as e:
        log.warning("server_props_unavailable type=%s",type(e).__name__)
        props = None

    cfg = {"temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k, "min_p": a.min_p,
           "max_tokens": a.max_tokens, "seed": a.seed, "thinking": a.thinking == "on",
           "cache_prompt": a.cache_prompt, "system_prompt": a.system_prompt, "max_turns": a.max_turns,
           "teacher_forcing": a.teacher_forcing}
    meta = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "argv": sys.argv, "config": cfg,
        "classes": classes, "n_tools": len(bfcl.tools_for(classes)), "task_ids": ids,
        "harness_commit": git_rev(Path(__file__).resolve().parent),
        "bfcl_commit": git_rev(bfcl.root), "host": platform.node(), "python": platform.python_version(),
        "server_props": props, "cold_start_server_ready_s": cold_start_s,
        "long_context_lists": long_context_lists(),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    log_req, f_req = jsonl_writer(out / "requests.jsonl")
    log_ses, f_ses = jsonl_writer(out / "sessions.jsonl")
    monitor = Monitor(server_pid=proc.pid if proc else None).start()
    try:
        for n, tid in enumerate(ids):
            log.info('task_start id=%s number=%d/%d',tid,n+1,len(ids))
            print(f"[{time.strftime('%H:%M:%S')}] {n + 1}/{len(ids)} {tid}", flush=True)
            s = run_session(tasks[tid], bfcl, server, classes, cfg, log_req, monitor)
            if n == 0 and cold_start_s is not None and s["warm_time_to_first_call_ms"] is not None:
                s["cold_time_to_first_call_ms"] = round(cold_start_s * 1000 + s["warm_time_to_first_call_ms"], 1)
            log_ses(s)
            log.info("task_finish id=%s success=%s passed=%d/%d duration_s=%.1f failure=%s",tid,s["session_success"],s["turns_passed"],s["n_turns_run"],s["session_wall_ms"]/1000,s["fail_reason"])
            print(f"    success={s['session_success']} turns_passed={s['turns_passed']}/{s['n_turns_run']} "
                  f"calls_valid={s['n_calls_valid']}/{s['n_calls']} wall={s['session_wall_ms'] / 1000:.0f}s",
                  flush=True)
    finally:
        log.info('cleanup_start')
        monitor.stop()
        f_req.close()
        f_ses.close()
        if proc:
            proc.terminate()

    if a.teacher_forcing and not a.no_score:
        # BFCL's checker replays the model's calls across the whole session, which does not
        # apply when every turn starts from the correct history. The harness's per-turn score is the result.
        log.info("official_score_skipped reason=teacher_forcing")
    elif not a.no_score:
        score_run_folder(out, bfcl, tasks)


@trace
def score_run_folder(out: Path, bfcl: BFCL, tasks: dict) -> None:
    """Score every session in `out` with BFCL's official checker; write and log the scores.

    A scoring error is logged and does not fail the run (the run's own files are already saved);
    the run can be scored again later with `python -m harness.official_score <out>`.
    """
    log.info("bfcl_score_start out=%s", out)
    try:
        summary = save_scores(out, score_sessions(out, bfcl, tasks))
    except Exception as e:
        log.error("bfcl_score_failed type=%s", type(e).__name__)
        return
    c = summary["tool_calls"]
    print(f"BFCL: session_success={summary['official_session_success_rate']} "
          f"per_turn_accuracy={summary['official_per_turn_accuracy']} "
          f"calls matched={c['matched']}/{c['expected']} wrong_args={c['wrong_args']} "
          f"missed={c['missed']} extra={c['extra']} -> {out / 'bfcl_scores.json'}", flush=True)


if __name__ == "__main__":
    main()
