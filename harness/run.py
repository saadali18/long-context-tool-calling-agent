"""Run the A1 agent on a set of tasks and log everything.

Baseline (spec Task 3): stateless client, prompt cache off, all tool schemas, no grammar.

  python -m harness.run --bfcl-root third_party/gorilla/berkeley-function-call-leaderboard \
      --split splits/tasks_TradingBot_MessageAPI.json --which test --out results/harness_baseline

Outputs in --out: metadata.json, requests.jsonl (one line per model request),
sessions.jsonl (one line per task). Summarise with `python -m harness.summarize <out>`.
"""

from __future__ import annotations

import argparse
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
from .pi_monitor import Monitor
from .runner import jsonl_writer, run_session


def load_extra_tasks(path: str) -> dict[str, Task]:
    """Our own long tasks: JSONL rows with id, question, initial_config, involved_classes, ground_truth."""
    out = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        out[r["id"]] = Task(id=r["id"], turns=r["question"], initial_config=r["initial_config"],
                            involved_classes=r["involved_classes"], ground_truth=r["ground_truth"],
                            excluded_functions=r.get("excluded_function", []), source=path)
    return out


def long_context_lists() -> dict:
    """Lengths of BFCL's long-context filler lists and the A1 cap, if applied (scripts/cap_long_context.py)."""
    import importlib
    L = importlib.import_module("bfcl_eval.eval_checker.multi_turn_eval.func_source_code.long_context")
    names = ["MA_5_EXTENSION", "MA_20_EXTENSION", "WATCH_LIST_EXTENSION", "TRANSACTION_HISTORY_EXTENSION"]
    out = {n: len(getattr(L, n)) for n in names}
    out["a1_list_cap"] = vars(L).get("_A1_LIST_CAP")
    return out


def git_rev(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bfcl-root", required=True)
    ap.add_argument("--classes", default="TradingBot,MessageAPI", help="the fixed tool subset")
    ap.add_argument("--split", help="file from harness.make_split")
    ap.add_argument("--which", default="test", choices=["test", "trace", "all"])
    ap.add_argument("--ids", help="comma-separated task ids (overrides --split)")
    ap.add_argument("--extra-tasks", help="JSONL of our own long tasks (added to the run)")
    ap.add_argument("--limit", type=int, help="run only the first N tasks")
    ap.add_argument("--max-turns", type=int, help="truncate each session to N user turns (Pi latency runs)")
    ap.add_argument("--server-url", default="http://127.0.0.1:8080")
    ap.add_argument("--launch-server", help="full llama-server command; harness starts it and measures cold start")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=512, help="decode budget per request")
    ap.add_argument("--thinking", choices=["on", "off"], default="off")
    ap.add_argument("--cache-prompt", action="store_true", help="allow prefix reuse (NOT for the baseline)")
    ap.add_argument("--system-prompt", choices=["none", "bfcl"], default="none")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    classes = a.classes.split(",")
    bfcl = BFCL(a.bfcl_root)
    tasks = bfcl.load_tasks()

    if a.ids:
        ids = a.ids.split(",")
    elif a.split:
        sp = json.load(open(a.split))
        if sp["classes"] != classes:
            sys.exit(f"--classes {classes} does not match the split's {sp['classes']}")
        ids = sp["test"] + sp["trace"] if a.which == "all" else sp[a.which]
    else:
        ids = sorted((t.id for t in tasks.values() if set(t.involved_classes) <= set(classes)),
                     key=lambda s: int(s.rsplit("_", 1)[1]))
    if a.extra_tasks:
        extra = load_extra_tasks(a.extra_tasks)
        tasks.update(extra)
        ids += list(extra)
    if a.limit:
        ids = ids[: a.limit]
    for i in ids:
        if not set(tasks[i].involved_classes) <= set(classes):
            sys.exit(f"task {i} uses {tasks[i].involved_classes}, outside the tool subset {classes}")

    server = LlamaServer(a.server_url)
    proc, cold_start_s = None, None
    if a.launch_server:
        t0 = time.monotonic()
        proc = subprocess.Popen(shlex.split(a.launch_server), stdout=open(out / "server.log", "w"),
                                stderr=subprocess.STDOUT)
        server.wait_ready(max_wait_s=1800)
        cold_start_s = time.monotonic() - t0
    elif not server.health():
        sys.exit(f"llama-server not reachable at {a.server_url} (start it, or use --launch-server)")

    try:
        with urllib.request.urlopen(a.server_url.rstrip("/") + "/props", timeout=10) as r:
            props = json.loads(r.read())
    except Exception:
        props = None

    cfg = {"temperature": a.temperature, "max_tokens": a.max_tokens, "thinking": a.thinking == "on",
           "cache_prompt": a.cache_prompt, "system_prompt": a.system_prompt, "max_turns": a.max_turns}
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
            print(f"[{time.strftime('%H:%M:%S')}] {n + 1}/{len(ids)} {tid}", flush=True)
            s = run_session(tasks[tid], bfcl, server, classes, cfg, log_req, monitor)
            if n == 0 and cold_start_s is not None and s["warm_time_to_first_call_ms"] is not None:
                s["cold_time_to_first_call_ms"] = round(cold_start_s * 1000 + s["warm_time_to_first_call_ms"], 1)
            log_ses(s)
            print(f"    success={s['session_success']} turns_passed={s['turns_passed']}/{s['n_turns_run']} "
                  f"calls_valid={s['n_calls_valid']}/{s['n_calls']} wall={s['session_wall_ms'] / 1000:.0f}s",
                  flush=True)
    finally:
        monitor.stop()
        f_req.close()
        f_ses.close()
        if proc:
            proc.terminate()


if __name__ == "__main__":
    main()
