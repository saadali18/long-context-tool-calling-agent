"""Check that capping BFCL's long-context filler lists does not change scoring.

Run on an UNCAPPED long_context.py (before cap_long_context.py, or after --undo).
The script replays every task's ground-truth calls twice in one process, once with
the original lists and once with the lists trimmed to --cap, and checks:

  1. Only get_stock_info / get_watchlist / get_transaction_history outputs change,
     and each changed output is the original with only filler removed from the end
     (price, percent_change and volume are untouched).
  2. The backend state after every turn is identical with and without the cap,
     so BFCL's state check gives the same verdicts.
  3. Scoring the ground truth against itself with the cap passes every turn.
  4. Tokens per task before/after (exact with --tokenizer, else a per-digit estimate).

  python3 scripts/verify_list_cap.py --bfcl-root third_party/gorilla/berkeley-function-call-leaderboard \
      --cap 600 --tokenizer ~/models/Qwen3-1.7B/tokenizer.json
"""

from __future__ import annotations

import argparse
import ast
import copy
import importlib
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import bfcl_env as env  # noqa: E402

LISTS = ["MA_5_EXTENSION", "MA_20_EXTENSION", "WATCH_LIST_EXTENSION", "TRANSACTION_HISTORY_EXTENSION"]
AFFECTED = {"get_stock_info", "get_watchlist", "get_transaction_history"}


def est_tokens(s: str) -> int:  # Qwen-like: one token per digit, one per word or symbol
    return sum(len(x) if x.isdigit() else 1 for x in re.findall(r"\d+|[A-Za-z]+|[^\sA-Za-z\d]", s))


def replay(bfcl, task, classes):
    """Run the ground truth; return per-call (name, result) and the public state after each turn."""
    inst = bfcl.new_instances(classes, task.initial_config)
    own = env.method_owner(inst)
    calls, states = [], []
    for turn in task.ground_truth:
        for c in turn:
            name, kw = env.parse_call_string(c, own)
            calls.append((name, env.execute(own, name, kw)))
        states.append({k: copy.deepcopy(env.public_state(v)) for k, v in inst.items()})
    return calls, states


def is_prefix_trim(name, before: str, after: str) -> bool:
    def parse(x):
        try:
            return json.loads(x)
        except Exception:
            # list outputs come back as Python reprs; BFCL's watch-list filler ends with a stray `...`
            return ast.literal_eval(x.replace("Ellipsis", "None"))
    try:
        b, a = parse(before), parse(after)
    except Exception:
        return False
    if isinstance(b, list):
        return isinstance(a, list) and b[: len(a)] == a and len(a) <= len(b)
    if name == "get_stock_info":
        same_scalars = all(a.get(k) == b.get(k) for k in b if not isinstance(b[k], list))
        lists_ok = all(b[k][: len(a[k])] == a[k] for k in b if isinstance(b[k], list))
        return same_scalars and lists_ok
    if name == "get_transaction_history":
        tb, ta = b["transaction_history"], a["transaction_history"]
        return tb[: len(ta)] == ta
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bfcl-root", required=True)
    ap.add_argument("--cap", type=int, default=600)
    ap.add_argument("--classes", default="TradingBot,MessageAPI")
    ap.add_argument("--tokenizer", help="Qwen3 tokenizer.json for exact counts")
    a = ap.parse_args()

    count = est_tokens
    if a.tokenizer:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(Path(a.tokenizer).expanduser()))
        count = lambda s: len(tok.encode(s, add_special_tokens=False).ids)  # noqa: E731

    bfcl = env.BFCL(a.bfcl_root)
    L = importlib.import_module("bfcl_eval.eval_checker.multi_turn_eval.func_source_code.long_context")
    if "_A1_LIST_CAP" in vars(L):
        sys.exit("long_context.py is already capped: run scripts/cap_long_context.py --undo first, then retry")
    lengths = {n: len(getattr(L, n)) for n in LISTS}

    classes = a.classes.split(",")
    tasks = [t for t in bfcl.load_tasks().values() if set(t.involved_classes) <= set(classes)]
    tasks.sort(key=lambda t: int(t.id.rsplit("_", 1)[1]))

    before = {t.id: replay(bfcl, t, classes) for t in tasks}
    saved = {n: list(getattr(L, n)) for n in LISTS}
    for n in LISTS:                       # trim in place: TradingBot holds the same list objects
        del getattr(L, n)[a.cap:]
    after = {t.id: replay(bfcl, t, classes) for t in tasks}

    problems, changed, per_task = [], 0, []
    for t in tasks:
        (cb, sb), (ca, sa) = before[t.id], after[t.id]
        for (n1, r1), (_, r2) in zip(cb, ca):
            if r1 == r2:
                continue
            changed += 1
            if n1 not in AFFECTED:
                problems.append(f"{t.id}: {n1} output changed")
            elif not is_prefix_trim(n1, r1, r2):
                problems.append(f"{t.id}: {n1} changed beyond trimming filler")
        for ti, (x, y) in enumerate(zip(sb, sa)):
            if x != y:
                problems.append(f"{t.id}: backend state differs after turn {ti}")
        per_task.append((t.id, sum(count(r) for _, r in cb), sum(count(r) for _, r in ca)))

    # self-scoring with the cap: ground truth as the "model" must pass every turn
    fails = 0
    for t in tasks:
        m = bfcl.new_instances(classes, t.initial_config); g = bfcl.new_instances(classes, t.initial_config)
        mo, go = env.method_owner(m), env.method_owner(g)
        results = []
        for turn in t.ground_truth:
            gt = []
            for c in turn:
                n, kw = env.parse_call_string(c, mo); results.append(env.execute(mo, n, kw))
                n, kw = env.parse_call_string(c, go); gt.append(env.execute(go, n, kw))
            ok = all(env.state_matches(m[c], g[c])[0] for c in t.involved_classes)
            if not ok or env.responses_contained(gt, results):
                fails += 1

    for n in LISTS:  # restore
        getattr(L, n)[:] = saved[n]

    b = [x[1] for x in per_task]; c = [x[2] for x in per_task]
    print(f"tasks: {len(tasks)}   cap: {a.cap}   original lengths: {lengths}")
    print(f"tokens counted with: {'Qwen3 tokenizer' if a.tokenizer else 'per-digit estimate'}")
    print(f"changed call outputs: {changed} (all should be get_stock_info / get_watchlist / get_transaction_history)")
    print(f"problems: {len(problems)}")
    for p in problems[:20]:
        print("   ", p)
    print(f"self-scoring failures with cap: {fails}")
    print(f"ground-truth tool output per task, tokens: median {st.median(b):.0f} -> {st.median(c):.0f}, "
          f"max {max(b)} -> {max(c)}")
    print(f"tasks whose tool output alone exceeds 16k: {sum(x > 16384 for x in b)} -> {sum(x > 16384 for x in c)}")
    print("VERDICT:", "scoring unaffected" if not problems and not fails else "CHECK THE PROBLEMS ABOVE")


if __name__ == "__main__":
    main()
