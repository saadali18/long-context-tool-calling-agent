"""Gap experiment: compare free-running runs with a teacher-forced run (Task 3).

Free-running: each turn starts from the model's own history, so an early mistake can carry into
later turns. Teacher-forced (--teacher-forcing): each turn starts from BFCL's correct history and
app state, so each turn is scored on its own.

    python3 scripts/gap_experiment.py --free results/gpu_bf16_think_18_tasks_seed0 \
        [--free ...seed1 ...seed2] --forced results/gpu_bf16_think_18_tf_seed0
"""
import argparse, collections, json
from pathlib import Path


def sessions(folder):
    return {s["task"]: s for s in map(json.loads, Path(folder, "sessions.jsonl").read_text().splitlines())}


def pct(a, b):
    return f"{a}/{b} = {100 * a / b:.1f}%" if b else "n/a"


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--free", action="append", required=True, help="free-running run folder (repeat per seed)")
ap.add_argument("--forced", action="append", required=True, help="teacher-forced run folder (repeat per seed)")
a = ap.parse_args()

for f in a.forced:
    meta = json.loads(Path(f, "metadata.json").read_text())
    if not meta["config"].get("teacher_forcing"):
        raise SystemExit(f"{f} was not run with --teacher-forcing")

free = [sessions(f) for f in a.free]
forced = [sessions(f) for f in a.forced]

# 1. Per-turn accuracy and session success, free-running vs teacher-forced
for name, runs in (("free-running", free), ("teacher-forced", forced)):
    played = passed = ok = n = 0
    by_turn = collections.defaultdict(lambda: [0, 0])
    for run in runs:
        for s in run.values():
            n += 1
            ok += bool(s["session_success"])
            for t in s["turns"]:
                played += 1
                passed += t["passed"]
                by_turn[t["turn"]][0] += t["passed"]
                by_turn[t["turn"]][1] += 1
    print(f"{name:>15}: per-turn {pct(passed, played)} | sessions {pct(ok, n)} | turns played {played}")
    print(" " * 17 + "by turn: " + "  ".join(f"t{i} {v[0]}/{v[1]}" for i, v in sorted(by_turn.items())))

# 2. Matched turns: the same task and turn, where the free-running run had already failed earlier
after_fail_free = [0, 0]
after_fail_forced = [0, 0]
for fr in free:
    for tid, s in fr.items():
        first = s["first_failed_turn"]
        if first is None:
            continue
        for t in s["turns"]:
            if t["turn"] <= first:
                continue
            after_fail_free[0] += t["passed"]
            after_fail_free[1] += 1
            for fo in forced:
                ft = next((x for x in fo.get(tid, {}).get("turns", []) if x["turn"] == t["turn"]), None)
                if ft is not None:
                    after_fail_forced[0] += ft["passed"]
                    after_fail_forced[1] += 1
print()
print("Turns that come after an earlier failed turn (in the free-running runs):")
print(f"   free-running   {pct(*after_fail_free)}")
print(f"   teacher-forced {pct(*after_fail_forced)}   (the same task and turn, from the correct history)")

# 3. If turns were independent: predicted session success from teacher-forced per-turn accuracy
p_turn = collections.defaultdict(list)
for fo in forced:
    for tid, s in fo.items():
        for t in s["turns"]:
            p_turn[tid].append(t["passed"])
tasks = next(iter(forced)).values()
p_all = sum(sum(v) for v in p_turn.values()) / max(1, sum(len(v) for v in p_turn.values()))
pred = sum(p_all ** s["n_turns_total"] for s in tasks) / max(1, len(tasks))
print()
print(f"Predicted session success if turns were independent (p = {100 * p_all:.1f}% per turn): {100 * pred:.1f}%")
