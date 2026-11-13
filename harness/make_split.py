"""Pick the A1 tool subset's tasks and hold out a fixed test split.

  python -m harness.make_split --bfcl-root third_party/gorilla/berkeley-function-call-leaderboard

Writes splits/tasks_<classes>.json with the trace and test task ids. Every reported
measurement uses the test ids; the trace ids are for calibration, distillation etc.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .bfcl_env import BFCL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfcl-root", required=True)
    ap.add_argument("--classes", default="TradingBot,MessageAPI")
    ap.add_argument("--test-fraction", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=624)
    ap.add_argument("--out-dir", default="splits")
    a = ap.parse_args()

    classes = a.classes.split(",")
    bfcl = BFCL(a.bfcl_root)
    tasks = bfcl.load_tasks()
    sel = sorted(t.id for t in tasks.values() if set(t.involved_classes) <= set(classes))
    n_funcs = len(bfcl.tools_for(classes))
    rng = random.Random(a.seed)
    shuffled = sel[:]
    rng.shuffle(shuffled)
    n_test = round(len(sel) * a.test_fraction)
    split = {
        "classes": classes, "n_functions": n_funcs, "seed": a.seed,
        "dataset": "BFCL_v4_multi_turn_long_context",
        "test": sorted(shuffled[:n_test], key=lambda s: int(s.rsplit("_", 1)[1])),
        "trace": sorted(shuffled[n_test:], key=lambda s: int(s.rsplit("_", 1)[1])),
    }
    out = Path(a.out_dir) / f"tasks_{'_'.join(classes)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        previous = json.loads(out.read_text(encoding="utf-8"))
        if "extra_tasks" in previous:
            # Keep the manually frozen custom-task holdout assignments intact.
            split["extra_tasks"] = previous["extra_tasks"]
    out.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    print(f"{n_funcs} functions, {len(sel)} tasks -> {len(split['test'])} test / "
          f"{len(split['trace'])} trace  ->  {out}")


if __name__ == "__main__":
    main()
