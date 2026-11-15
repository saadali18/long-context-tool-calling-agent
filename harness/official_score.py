"""Score a finished harness run a second time with BFCL's own official checker.

What this file is for
---------------------
`harness.runner` already scores every session while it runs, with its own copy of BFCL's
checking logic (state check + response check after each turn, tools run by method lookup).
This file re-scores the same run with BFCL's original code, unchanged:

- Session success comes from BFCL's `multi_turn_checker`, the function the official
  leaderboard uses. It stops at the first wrong turn.
- Per-turn correctness uses BFCL's own `state_checker` and `response_checker` after every
  turn, without stopping at the first wrong one (the official checker cannot report this).

Both scorers then read the same outputs (`requests.jsonl` + `sessions.jsonl`), so their
numbers can be compared directly. Any disagreement is printed and points at a difference
between the harness's scoring and BFCL's.

How a run is turned back into BFCL's input
------------------------------------------
BFCL's checker wants the model's calls as turn -> step -> call strings, e.g.
`[[["get_stock_info(symbol='AAPL')"], ["place_order(...)"]], ...]`. They are rebuilt from
`requests.jsonl`, where each line is one model request (one step) with its parsed calls:

- A call is kept only when its arguments were valid JSON, its name is one of the task's tools
  and every argument name is a plain identifier. BFCL runs call strings with `eval`, so this
  stops anything else from being run. A dropped call only ever produced an error message in
  the harness, so dropping it does not change the backend state.
- A step with no kept calls is left out, as BFCL's own evaluation does with steps that hold no
  decodable call (e.g. plain text answers).
- A session that was aborted, or a turn that used all 20 steps, fails as
  "multi_turn:force_terminated", like in BFCL. A run cut short on purpose (`--max-turns`)
  is scored on the turns that were played; its session success is False, as in the harness.

Usage
-----
    python3 -m harness.official_score results/harness_bf16_143_full \\
        --bfcl-root third_party/gorilla/berkeley-function-call-leaderboard

Writes `official_scores.jsonl` (one line per task) into the run folder and prints both
scorers side by side.
"""

from __future__ import annotations

import argparse
import importlib
import json
import uuid
from collections import defaultdict
from pathlib import Path

from . import bfcl_env as env
from .bfcl_env import BFCL, Task
from .run import load_extra_tasks


def to_call_string(name: str, args: dict) -> str:
    """Write one call in the Python-call text form that BFCL runs and checks.

    Example: to_call_string("get_stock_info", {"symbol": "AAPL"}) -> "get_stock_info(symbol='AAPL')".
    Values are written with `repr`, so strings get quotes and numbers, lists, dicts,
    True/False/None stay Python values.
    """
    return f"{name}({','.join(f'{k}={v!r}' for k, v in args.items())})"


def task_tool_names(bfcl: BFCL, task: Task) -> set[str]:
    """Names of the tools a task may call: the tools of its own backends, minus excluded ones.

    BFCL only creates the task's own backends, so a call to any other name could not reach a
    tool there. Keeping only these names is also what makes `eval` safe.
    """
    return {t["function"]["name"] for t in bfcl.tools_for(task.involved_classes, task.excluded_functions)}


def rebuild_model_calls(requests: list[dict], n_turns: int, allowed: set[str]) -> list[list[list[str]]]:
    """Turn a session's request records back into BFCL's turn -> step -> call-string list.

    Args:
        requests: The `requests.jsonl` lines of one task (each has "turn", "step", "calls").
        n_turns: Number of turns that were played (`n_turns_run` in sessions.jsonl).
        allowed: Tool names the task may call (see `task_tool_names`).

    Returns:
        One list per played turn; each holds the steps that had at least one kept call.
    """
    by_turn: dict[int, list] = defaultdict(list)
    for rec in sorted(requests, key=lambda r: (r["turn"], r["step"])):
        step = []
        for c in rec["calls"]:
            args = c.get("args")
            if not (c.get("well_formed") and c.get("name") in allowed and isinstance(args, dict)):
                continue
            if not all(isinstance(k, str) and k.isidentifier() for k in args):
                continue
            step.append(to_call_string(c["name"], args))
        if step:
            by_turn[rec["turn"]].append(step)
    return [by_turn[t] for t in range(n_turns)]


def hit_step_limit(requests: list[dict]) -> bool:
    """True when some turn used all 20 steps and the model was still calling tools.

    BFCL force-terminates such a session; the harness simply moves on to the next turn.
    """
    last = {}
    for rec in requests:
        if rec["turn"] not in last or rec["step"] > last[rec["turn"]]["step"]:
            last[rec["turn"]] = rec
    return any(r["step"] >= env.MAX_STEPS_PER_TURN - 1 and r["calls"] for r in last.values())


class OfficialChecker:
    """BFCL's official checking code, loaded from the BFCL copy that `BFCL(root)` put on sys.path.

    The BFCL modules are imported here, after `BFCL(root)` was created, so the checker uses
    exactly the same BFCL copy (and the same long-context list cap) as the harness run.
    """

    def __init__(self):
        self.checker = importlib.import_module("bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker")
        self.utils = importlib.import_module("bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils")

    def _release(self, prefix: str) -> None:
        """Delete the backend objects BFCL kept in memory under names starting with `prefix`.

        BFCL keeps its backends as global variables named after the run, so that state carries
        over between turns. Every scoring uses a fresh unique prefix and removes it afterwards,
        so no state leaks from one task into the next.
        """
        for key in [k for k in vars(self.utils) if k.startswith(prefix)]:
            del vars(self.utils)[key]

    def session(self, task: Task, model_calls: list[list[list[str]]]) -> dict:
        """Run BFCL's `multi_turn_checker` on one session.

        Args:
            task: The task (ground truth, starting state, backends).
            model_calls: Output of `rebuild_model_calls`, one entry per ground-truth turn.

        Returns:
            BFCL's result dict: "valid" plus, on failure, "error_type" and "error_message".
        """
        entry = {"id": task.id, "initial_config": task.initial_config, "involved_classes": task.involved_classes}
        name = f"official_{uuid.uuid4().hex[:8]}"
        try:
            return self.checker.multi_turn_checker(model_calls, task.ground_truth, entry, "multi_turn_long_context", name)
        finally:
            self._release(name)

    def per_turn(self, task: Task, model_calls: list[list[list[str]]]) -> list[bool]:
        """Check every played turn with BFCL's own `state_checker` and `response_checker`.

        Two copies of the backends are kept, one changed only by the model's calls and one only
        by the ground-truth calls, and compared after each turn, like `multi_turn_checker` does,
        but without stopping at the first wrong turn. Rules taken from BFCL:

        - a turn whose ground truth has calls fails when the model made none;
        - a turn whose ground truth is empty is not checked (counts as correct);
        - the response check looks at all of the model's outputs so far, not only this turn's.

        After a wrong turn the model continues from its own state, so later turns are judged on
        what the model actually did (the harness does the same).
        """
        name = f"turncheck_{uuid.uuid4().hex[:8]}"
        run = self.utils.execute_multi_turn_func_call
        flags, all_outputs = [], []
        try:
            for i, steps in enumerate(model_calls):
                expected = task.ground_truth[i]
                model_inst = {}
                for step in steps:
                    out, model_inst = run(step, task.initial_config, task.involved_classes,
                                          f"{name}_model", task.id, long_context=True)
                    all_outputs.extend(out)
                gt_out, gt_inst = run(expected, task.initial_config, task.involved_classes,
                                      f"{name}_truth", task.id, long_context=True)
                if not expected:
                    flags.append(True)
                elif not steps:
                    flags.append(False)
                else:
                    flags.append(self.checker.state_checker(model_inst, gt_inst)["valid"]
                                 and self.checker.response_checker(all_outputs, gt_out, i)["valid"])
        finally:
            self._release(name)
        return flags


def score_session(checker: OfficialChecker, bfcl: BFCL, task: Task, session: dict, requests: list[dict]) -> dict:
    """Score one harness session with BFCL's code and put both scorers' results side by side.

    Args:
        checker: The loaded official checker.
        bfcl: The BFCL copy (for the task's tool names).
        task: The task.
        session: The task's line from sessions.jsonl (the harness's own scores).
        requests: The task's lines from requests.jsonl.

    Returns:
        A dict with the task id, the official results (official_session_success,
        official_error_type, official_turns) and the harness's results (harness_session_success,
        harness_turns), plus `agree` (True when both give the same session result and the same
        flag for every turn).
    """
    n = session["n_turns_run"]
    model_calls = rebuild_model_calls(requests, n, task_tool_names(bfcl, task))
    turns = checker.per_turn(task, model_calls)

    if session.get("aborted") or hit_step_limit(requests):
        official = {"valid": False, "error_type": "multi_turn:force_terminated"}
    elif n < len(task.ground_truth):  # cut short on purpose with --max-turns
        official = {"valid": False, "error_type": "truncated"}
    else:
        official = checker.session(task, model_calls)

    harness_turns = [t["passed"] for t in session["turns"]]
    result = {
        "task": task.id,
        "n_turns_run": n,
        "official_session_success": official["valid"],
        "official_error_type": official.get("error_type"),
        "official_error_message": official.get("error_message"),
        "official_turns": turns,
        "harness_session_success": session["session_success"],
        "harness_fail_reason": session.get("fail_reason"),
        "harness_turns": harness_turns,
        "model_calls": model_calls,
    }
    result["agree"] = (official["valid"] == session["session_success"] and turns == harness_turns)
    return result


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSON Lines file into a list of dicts (empty lines skipped)."""
    return [json.loads(line) for line in open(path) if line.strip()]


def score_run(run_dir: str | Path, bfcl_root: str | Path, extra_tasks: str | None = None) -> list[dict]:
    """Score every session of a harness run folder with BFCL's code.

    Args:
        run_dir: Folder written by `harness.run` (needs requests.jsonl and sessions.jsonl).
        bfcl_root: The BFCL folder (berkeley-function-call-leaderboard) to load tasks and code from.
        extra_tasks: Our own task file, if the run used `--extra-tasks`.

    Returns:
        One result per session (see `score_session`), in sessions.jsonl order.
    """
    run_dir = Path(run_dir)
    bfcl = BFCL(bfcl_root)
    tasks = bfcl.load_tasks()
    if extra_tasks:
        tasks.update(load_extra_tasks(extra_tasks))
    checker = OfficialChecker()

    requests_by_task = defaultdict(list)
    for rec in _read_jsonl(run_dir / "requests.jsonl"):
        requests_by_task[rec["task"]].append(rec)
    return [score_session(checker, bfcl, tasks[s["task"]], s, requests_by_task[s["task"]])
            for s in _read_jsonl(run_dir / "sessions.jsonl")]


def compare(results: list[dict]) -> dict:
    """Add up both scorers over a run: session success rate, per-turn accuracy and agreement."""
    def rate(values):
        return round(sum(values) / len(values), 4) if values else None

    off_turns = [x for r in results for x in r["official_turns"]]
    har_turns = [x for r in results for x in r["harness_turns"]]
    return {
        "num_sessions": len(results),
        "official_session_success_rate": rate([r["official_session_success"] for r in results]),
        "harness_session_success_rate": rate([r["harness_session_success"] for r in results]),
        "official_per_turn_accuracy": rate(off_turns),
        "harness_per_turn_accuracy": rate(har_turns),
        "sessions_in_agreement": sum(r["agree"] for r in results),
        "disagreements": [r["task"] for r in results if not r["agree"]],
    }


def main():
    """Command line: score a run folder, write official_scores.jsonl, print the comparison."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="folder written by harness.run")
    ap.add_argument("--bfcl-root", required=True, help="berkeley-function-call-leaderboard folder")
    ap.add_argument("--extra-tasks", help="our own task file, if the run used --extra-tasks")
    a = ap.parse_args()

    results = score_run(a.run_dir, a.bfcl_root, a.extra_tasks)
    with open(Path(a.run_dir) / "official_scores.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    for r in results:
        print(f"{r['task']}: official {'PASS' if r['official_session_success'] else 'FAIL'} "
              f"({r['official_error_type']}) turns {r['official_turns']} | harness "
              f"{'PASS' if r['harness_session_success'] else 'FAIL'} turns {r['harness_turns']}"
              f"{'' if r['agree'] else '  <-- DISAGREE'}")
    print(json.dumps(compare(results), indent=2))


if __name__ == "__main__":
    main()
