"""Which expected tool calls the model made, and which it missed, turn by turn.

BFCL's checker only says whether a turn passed (the backend state and tool outputs match the
ground truth). This file adds a call-level view of the same session, for reading results:

- matched:    the model made the expected call with the same arguments;
- wrong_args: the model called the expected tool, but with different arguments;
- missed:     the model never called the expected tool in that turn;
- extra:      the model made a call that is not in the ground truth for that turn.

Arguments are compared after filling in the tool's default values, so `f()` and `f(x=<default>)`
count as the same call. Calls are matched inside one turn, in any order. A turn can still pass
BFCL's check with extra or reordered calls, so these counts explain a score, they do not replace it.
"""

from __future__ import annotations

import inspect

from . import bfcl_env as env
from .debug_log import trace


@trace
def normalize(call: str, owner: dict) -> tuple[str, dict]:
    """Parse a call string into (name, arguments with defaults filled in)."""
    name, kwargs = env.parse_call_string(call, owner)
    obj = owner.get(name)
    if obj is not None:
        try:
            bound = inspect.signature(getattr(obj, name)).bind(**kwargs)
            bound.apply_defaults()
            kwargs = dict(bound.arguments)
        except TypeError:
            pass  # unknown or missing arguments: compare them as written
    return name, kwargs


@trace
def match_turn(expected: list[str], made: list[str], owner: dict) -> dict:
    """Match one turn's ground-truth calls against the model's calls.

    Args:
        expected: Ground-truth call strings for the turn.
        made: The model's call strings for the turn (all steps, in order).
        owner: Method name -> backend object (see `bfcl_env.method_owner`).

    Returns:
        {"matched", "wrong_args", "missed", "extra"}; wrong_args holds {"expected", "made"} pairs.
    """
    exp = [(c, *normalize(c, owner)) for c in expected]
    mod = [(c, *normalize(c, owner)) for c in made]
    used = [False] * len(mod)

    def take(name, args=None):
        for j, (_, n, a) in enumerate(mod):
            if not used[j] and n == name and (args is None or a == args):
                used[j] = True
                return mod[j][0]
        return None

    matched, rest = [], []
    for c, n, a in exp:  # first pass: same tool and same arguments
        if take(n, a):
            matched.append(c)
        else:
            rest.append((c, n))
    wrong_args, missed = [], []
    for c, n in rest:  # second pass: same tool, other arguments
        hit = take(n)
        if hit:
            wrong_args.append({"expected": c, "made": hit})
        else:
            missed.append(c)
    extra = [mod[j][0] for j in range(len(mod)) if not used[j]]
    return {"matched": matched, "wrong_args": wrong_args, "missed": missed, "extra": extra}


@trace
def session_coverage(bfcl: env.BFCL, task: env.Task, model_calls: list[list[list[str]]]) -> dict:
    """Call-level comparison for every played turn of one session.

    Args:
        bfcl: The BFCL copy (to build the task's backends for argument names and defaults).
        task: The task with its ground truth.
        model_calls: Turn -> step -> call strings (from `official_score.rebuild_model_calls`).

    Returns:
        Totals over the played turns (expected, made, matched, wrong_args, missed, extra,
        call_recall = matched / expected, call_precision = matched / made) and "turns", one
        entry per played turn with the call lists.
    """
    owner = env.method_owner(bfcl.new_instances(task.involved_classes, task.initial_config))
    turns = []
    for i, steps in enumerate(model_calls):
        made = [c for step in steps for c in step]
        turns.append({"turn": i, "expected": list(task.ground_truth[i]), "made": made,
                      **match_turn(task.ground_truth[i], made, owner)})
    totals = {k: sum(len(t[k]) for t in turns) for k in ("expected", "made", "matched", "wrong_args", "missed", "extra")}
    totals["call_recall"] = round(totals["matched"] / totals["expected"], 4) if totals["expected"] else None
    totals["call_precision"] = round(totals["matched"] / totals["made"], 4) if totals["made"] else None
    return {**totals, "turns": turns}
