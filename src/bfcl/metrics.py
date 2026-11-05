"""Final numbers for a whole run.

What this file is for
---------------------
`evaluator.evaluate_task` marks one task at a time. This file adds those marks up over all
tasks of a run, giving the numbers that are reported and compared between runs:

- session_success_rate: share of tasks where every turn was correct (BFCL's main metric).
- per_turn_accuracy: share of all turns (over all tasks) that were correct.
- num_tool_calls: how many tool calls were made in total.
- mean_context_chars / max_context_chars: how long the sessions got, in characters.
"""

from statistics import mean


def _safe_mean(values: list[float]) -> float | None:
    """Return the average of a list rounded to 4 decimals, or None if the list is empty.

    Used so that a run with no tasks (or no turns) gives None instead of crashing.
    """
    return round(mean(values), 4) if values else None


def summarize(task_results: list[dict]) -> dict:
    """Add up the marks of all tasks into the final numbers for a run.

    Args:
        task_results: One dict per task, as returned by `evaluator.evaluate_task`.

    Returns:
        A dict with:
            num_tasks: number of tasks in the run.
            session_success_rate: fraction of tasks with session_success = True.
            per_turn_accuracy: fraction of correct turns, counting every turn of every task.
            num_tool_calls: total tool calls made (including ones that were rejected).
            mean_context_chars, max_context_chars: average and largest session size in
                characters (tool descriptions + full history). Roughly 3 to 4 characters
                make one token, so this shows how long the context gets without a tokenizer.
    """
    turn_flags = [ok for r in task_results for ok in r["turn_correct"]]
    calls = [c for r in task_results for s in r["steps"] for c in s["tool_calls"]]

    return {
        "num_tasks": len(task_results),
        "session_success_rate": _safe_mean([float(r["session_success"]) for r in task_results]),
        "per_turn_accuracy": _safe_mean([float(ok) for ok in turn_flags]),
        "num_tool_calls": len(calls),
        "mean_context_chars": _safe_mean([r["context_chars"] for r in task_results]),
        "max_context_chars": max((r["context_chars"] for r in task_results), default=0),
    }
