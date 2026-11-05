"""Marking a finished session against BFCL's correct answers.

What this file is for
---------------------
After `task_runner.run_task` has played out a task, this file decides how well it went.
It produces the two metrics the project asks for:

- Session success: did the whole task go right? This uses BFCL's own official checker
  (`multi_turn_checker`) unchanged, so our numbers match BFCL's definition.
- Per-turn accuracy: which individual turns went right? BFCL's checker stops at the first
  wrong turn, so it cannot tell us this. `check_each_turn` runs the same checks as BFCL,
  but for every turn.

How BFCL decides a turn is correct
----------------------------------
BFCL does not compare the call text directly (there are many correct ways to call tools).
Instead it runs the model's calls on one fresh copy of the tools and the correct calls on
another copy, then checks two things after each turn:

1. State: the tools end up in the same state (same orders, balance, messages, ...).
2. Outputs: every output the correct calls produced also appears in the model's outputs
   (so the model actually looked up what the user asked for).
"""

import uuid

from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
    multi_turn_checker,
    response_checker,
    state_checker,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call

from bfcl.loader import BFCLTask
from bfcl.task_runner import release_backends


def _test_entry(task: BFCLTask) -> dict:
    """Pack the task fields that BFCL's checker needs (id, starting state, tool groups) into a dict.

    BFCL's checker expects the task in its original dict form, not as a BFCLTask.
    """
    return {"id": task.id, "initial_config": task.initial_config, "involved_classes": task.involved_classes}


def _execute(task: BFCLTask, calls: list[str], name: str, long_context: bool):
    """Run a list of call strings on BFCL's tools for this task.

    The tools are created from the task's starting state the first time a `name` is used, and
    kept in memory afterwards, so later calls with the same `name` continue from the changed
    state (this is how state carries over from turn to turn). Only the task's own tool groups
    are created, like in BFCL's checker.

    Args:
        task: The task (starting state and tool groups).
        calls: Call strings, e.g. ["get_stock_info(symbol='AAPL')"].
        name: Name that identifies this copy of the tools.
        long_context: Run the tools in long-context mode.

    Returns:
        (list of output texts, dict of tool objects by tool-group name). The tool objects
        are used to compare state.
    """
    return execute_multi_turn_func_call(
        func_call_list=calls,
        initial_config=task.initial_config,
        involved_classes=task.involved_classes,
        model_name=name,
        test_entry_id=task.id,
        long_context=long_context,
    )


def check_each_turn(task: BFCLTask, model_calls: list[list[list[str]]], long_context: bool) -> list[bool]:
    """Decide for every turn separately whether it was correct.

    Two copies of the tools are kept: one changed only by the model's calls and one changed
    only by the correct calls. After each turn they are compared with BFCL's own
    `state_checker` (same state) and `response_checker` (the correct outputs appear among
    all of the model's outputs so far). Special cases:

    - If the correct answer for a turn has no calls (the request cannot be done), the turn
      is right only if the model made no calls either.
    - If the correct answer has calls but the model made none, the turn is wrong.

    Unlike BFCL's checker, this does not stop at the first wrong turn. Note that after a wrong
    turn the model continues from its own (different) state, so later turns are judged on
    what the model actually did.

    Args:
        task: The task.
        model_calls: The calls that were run, as turn -> step -> call strings
            (`model_calls` from `run_task`).
        long_context: Run the tools in long-context mode.

    Returns:
        One True/False per turn of the task.
    """
    name = f"turncheck_{uuid.uuid4().hex[:8]}"
    turn_correct = []
    all_model_outputs: list[str] = []
    try:
        for turn_index, expected_calls in enumerate(task.ground_truth):
            turn_steps = model_calls[turn_index] if turn_index < len(model_calls) else []

            model_instances = {}
            for step_calls in turn_steps:
                outputs, model_instances = _execute(task, step_calls, f"{name}_model", long_context)
                all_model_outputs.extend(outputs)
            expected_outputs, truth_instances = _execute(task, expected_calls, f"{name}_truth", long_context)

            if not expected_calls:  # turn where the model should NOT call anything
                turn_correct.append(not turn_steps)
                continue
            if not turn_steps:  # model made no call where one was needed
                turn_correct.append(False)
                continue
            state_ok = state_checker(model_instances, truth_instances)["valid"]
            response_ok = response_checker(all_model_outputs, expected_outputs, turn_index)["valid"]
            turn_correct.append(state_ok and response_ok)
    finally:
        release_backends(name)
    return turn_correct


def evaluate_task(task: BFCLTask, trace: dict, category: str) -> dict:
    """Mark one finished task: session success and per-turn correctness.

    Session success comes from BFCL's official `multi_turn_checker`: the session passes only
    if every turn is correct. If the session was cut off (step limit reached, so fewer turns
    were played than the task has), it fails with "multi_turn:force_terminated", exactly as
    BFCL does.

    Args:
        task: The task.
        trace: The output of `run_task` for this task.
        category: BFCL category name. Long-context mode is used when it contains "long_context".

    Returns:
        The trace with four keys added:
            session_success: True/False.
            turn_correct: one True/False per turn (from `check_each_turn`).
            error_type: BFCL's error category when the session failed, else None.
            error_message: BFCL's explanation when the session failed, else None.
    """
    long_context = "long_context" in category
    model_calls = trace["model_calls"]

    if trace["force_terminated"] or len(model_calls) != len(task.ground_truth):
        official = {"valid": False, "error_type": "multi_turn:force_terminated"}
    else:
        name = f"official_{uuid.uuid4().hex[:8]}"
        try:
            official = multi_turn_checker(model_calls, task.ground_truth, _test_entry(task), category, name)
        finally:
            release_backends(name)

    return {
        **trace,
        "session_success": official["valid"],
        "error_type": official.get("error_type"),
        "error_message": official.get("error_message"),
        "turn_correct": check_each_turn(task, model_calls, long_context),
    }
