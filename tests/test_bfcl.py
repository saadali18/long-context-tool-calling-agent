"""Automatic checks for the BFCL benchmark setup. Run with `pytest` from the repo root.

No model or llama-server is needed: a fake in-process server replays BFCL's correct calls
(or deliberately wrong ones) through the real agent loop (`harness.runner.run_session`).
Each session is then scored twice, by the harness itself and by BFCL's official checker
(`harness.official_score`), and the tests check that:

- the tool subset and the train/test split are what the project expects;
- tool descriptions use standard JSON Schema types;
- long-context mode really makes tool outputs longer;
- correct calls pass and a silent model fails, under both scorers;
- both scorers agree when the model makes mistakes;
- unsafe calls are never handed to BFCL's `eval`;
- our own long tasks (data/long_tasks.jsonl) are solvable by their ground truth.

The BFCL copy is taken from the BFCL_ROOT environment variable, or from
third_party/gorilla/berkeley-function-call-leaderboard. Without it the tests are skipped.
"""

import importlib
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `harness` imports from any folder
from harness import bfcl_env as env  # noqa: E402
from harness.official_score import OfficialChecker, rebuild_model_calls, score_session  # noqa: E402
from harness.run import load_extra_tasks  # noqa: E402
from harness.runner import run_session  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BFCL_ROOT = Path(os.environ.get("BFCL_ROOT", ROOT / "third_party/gorilla/berkeley-function-call-leaderboard"))
SPLIT = json.loads((ROOT / "splits/tasks_TradingBot_MessageAPI.json").read_text())
CLASSES = SPLIT["classes"]
LONG_TASKS = ROOT / "data/long_tasks.jsonl"
CFG = {"temperature": 0.0, "max_tokens": 512, "cache_prompt": False, "thinking": False,
       "system_prompt": "none", "max_turns": None}

pytestmark = pytest.mark.skipif(not (BFCL_ROOT / "bfcl_eval").is_dir(),
                                reason=f"BFCL not found at {BFCL_ROOT} (set BFCL_ROOT)")


@pytest.fixture(scope="module")
def bfcl():
    """The BFCL copy (puts it on sys.path, so BFCL's own code is imported from it)."""
    return env.BFCL(BFCL_ROOT)


@pytest.fixture(scope="module")
def tasks(bfcl):
    """All BFCL long-context tasks by id."""
    return bfcl.load_tasks()


@pytest.fixture(scope="module")
def checker(bfcl):
    """BFCL's official checker, loaded from the same BFCL copy."""
    return OfficialChecker()


class FakeServer:
    """Stands in for llama-server: answers each request with the task's ground-truth calls.

    Each turn it sends the ground-truth calls one per step, then a plain text answer to end the
    turn. Modes:
        gt: the correct calls.
        silent: never calls a tool.
        wrong: in turn 1, adds "X" to every text argument (wrong state and outputs).
        unsafe: in turn 0, first sends calls that try to run Python code, then the correct calls.
    """

    def __init__(self, task, bfcl, mode="gt"):
        """Work out the per-turn plan of (name, arguments) from the ground truth."""
        owner = env.method_owner(bfcl.new_instances(task.involved_classes, task.initial_config))
        self.plan = [[env.parse_call_string(c, owner) for c in turn] for turn in task.ground_truth]
        self.mode = mode

    def chat(self, messages, tools, **_):
        """Return one OpenAI-style chat reply, like llama-server's /v1/chat/completions."""
        turn = sum(m["role"] == "user" for m in messages) - 1
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        done = sum(m["role"] == "tool" and not m["tool_call_id"].startswith("bad")
                   for m in messages[last_user:])
        plan = self.plan[turn]
        if self.mode == "silent" or done >= len(plan):
            return self._reply({"role": "assistant", "content": "Done."}, "stop")
        name, kwargs = plan[done]
        if self.mode == "wrong" and turn == 1:
            kwargs = {k: (v + "X" if isinstance(v, str) else v) for k, v in kwargs.items()}
        calls = [{"id": f"c{turn}_{done}", "type": "function",
                  "function": {"name": name, "arguments": json.dumps(kwargs)}}]
        if self.mode == "unsafe" and turn == 0 and done == 0:
            calls = [
                {"id": "bad1", "type": "function",
                 "function": {"name": "__import__('os').getcwd", "arguments": "{}"}},
                {"id": "bad2", "type": "function",
                 "function": {"name": name, "arguments": json.dumps({"x=__import__('os').getcwd(),y": 1})}},
            ] + calls
        return self._reply({"role": "assistant", "content": "", "tool_calls": calls}, "tool_calls")

    @staticmethod
    def _reply(message, finish):
        """Wrap a message in the response shape the harness reads."""
        return {"choices": [{"message": message, "finish_reason": finish}], "_client_wall_ms": 0.0}


def play_and_score(task, bfcl, checker, mode="gt", **cfg):
    """Play one task through the real agent loop with the fake server and score it both ways.

    Returns:
        The comparison dict from `score_session` (official and harness results side by side).
    """
    requests = []
    session = run_session(task, bfcl, FakeServer(task, bfcl, mode), CLASSES, {**CFG, **cfg}, requests.append)
    return score_session(checker, bfcl, task, session, requests)


# ---------------------------------------------------------------- setup


def test_split_and_tool_subset(bfcl, tasks):
    """TradingBot + MessageAPI give 30 tools; the split has 13 test and 20 trace tasks, no overlap."""
    assert len(bfcl.tools_for(CLASSES)) == 30
    assert len(SPLIT["test"]) == 13 and len(SPLIT["trace"]) == 20
    assert not set(SPLIT["test"]) & set(SPLIT["trace"])
    for tid in SPLIT["test"] + SPLIT["trace"]:
        assert set(tasks[tid].involved_classes) <= set(CLASSES)


def test_tools_use_json_schema_types(bfcl):
    """BFCL-only type names (dict, float, tuple, any) must be converted to JSON Schema names."""
    text = json.dumps(bfcl.tools_for(CLASSES))
    for bfcl_type in ("dict", "float", "tuple", "any"):
        assert f'"type": "{bfcl_type}"' not in text


def test_long_context_mode_makes_outputs_longer(bfcl, tasks):
    """The watch list in long-context mode must be much longer (over 5x) than in normal mode."""
    task = tasks[SPLIT["test"][0]]
    module = importlib.import_module(bfcl.class_module["TradingBot"])
    normal = module.TradingBot()
    normal._load_scenario(json.loads(json.dumps(task.initial_config.get("TradingBot", {}))), long_context=False)
    long = bfcl.new_instances(["TradingBot"], task.initial_config)["TradingBot"]
    assert len(str(long.get_watchlist())) > 5 * len(str(normal.get_watchlist()))


# ---------------------------------------------------------------- both scorers


@pytest.mark.parametrize("tid", SPLIT["test"])
def test_ground_truth_passes_both_scorers(tid, bfcl, tasks, checker):
    """Replaying BFCL's correct calls must pass every turn, for the harness and for BFCL."""
    r = play_and_score(tasks[tid], bfcl, checker, "gt")
    assert r["official_session_success"], r["official_error_message"]
    assert r["harness_session_success"]
    assert all(r["official_turns"]) and r["agree"]


def test_silent_model_fails_both_scorers(bfcl, tasks, checker):
    """A model that never calls a tool must fail every turn (the checkers are not too easy)."""
    r = play_and_score(tasks[SPLIT["test"][0]], bfcl, checker, "silent")
    assert not r["official_session_success"] and not r["harness_session_success"]
    assert r["official_error_type"] == "multi_turn:empty_turn_model_response"
    assert not any(r["official_turns"]) and r["agree"]


@pytest.mark.parametrize("tid", SPLIT["test"])
def test_wrong_arguments_scorers_agree(tid, bfcl, tasks, checker):
    """With wrong arguments in turn 1, both scorers must give the same session and turn results."""
    r = play_and_score(tasks[tid], bfcl, checker, "wrong")
    assert r["agree"], r


def test_truncated_run_scores_played_turns_only(bfcl, tasks, checker):
    """With --max-turns 1, only turn 0 is scored and the session counts as not successful."""
    r = play_and_score(tasks[SPLIT["test"][0]], bfcl, checker, "gt", max_turns=1)
    assert r["official_turns"] == [True] and r["harness_turns"] == [True]
    assert not r["official_session_success"] and r["agree"]


# ---------------------------------------------------------------- safety


def test_unsafe_calls_are_never_given_to_eval(bfcl, tasks, checker):
    """Calls that try to run code through the tool name or an argument name are dropped."""
    task = tasks[SPLIT["test"][0]]
    r = play_and_score(task, bfcl, checker, "unsafe")
    first_step = r["model_calls"][0][0]
    assert not any("__import__" in call for call in first_step)
    assert r["official_session_success"] and r["agree"]


def test_rebuild_keeps_only_task_tools():
    """A call to a name outside the task's tools, or with bad JSON, is not rebuilt."""
    requests = [{"turn": 0, "step": 0, "calls": [
        {"name": "get_watchlist", "well_formed": True, "args": {}},
        {"name": "exit", "well_formed": True, "args": {}},
        {"name": "get_watchlist", "well_formed": False},
    ]}, {"turn": 0, "step": 1, "calls": []}]
    assert rebuild_model_calls(requests, 1, {"get_watchlist"}) == [[["get_watchlist()"]]]


# ---------------------------------------------------------------- our own long tasks


@pytest.mark.skipif(not LONG_TASKS.exists(), reason="data/long_tasks.jsonl not present")
def test_long_tasks_ground_truth_passes(bfcl, checker):
    """Every task in data/long_tasks.jsonl must pass both scorers when its ground truth is replayed."""
    for task in load_extra_tasks(str(LONG_TASKS)).values():
        r = play_and_score(task, bfcl, checker, "gt")
        assert r["official_session_success"], (task.id, r["official_error_message"])
        assert r["harness_session_success"] and r["agree"], task.id
