"""Automatic checks for the BFCL benchmark code. Run with `pytest` from the repo root.

No model or server is needed. The tests check that the data is loaded as expected, that
correct answers pass, that wrong behaviour fails, that unsafe calls are blocked, and that
long-context mode really makes tool outputs longer.
"""

import json

import pytest
import yaml

from bfcl.evaluator import evaluate_task
from bfcl.loader import load_function_docs, load_tasks, to_openai_tools
from bfcl.task_runner import GroundTruthCaller, ModelReply, ToolCall, run_task, to_call_string

CONFIG = yaml.safe_load(open("configs/bfcl.yaml"))
CATEGORY, CLASSES = CONFIG["category"], CONFIG["classes"]


@pytest.fixture(scope="module")
def tasks():
    """All tasks of our tool subset (loaded once for the whole test file)."""
    return load_tasks(CATEGORY, CLASSES)


@pytest.fixture(scope="module")
def docs():
    """Tool descriptions of our tool subset, as BFCL writes them."""
    return load_function_docs(CLASSES)


@pytest.fixture(scope="module")
def tools(docs):
    """Tool descriptions converted to the OpenAI tools format."""
    return to_openai_tools(docs)


def run_and_score(task, caller, tools):
    """Helper: play one task with the given caller in long-context mode and mark it."""
    trace = run_task(task, caller, tools, CLASSES, long_context=True)
    return evaluate_task(task, trace, CATEGORY)


# ---------------------------------------------------------------- loader


def test_subset_has_expected_size(tasks, tools):
    """TradingBot + MessageAPI must give 30 tools and 33 tasks, all inside the subset."""
    assert len(tools) == 30
    assert len(tasks) == 33
    assert all(set(t.involved_classes) <= set(CLASSES) for t in tasks)


def test_tools_use_json_schema_types(tools):
    """BFCL-only type names (dict, float) must be converted to standard names."""
    text = json.dumps(tools)
    assert '"type": "dict"' not in text and '"type": "float"' not in text


# ---------------------------------------------------------------- execution


def test_call_string_format():
    """Tool calls must be written in the exact Python-call text form BFCL runs and checks."""
    assert to_call_string("place_order", {"symbol": "AAPL", "amount": 10}) == "place_order(symbol='AAPL',amount=10)"


# ---------------------------------------------------------------- end to end


def test_ground_truth_replay_passes(tasks, docs, tools):
    """Replaying BFCL's correct calls must pass the session and every turn."""
    for task in tasks[:3]:
        result = run_and_score(task, GroundTruthCaller(task, docs), tools)
        assert result["session_success"], result["error_message"]
        assert all(result["turn_correct"])


class SilentCaller:
    """A fake model that never calls a tool. Used to show that doing nothing fails."""

    def next_message(self, messages, tools):
        """Always answer with plain text and no tool calls."""
        return ModelReply(message={"role": "assistant", "content": "I can't help."})


def test_silent_model_fails(tasks, tools):
    """A model that does nothing must fail the session and every turn (the checker is not too easy)."""
    result = run_and_score(tasks[0], SilentCaller(), tools)
    assert not result["session_success"]
    assert not any(result["turn_correct"])


class UnsafeCaller:
    """A fake model that tries to run Python code through the tool name and an input name."""

    def __init__(self):
        """Start with nothing sent yet."""
        self.sent = False

    def next_message(self, messages, tools):
        """Send the two harmful calls once, then answer with plain text."""
        if self.sent:
            return ModelReply(message={"role": "assistant", "content": "ok"})
        self.sent = True
        calls = [
            ToolCall("a", "__import__('os').getcwd", "{}"),
            ToolCall("b", "get_stock_info", '{"symbol=__import__(\'os\').getcwd(),x": 1}'),
        ]
        return ModelReply(message={"role": "assistant", "content": ""}, tool_calls=calls)


def test_unsafe_calls_are_not_executed(tasks, tools):
    """Harmful calls must not run: each gets an error output and nothing is recorded as a call."""
    trace = run_task(tasks[0], UnsafeCaller(), tools, CLASSES, long_context=True)
    first_step_calls = trace["steps"][0]["tool_calls"]
    assert all(c["output"].startswith("Error:") for c in first_step_calls)
    assert trace["model_calls"][0] == []


def test_long_context_mode_makes_outputs_longer(tasks, tools):
    """The watch list output in long-context mode must be much longer (over 5x) than in normal mode."""
    task = next(t for t in tasks if "TradingBot" in t.involved_classes)

    class WatchlistCaller(UnsafeCaller):
        """A fake model that calls get_watchlist once."""

        def next_message(self, messages, tools):
            """Call get_watchlist once, then answer with plain text."""
            if self.sent:
                return ModelReply(message={"role": "assistant", "content": "ok"})
            self.sent = True
            return ModelReply(message={"role": "assistant"}, tool_calls=[ToolCall("a", "get_watchlist", "{}")])

    short = run_task(task, WatchlistCaller(), tools, CLASSES, long_context=False)
    long = run_task(task, WatchlistCaller(), tools, CLASSES, long_context=True)
    get_output = lambda trace: trace["steps"][0]["tool_calls"][0]["output"]  # noqa: E731
    assert len(get_output(long)) > 5 * len(get_output(short))
