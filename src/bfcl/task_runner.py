"""Playing out one BFCL session (conversation) turn by turn.

A BFCL task is a conversation of several user turns. In each turn, whoever is answering
(the "caller") may call tools; the tools run on BFCL's simulated backends (a fake trading
account, a fake messaging app, ...) and their outputs are added to the conversation, so the
caller can see them and call more tools. The turn ends when the caller replies without any
tool call. This file runs that loop and records everything that happened.

How one session runs
--------------------
    for each user turn:
        add the user message to the history
        repeat (at most max_steps_per_turn times):
            ask the caller for its next message
            no tool calls  -> the turn is over
            tool calls     -> run them on the BFCL tools, add the outputs to the history, ask again

Long-context mode
-----------------
With `long_context=True`, BFCL's tools return much longer outputs (e.g. TradingBot adds
many extra stocks to the watch list and long transaction histories). This is what makes
the sessions grow to many thousands of tokens.

Callers
-------
A caller is any object with a method `next_message(messages, tools) -> ModelReply`.
`GroundTruthCaller` (below) replays BFCL's correct calls; it is used to check that the
benchmark itself works (it must pass every task).
"""

import ast
import json
import uuid
from dataclasses import dataclass, field

from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call

from bfcl.loader import BFCLTask


@dataclass
class ToolCall:
    """One tool call made by the caller.

    Attributes:
        id: An id for this call. The tool's output message points back to it with
            `tool_call_id`, so the caller knows which output belongs to which call.
        name: The tool name, e.g. "place_order".
        raw_arguments: The inputs as a JSON string, exactly as the caller produced it,
            e.g. '{"symbol": "AAPL", "amount": 10}'. It is kept as text (not parsed) because
            a model may produce broken JSON, and that must be handled, not crash the run.
    """

    id: str
    name: str
    raw_arguments: str


@dataclass
class ModelReply:
    """One answer from the caller.

    Attributes:
        message: The assistant message to add to the conversation history (an OpenAI-style
            chat message, with "tool_calls" inside it when there are calls).
        tool_calls: The tool calls in that message, as ToolCall objects. Empty means the
            caller is done with the current turn.
    """

    message: dict
    tool_calls: list[ToolCall] = field(default_factory=list)


class GroundTruthCaller:
    """A fake "perfect model" that answers every turn with BFCL's correct tool calls.

    Why it exists: before testing any real model, we need to know that the benchmark itself is
    right (data loading, running the tools, converting calls, scoring). If this caller does
    not score 100% session success, the problem is in the benchmark code, not in a model.

    How it answers: when the last message is from the user, it returns all correct calls of
    the current turn at once. When the last message is a tool output, those calls have run,
    so it replies "Done." with no calls, which ends the turn.
    """

    def __init__(self, task: BFCLTask, function_docs: list[dict]):
        """Prepare the caller for one task.

        Args:
            task: The task whose correct calls will be replayed.
            function_docs: Tool descriptions from `loader.load_function_docs`. Used to know
                the order of each tool's inputs, because a few correct calls are written
                without input names (e.g. get_stock_info('OMEG')), and we need the name
                ("symbol") to build a proper tool call.
        """
        self.task = task
        self.param_order = {d["name"]: list(d["parameters"]["properties"]) for d in function_docs}

    def next_message(self, messages: list[dict], tools: list[dict]) -> ModelReply:
        """Return the next answer: the correct calls of this turn, or "Done." after they ran.

        Args:
            messages: The conversation so far. The current turn number is found by counting
                the user messages in it.
            tools: The offered tools (not needed here, but part of the caller interface).

        Returns:
            A ModelReply with the correct calls, or with no calls to end the turn.
        """
        if messages[-1]["role"] == "tool":
            return ModelReply(message={"role": "assistant", "content": "Done."})

        turn_index = sum(1 for m in messages if m["role"] == "user") - 1
        tool_calls = [
            self._to_tool_call(call_string, i)
            for i, call_string in enumerate(self.task.ground_truth[turn_index])
        ]
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.raw_arguments}}
                for c in tool_calls
            ],
        }
        return ModelReply(message=message, tool_calls=tool_calls)

    def _to_tool_call(self, call_string: str, index: int) -> ToolCall:
        """Turn one correct-answer string into a ToolCall, the same shape a model would produce.

        Example: "place_order(symbol='AAPL', amount=10)" becomes
        ToolCall(name="place_order", raw_arguments='{"symbol": "AAPL", "amount": 10}').
        The string is read with Python's `ast` module (safe parsing, nothing is executed).
        Inputs given without a name are named using the tool's input order.
        """
        node = ast.parse(call_string, mode="eval").body
        name = node.func.id
        args = {self.param_order[name][i]: ast.literal_eval(a) for i, a in enumerate(node.args)}
        args.update({kw.arg: ast.literal_eval(kw.value) for kw in node.keywords})
        return ToolCall(id=f"call_{index}", name=name, raw_arguments=json.dumps(args))


# ---------------------------------------------------------------- running tool calls


def to_call_string(name: str, args: dict) -> str:
    """Write a tool call as the Python call text that BFCL runs and checks.

    Example: to_call_string("get_stock_info", {"symbol": "AAPL"}) -> "get_stock_info(symbol='AAPL')".
    Values are written with `repr`, so strings get quotes and numbers stay numbers.
    """
    return f"{name}({','.join(f'{k}={v!r}' for k, v in args.items())})"


def release_backends(run_name: str) -> None:
    """Delete the tool objects BFCL kept in memory for one run.

    BFCL stores its tool objects (e.g. the TradingBot account) as global variables named
    after the run, so that state carries over from one turn to the next. If they were not
    removed, a later run with the same name would continue from old state, and memory would
    keep growing. Every run uses a unique name and calls this at the end.

    Args:
        run_name: The name prefix used for the run.
    """
    for key in [k for k in vars(multi_turn_utils) if k.startswith(run_name)]:
        del vars(multi_turn_utils)[key]


def _parse_arguments(call: ToolCall, tool_names: set[str]) -> tuple[dict | None, str]:
    """Check that a tool call is safe and well-formed before running it.

    BFCL runs tool calls by evaluating Python text (`eval`). A model could produce a tool name
    or input name that contains code, so only these calls are allowed:
    the tool name is one of the offered tools, the inputs are a valid JSON object, and every
    input name is a plain name (letters, digits, underscore). Input values are safe because
    `to_call_string` writes them with `repr`.

    Args:
        call: The tool call to check.
        tool_names: Names of the offered tools.

    Returns:
        (inputs as a dict, "") when the call can run, or (None, error message) when it
        cannot. The error message is shown to the caller as the tool's output.
    """
    if call.name not in tool_names:
        return None, f"Error: '{call.name}' is not an available tool."
    try:
        args = json.loads(call.raw_arguments or "{}")
    except json.JSONDecodeError:
        args = None
    if not isinstance(args, dict):
        return None, "Error: tool call arguments are not a valid JSON object."
    if not all(isinstance(k, str) and k.isidentifier() for k in args):
        return None, "Error: invalid argument name."
    return args, ""


def _execute_step(task, tool_calls, tool_names, backend_classes, run_name, long_context):
    """Run all tool calls of one step on the BFCL tools.

    Each call is checked with `_parse_arguments`; the ones that pass are run together through
    BFCL's own `execute_multi_turn_func_call`, the same function BFCL uses in its official runs.

    Args:
        task: The task (gives the starting state of the tools and the task id).
        tool_calls: The ToolCall objects of this step.
        tool_names: Names of the offered tools.
        backend_classes: The tool groups to create, e.g. ["TradingBot", "MessageAPI"].
        run_name: Unique name of this run, so the tools keep their state between steps.
        long_context: Run the tools in long-context mode (longer outputs).

    Returns:
        (call strings that were run, one output text per tool call in the same order).
        These call strings are what BFCL's checker scores later.
    """
    call_strings, outputs, runnable_indices = [], [], []
    for i, call in enumerate(tool_calls):
        args, error = _parse_arguments(call, tool_names)
        outputs.append(error)
        if args is not None:
            call_strings.append(to_call_string(call.name, args))
            runnable_indices.append(i)

    if call_strings:
        results, _ = execute_multi_turn_func_call(
            func_call_list=call_strings,
            initial_config=task.initial_config,
            involved_classes=backend_classes,
            model_name=run_name,
            test_entry_id=task.id,
            long_context=long_context,
        )
        for i, result in zip(runnable_indices, results):
            outputs[i] = result
    return call_strings, outputs


def run_task(
    task: BFCLTask,
    caller,
    tools: list[dict],
    backend_classes: list[str],
    long_context: bool,
    max_steps_per_turn: int = 20,
) -> dict:
    """Play out one full task (all turns) and return a record of what happened.

    For each turn the user message is added, then the caller is asked for its next message
    again and again: its tool calls are run and their outputs added, until it answers without
    a tool call. If a turn reaches `max_steps_per_turn` steps, the session stops early and is
    marked `force_terminated` (BFCL counts that as a failed session).

    Args:
        task: The task to play.
        caller: Anything with `next_message(messages, tools) -> ModelReply`.
        tools: The offered tools in OpenAI format (from `loader.to_openai_tools`).
        backend_classes: The tool groups to create (our subset).
        long_context: Run the tools in long-context mode.
        max_steps_per_turn: Most caller answers allowed in one turn (BFCL uses 20).

    Returns:
        A dict with:
            id: the task id.
            model_calls: the calls that were run, as turn -> step -> list of call strings.
                This is the exact shape BFCL's checker expects.
            force_terminated: True if a turn hit the step limit.
            steps: one entry per caller answer (turn, step, text, and each tool call with
                its inputs and output), for reading and debugging.
            context_chars: size of the whole session (tool descriptions + history) in
                characters, as a tokenizer-free measure of how long the context got.
    """
    run_name = f"run_{uuid.uuid4().hex[:8]}"
    tool_names = {t["function"]["name"] for t in tools}

    messages: list[dict] = []
    model_calls: list[list[list[str]]] = []
    steps: list[dict] = []
    force_terminated = False

    try:
        for turn_index, user_messages in enumerate(task.turns):
            messages.extend(user_messages)
            turn_calls = []
            for step_index in range(max_steps_per_turn):
                reply = caller.next_message(messages, tools)
                messages.append(reply.message)

                call_strings, outputs = _execute_step(
                    task, reply.tool_calls, tool_names, backend_classes, run_name, long_context
                )
                for call, output in zip(reply.tool_calls, outputs):
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
                if call_strings:
                    turn_calls.append(call_strings)

                steps.append(
                    {
                        "turn": turn_index,
                        "step": step_index,
                        "content": reply.message.get("content", ""),
                        "tool_calls": [
                            {"name": c.name, "arguments": c.raw_arguments, "output": o}
                            for c, o in zip(reply.tool_calls, outputs)
                        ],
                    }
                )
                if not reply.tool_calls:
                    break
            else:
                force_terminated = True

            model_calls.append(turn_calls)
            if force_terminated:
                break
    finally:
        release_backends(run_name)

    return {
        "id": task.id,
        "model_calls": model_calls,
        "force_terminated": force_terminated,
        "steps": steps,
        "context_chars": len(json.dumps(tools)) + len(json.dumps(messages)),
    }
