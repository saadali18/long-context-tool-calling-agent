"""Loading the BFCL benchmark data.

This is the first step of the benchmark. It reads everything we need from BFCL:

1. The tasks: multi-turn conversations where a user asks for things over several turns
   (e.g. "buy 10 shares of AAPL", then "send the order id to my manager").
2. The ground truth: the correct tool calls for every turn of every task.
3. The tool descriptions: name, purpose and inputs of every tool (e.g. `place_order`).

It also keeps only the tasks that fit our tool subset (TradingBot + MessageAPI), and turns
BFCL's tool descriptions into the standard "tools" format that chat models understand.

Where the data comes from
-------------------------
Everything is read from the installed `bfcl-eval` pip package (pinned version, installed
with --no-deps). The files are inside `bfcl_eval/data/`, so every group member reads exactly
the same data and nothing needs to be downloaded separately.
"""

import copy
import json
from dataclasses import dataclass, field
from importlib import resources

from bfcl_eval.constants.executable_backend_config import MULTI_TURN_FUNC_DOC_FILE_MAPPING

# BFCL writes parameter types with Python-style names; the standard tools format (JSON Schema)
# uses different names for some of them. Types not listed here are already the same.
BFCL_TO_JSON_TYPE = {
    "dict": "object",
    "float": "number",
    "tuple": "array",
    "any": "string",
}


@dataclass
class BFCLTask:
    """One BFCL task: a multi-turn conversation, its starting state and its correct answers.

    A task is what one "session" of the benchmark plays out. Each turn the user says
    something, and the model must respond with the right tool calls.

    Attributes:
        id: Task id, e.g. "multi_turn_long_context_100".
        turns: The user messages, one list per turn. Each message is a chat message dict
            like {"role": "user", "content": "..."}.
        initial_config: Starting state of each tool group (backend), e.g. the account
            balance, existing orders and watch list for TradingBot. The tools are created
            from this state at the start of the session.
        involved_classes: The tool groups this task uses, e.g. ["TradingBot", "MessageAPI"].
        ground_truth: The correct tool calls, one list per turn, written as Python call
            strings, e.g. "get_stock_info(symbol='AAPL')". BFCL's checker compares the
            model's calls against these by running both and comparing the results.
        excluded_functions: Tools BFCL hides for this task. Kept for reference only; we
            offer the same full tool set for every task.
    """

    id: str
    turns: list[list[dict]]
    initial_config: dict
    involved_classes: list[str]
    ground_truth: list[list[str]]
    excluded_functions: list[str] = field(default_factory=list)

    @property
    def num_turns(self) -> int:
        """Number of user turns in the task (same as the number of ground-truth turns)."""
        return len(self.turns)


def _data_dir():
    """Return the folder that holds the BFCL data files inside the installed `bfcl_eval` package.

    Using `importlib.resources` means the code works wherever the package is installed
    (any virtual environment, any machine), without hard-coding a path.
    """
    return resources.files("bfcl_eval") / "data"


def _read_jsonl(path) -> list[dict]:
    """Read a JSON Lines file (one JSON object per line) and return a list of dicts.

    All BFCL data files use this format. Empty lines are skipped.
    """
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_tasks(category: str, classes: list[str] | None = None) -> list[BFCLTask]:
    """Load the tasks of a BFCL category together with their correct answers.

    BFCL stores the tasks and the correct answers in two separate files with the same name
    (`BFCL_v4_<category>.json` and `possible_answer/BFCL_v4_<category>.json`). This function
    reads both and joins them by task id.

    Args:
        category: BFCL category name, e.g. "multi_turn_long_context".
        classes: The tool groups in our subset, e.g. ["TradingBot", "MessageAPI"]. If given,
            a task is kept only when every tool group it uses is inside this list, so the
            model always has all the tools a task needs. If None, all tasks are kept.

    Returns:
        A list of BFCLTask objects, in BFCL's file order.
    """
    data_dir = _data_dir()
    entries = _read_jsonl(data_dir / f"BFCL_v4_{category}.json")
    answers = _read_jsonl(data_dir / "possible_answer" / f"BFCL_v4_{category}.json")
    ground_truth = {a["id"]: a["ground_truth"] for a in answers}

    tasks = []
    for entry in entries:
        if classes is not None and not set(entry["involved_classes"]) <= set(classes):
            continue
        tasks.append(
            BFCLTask(
                id=entry["id"],
                turns=entry["question"],
                initial_config=entry["initial_config"],
                involved_classes=entry["involved_classes"],
                ground_truth=ground_truth[entry["id"]],
                excluded_functions=entry.get("excluded_function", []),
            )
        )
    return tasks


def load_function_docs(classes: list[str]) -> list[dict]:
    """Load the description of every tool in the given tool groups.

    Each tool group (backend) has its own description file in `data/multi_turn_func_doc/`
    (e.g. TradingBot -> `trading_bot.json`). BFCL's own mapping is used to find the file.

    Args:
        classes: The tool groups, e.g. ["TradingBot", "MessageAPI"].

    Returns:
        One dict per tool, exactly as BFCL writes it: "name", "description", "parameters"
        (the inputs, with BFCL type names) and "response" (what the tool returns).
    """
    docs = []
    for class_name in classes:
        file_name = MULTI_TURN_FUNC_DOC_FILE_MAPPING[class_name]
        docs.extend(_read_jsonl(_data_dir() / "multi_turn_func_doc" / file_name))
    return docs


def _to_json_schema(schema: dict) -> dict:
    """Rename BFCL type names to standard JSON Schema type names in one parameter schema.

    For example {"type": "dict", "properties": {"price": {"type": "float"}}} becomes
    {"type": "object", "properties": {"price": {"type": "number"}}}. Nested properties and
    list items are converted too (the function calls itself on them). The input is not changed;
    a converted copy is returned.
    """
    schema = dict(schema)
    if "type" in schema:
        schema["type"] = BFCL_TO_JSON_TYPE.get(schema["type"], schema["type"])
    if "properties" in schema:
        schema["properties"] = {k: _to_json_schema(v) for k, v in schema["properties"].items()}
    if "items" in schema:
        schema["items"] = _to_json_schema(schema["items"])
    return schema


def to_openai_tools(function_docs: list[dict]) -> list[dict]:
    """Convert BFCL tool descriptions to the standard OpenAI "tools" format.

    This is the format chat models (and llama.cpp's server) expect in the `tools` field of a
    chat request: a list of {"type": "function", "function": {name, description, parameters}}.
    The parameter types are converted with `_to_json_schema`. BFCL's "response" field is
    dropped because it only describes the return value, which the model does not need.

    Args:
        function_docs: The output of `load_function_docs`.

    Returns:
        One tool dict per function, in the same order.
    """
    tools = []
    for doc in copy.deepcopy(function_docs):
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": doc["name"],
                    "description": doc["description"],
                    "parameters": _to_json_schema(doc["parameters"]),
                },
            }
        )
    return tools
