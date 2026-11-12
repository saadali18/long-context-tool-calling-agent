"""BFCL multi-turn environment for the A1 harness.

Loads BFCL tasks, tool schemas and the simulated backends (TradingBot, MessageAPI,
...) straight from the BFCL source, and scores sessions the same way BFCL's
multi_turn_checker does (state check + response check after every turn).

Differences from BFCL's own code, on purpose:
  * model calls arrive as JSON tool calls and are executed by method lookup,
    not by eval() of a string;
  * every turn is checked even after a failure, so per-turn accuracy can be
    reported next to session success (BFCL stops at the first failed turn).
"""

from __future__ import annotations

import ast
import copy
import importlib
import inspect
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# BFCL type names -> JSON-schema types (same table as BFCL's GORILLA_TO_OPENAPI)
_TYPE_MAP = {
    "integer": "integer", "number": "number", "float": "number", "string": "string",
    "boolean": "boolean", "bool": "boolean", "array": "array", "list": "array",
    "dict": "object", "object": "object", "tuple": "array", "any": "string",
}

DATASET = "BFCL_v4_multi_turn_long_context.json"
STATELESS = {"MathAPI"}
MAX_STEPS_PER_TURN = 20  # BFCL's MAXIMUM_STEP_LIMIT


@dataclass
class Task:
    id: str
    turns: list            # list of turns; each turn = list of {"role","content"} messages
    initial_config: dict
    involved_classes: list
    ground_truth: list     # list (per turn) of call strings, e.g. "get_watchlist()"
    excluded_functions: list = field(default_factory=list)
    source: str = "bfcl"


class BFCL:
    """Access to one BFCL checkout (berkeley-function-call-leaderboard folder)."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not (self.root / "bfcl_eval").is_dir():
            raise FileNotFoundError(f"{self.root} has no bfcl_eval/ folder")
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        cfg = importlib.import_module("bfcl_eval.constants.executable_backend_config")
        self.class_module = cfg.CLASS_FILE_PATH_MAPPING
        self.doc_file = cfg.MULTI_TURN_FUNC_DOC_FILE_MAPPING
        self.data = self.root / "bfcl_eval" / "data"

    # ---------- tasks ----------
    def load_tasks(self) -> dict[str, Task]:
        gt = {}
        for line in open(self.data / "possible_answer" / DATASET):
            row = json.loads(line)
            gt[row["id"]] = row["ground_truth"]
        tasks = {}
        for line in open(self.data / DATASET):
            r = json.loads(line)
            tasks[r["id"]] = Task(
                id=r["id"], turns=r["question"], initial_config=r["initial_config"],
                involved_classes=r["involved_classes"], ground_truth=gt[r["id"]],
                excluded_functions=r.get("excluded_function", []),
            )
        return tasks

    # ---------- tool schemas ----------
    def func_docs(self, class_name: str) -> list[dict]:
        path = self.data / "multi_turn_func_doc" / self.doc_file[class_name]
        return [json.loads(line) for line in open(path)]

    def tools_for(self, classes: list[str], excluded: list[str] = ()) -> list[dict]:
        """OpenAI-style tool list for the given classes (BFCL-equivalent conversion)."""
        tools = []
        for c in classes:
            for doc in self.func_docs(c):
                if doc["name"] in excluded:
                    continue
                params = copy.deepcopy(doc["parameters"])
                params["type"] = "object"
                params["properties"] = _cast_types(params.get("properties", {}))
                tools.append({"type": "function", "function": {
                    "name": doc["name"], "description": doc["description"],
                    "parameters": params}})
        return tools

    # ---------- backends ----------
    def new_instances(self, classes: list[str], initial_config: dict) -> dict:
        """Fresh backend objects in long-context mode, loaded with the task's scenario."""
        out = {}
        for c in classes:
            cls = getattr(importlib.import_module(self.class_module[c]), c)
            obj = cls()
            if c not in STATELESS:
                obj._load_scenario(copy.deepcopy(initial_config.get(c, {})), long_context=True)
            out[c] = obj
        return out


def _cast_types(props: dict) -> dict:
    for key, val in props.items():
        t = val.get("type", "string")
        if t == "float":
            val["format"] = "float"
            val["description"] = val.get("description", "") + " This is a float type value."
        val["type"] = _TYPE_MAP.get(t, "string")
        if val["type"] in ("array", "object"):
            if "properties" in val:
                val["properties"] = _cast_types(val["properties"])
            elif "items" in val and "type" in val["items"]:
                it = val["items"]
                it["type"] = _TYPE_MAP.get(it["type"], "string")
                if it["type"] == "object" and "properties" in it:
                    it["properties"] = _cast_types(it["properties"])
                elif it["type"] == "array" and "items" in it and "type" in it["items"]:
                    it["items"]["type"] = _TYPE_MAP.get(it["items"]["type"], "string")
    return props


# ---------- executing calls ----------
def method_owner(instances: dict) -> dict:
    """public method name -> backend object (raises on a name clash)."""
    owner = {}
    for obj in instances.values():
        for name, _ in inspect.getmembers(obj, predicate=inspect.ismethod):
            if name.startswith("_"):
                continue
            if name in owner and owner[name] is not obj:
                raise ValueError(f"method name clash in tool subset: {name}")
            owner[name] = obj
    return owner


def _to_text(result) -> str:
    # same conversion as BFCL's execute_multi_turn_func_call
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        try:
            return json.dumps(result)
        except Exception:
            return str(result)
    return str(result)


BLOCKED = {"kill", "exit", "quit", "remove", "unlink", "popen", "Popen", "run"}


def execute(owner: dict, name: str, kwargs: dict) -> str:
    if name in BLOCKED:
        return f"Error during execution: Function call {name} is not allowed."
    obj = owner.get(name)
    if obj is None:
        return f"Error during execution: name '{name}' is not defined"
    try:
        return _to_text(getattr(obj, name)(**kwargs))
    except Exception as e:  # noqa: BLE001 - BFCL returns the error text to the model
        return f"Error during execution: {e}"


def parse_call_string(call: str, owner: dict) -> tuple[str, dict]:
    """'get_stock_info(symbol="X")' or "get_stock_info('X')" -> (name, kwargs). No eval."""
    node = ast.parse(call.strip(), mode="eval").body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError(f"not a simple call: {call}")
    name = node.func.id
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
    if node.args:
        obj = owner.get(name)
        if obj is None:
            raise ValueError(f"unknown function in ground truth: {name}")
        params = [p for p in inspect.signature(getattr(obj, name)).parameters]
        for p, a in zip(params, node.args):
            kwargs[p] = ast.literal_eval(a)
    return name, kwargs


# ---------- validity of a model call ----------
def check_call(tools_by_name: dict, name: str, raw_args) -> dict:
    """Spec's call validity: well formed, existing tool, schema-valid arguments."""
    res = {"name": name, "well_formed": True, "exists": True, "schema_valid": True, "problems": []}
    args = raw_args
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError:
            res.update(well_formed=False, schema_valid=False)
            res["problems"].append("arguments are not valid JSON")
            return res
    if not isinstance(args, dict):
        res.update(well_formed=False, schema_valid=False)
        res["problems"].append("arguments are not a JSON object")
        return res
    res["args"] = args
    tool = tools_by_name.get(name)
    if tool is None:
        res.update(exists=False, schema_valid=False)
        res["problems"].append("no such tool")
        return res
    schema = tool["function"]["parameters"]
    props = schema.get("properties", {})
    for req in schema.get("required", []):
        if req not in args:
            res["problems"].append(f"missing required '{req}'")
    for k, v in args.items():
        if k not in props:
            res["problems"].append(f"unknown argument '{k}'")
        elif not _type_ok(v, props[k]):
            res["problems"].append(f"'{k}' should be {props[k].get('type')}")
    res["schema_valid"] = not res["problems"]
    return res


def _type_ok(v, spec: dict) -> bool:
    t = spec.get("type")
    if v is None:
        return True  # optional args are often passed as null
    ok = {
        "string": isinstance(v, str),
        "integer": isinstance(v, int) and not isinstance(v, bool),
        "number": isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": isinstance(v, bool),
        "array": isinstance(v, list),
        "object": isinstance(v, dict),
    }.get(t, True)
    if ok and t == "array" and "items" in spec:
        return all(_type_ok(x, spec["items"]) for x in v)
    return ok


# ---------- scoring (BFCL multi_turn_checker logic) ----------
def public_state(obj) -> dict:
    return {k: v for k, v in vars(obj).items() if not k.startswith("_")}


def state_matches(model_obj, gt_obj) -> tuple[bool, dict]:
    diffs = {}
    for k in vars(gt_obj):
        if k.startswith("_"):
            continue
        if getattr(model_obj, k) != getattr(gt_obj, k):
            diffs[k] = {"model": getattr(model_obj, k), "ground_truth": getattr(gt_obj, k)}
    return not diffs, diffs


def responses_contained(gt_results: list, model_results_so_far: list) -> list:
    """Return GT results missing from the model's results (unordered, with multiplicity)."""
    pool = list(model_results_so_far)
    missing = []
    for item in gt_results:
        try:
            pool.remove(item)
        except ValueError:
            missing.append(item)
    return missing
