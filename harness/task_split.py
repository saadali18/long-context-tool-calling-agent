"""Select BFCL and custom long-task IDs without train/test leakage.

The existing split's test and trace lists contain the original BFCL
identifiers. extra_tasks contains matching assignments for the custom tasks.
trace is the development/tuning set; test is held out.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Collection, Mapping
from pathlib import Path


class TaskSplitError(ValueError):
    """Split configuration is missing or inconsistent with supplied tasks."""


def _ids(mapping: Mapping, key: str, label: str) -> list[str]:
    ids = mapping.get(key)
    if not isinstance(ids, list) or any(not isinstance(i, str) or not i for i in ids):
        raise TaskSplitError(f"{label}.{key} must be a list of nonempty task IDs")
    if len(ids) != len(set(ids)):
        raise TaskSplitError(f"duplicate task IDs in {label}.{key}")
    return ids


def select_task_ids(split: Mapping, which: str, extra_task_ids: Collection[str] | None = None) -> list[str]:
    """Select IDs for test, trace/dev, or all, verifying all extra assignments.

    Without --extra-tasks the original BFCL-only split remains unchanged.
    If custom tasks are supplied, every task MUST be assigned to exactly one
    split, or the run fails before inference starts.
    """
    if which not in ("test", "trace", "dev", "all"):
        raise TaskSplitError(f"invalid split {which!r}; expected test, trace, dev or all")
    if not isinstance(split, Mapping):
        raise TaskSplitError("split must be a JSON object")
    which = "trace" if which == "dev" else which
    base_test = _ids(split, "test", "split")
    base_trace = _ids(split, "trace", "split")
    if set(base_test) & set(base_trace):
        raise TaskSplitError("BFCL task assigned to both test and trace")
    chosen = base_test + base_trace if which == "all" else (base_test if which == "test" else base_trace)

    if extra_task_ids is None:
        return list(chosen)

    extra = split.get("extra_tasks")
    if not isinstance(extra, Mapping):
        raise TaskSplitError("--extra-tasks requires extra_tasks assignments in the split file")
    custom_test = _ids(extra, "test", "extra_tasks")
    custom_trace = _ids(extra, "trace", "extra_tasks")
    if set(custom_test) & set(custom_trace):
        raise TaskSplitError("custom task assigned to both test and trace")
    supplied = list(extra_task_ids)
    if len(supplied) != len(set(supplied)):
        raise TaskSplitError("duplicate custom task IDs in --extra-tasks")
    assigned = set(custom_test) | set(custom_trace)
    base = set(base_test) | set(base_trace)
    if assigned & base:
        raise TaskSplitError("custom task IDs overlap BFCL task IDs")
    if assigned != set(supplied):
        raise TaskSplitError(
            "custom task assignment mismatch: "
            f"unassigned={sorted(set(supplied) - assigned)}, "
            f"not_loaded={sorted(assigned - set(supplied))}"
        )
    chosen_extra = custom_test + custom_trace if which == "all" else (
        custom_test if which == "test" else custom_trace
    )
    return list(chosen) + chosen_extra


def load_task_ids(path: str | Path) -> list[str]:
    """Read custom task IDs from their JSONL source, for dry-run validation."""
    ids = []
    with open(path, encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
                raise TaskSplitError(f"invalid task id in {path} line {line_no}")
            ids.append(row["id"])
    if len(ids) != len(set(ids)):
        raise TaskSplitError(f"duplicate custom task IDs in {path}")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate task splits without starting a model or BFCL")
    parser.add_argument("--split", default="splits/tasks_TradingBot_MessageAPI.json")
    parser.add_argument("--extra-tasks", default="data/long_tasks.jsonl")
    parser.add_argument("--which", choices=["test", "trace", "dev", "all"], default="test")
    args = parser.parse_args()
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    ids = select_task_ids(split, args.which, load_task_ids(args.extra_tasks))
    print(f"{args.which}: {len(ids)} tasks")
    for task_id in ids:
        print(task_id)


if __name__ == "__main__":
    main()
