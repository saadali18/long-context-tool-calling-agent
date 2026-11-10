"""Command-line entry point: run the BFCL benchmark on our tool subset and score it.

What this script does
---------------------
1. Reads the config (BFCL category, tool subset, step limit).
2. Loads the tasks and tool descriptions (src/bfcl/loader.py).
3. For every task: plays the session with the chosen caller (src/bfcl/task_runner.py) and
   marks it (src/bfcl/evaluator.py). One line per task is printed (PASS/FAIL, turns correct).
4. Adds up all marks (src/bfcl/metrics.py), prints the summary, and saves everything to
   outputs/<name>/ (config.yaml, results.jsonl, summary.json).

Example:
    # Check the benchmark by replaying BFCL's correct calls; every task must pass.
    python scripts/run_bfcl.py --caller ground_truth --name gt_all
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from bfcl.evaluator import evaluate_task
from bfcl.loader import load_function_docs, load_task_file, load_tasks, to_openai_tools
from bfcl.metrics import summarize
from bfcl.task_runner import GroundTruthCaller, run_task


def parse_args():
    """Read the command-line options.

    Options:
        --config: config file (default configs/bfcl.yaml).
        --caller: who makes the tool calls; "ground_truth" replays BFCL's correct calls.
        --limit: run only the first N tasks.
        --ids: run only the given task ids.
        --name: name of the output folder (default <caller>_<date>_<time>).
        --out-dir: parent folder for output folders (default outputs).
        --tasks: which tasks to run: all (default), bfcl (BFCL's only) or extra (our files only).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/bfcl.yaml")
    parser.add_argument("--caller", choices=["ground_truth"], default="ground_truth")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N tasks")
    parser.add_argument("--ids", nargs="*", help="run only these task ids")
    parser.add_argument("--name", default=None, help="run folder name (default: <caller>_<time>)")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument(
        "--tasks",
        choices=["all", "bfcl", "extra"],
        default="all",
        help="all = BFCL tasks + our task files; bfcl = BFCL tasks only; extra = our task files only",
    )
    return parser.parse_args()


def make_caller(name: str, task, function_docs):
    """Create the caller for one task, chosen by the --caller option.

    A new caller is made for every task because a caller may keep per-task information
    (GroundTruthCaller holds the task's correct answers). New caller types are added here.

    Args:
        name: The --caller value.
        task: The task about to be run.
        function_docs: Tool descriptions (needed by GroundTruthCaller).

    Returns:
        An object with `next_message(messages, tools)`.
    """
    if name == "ground_truth":
        return GroundTruthCaller(task, function_docs)
    raise ValueError(f"unknown caller: {name}")


def main():
    """Run every selected task, mark it, print progress and save the results and summary."""
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    long_context = "long_context" in cfg["category"]

    function_docs = load_function_docs(cfg["classes"])
    tools = to_openai_tools(function_docs)
    extra_files = cfg.get("extra_task_files", []) if args.tasks != "bfcl" else []
    tasks = load_tasks(cfg["category"], cfg["classes"], extra_files)
    if args.tasks == "extra":
        extra_ids = {t.id for path in extra_files for t in load_task_file(path, cfg["classes"])}
        tasks = [t for t in tasks if t.id in extra_ids]
    if args.ids:
        tasks = [t for t in tasks if t.id in set(args.ids)]
    tasks = tasks[: args.limit] if args.limit else tasks

    run_name = args.name or f"{args.caller}_{datetime.now():%Y%m%d_%H%M%S}"
    out = Path(args.out_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out / "config.yaml")
    print(f"{len(tasks)} tasks, {len(tools)} tools -> {out}")

    results = []
    with open(out / "results.jsonl", "w") as f:
        for i, task in enumerate(tasks, 1):
            caller = make_caller(args.caller, task, function_docs)
            trace = run_task(task, caller, tools, cfg["classes"], long_context, cfg["max_steps_per_turn"])
            result = evaluate_task(task, trace, cfg["category"])
            results.append(result)
            f.write(json.dumps(result) + "\n")
            turns = f"{sum(result['turn_correct'])}/{len(result['turn_correct'])}"
            status = "PASS" if result["session_success"] else f"FAIL ({result['error_type']})"
            print(f"[{i}/{len(tasks)}] {task.id}: {status}, turns correct {turns}")

    summary = summarize(results)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
