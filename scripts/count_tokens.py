"""Count real Qwen3 tokens in the sessions of a finished run (Task 1: long-session sizes).

`run_bfcl.py` measures session size in characters. Characters are only a rough guide: the
Qwen tokenizer splits numbers into single digits, and BFCL's long-context outputs are mostly
numbers, so "characters / 4" can badly underestimate the token count. This script re-counts
a run with the real tokenizer.

What is counted, per task: the tool descriptions, every user message, every model message
and tool call, and every tool output, in session order. The chat template adds a few tokens
per message on top of this, so the server's own count will be slightly higher.

Setup (once):
    pip install tokenizers huggingface_hub
    hf download Qwen/Qwen3-1.7B tokenizer.json --local-dir models/qwen3-tokenizer

Example:
    python scripts/count_tokens.py outputs/gt_long
"""

import argparse
import json
from pathlib import Path

import yaml
from tokenizers import Tokenizer

from bfcl.loader import load_function_docs, load_tasks, to_openai_tools


def parse_args():
    """Read the command-line options (run folder, tokenizer file, config)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="run folder written by run_bfcl.py, e.g. outputs/gt_long")
    parser.add_argument("--tokenizer", default="models/qwen3-tokenizer/tokenizer.json")
    parser.add_argument("--config", default="configs/bfcl.yaml")
    return parser.parse_args()


def count_session(result: dict, task, tools_text: str, tok: Tokenizer) -> dict:
    """Count tokens for one task's session.

    Args:
        result: One line of results.jsonl (has the steps: model text, tool calls, outputs).
        task: The BFCLTask (has the user messages).
        tools_text: The tool descriptions as sent to the model (JSON text).
        tok: The Qwen tokenizer.

    Returns:
        A dict with the total tokens, total characters, tokens after each turn, and the
        largest single tool output (tokens and which call produced it).
    """

    def n(text: str) -> int:
        return len(tok.encode(text).ids) if text else 0

    total_tokens, total_chars = n(tools_text), len(tools_text)
    tokens_after_turn = []
    largest = (0, "")
    per_tool: dict[str, list[int]] = {}

    for turn_index, user_messages in enumerate(task.turns):
        for message in user_messages:
            total_tokens += n(message["content"])
            total_chars += len(message["content"])
        for step in (s for s in result["steps"] if s["turn"] == turn_index):
            total_tokens += n(step.get("content") or "")
            total_chars += len(step.get("content") or "")
            for call in step["tool_calls"]:
                call_text = f'{call["name"]}({call["arguments"]})'
                output = call["output"] if isinstance(call["output"], str) else json.dumps(call["output"])
                out_tokens = n(output)
                total_tokens += n(call_text) + out_tokens
                total_chars += len(call_text) + len(output)
                per_tool.setdefault(call["name"], []).append(out_tokens)
                if out_tokens > largest[0]:
                    largest = (out_tokens, f"turn {turn_index + 1}: {call['name']}")
        tokens_after_turn.append(total_tokens)

    return {
        "tokens": total_tokens,
        "chars": total_chars,
        "tokens_after_turn": tokens_after_turn,
        "largest_output": largest,
        "per_tool": per_tool,
    }


def main():
    """Count every task in the run, print a table and save token_counts.json in the run folder."""
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    tok = Tokenizer.from_file(args.tokenizer)

    tools_text = json.dumps(to_openai_tools(load_function_docs(cfg["classes"])))
    tasks = {t.id: t for t in load_tasks(cfg["category"], cfg["classes"], cfg.get("extra_task_files", []))}
    results = [json.loads(line) for line in (Path(args.run_dir) / "results.jsonl").read_text().splitlines() if line]

    rows = {}
    print(f"tool descriptions alone: {len(tok.encode(tools_text).ids)} tokens\n")
    print(f"{'task':32s} {'tokens':>8s} {'chars':>8s} {'chars/token':>11s}  largest output")
    for result in results:
        row = count_session(result, tasks[result["id"]], tools_text, tok)
        rows[result["id"]] = row
        size, where = row["largest_output"]
        print(f"{result['id']:32s} {row['tokens']:8d} {row['chars']:8d} {row['chars'] / row['tokens']:11.2f}  {size} ({where})")
        print(f"{'':32s} tokens after each turn: {row['tokens_after_turn']}")

    # Output size of each tool across the whole run: this is what decides how long a session gets.
    per_tool: dict[str, list[int]] = {}
    for row in rows.values():
        for name, sizes in row["per_tool"].items():
            per_tool.setdefault(name, []).extend(sizes)
    print(f"\n{'tool':32s} {'calls':>6s} {'min':>7s} {'mean':>7s} {'max':>7s}   (output tokens per call)")
    for name, sizes in sorted(per_tool.items(), key=lambda item: -max(item[1])):
        print(f"{name:32s} {len(sizes):6d} {min(sizes):7d} {sum(sizes) // len(sizes):7d} {max(sizes):7d}")

    tokens = sorted(row["tokens"] for row in rows.values())
    print(f"\nsessions: {len(tokens)}, median {tokens[len(tokens) // 2]} tokens, max {tokens[-1]} tokens")
    for limit in (16384, 32768):
        print(f"  over {limit} tokens: {sum(t > limit for t in tokens)} of {len(tokens)}")

    (Path(args.run_dir) / "token_counts.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
