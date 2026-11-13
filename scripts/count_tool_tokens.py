"""Count the real tokens of every BFCL tool output, per tool and per task.

Replays each task's ground-truth calls on the BFCL backends in long-context mode
(the same outputs the model would read), then tokenizes each output with the
Qwen3 tokenizer.

  python3 scripts/count_tool_tokens.py \
      --bfcl-root third_party/gorilla/berkeley-function-call-leaderboard \
      --tokenizer ~/models/Qwen3-1.7B/tokenizer.json

Our own long tasks: add --extra-tasks data/long_tasks.jsonl (counted after BFCL's;
use --only-extra to count only them).

Optional: --compare-tiktoken also counts with OpenAI's o200k tokenizer (which groups
up to 3 digits per token), to show why other token counts can be much lower.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.bfcl_env import BFCL, execute, method_owner, parse_call_string  # noqa: E402
from harness.run import load_extra_tasks  # noqa: E402


def make_counter(spec: str):
    if spec == "chars":  # for testing without a tokenizer file
        return lambda s: len(s)
    from tokenizers import Tokenizer  # pip install tokenizers
    tok = Tokenizer.from_file(str(Path(spec).expanduser()))
    return lambda s: len(tok.encode(s, add_special_tokens=False).ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfcl-root", required=True)
    ap.add_argument("--tokenizer", required=True, help="path to Qwen3 tokenizer.json (or 'chars')")
    ap.add_argument("--classes", default="TradingBot,MessageAPI")
    ap.add_argument("--split", help="optional split file: count only its tasks")
    ap.add_argument("--which", default="all", choices=["test", "trace", "all"])
    ap.add_argument("--extra-tasks", help="JSONL of our own long tasks, counted after BFCL's")
    ap.add_argument("--only-extra", action="store_true", help="count only the --extra-tasks tasks")
    ap.add_argument("--compare-tiktoken", action="store_true")
    ap.add_argument("--out", help="write a JSON report here too")
    a = ap.parse_args()

    count = make_counter(a.tokenizer)
    tk = None
    if a.compare_tiktoken:
        import tiktoken  # pip install tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        tk = lambda s: len(enc.encode(s))  # noqa: E731

    classes = a.classes.split(",")
    bfcl = BFCL(a.bfcl_root)
    tasks = bfcl.load_tasks()
    ids = sorted((t.id for t in tasks.values() if set(t.involved_classes) <= set(classes)),
                 key=lambda s: int(s.rsplit("_", 1)[1]))
    if a.split:
        sp = json.load(open(a.split))
        keep = set(sp["test"] + sp["trace"]) if a.which == "all" else set(sp[a.which])
        ids = [i for i in ids if i in keep]
    if a.extra_tasks:
        extra = load_extra_tasks(a.extra_tasks)
        tasks.update(extra)
        ids = list(extra) if a.only_extra else ids + list(extra)

    tools_tok = count(json.dumps(bfcl.tools_for(classes)))
    per_tool = defaultdict(list)            # name -> [(tokens, chars, digits, tiktoken)]
    per_task = []
    for tid in ids:
        t = tasks[tid]
        inst = bfcl.new_instances(classes, t.initial_config)
        owner = method_owner(inst)
        out_tok = out_tk = 0
        user_tok = sum(count(m["content"]) for turn in t.turns for m in turn)
        for turn in t.ground_truth:
            for call in turn:
                name, kwargs = parse_call_string(call, owner)
                text = execute(owner, name, kwargs)
                n = count(text)
                k = tk(text) if tk else None
                per_tool[name].append((n, len(text), sum(c.isdigit() for c in text), k))
                out_tok += n
                out_tk += k or 0
        per_task.append({"task": tid, "tool_output_tokens": out_tok, "user_tokens": user_tok,
                         "min_final_context": tools_tok + user_tok + out_tok,
                         "tool_output_tiktoken": out_tk if tk else None})

    print(f"tokenizer: {a.tokenizer}   tasks: {len(ids)}   classes: {classes}")
    print(f"tool schemas (all {len(bfcl.tools_for(classes))} functions, as JSON): {tools_tok} tokens\n")
    hdr = f"{'tool':28s} {'calls':>5s} {'min':>7s} {'mean':>7s} {'max':>7s} {'chars':>7s} {'digits':>7s} {'chars/tok':>9s}"
    if tk:
        hdr += f" {'o200k':>7s}"
    print(hdr + "   (tokens per call)")
    for name, v in sorted(per_tool.items(), key=lambda x: -st.mean(r[0] for r in x[1])):
        toks = [r[0] for r in v]
        chars = st.mean(r[1] for r in v)
        line = (f"{name:28s} {len(v):5d} {min(toks):7d} {st.mean(toks):7.0f} {max(toks):7d} "
                f"{chars:7.0f} {st.mean(r[2] for r in v):7.0f} {chars / max(st.mean(toks), 1):9.2f}")
        if tk:
            line += f" {st.mean(r[3] for r in v):7.0f}"
        print(line)

    outs = [p["tool_output_tokens"] for p in per_task]
    finals = [p["min_final_context"] for p in per_task]
    print(f"\nper task, ground-truth tool output: median {st.median(outs):.0f}, max {max(outs)} tokens")
    if tk:
        print(f"  same with o200k (digit-grouping) tokenizer: median "
              f"{st.median(p['tool_output_tiktoken'] for p in per_task):.0f} tokens")
    print(f"per task, minimum final context (schemas + user turns + tool outputs): "
          f"median {st.median(finals):.0f}, max {max(finals)} tokens")
    for limit in (16384, 32768):
        print(f"  tasks over {limit}: {sum(f > limit for f in finals)} of {len(finals)}")
    print("  (the real context is larger still: it also holds the model's own calls and the chat template)")

    if a.out:
        Path(a.out).write_text(json.dumps({
            "tokenizer": a.tokenizer, "classes": classes, "tool_schema_tokens": tools_tok,
            "per_tool": {k: {"calls": len(v), "mean_tokens": st.mean(r[0] for r in v),
                             "max_tokens": max(r[0] for r in v)} for k, v in per_tool.items()},
            "per_task": per_task}, indent=2))
        print(f"\nreport -> {a.out}")


if __name__ == "__main__":
    main()
