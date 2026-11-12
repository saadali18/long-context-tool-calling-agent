"""Cap BFCL's long-context filler lists in the installed bfcl-eval (run once per environment).

Why
---
In long-context mode, BFCL's TradingBot pads tool outputs with filler lists from
`bfcl_eval/eval_checker/multi_turn_eval/func_source_code/long_context.py`. Measured with the
Qwen3 tokenizer (scripts/count_tokens.py), the full lists make one get_stock_info output
~31.7k tokens and one get_transaction_history output ~49.4k tokens, so most sessions exceed 32k.

On our TA's suggestion, the first 500 items of four lists are kept. The real account data is
always returned before the filler (get_watchlist, get_transaction_history), get_stock_info's
MA lists are the same for every stock, and none of the lists is stored in the backend state
BFCL's checker compares. The ground-truth replay passes all tasks with the cap.

Every copy of BFCL's tools must carry the same cap: the bfcl-eval used by BFCL's checker
(this repo's .venv) and the agent's tool environment. Run this script in each environment,
right after `pip install --no-deps bfcl-eval==2026.3.23`.

What it does
------------
Appends a marked block to the installed long_context.py (a backup is kept as
long_context.py.orig) and prints the new list lengths. Running it again changes nothing.
To undo: copy long_context.py.orig back over long_context.py.

Example:
    python scripts/cap_long_context.py
"""

import argparse
import importlib
import os
import shutil

import bfcl_eval

LISTS = ("MA_5_EXTENSION", "MA_20_EXTENSION", "WATCH_LIST_EXTENSION", "TRANSACTION_HISTORY_EXTENSION")
MARKER = "# Project: cap the long-context filler lists"


def main():
    """Find the installed long_context.py, append the cap block if missing, print the lengths."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cap", type=int, default=500, help="items kept in each list (team value: 500)")
    args = parser.parse_args()

    path = os.path.join(
        os.path.dirname(bfcl_eval.__file__), "eval_checker", "multi_turn_eval", "func_source_code", "long_context.py"
    )
    text = open(path).read()
    if MARKER in text:
        print(f"already capped: {path}")
    else:
        shutil.copyfile(path, path + ".orig")
        block = f"\n\n{MARKER} (TA suggestion). Original: long_context.py.orig\n"
        block += "".join(f"{name} = {name}[:{args.cap}]\n" for name in LISTS)
        with open(path, "a") as f:
            f.write(block)
        print(f"capped to {args.cap} items: {path}")

    module = importlib.import_module("bfcl_eval.eval_checker.multi_turn_eval.func_source_code.long_context")
    for name in LISTS:
        print(f"  {name}: {len(getattr(module, name))} items")


if __name__ == "__main__":
    main()
