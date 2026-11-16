"""Cap BFCL's long-context filler lists (TA guidance, Nov 11 and Nov 16 2026).

Caps (TA, Nov 16): MA_5, MA_20 and WATCH_LIST at 500 items, TRANSACTION_HISTORY at 100.

In long-context mode, TradingBot pads four outputs with filler lists from
long_context.py. With Qwen3's tokenizer (one token per digit) they push sessions
to ~40k tokens. The real, task-relevant entries always come first (the filler is
appended after them, or replaces MA(5)/MA(20) entirely), so capping the filler
lists does not remove anything a task depends on.

This script appends a small, clearly marked block to long_context.py that trims
the four lists in place when the module is imported. It is idempotent, and
--undo removes it. Apply it to EVERY copy of long_context.py that is used: our
harness uses one copy (third_party/...), but a pip-installed bfcl_eval has its own.

  python3 scripts/cap_long_context.py                                # TA caps: 500 / 500 / 500 / 100
  python3 scripts/cap_long_context.py --cap 500 --txn-cap 100 --file path/to/long_context.py
  python3 scripts/cap_long_context.py --undo
  python3 scripts/cap_long_context.py --status
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

LISTS = ["MA_5_EXTENSION", "MA_20_EXTENSION", "WATCH_LIST_EXTENSION", "TRANSACTION_HISTORY_EXTENSION"]
BEGIN = "# >>> A1 cap of long-context filler lists >>>"
END = "# <<< A1 cap of long-context filler lists <<<"
REL = "bfcl_eval/eval_checker/multi_turn_eval/func_source_code/long_context.py"


def find_copies() -> list[Path]:
    found = []
    repo_copy = Path(__file__).resolve().parents[1] / "third_party/gorilla/berkeley-function-call-leaderboard" / REL
    if repo_copy.exists():
        found.append(repo_copy)
    try:  # a pip-installed bfcl_eval, if any
        out = subprocess.run([sys.executable, "-I", "-c",
                              "import bfcl_eval, os; print(os.path.dirname(bfcl_eval.__file__))"],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        if out:
            p = Path(out).parent / REL
            if p.exists() and p.resolve() not in [f.resolve() for f in found]:
                found.append(p)
    except Exception:
        pass
    return found


def strip_block(src: str) -> str:
    # removes exactly what block() added, so --undo restores the original file byte for byte
    return re.sub(rf"\n{re.escape(BEGIN)}\n.*?{re.escape(END)}\n", "", src, flags=re.S)


def caps_for(cap: int, txn_cap: int) -> dict:
    return {n: (txn_cap if n == "TRANSACTION_HISTORY_EXTENSION" else cap) for n in LISTS}


def block(cap: int, txn_cap: int) -> str:
    caps = caps_for(cap, txn_cap)
    return (f"\n{BEGIN}\n"
            f"# Added by long-context-tool-calling-agent/scripts/cap_long_context.py. Remove with --undo.\n"
            f"_A1_LIST_CAP = {cap}\n"
            f"_A1_LIST_CAPS = {caps!r}\n"
            f"for _a1_name, _a1_n in _A1_LIST_CAPS.items():\n"
            f"    del globals()[_a1_name][_a1_n:]   # trim in place: TradingBot holds the same list objects\n"
            f"{END}\n")


def status(path: Path) -> str:
    src = path.read_text()
    if BEGIN not in src:
        return "not capped"
    m = re.search(r"_A1_LIST_CAPS = (\{.*?\})", src)
    if m:
        caps = eval(m.group(1), {})  # our own dict literal
        return "capped " + ", ".join(f"{k.replace('_EXTENSION', '')}={v}" for k, v in caps.items())
    m = re.search(r"_A1_LIST_CAP = (\d+)", src)
    return f"capped (old block) all={m.group(1)}; rerun to apply the Nov 16 caps"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap", type=int, default=500, help="MA_5, MA_20 and WATCH_LIST cap (TA: 500)")
    ap.add_argument("--txn-cap", type=int, default=100, help="TRANSACTION_HISTORY cap (TA: 100)")
    ap.add_argument("--file", action="append", help="path to a long_context.py (repeatable)")
    ap.add_argument("--undo", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    files = [Path(f) for f in a.file] if a.file else find_copies()
    if not files:
        sys.exit("no long_context.py found; pass --file")
    for f in files:
        if a.status:
            print(f"{status(f)}  {f}")
            continue
        src = strip_block(f.read_text())
        if not a.undo:
            src += block(a.cap, a.txn_cap)
        f.write_text(src)
        print(f"{status(f)}  {f}")


if __name__ == "__main__":
    main()
