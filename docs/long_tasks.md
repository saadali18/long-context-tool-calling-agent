# Long-session tasks (Task 1) and long-context list cap

Author: Eman Fatima · Branch: `long-tasks` · Benchmark: `bfcl-eval==2026.3.23` · Tokenizer: Qwen3-1.7B (`tokenizer.json`)

## What the project asks for (Task 1)

At least 5 tasks over the same backends that reach **16k–32k tokens** through many calls and
verbose outputs, each with a **functional success check**; at least **two** must need a detail
from **early in the session**. A test split is held out (still to do, together with the BFCL split).

## Real token counts and the list cap

`scripts/count_tokens.py` re-counts a finished run with the Qwen3 tokenizer. Estimating tokens
as characters ÷ 4 badly underestimates these sessions: the long-context outputs are mostly
numbers, and Qwen splits every digit into its own token (≈1.3–1.5 characters per token).

In long-context mode, BFCL's TradingBot pads outputs with filler lists from
`bfcl_eval/eval_checker/multi_turn_eval/func_source_code/long_context.py`. On our TA's
suggestion, four of them are capped to their **first 500 items**: `MA_5_EXTENSION`,
`MA_20_EXTENSION`, `WATCH_LIST_EXTENSION`, `TRANSACTION_HISTORY_EXTENSION`.

Why it is safe (from `trading_bot.py`):
- `get_watchlist` and `get_transaction_history` return the account's real data **first** and
  append the filler after it, so the cap only removes filler.
- `get_stock_info` replaces MA(5)/MA(20) with the filler lists, which are the same for every stock.
- None of these lists is stored in the backend state that BFCL's checker compares.
- Check: the ground-truth replay passes all tasks with and without the cap (success rate 1.0).

| Output tokens per call | Full lists | Capped (500) |
|---|---|---|
| `get_transaction_history` | 49,397 | 24,741 |
| `get_stock_info` | 31,664 | 7,943 |
| `get_watchlist` | 3,965 | 1,971 |
| `get_available_stocks` (not capped) | 3,982 | 3,982 |
| `get_order_details` (not capped) | 1,576 | 1,576 |
| Tool descriptions (30 tools, every session) | 2,925 | 2,925 |

| 33 BFCL tasks (TradingBot + MessageAPI) | Full lists | Capped (500) |
|---|---|---|
| Over 32k tokens | 28 | 0 |
| Over 16k tokens | 28 | 7 |

**Every environment that runs BFCL's tools must carry the same cap** (this repo's `.venv`,
which BFCL's checker uses, and the agent's tool environment). Run once after installing bfcl-eval:

```bash
pip install --no-deps bfcl-eval==2026.3.23
python scripts/cap_long_context.py      # appends the cap block; backup kept as long_context.py.orig
```

## The long tasks (`data/long_tasks.jsonl`)

One task per line, BFCL format with the correct calls in the same entry (`ground_truth`).
Ids contain `long_context` so BFCL runs the tools in long-context mode. The success check is
BFCL's own: the final backend state (orders, watch list, balance, sent messages) must match
the state the correct calls produce. Every message the model sends has fixed text, and each
task sets its own `order_counter`, so new order ids are predictable.

| Task | What the user asks | Big outputs | Early detail |
|---|---|---|---|
| `a1_long_context_1` Portfolio audit | Order 12345 → full transaction history → check order 12447 → cancel it and sell "the same stock and number of shares as the first order I asked about" | transaction history | ✅ from a tool output (turn 1: AAPL, 15 shares) |
| `a1_long_context_2` Tech scan | Tech stocks → MSFT info → NVDA info → buy 10 of the cheaper one | tech list, 2× stock info | – |
| `a1_long_context_3` Watchlist cleanup | Watchlist → ZETA info → remove ZETA, TSLA info → add TSLA, show watchlist → message Bob a fixed text | 2× stock info, 2× watchlist | – |
| `a1_long_context_4` Standing rule | Turn 1 sets a rule: after any sell, message Catherine "Sell order placed: <ID>"; research turns; turn 5 only says "sell 5 TSLA at $670" | 2× sector list, stock info, order details | ✅ from a user message (turn 1 rule) |
| `a1_long_context_5` Order trail | Buy MSFT → buy OMEG → AAPL info → GOOG info → watchlist → "cancel the first order I placed today" | 2× stock info, watchlist | ✅ from a tool output (turn 1 order id) |
| `a1_long_context_6` Deposit check | Turn 1 mentions a $3,750 deposit → account info → transaction history → order details → "make the deposit I mentioned at the start" | transaction history | ✅ from a user message (turn 1 amount) |

## Results (ground-truth replay, capped lists)

All 6 tasks pass (`session_success_rate` 1.0, `per_turn_accuracy` 1.0, 34 tool calls) and all
are between 16k and 32k tokens:

| Task | Turns | Tokens | Tokens after each turn | Largest output |
|---|---|---|---|---|
| `a1_long_context_1` | 4 | 31,080 | 4,544 / 29,317 / 30,926 / 31,080 | 24,741 (transaction history) |
| `a1_long_context_2` | 4 | 22,965 | 6,935 / 14,901 / 22,866 / 22,965 | 7,944 (stock info) |
| `a1_long_context_3` | 5 | 23,030 | 4,920 / 12,885 / 20,893 / 22,921 / 23,030 | 7,944 (stock info) |
| `a1_long_context_4` | 5 | 20,776 | 6,983 / 10,997 / 18,965 / 20,597 / 20,776 | 7,944 (stock info) |
| `a1_long_context_5` | 6 | 21,073 | 3,017 / 3,111 / 11,077 / 19,041 / 21,025 / 21,073 | 7,945 (stock info) |
| `a1_long_context_6` | 4 | 29,421 | 3,010 / 27,765 / 29,370 / 29,421 | 24,741 (transaction history) |

Median 23,030 tokens, max 31,080. Counts cover tool descriptions, user messages, tool calls
and tool outputs; the chat template adds a few tokens per message on top.

## Reproduce

```bash
source .venv/bin/activate
python scripts/cap_long_context.py                                   # once per environment
python scripts/run_bfcl.py --caller ground_truth --tasks extra --name gt_long6
python scripts/count_tokens.py outputs/gt_long6                      # needs models/qwen3-tokenizer/tokenizer.json
```

Tokenizer setup (once): `pip install tokenizers huggingface_hub` and
`hf download Qwen/Qwen3-1.7B tokenizer.json --local-dir models/qwen3-tokenizer`.

`run_bfcl.py --tasks` selects `all` (BFCL + ours, default), `bfcl` or `extra` (ours only).
