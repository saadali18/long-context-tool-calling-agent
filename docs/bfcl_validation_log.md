# BFCL benchmark setup log

Branch: `bfcl_validation`

## Step 1: Requirements (Project A1, Benchmark)

- BFCL multi-turn backends in long-context mode (category `multi_turn_long_context`).
- 20 to 30 functions, must include TradingBot.
- Per-turn accuracy for call accuracy, session success on tasks that only use those functions.

## Step 2: Create the branch

```bash
git clone https://github.com/saadali18/long-context-tool-calling-agent
cd long-context-tool-calling-agent
git checkout -b bfcl_validation
```

## Step 3: Get BFCL

BFCL is in the Gorilla repo (https://github.com/ShishirPatil/gorilla, folder
`berkeley-function-call-leaderboard/`) and on PyPI as `bfcl-eval`. The PyPI package contains
the data, the backends and the checker, so no separate download is needed.

```bash
pip index versions bfcl-eval      # latest: 2026.3.23
```

Files used, inside the installed package `bfcl_eval/`:

| What | Path |
|---|---|
| Tasks | `data/BFCL_v4_multi_turn_long_context.json` (200 tasks) |
| Ground truth | `data/possible_answer/BFCL_v4_multi_turn_long_context.json` |
| Tool schemas | `data/multi_turn_func_doc/<backend>.json` |
| Backends | `eval_checker/multi_turn_eval/func_source_code/*.py` |
| Checker | `eval_checker/multi_turn_eval/multi_turn_checker.py` |

Error: a normal install did not work.

```bash
pip install --dry-run bfcl-eval==2026.3.23
```

It pulls in about 150 packages (torch, CUDA, transformers, faiss, many cloud SDKs) and pins
`numpy==1.26.4`, which has no wheel for Python 3.13. The multi-turn backends only need the
standard library and `mpmath`.

Fix: install without dependencies.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --no-deps bfcl-eval==2026.3.23
pip install -e ".[dev]"          # mpmath, pyyaml, pytest
```

## Step 4: Choose the tool subset

Functions per backend:

| Backend | Functions |
|---|---|
| VehicleControlAPI | 22 |
| TradingBot | 20 |
| GorillaFileSystem | 18 |
| TravelAPI | 18 |
| MathAPI | 17 |
| TwitterAPI | 14 |
| MessageAPI | 10 |
| TicketAPI | 9 |

TradingBot (20) is required, so at most 10 more functions fit:

| Subset | Functions | Tasks |
|---|---|---|
| TradingBot + MessageAPI | 30 | 33 |
| TradingBot + TicketAPI | 29 | 29 |
| TradingBot only | 20 | 20 |

Chosen: TradingBot + MessageAPI (`configs/bfcl.yaml`).

## Step 5: Write the code

`src/bfcl/loader.py`, `task_runner.py`, `evaluator.py`, `metrics.py`, `scripts/run_bfcl.py`,
`configs/bfcl.yaml`, `tests/test_bfcl.py`.

- BFCL tool schemas use Python type names (`dict`, `float`, `tuple`, `any`); converted to
  JSON Schema names (`object`, `number`, `array`, `string`).
- Session success uses BFCL's `multi_turn_checker` unchanged. Per-turn accuracy runs the same
  state and output checks on every turn.

Errors found and fixed:

- Two ground-truth calls use positional arguments (`get_stock_info('OMEG')`,
  `get_available_stocks('Technology')`). Fixed by naming them from the schema's parameter order.
- BFCL runs calls with `eval()`. Fixed by only running calls to offered tools with plain
  argument names.
- BFCL keeps backend state in module globals keyed by a name, so re-runs reused old state.
  Fixed by giving each run a unique name and deleting its state afterwards.

## Step 6: Validate with the ground truth

```bash
pytest
python scripts/run_bfcl.py --caller ground_truth --name gt_all
```

Result on all 33 tasks:

```
session_success_rate  1.0
per_turn_accuracy     1.0
num_tool_calls        162
mean_context_chars    52,884  (max 84,104)
```

All 7 tests pass.
