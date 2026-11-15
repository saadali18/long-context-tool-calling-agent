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
`berkeley-function-call-leaderboard/`). It is cloned at a fixed commit, the same one the
harness on `main` uses:

```bash
git clone --filter=blob:none --sparse https://github.com/ShishirPatil/gorilla.git third_party/gorilla
git -C third_party/gorilla sparse-checkout set berkeley-function-call-leaderboard
git -C third_party/gorilla checkout 6ea57973c7a6097fd7c5915698c54c17c5b1b6c8
B=third_party/gorilla/berkeley-function-call-leaderboard
```

`third_party/` is in `.gitignore`. The multi-turn backends only need the Python standard library.

Files used, inside `$B/bfcl_eval/`:

| What | Path |
|---|---|
| Tasks | `data/BFCL_v4_multi_turn_long_context.json` (200 tasks) |
| Ground truth | `data/possible_answer/BFCL_v4_multi_turn_long_context.json` |
| Tool schemas | `data/multi_turn_func_doc/<backend>.json` |
| Backends | `eval_checker/multi_turn_eval/func_source_code/*.py` |
| Checker | `eval_checker/multi_turn_eval/multi_turn_checker.py` |

Earlier the PyPI package `bfcl-eval==2026.3.23` was used (installed with `--no-deps`, because
a normal install pulls in about 150 packages including torch and CUDA). It was compared with
the GitHub commit:

```bash
pip download --no-deps bfcl-eval==2026.3.23 -d /tmp/bfcl_pip
cd /tmp/bfcl_pip && unzip -q bfcl_eval-2026.3.23-py3-none-any.whl -d whl
diff -rq whl/bfcl_eval $B/bfcl_eval -x __pycache__
```

Result: no differences. The commit is dated Mar 23 2026, the same day as the release, so the
data, backends and checker are identical. Only the location changed.

Long-context list cap (four filler lists cut to 500 items, see `docs/long_tasks.md`):

```bash
python3 scripts/cap_long_context.py --cap 500
python3 scripts/cap_long_context.py --status
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

Chosen: TradingBot + MessageAPI (`--classes TradingBot,MessageAPI`, the harness default).

## Step 5: Merge the harness from main

```bash
git checkout bfcl_validation
git merge --no-commit -X theirs origin/main
```

`main` has the agent harness (`harness/`): agent loop, tool execution by method lookup,
per-request logging, its own scorer with BFCL's checking logic, the test split
(`splits/tasks_TradingBot_MessageAPI.json`, seed 624: 13 test and 20 trace tasks) and a mock
llama-server (`tests/mock_server.py`, `tests/run_mock_tests.sh`).

The harness is kept as the runner. From this branch only the official scoring and the tests
are kept, rewritten on top of it. Removed because the harness does the same job:
`src/bfcl/` (loader, task runner, evaluator, metrics), `scripts/run_bfcl.py`,
`configs/bfcl.yaml`, `pyproject.toml`, `scripts/count_tokens.py`.

## Step 6: Official BFCL scorer

`harness/official_score.py` scores a finished harness run again with BFCL's unchanged code:
`multi_turn_checker` for session success, and `state_checker` + `response_checker` after every
turn for per-turn accuracy.

- The model's calls are rebuilt from `requests.jsonl` as call strings (turn, step, call).
- BFCL runs call strings with `eval()`. Only calls to the task's own tools with plain argument
  names are kept.
- BFCL keeps backend state in module globals keyed by a name. Each scoring uses a unique name
  and deletes its state afterwards.
- Two ground-truth calls use positional arguments (`get_stock_info('OMEG')`,
  `get_available_stocks('Technology')`); BFCL runs these as they are.

Other changes:

- `scripts/count_tool_tokens.py`: `--extra-tasks` and `--only-extra` to count our own tasks.
- `harness/run.py`: with `--ids`, only the listed ids run, so `--ids a1_long_context_1
  --extra-tasks data/long_tasks.jsonl` runs one of our tasks.
- `tests/run_mock_tests.sh`: runs the official scorer after each mode.

## Step 7: Validate

```bash
bash tests/run_mock_tests.sh
pip install pytest
python3 -m pytest tests
python3 -m harness.official_score results/harness_bf16_143_full --bfcl-root $B
python3 -m harness.official_score results/harness_bf16_143_t1 --bfcl-root $B
```

Results:

| Run | Harness session / per-turn | Official session / per-turn | Agree |
|---|---|---|---|
| mock `gt` (13 test tasks) | 1.0 / 1.0 | 1.0 / 1.0 | 13 of 13 |
| mock `noisy` | 1.0 / 1.0 | 1.0 / 1.0 | 13 of 13 |
| mock `silent` | 0.0 / 0.0 | 0.0 / 0.0 | 13 of 13 |
| mock `wrong` | 0.308 / 0.463 | 0.308 / 0.463 | 13 of 13 |
| `harness_bf16_143_full` (Qwen3-1.7B BF16) | 0.0 / 0.25 | 0.0 / 0.25 | 1 of 1 |
| `harness_bf16_143_t1` (1 turn) | 0.0 / 1.0 | 0.0 / 1.0 | 1 of 1 |

On task 143 both scorers fail the session at turn 1 for the same reason
(`execution_response_mismatch`).

All 34 pytest tests pass.
