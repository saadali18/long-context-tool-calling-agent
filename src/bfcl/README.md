# BFCL benchmark

BFCL multi-turn backends in long-context mode, on a 30-function subset that includes
TradingBot, scored with per-turn accuracy and session success.

Setup steps are in [`docs/bfcl_validation_log.md`](../../docs/bfcl_validation_log.md).

## Settings

| Setting | Value |
|---|---|
| BFCL version | `bfcl-eval==2026.3.23` |
| Category | `multi_turn_long_context` |
| Tool subset | TradingBot (20) + MessageAPI (10) = 30 functions |
| Tasks | 33 (tasks that only use those two backends) |

Change the subset in `configs/bfcl.yaml`.

## Files

| File | Purpose |
|---|---|
| `loader.py` | Loads tasks, ground truth and tool schemas; keeps tasks inside the subset. |
| `task_runner.py` | Runs a session: executes each tool call on the BFCL backends and feeds the output back. |
| `evaluator.py` | Session success (BFCL checker) and per-turn correctness. |
| `metrics.py` | Run summary. |
| `scripts/run_bfcl.py` | Command-line entry point. |

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --no-deps bfcl-eval==2026.3.23
pip install -e ".[dev]"
```

## Run

```bash
pytest
python scripts/run_bfcl.py --caller ground_truth --name gt_all
```

Expected: `session_success_rate: 1.0` and `per_turn_accuracy: 1.0` on all 33 tasks.

### Arguments of `scripts/run_bfcl.py`

| Argument | Default | What it does |
|---|---|---|
| `--config` | `configs/bfcl.yaml` | Config file to use (category, tool subset, step limit). |
| `--caller` | `ground_truth` | Who produces the tool calls. `ground_truth` replays BFCL's correct calls, which validates the benchmark setup. |
| `--limit N` | all tasks | Run only the first N tasks. Useful for a quick check, e.g. `--limit 3`. |
| `--ids ID [ID ...]` | all tasks | Run only these task ids, e.g. `--ids multi_turn_long_context_100 multi_turn_long_context_103`. |
| `--name NAME` | `<caller>_<date>_<time>` | Name of the output folder for this run. |
| `--out-dir DIR` | `outputs` | Parent folder where run folders are created. |

Example: run two specific tasks into `outputs/check_two/`:

```bash
python scripts/run_bfcl.py --ids multi_turn_long_context_100 multi_turn_long_context_103 --name check_two
```

### Config keys (`configs/bfcl.yaml`)

| Key | Meaning |
|---|---|
| `category` | BFCL category. Long-context mode is turned on when the name contains `long_context`. |
| `classes` | Backends whose functions form the tool subset. Only tasks using these backends are run. |
| `max_steps_per_turn` | Maximum tool-call steps in one turn (BFCL uses 20). Hitting it fails the session. |

### Output

Each run writes to `outputs/<name>/`:

| File | Content |
|---|---|
| `config.yaml` | Copy of the config used. |
| `results.jsonl` | One line per task: the calls made and their outputs per step, `session_success`, `turn_correct` (one flag per turn) and BFCL's `error_type` on failure. |
| `summary.json` | Run-level metrics (below). |

The console prints one line per task (`PASS`/`FAIL`, turns correct) and then the summary.

## Metrics

| Metric | Meaning |
|---|---|
| `session_success_rate` | All turns correct (BFCL official). |
| `per_turn_accuracy` | Fraction of turns with correct backend state and outputs. |
| `num_tool_calls` | Tool calls made. |
| `mean/max_context_chars` | Session size in characters. |
