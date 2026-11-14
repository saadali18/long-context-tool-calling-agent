# A1 harness: BFCL long-context agent loop for llama-server

The client program from Task 2 of the A1 spec. It runs the agent loop against `llama-server`, executes tools on BFCL's simulated backends, logs every request, and scores sessions the way BFCL's multi-turn checker does.

Pure Python standard library (Python 3.10+). Nothing to `pip install`.

## What it logs (spec Task 2)

| Spec requirement | Field | File |
| --- | --- | --- |
| Tokens sent | `prompt_tokens` | requests.jsonl |
| Tokens reprocessed by the model | `reprocessed_tokens` (llama-server `timings.prompt_n`), `cached_tokens` | requests.jsonl |
| Prefill time / decode time | `prefill_ms`, `decode_ms`, `decode_tokens` | requests.jsonl |
| Tool time | `tool_ms` per call and per request | requests.jsonl |
| Call validity | `calls[].well_formed`, `.exists`, `.schema_valid`, `.problems` | requests.jsonl |
| Session success, per-turn result, failure reason | `session_success`, `turns[]`, `fail_reason` | sessions.jsonl |
| Time per turn and per session, warm/cold time to first call | `turns[].wall_ms`, `session_wall_ms`, `warm_time_to_first_call_ms`, `cold_time_to_first_call_ms` | sessions.jsonl |
| Peak memory, energy, temperature, throttling (Pi only) | `peak_server_rss_mib`, `energy_j_pmic`, `max_temp_c`, `throttled_during` | sessions.jsonl |

`metadata.json` records the arguments, harness and BFCL commits, host, and llama-server's `/props` (model path, context size, build).

## Baseline settings (spec Task 3)

The defaults are the baseline: a **stateless** client (the full history is re-sent on every request), **`cache_prompt: false`** (no prefix reuse), **all 30 tool schemas** on every request (the chat template puts them in the system prompt), and **no grammar** (the model may write anything). Do not pass `--cache-prompt` for baseline runs.

Defaults the group still has to confirm: tool subset `TradingBot,MessageAPI` (30 functions, 33 BFCL tasks, 13 held out for test), `--temperature 0`, `--max-tokens 512`, `--thinking off`, no extra system prompt.

## Setup (on the Pi, and on any machine used for accuracy runs)

```bash
cd ~/long-context-tool-calling-agent
# BFCL source, pinned (tasks, tool docs, backends, ground truth)
git clone --filter=blob:none --sparse https://github.com/ShishirPatil/gorilla.git third_party/gorilla
git -C third_party/gorilla sparse-checkout set berkeley-function-call-leaderboard
git -C third_party/gorilla checkout 6ea57973c7a6097fd7c5915698c54c17c5b1b6c8
echo "third_party/" >> .gitignore
B=third_party/gorilla/berkeley-function-call-leaderboard

# self-test with a fake server (about 1 minute, no model needed)
bash tests/run_mock_tests.sh
```

`splits/tasks_TradingBot_MessageAPI.json` is already in the repo (seed 624). To regenerate it: `python3 -m harness.make_split --bfcl-root $B`.

## Running

Start the server (baseline), then the harness:

```bash
~/llama.cpp/build-baseline/bin/llama-server -m ~/models/qwen3-1.7b-bf16.gguf \
  -c 16384 -t 4 --jinja -np 1 --host 127.0.0.1 --port 8080

# Pi latency run: one test task, first 2 user turns only
nohup python3 -m harness.run --bfcl-root $B --split splits/tasks_TradingBot_MessageAPI.json \
  --which test --limit 1 --max-turns 2 --out results/harness_baseline_pi_short > results/harness_short.log 2>&1 &

# full test split (accuracy): run where it is fast enough, same GGUF and same llama.cpp commit
python3 -m harness.run --bfcl-root $B --split splits/tasks_TradingBot_MessageAPI.json \
  --which test --out results/harness_baseline_test

python3 -m harness.summarize results/harness_baseline_test
```

- **Cold start:** add `--launch-server "<the full llama-server command>"`. The harness starts the server itself and reports `cold_time_to_first_call_ms` for the first task.
- **Our own long tasks:** `--extra-tasks tasks/long_tasks.jsonl`, one JSON object per line with `id`, `question` (a list of turns, each a list of `{"role":"user","content":...}`), `initial_config`, `involved_classes` and `ground_truth` (a list of call strings per turn, e.g. `"get_watchlist()"`). Success is the same end-state check as BFCL.
- **Optimised runs later:** the same command with `--cache-prompt`, a different model file, or a different server build. Compare with `summarize`.

## How it scores

After every user turn, the harness runs that turn's ground-truth calls on a second copy of the backends. A turn passes when (1) the backends involved have the same public state as the ground-truth copy, and (2) every ground-truth result appears among the model's results so far. That is BFCL's `multi_turn_checker` logic. A session succeeds only if every turn passes.

The same run can be scored a second time with BFCL's own, unchanged code (see [Official BFCL scorer](#official-bfcl-scorer)), so the two can be compared.

Unlike BFCL, it keeps scoring after the first failed turn, so `per_turn_accuracy` and `turn_accuracy_by_index` show where sessions start failing. After a failure the backend state has diverged, so later turns mostly fail too. Measuring single-turn accuracy cleanly, for the gap experiment, needs a teacher-forced mode (reset to the ground-truth state and history before each turn). That mode is not built yet.

## Scale on the Pi (BF16 baseline)

Replaying the ground truth through the harness gives these sizes, estimated at about 4 characters per token. The harness logs real token counts.

| Quantity (13 test tasks) | Estimate |
| --- | --- |
| Tool schemas | about 3,200 tokens, so about 38 minutes of prefill before the first call |
| Largest context in a session | median 13,700 tokens, max 16,100 (2 of the 33 tasks exceed 16k) |
| Tokens sent per session (stateless) | median about 79,000 |
| Prefill for the whole test split at 1.4 tokens/s | about 230 hours |

So the full baseline test split cannot run on the Pi in BF16. Measure latency on the Pi with `--limit`/`--max-turns`, and run accuracy elsewhere with identical weights, which the spec allows.

## Official BFCL scorer

`harness/official_score.py` re-scores a finished run folder with BFCL's original checker code, loaded from the same `--bfcl-root` (so the same commit and the same list cap):

```bash
python3 -m harness.official_score results/harness_baseline_test --bfcl-root $B
# runs that used --extra-tasks:
python3 -m harness.official_score <run_dir> --bfcl-root $B --extra-tasks data/long_tasks.jsonl
```

| Argument | What it does |
| --- | --- |
| `run_dir` | Folder written by `harness.run` (reads `requests.jsonl` and `sessions.jsonl`). |
| `--bfcl-root` | BFCL folder (`berkeley-function-call-leaderboard`). Required. |
| `--extra-tasks` | Our own task file, when the run used `--extra-tasks`. |

- **Session success:** BFCL's `multi_turn_checker`, exactly as the leaderboard runs it (stops at the first failed turn).
- **Per-turn accuracy:** BFCL's `state_checker` and `response_checker` after every turn, without stopping.
- The model's calls are rebuilt from `requests.jsonl` as BFCL call strings. Only calls with valid JSON arguments, a name from the task's tools and plain argument names are kept, because BFCL runs them with `eval`. Steps without a kept call are left out, as BFCL does.
- An aborted session, or a turn that used all 20 steps, fails as `multi_turn:force_terminated`. A `--max-turns` run is scored on the turns that were played.

It writes `official_scores.jsonl` into the run folder (one line per task: official and harness results side by side, the rebuilt calls, and `agree`) and prints both session success rates, both per-turn accuracies and any task where they disagree.

Checked so far: the four mock modes (13 test tasks each) and `results/harness_bf16_143_full` / `_t1` give identical session and per-turn results under both scorers.

## Tests

`tests/mock_server.py` imitates llama-server and replays the ground truth, in four modes:

| Mode | Expected result |
| --- | --- |
| `gt` | every session succeeds |
| `noisy` | invalid calls counted (`call_validity` < 1); sessions still succeed |
| `silent` | `empty_turn_model_response` failures |
| `wrong` | state or response mismatches from turn 1 |

`tests/run_mock_tests.sh` also runs the official scorer on each mode; the last line must be `"disagreements": []`.

`tests/test_bfcl.py` (pytest) drives `harness.runner.run_session` with an in-process fake server and scores every session with both scorers. It checks the split (30 tools, 13 test / 20 trace tasks, no overlap), JSON Schema types, that long-context outputs are longer, that the ground truth passes both scorers on every test task and on `data/long_tasks.jsonl`, that a silent model fails, that both scorers agree when arguments are wrong, that `--max-turns` runs are scored on the played turns, and that unsafe calls never reach `eval`.

```bash
pip install pytest                      # the only package needed, for the tests only
BFCL_ROOT=$B python3 -m pytest tests    # BFCL_ROOT defaults to the path above
```

## Limitations

- Power comes from the PMIC rails (`vcgencmd pmic_read_adc`), not the whole board.
- Token counts depend on llama-server returning `timings`. If it doesn't, `reprocessed_tokens` is empty.
- A call that names a real tool but has bad arguments is still executed. The backend's error text goes back to the model, as in BFCL.
