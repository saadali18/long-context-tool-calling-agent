# Split-aware long-task evaluation

This change keeps the existing 33 BFCL task assignments fixed while including Eman's
six custom long-session tasks in an explicit, leakage-safe split. **`trace` means
development/tuning; `test` is held out for final comparisons.** This PR does not
alter BFCL scoring, model execution, prompt caching, or quantization.

## Task allocation

| Dataset | Trace / development | Held-out test |
| --- | ---: | ---: |
| TradingBot + MessageAPI BFCL | 20 | 13 |
| Custom long-context tasks | 1 (L4) | 5 (L1, L2, L3, L5, L6) |
| **Total** | **21** | **18** |

L4 is the early-rule-recall task. This allocation is a proposed fixed evaluation
policy: **confirm the split with the team before tuning or publishing final scores**.

## Files

- `data/long_tasks.jsonl` is copied unchanged from Eman's `long-tasks` branch.
- `splits/tasks_TradingBot_MessageAPI.json` keeps its original BFCL IDs and has a
  new `extra_tasks.test`/`extra_tasks.trace` mapping.
- `harness/task_split.py` validates the manifest and selects exact IDs.
- `harness/run.py` uses that selector when custom tasks are provided.
- `harness/make_split.py` retains existing custom-task assignments if the BFCL
  split manifest is regenerated.
- `tests/test_long_task_split.py` validates separation and common error cases.

## Dry-run the selection: no model, Pi, or BFCL installation necessary

Run from repository root:

```bash
python3 -m unittest discover -s tests -p 'test_long_task_split.py' -v
python3 -m harness.task_split --which test
python3 -m harness.task_split --which trace
```

Expected: **11 tests pass**, test lists **18 IDs**, and trace lists **21 IDs**.
Test and trace are disjoint. `--which dev` is an alias for `trace`.

## Real-model runs (requires BFCL checkout + llama-server)

```bash
B=third_party/gorilla/berkeley-function-call-leaderboard

# Development set (21 tasks total). Can limit runs for smoke tests.
python3 -m harness.run --bfcl-root "$B" \
  --split splits/tasks_TradingBot_MessageAPI.json --which trace \
  --extra-tasks data/long_tasks.jsonl --limit 1 \
  --out results/long_trace_smoke

# Held-out test set (18 tasks). Keep for final, frozen evaluation.
python3 -m harness.run --bfcl-root "$B" \
  --split splits/tasks_TradingBot_MessageAPI.json --which test \
  --extra-tasks data/long_tasks.jsonl \
  --out results/long_test_final
```

The existing BFCL-only command (without `--extra-tasks`) still selects the
original 13 test or 20 trace tasks.

If `--extra-tasks` is supplied with `--split`, every custom task ID must
appear in exactly one set. Missing, duplicated, or unexpected IDs cause an
error **before inference**. To deliberately debug an individual task, use
`--ids a1_long_context_4 --extra-tasks data/long_tasks.jsonl`; explicit
IDs bypass the split and should never be used to produce held-out results.

## Integration with Obaid's official scorer

This branch is intentionally small and independent: it changes **task
selection only**. The official BFCL scorer can be integrated later without
changing these assignments. After integration, replay all six ground-truth
tasks and compare local and official scoring on the **same** traces.

**Not yet verified in this PR:** full BFCL/llama-server model runs, scorer
parity, actual long-task model accuracy, Pi throughput, memory, or energy.
The no-model split tests validate task selection, not these metrics.
