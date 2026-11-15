# Long-context task split: integration with main

This PR is synchronized with Maria's merged split-aware runner and Obaid's official
BFCL scorer. It **does not replace their code or re-import Eman's tasks**. It adds
validation, regression tests, and protections against accidentally changing a
frozen evaluation split.

## Frozen allocation (from main)

| Dataset | Development / trace | Held-out test |
| --- | ---: | ---: |
| BFCL TradingBot + MessageAPI | 20 | 13 |
| Eman's custom tasks | L5 | L1, L2, L3, L4, L6 |
| **Total** | **21** | **18** |

The source of truth is `splits/tasks_TradingBot_MessageAPI.json`, using keys
`test`, `trace`, `extra_test`, and `extra_trace`. The existing five-task
held-out set is preserved; **do not tune on these held-out long tasks**.

## Safeguards

- `harness.task_split.select_task_ids` checks both BFCL groups for overlap,
  all custom assignments, duplicate IDs, unexpected/unloaded custom tasks, and
  accidental assignment of the same task to two groups.
- `harness.run.select_ids` calls that validator when a split is supplied, while
  retaining the already-merged behavior for explicit `--ids` and unsplit runs.
- `harness.run.load_extra_tasks` rejects duplicate IDs in custom JSONL files.
- `harness.make_split` preserves `extra_test`, `extra_trace`, and
  `extra_note` if a split manifest is regenerated for the same tool classes.

## Validate without the Pi or model server

From repository root:

```bash
python3 -m unittest discover -s tests -p 'test_long_task_split.py' -v
python3 -m harness.task_split --which test
python3 -m harness.task_split --which trace
```

Expected: **14 unit tests**, **18 test tasks**, **21 development tasks**, no
overlap. `--which dev` is an alias for `--which trace`.

The existing `tests/test_bfcl.py` suite still tests Eman's tasks and BFCL
official scoring separately; it requires the BFCL checkout.

## Real-model example (requires BFCL checkout + llama-server)

```bash
B=third_party/gorilla/berkeley-function-call-leaderboard

python3 -m harness.run --bfcl-root "$B" \
  --split splits/tasks_TradingBot_MessageAPI.json --which trace \
  --extra-tasks data/long_tasks.jsonl --out results/long_trace

# Held-out test; run only after the pipeline is finalized.
python3 -m harness.run --bfcl-root "$B" \
  --split splits/tasks_TradingBot_MessageAPI.json --which test \
  --extra-tasks data/long_tasks.jsonl --out results/long_test
```

**Not established by these unit tests:** Qwen3 model accuracy, timing or memory
on the Raspberry Pi, or scorer parity on new real-model traces. Run the BFCL
integration tests and actual device measurements separately.
