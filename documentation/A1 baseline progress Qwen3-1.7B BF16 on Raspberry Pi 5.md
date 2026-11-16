# A1 baseline progress: Qwen3-1.7B BF16 on Raspberry Pi 5

## Summary

Three of the five baseline requirements are complete. The other two are partly complete. The baseline is Qwen3-1.7B in BF16, served by llama.cpp on a Raspberry Pi 5 with 8 GB of RAM.

The most important finding so far is that BF16 is limited by computation on the Pi, not by memory bandwidth. The Pi's Cortex-A76 cores cannot do BF16 arithmetic directly. Prefill runs at about 1.4 tokens per second. Decode runs at about 1.1 tokens per second at the start of a session.

The second finding is that the stateless baseline client wastes most of its time. In task 143, the model read 36,141 tokens to handle a conversation that was only 5,709 tokens long. Prefill took 97% of the 7.2-hour session.

| #   | Requirement                                                    | Status          | Key number                                                                                             |
| --- | -------------------------------------------------------------- | --------------- | ------------------------------------------------------------------------------------------------------ |
| 1   | Prefill and decode throughput at three or more context lengths | Complete        | Decode falls from 1.11 to 0.71 tokens/s between 0 and 8k context                                       |
| 2   | Memory breakdown                                               | Complete        | Peak 5.6 GiB with a 16k context                                                                        |
| 3   | Is decode bandwidth-bound?                                     | Complete        | No. Decode uses 28–36% of the measured 12.6 GB/s                                                       |
| 4   | Per-turn and per-session wall-clock time                       | Partly complete | Three sessions measured: 7.2 h, 3.1 h and 4.5 h. Cold start not measured yet                           |
| 5   | Single-turn accuracy vs session success, with the cause tested | Partly complete | 1 of 10 turns passed across three sessions, and that one only by luck. The experiment is not built yet |

## Setup

Every measurement in this document uses the settings below. Changing any of them means the baseline must be measured again.

| Part             | Setting                                                                             |
| ---------------- | ----------------------------------------------------------------------------------- |
| Board            | Raspberry Pi 5 Model B Rev 1.0, 8 GB RAM, 117 GB SD card                            |
| Operating system | Raspberry Pi OS 64-bit (Debian Trixie), desktop turned off (`multi-user.target`)    |
| Cooling          | Active cooler, enabled with `dtparam=cooling_fan=on` in `/boot/firmware/config.txt` |
| Inference engine | llama.cpp from our fork, commit `efbf6e80a` (build 12115), default Release build    |
| Model            | `Qwen/Qwen3-1.7B`, converted by us to BF16 GGUF: 3.78 GiB, 2.03B parameters         |
| Server command   | `llama-server -c 16384 -t 4 --jinja -np 1`                                          |
| Client           | Our harness (`harness/`), stateless: the full history is re-sent on every request   |
| Prompt cache     | Off: every request sends `"cache_prompt": false`                                    |
| Tool schemas     | All 30 tools on every request, no grammar                                           |
| Sampling         | Temperature 0, at most 512 output tokens per request, Qwen3 thinking mode off       |

The fan setting matters. Without it, the fan never ran and the CPU throttled at 90°C. One early benchmark run was discarded for that reason.

**Thinking mode.** Qwen3's thinking mode is off for the baseline and will stay off for every later variant. The TA confirmed this choice on Nov 13. The tasks do not need step-by-step reasoning. Qwen3 also tends to overthink in this mode. Reasoning quality also drops more than non-reasoning quality after quantization, so keeping it off makes the later quantized comparisons fair.

## Benchmark and token budget

The benchmark is BFCL's multi-turn long-context category. We use the TradingBot and MessageAPI apps, which together have 30 functions. Thirty-three BFCL tasks use only these two apps.

Thirteen of the 33 tasks are held out as the test set. Every reported number is measured on them. The other 20 tasks form the trace set, which is only used for tuning. The split was drawn at random with seed 624. BFCL is pinned at commit `6ea5797`.

**The token problem.** With Qwen3's tokenizer, BFCL's long-context outputs were far larger than the project manual expected. Qwen3 turns every digit into its own token. One `get_stock_info` call returned 31,664 tokens.

**The fix.** On the TA's advice, we cap four filler lists in BFCL's `long_context.py` at 500 items. The capped lists are `MA_5_EXTENSION`, `MA_20_EXTENSION`, `WATCH_LIST_EXTENSION` and `TRANSACTION_HISTORY_EXTENSION`. A verification script confirmed that the cap does not change any score. Every harness run records the cap in its `metadata.json`.

The table shows real Qwen3 token counts for the 33 tasks.

| Quantity                                | Without cap   | With cap 500  |
| --------------------------------------- | ------------- | ------------- |
| One `get_stock_info` call               | 31,664 tokens | 7,944 tokens  |
| Tool output per task, median            | 35,695 tokens | 9,986 tokens  |
| Tool output per task, maximum           | 67,307 tokens | 17,871 tokens |
| Smallest possible full session, median  | not measured  | 13,114 tokens |
| Smallest possible full session, maximum | not measured  | 20,945 tokens |
| Tasks over 16k tokens                   | 28 of 33      | 7 of 33       |

The 30 tool schemas alone take 3,001 tokens. They are sent with every request.

With the cap, 11 of the 13 test tasks fit in the 16k context. Tasks 131 and 116 do not fit. Following the TA's guidance, they count as baseline failures.

## Requirement 1: throughput at three context lengths

Prefill speed stays flat at about 1.4 tokens per second. Decode speed falls as the context grows. These numbers come from `llama-bench` with 64 prompt tokens and 16 output tokens at each depth.

"Depth" is the number of tokens already in the context before the measurement starts.

| Depth (tokens)        | Prefill (tokens/s) | Decode (tokens/s) | Decode time per token |
| --------------------- | ------------------ | ----------------- | --------------------- |
| 0                     | 1.40               | 1.11              | 901 ms                |
| 2,048                 | 1.44               | 1.00              | 1,000 ms              |
| 8,192                 | 1.39               | 0.71              | 1,408 ms              |
| 16,384 (extrapolated) | about 1.4          | about 0.52        | about 1,920 ms        |

Decode time per token follows a straight line: about 888 ms plus 0.063 ms for every token of context. This line matches all three measurements within 2%. We used it to extrapolate to 16k, as the TA allowed.

At 1.4 tokens per second, reading a full 16k context takes about 3.2 hours. A 32k context is not possible in BF16, because it would need about 7.3 GiB of RAM for the weights and KV cache alone.

The run took 2 hours on Nov 6, from 15:13 to 17:18. The CPU was never throttled. The highest temperature was 75.7°C. Average power on the PMIC rails was 7.7 W, and total energy was 57.6 kJ.

## Requirement 2: memory breakdown

The server's peak memory is 5,739 MiB with a 16k context. The weights and the KV cache account for almost all of it.

| Part                                  | Size          | How we know                                                          |
| ------------------------------------- | ------------- | -------------------------------------------------------------------- |
| Weights (BF16)                        | 3,871 MiB     | Size of the GGUF file                                                |
| KV cache, 16,384 tokens, 16-bit       | 1,792 MiB     | 28 layers × 8 KV heads × 128 × 2 (K and V) × 2 bytes × 16,384 tokens |
| Compute buffers and other             | about 76 MiB  | Peak minus weights minus KV cache                                    |
| **Server peak (VmHWM)**               | **5,739 MiB** | Measured: 5,876,960 kB                                               |
| Operating system and background, idle | 604 MiB       | `free -m` before the server starts                                   |

Two checks agree with this breakdown. First, memory marked "used" rose by 1,776 MiB when the server started, which matches the KV cache. Second, the weights appear as 4,155 MiB of file cache, because llama.cpp memory-maps the model file.

The KV cache costs 112 KiB per token. llama.cpp reserves the full 16k cache when the server starts, so the memory does not grow during a session.

The full harness session of task 143 peaked at 5,807 MiB. That is close to the startup figure, which confirms the cache is reserved up front.

## Requirement 3: is decode bandwidth-bound?

No. In BF16, decode uses only about one third of the Pi's memory bandwidth.

We measured the bandwidth with `sysbench`, reading 512 MB blocks with 4 threads. The result was 12,010 MiB/s, which is 12.6 GB/s. An earlier test with 1 MB blocks gave 68 GB/s. That number measured the CPU cache, not RAM, so we discarded it.

To make each token, decode must read all the weights and the whole KV cache once. The table multiplies those bytes by the measured decode speed.

| Depth                          | Bytes read per token | Decode speed        | Data moved     | Share of 12.6 GB/s |
| ------------------------------ | -------------------- | ------------------- | -------------- | ------------------ |
| 0                              | 4.06 GB              | 1.11 tokens/s       | 4.5 GB/s       | 36%                |
| 2,048                          | 4.29 GB              | 1.00 tokens/s       | 4.3 GB/s       | 34%                |
| about 4,500 (task 143 session) | about 4.58 GB        | about 0.85 tokens/s | about 3.9 GB/s | about 31%          |
| 8,192                          | 5.00 GB              | 0.71 tokens/s       | 3.6 GB/s       | 28%                |

If decode were limited by memory, it would reach about 3.1 tokens per second at depth 0. It reaches only 1.11.

The likely cause is BF16 itself. The Cortex-A76 cores have no BF16 instructions. llama.cpp must convert every weight to 32-bit before it can use it, and that conversion is the bottleneck.

Prefill is also limited by computation. At 1.44 tokens per second, it performs about 6 billion floating-point operations per second. That is a small fraction of what the four cores can do in 32-bit arithmetic, so the same BF16 conversion likely slows prefill too.

This gives us a testable prediction for later. A format the cores handle natively, such as FP16 or Q8_0, should speed up decode until it reaches the 12.6 GB/s limit. At that point decode would become bandwidth-bound.

## Requirement 4: wall-clock time per turn and per session

Three full sessions are measured: test tasks 143, 104 and 146. All three were warm runs. Task 143 is described in detail first. Tasks 104 and 146 are compared after it.

Task 143 has 4 user turns. The server was already running when the session started, so these are warm times.

| Turn              | Wall-clock time                   |
| ----------------- | --------------------------------- |
| 1                 | 74 min (4,417 s)                  |
| 2                 | 101 min (6,050 s)                 |
| 3                 | 124 min (7,428 s)                 |
| 4                 | 134 min (8,018 s)                 |
| **Whole session** | **432 min (25,913 s), 7.2 hours** |

Each turn takes longer than the one before. This happens because the client re-sends the whole conversation with every request, and the conversation keeps growing.

| Session measure                    | Value                                        |
| ---------------------------------- | -------------------------------------------- |
| Time to the first tool call (warm) | 36.5 min                                     |
| Model requests                     | 8                                            |
| Prompt size per request            | from 3,126 to 5,709 tokens                   |
| Tokens sent in total               | 36,141                                       |
| Tokens re-read by the model        | 36,141 (0 taken from a cache)                |
| Prefill time                       | 418 min, 96.8% of the session                |
| Decode                             | 709 tokens in 13.9 min (about 0.85 tokens/s) |
| Peak server memory                 | 5,807 MiB                                    |
| Energy (PMIC rails)                | 195 kJ at an average of 7.5 W                |
| Highest temperature                | 76.8°C, never throttled                      |

The largest context in the session was 5,709 tokens. Yet the model read 36,141 tokens in total, about 6.3 times as much. Prompt caching would remove most of this repeated reading. That makes it our first planned improvement.

Prefill ran at 1.44 tokens per second for every request, from 3.1k to 5.7k tokens. This matches the `llama-bench` result, so the benchmark and the real sessions agree.

**Tasks 104 and 146.** Both sessions used the same settings as task 143. Both finished much sooner than estimated. The reason is that the model made fewer tool calls than the correct answer needs, so the conversation stayed small.

| Measure                            | Task 104                     | Task 146                      |
| ---------------------------------- | ---------------------------- | ----------------------------- |
| User turns                         | 2                            | 4                             |
| Model requests                     | 4                            | 7                             |
| Tool calls: model / correct answer | 3 / 3                        | 3 / 5                         |
| Time per turn                      | 75, 113 min                  | 37, 75, 78, 80 min            |
| Largest context                    | 5,424 tokens                 | 3,467 tokens                  |
| Tokens re-read in total            | 15,239 (0 from a cache)      | 23,053 (0 from a cache)       |
| Prefill share of the session       | 93.2%                        | 98.5%                         |
| Decode                             | 642 tokens                   | 238 tokens                    |
| Time to the first tool call (warm) | 37.1 min                     | 73.4 min                      |
| **Whole session**                  | **3.1 h (11,317 s)**         | **4.5 h (16,175 s)**          |
| Peak server memory                 | 5,804 MiB                    | 5,806 MiB                     |
| Energy (PMIC rails)                | 84 kJ at an average of 7.4 W | 119 kJ at an average of 7.4 W |
| Highest temperature                | 73.6°C, never throttled      | 71.9°C, never throttled       |

Prefill ran at 1.44–1.45 tokens per second in both sessions, the same as in task 143.

Task 146's time to the first tool call is long for a simple reason. The model answered its first request in text, without a tool call. So the first call came only in the second request.

A shorter session here does not mean a faster system. It means the model skipped work that the correct answer does. When we compare variants later, we must compare the same tasks, and we must also report how many requests each session made.

**Other test tasks.** We will not run all 13 test tasks on the Pi. As the TA allowed, the rest are extrapolated from the measured speeds. The estimates below assume the model makes the same calls as the correct answer.

| Task          | Turns | Final context | Estimated Pi time     | Plan                                  |
| ------------- | ----- | ------------- | --------------------- | ------------------------------------- |
| 143           | 4     | 5.7k          | 9 h (measured: 7.2 h) | Done                                  |
| 104           | 2     | 13.6k         | 10 h                  | Done (measured: 3.1 h)                |
| 146           | 4     | 15.5k         | 16 h                  | Done (measured: 4.5 h)                |
| 100           | 2     | 11.3k         | 7 h                   | Extrapolate                           |
| 132           | 2     | 13.0k         | 10 h                  | Extrapolate                           |
| 120           | 3     | 13.1k         | 15 h                  | Extrapolate                           |
| 107, 138, 145 | 5     | 13.4k         | 22 h each             | Extrapolate                           |
| 127           | 4     | 15.5k         | 16 h                  | Extrapolate                           |
| 118           | 5     | 15.7k         | 22 h                  | Extrapolate                           |
| 131, 116      | 6–7   | over 16k      | —                     | Context overflow, counted as failures |

## Requirement 5: accuracy and the gap

In task 143, the session failed and 1 of 4 turns passed. All 7 tool calls were valid: well formed, using a real tool, with arguments that fit the tool's schema. So the model's problem is choosing the right action, not formatting.

| Turn | User asked                                        | Model did                                                                                     | Result                                                |
| ---- | ------------------------------------------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| 1    | Add Zeta Corp to my watchlist                     | Called `add_to_watchlist('Zeta Corp')` instead of `'ZETA'`                                    | Passed only because ZETA was already in the watchlist |
| 2    | Show me the large order I placed                  | Looked up order 12345 instead of 12446                                                        | Failed                                                |
| 3    | Cancel the order we just reviewed                 | Tried to cancel 12345                                                                         | Failed                                                |
| 4    | Show my account, log in as USR001, message USR003 | Deposited $10,000 nobody asked for, logged in as USR003, sent "$10000" instead of "$10000.00" | Failed                                                |

**Why turn 2 failed.** The user did not give an order ID. The model had a tool, `get_order_history`, that lists the orders: 12345 and 12446. The model never called it. It went straight to order 12345, an ID that had not appeared anywhere in the conversation. The correct order was 12446, a sell of 5 GOOG shares worth about $14,203.

**Which errors follow from that one.** Turn 3 is a direct consequence. Cancelling "the order we just reviewed" was the right action for the order the model thought it had reviewed. Turn 4 failed for independent reasons. The deposit, the wrong login and the missing account lookup have nothing to do with the order mix-up.

So this session shows both kinds of failure. One early mistake cascaded into the next turn. The last turn also failed on its own.

**Tasks 104 and 146.** Both sessions failed, and no turn passed in either. All 6 tool calls were valid. Across the three sessions, 1 of 10 turns passed, and that one passed only by luck.

In task 104, turn 0 failed because the model's reply did not contain the stock data that the correct answer returns. Turn 1 failed because the watchlist ended up different from the correct one.

Task 146 shows the failures most clearly.

| Turn | User asked                                   | Correct answer                                                  | Model did                                                                                    | Result |
| ---- | -------------------------------------------- | --------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ------ |
| 0    | Which stocks are on my watchlist?            | `get_watchlist()`                                               | Asked the user to list the symbols. Called no tool                                           | Failed |
| 1    | Remove 'Zeta Corp' from the list             | `remove_stock_from_watchlist(symbol='ZETA')`                    | The same call with `'Zeta Corp'`. The tool said it was not on the list                       | Failed |
| 2    | Buy 50 shares of 'Apple' at the market price | `get_stock_info('AAPL')`, then `place_order` for AAPL at 227.16 | `place_order` with the symbol `'Apple'` and a made-up price of 190.5. The order was rejected | Failed |
| 3    | Show my most recent order                    | `get_order_details(order_id=12446)`                             | `get_order_history()`, then reported order 12345                                             | Failed |

**What the three sessions show.** Three patterns repeat.

- The model passes a company name where a tool needs a stock symbol. It did this with 'Zeta Corp' in tasks 143 and 146, and with 'Apple' in task 146. The model does know the symbols. In task 146 it told the user that Apple is "commonly referred to as AAPL", but it did not use that knowledge in the call.
- The model skips the calls that gather information. It asked the user for the watchlist instead of calling `get_watchlist`. It guessed a price instead of calling `get_stock_info`.
- The model makes at most one tool call per turn and then replies. Some correct answers need two calls in a row.

The correct answers never use the lookup tool `get_symbol_by_name`. They expect the model to supply the symbol itself, from its own knowledge or from an earlier tool output.

Task 146 also shows cascades. Turn 3 could not pass, because order 12446 was never created after the order in turn 2 failed. Turn 1 probably depends on turn 0 as well. If the model had called `get_watchlist`, it would have seen 'ZETA' in the output. The teacher-forcing experiment will test both claims.

**What is still needed.** One session is not enough to measure the gap. We need session success and per-turn accuracy on all 13 test tasks. These runs will happen on a faster machine with the same model file and llama.cpp commit.

The cause must also be tested by experiment, not only by reading logs. The planned experiment is teacher forcing. Before each turn, the harness resets the conversation and the app state to the correct ones. Then the model handles that turn only. If a turn passes under teacher forcing but failed in the normal run, its failure was a cascade. The difference between the two runs is the accuracy gap. The harness does not have this mode yet.

## Remaining work

Most of the Pi work is done. Three kinds of work remain: accuracy on the full test split, the accuracy-gap experiment, and two short Pi runs. Most group decisions are now settled.

**To complete the baseline:**

- [x] Run task 104 on the Pi (3.1 h).
- [x] Run task 146 on the Pi (4.5 h).
- [ ] Measure the cold-start time to the first call once on the Pi. The run uses the first turn of task 143, with the file cache cleared before the server starts. The project manual lists cold and warm time to the first call among the metrics for every change.
- [ ] Score tasks 104 and 146 with BFCL's official checker on the Pi, and commit the result. A check in a separate copy already showed that both scorers agree on every turn.
- [ ] Measure accuracy on the full test split on a faster machine, with the same model file, llama.cpp commit and BFCL copy. Run the 13 BFCL test tasks now. Run the 5 long test tasks once the long-task split is frozen.
- [ ] Build the teacher-forcing mode in the harness, and run the accuracy-gap experiment.
- [ ] Run one short Pi session that shows the long tasks overflow the 16k context. They then count as baseline failures.

**Group decisions:**

- [x] Use one harness. Our runner stays, and BFCL's official checker and its tests were added to it (PR #3, merged). The two scorers agree on all 10 turns from the Pi runs.
- [x] Use one BFCL source: the git clone at commit `6ea5797`. The harness and the official checker load the same copy.
- [x] Keep one list-cap script: `scripts/cap_long_context.py`, with a cap of 500 items.
- [x] Use one tool-schema format: the harness's, at 3,001 tokens. The second harness was removed.
- [ ] Finalize the 6 long tasks. They must stay between 16k and 32k tokens, as the manual requires. Proposal: replace `get_transaction_history` in L1 and L6 with lighter calls, which keeps them at about 17–20k tokens.
- [ ] Freeze the long-task split before any long-task accuracy run. PR #2 adds tested split selection. It needs a merge with `main` and a group choice of the trace task, L4 or L5. The test set will then have 18 tasks.
