#!/usr/bin/env bash
# Harness self-test against a fake llama-server (no model needed, ~1 min).
# Expected: gt -> all sessions succeed; noisy -> call_validity < 1 but sessions succeed;
#           silent -> empty_turn failures; wrong -> state/response mismatches from turn 1.
# Each run is then scored again with BFCL's official checker (harness.official_score);
# both scorers must agree on every session (last line: "disagreements": []).
set -u
cd "$(dirname "$0")/.."
B=${BFCL_ROOT:-third_party/gorilla/berkeley-function-call-leaderboard}
SPLIT=splits/tasks_TradingBot_MessageAPI.json
port=8190
for mode in gt noisy silent wrong; do
  python3 tests/mock_server.py --bfcl-root "$B" --port $port --mode $mode >/dev/null 2>&1 &
  pid=$!; sleep 1.5
  rm -rf /tmp/a1_mock_$mode
  python3 -m harness.run --bfcl-root "$B" --split $SPLIT --which test \
    --server-url http://127.0.0.1:$port --out /tmp/a1_mock_$mode >/dev/null
  echo "== $mode"; python3 -m harness.summarize /tmp/a1_mock_$mode | sed -n '1,4p;8p'
  python3 -m harness.official_score /tmp/a1_mock_$mode --bfcl-root "$B" | tail -9
  kill $pid; port=$((port+1))
done
