#!/usr/bin/env bash
# baseline_bench.sh - Task 3 items 1-3 for the A1 BF16 baseline on the Pi 5
#   1. prefill + decode throughput at several context depths
#   2. memory breakdown (weights / KV cache / compute buffers / rest)
#   3. measured memory bandwidth (for the bandwidth-bound check)
# Logs temperature, throttling and PMIC power every 5 s for the whole run.
#
# Run (survives SSH drops):
#   nohup bash ~/baseline_bench.sh > ~/bench_run.log 2>&1 &
# Override defaults like:  DEPTHS=0,2048,8192,15360 bash ~/baseline_bench.sh

set -uo pipefail

MODEL=${MODEL:-$HOME/models/qwen3-1.7b-bf16.gguf}
BIN=${BIN:-$HOME/llama.cpp/build-baseline/bin}
THREADS=${THREADS:-4}
CTX=${CTX:-16384}
DEPTHS=${DEPTHS:-0,2048,8192}
OUT=${OUT:-$HOME/results/baseline_$(date +%Y%m%d_%H%M)}
mkdir -p "$OUT"
echo "Results -> $OUT"

# ---------- 0. metadata (reproducibility record) ----------
{
  echo "date:        $(date -Is)"
  echo "model:       $MODEL"
  sha256sum "$MODEL"
  echo "llama.cpp:   $(git -C "$HOME/llama.cpp" rev-parse HEAD 2>/dev/null)"
  "$BIN/llama-server" --version 2>&1 | head -4
  grep -E "^(GGML_NATIVE|CMAKE_BUILD_TYPE)" "$HOME/llama.cpp/build-baseline/CMakeCache.txt"
  echo "board:       $(tr -d '\0' < /proc/device-tree/model)"
  echo "kernel:      $(uname -r)"
  echo "boot target: $(systemctl get-default)"
  echo "--- config.txt (active lines) ---"
  grep -vE '^#|^$' /boot/firmware/config.txt
  echo "throttled at start: $(vcgencmd get_throttled)"
} > "$OUT/metadata.txt" 2>&1

# ---------- background logger: temp, throttling, power ----------
log_loop() {
  echo "time,temp_c,throttled,pmic_power_w" > "$OUT/thermal_power.csv"
  while true; do
    t=$(vcgencmd measure_temp | grep -oE '[0-9]+\.[0-9]+')
    th=$(vcgencmd get_throttled | cut -d= -f2)
    # sum V*I over the PMIC rails (rail_A current x rail_V voltage)
    p=$(vcgencmd pmic_read_adc | awk '
      /current/ { n=substr($1,1,length($1)-2); split($2,b,"="); I[n]=b[2]+0 }
      /volt/    { n=substr($1,1,length($1)-2); split($2,b,"="); V[n]=b[2]+0 }
      END { s=0; for (k in I) if (k in V) s+=I[k]*V[k]; printf "%.3f", s }')
    echo "$(date +%T),$t,$th,$p" >> "$OUT/thermal_power.csv"
    sleep 5
  done
}
log_loop &
LOGPID=$!
trap 'kill $LOGPID 2>/dev/null' EXIT

# ---------- 2. memory breakdown (quick, ~2 min) ----------
echo "[$(date +%T)] memory breakdown..."
free -m > "$OUT/memory_idle.txt"
"$BIN/llama-server" -m "$MODEL" -c "$CTX" -t "$THREADS" --jinja --port 8080 \
  > "$OUT/server.log" 2>&1 &
SPID=$!
for _ in $(seq 1 300); do
  grep -qi "listening" "$OUT/server.log" && break
  sleep 1
done
{
  echo "=== process memory (VmRSS = now, VmHWM = peak) ==="
  grep -E "VmRSS|VmHWM" "/proc/$SPID/status"
  echo; echo "=== system memory with server loaded ==="
  free -m
  echo; echo "=== llama.cpp allocations from server log ==="
  grep -iE "KV|buffer size|model size|mapped|compute" "$OUT/server.log"
} > "$OUT/memory.txt" 2>&1
kill "$SPID"; wait "$SPID" 2>/dev/null

# ---------- 3. memory bandwidth (quick, ~1 min) ----------
echo "[$(date +%T)] memory bandwidth..."
if command -v sysbench >/dev/null; then
  for th in 1 4; do
    echo "=== read, $th thread(s) ===" >> "$OUT/bandwidth.txt"
    sysbench memory --memory-block-size=1M --memory-total-size=20G \
      --memory-oper=read --threads=$th run >> "$OUT/bandwidth.txt" 2>&1
  done
else
  echo "sysbench not installed (sudo apt install -y sysbench)" > "$OUT/bandwidth.txt"
fi

# ---------- 1. throughput at several depths (long: hours) ----------
echo "[$(date +%T)] throughput at depths $DEPTHS ..."
if "$BIN/llama-bench" --help 2>&1 | grep -q -- "n-depth"; then
  "$BIN/llama-bench" -m "$MODEL" -t "$THREADS" -p 64 -n 16 -d "$DEPTHS" -r 1 -o md \
    2>&1 | tee "$OUT/speed.md"
else
  echo "(llama-bench has no -d; falling back to prompt lengths, decode at depth 0 only)" \
    | tee "$OUT/speed.md"
  "$BIN/llama-bench" -m "$MODEL" -t "$THREADS" -p "64,${DEPTHS#0,}" -n 16 -r 1 -o md \
    2>&1 | tee -a "$OUT/speed.md"
fi

echo "throttled at end: $(vcgencmd get_throttled)" >> "$OUT/metadata.txt"
echo "[$(date +%T)] done -> $OUT"
