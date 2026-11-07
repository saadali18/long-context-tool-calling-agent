# long-context-tool-calling-agent




# A1 Pi 5 setup and baseline


## Current Status

The Pi is set up and reachable from anywhere, llama.cpp is built, and the Qwen3-1.7B BF16 baseline model is ready. The first baseline benchmark took above 2 hours. The blocker: BF16 runs at about 1.4 tokens/s, too slow to run full agent sessions on the Pi.

| Item | State |
| --- | --- |
| Pi 5 (8 GB), Raspberry Pi OS 64-bit (Trixie) | Working; desktop turned off for measurements |
| Access | SSH over home WiFi, phone hotspot or Tailscale (works from uni) |
| llama.cpp | Built from fork, which is default Release build |
| Baseline model | `Qwen/Qwen3-1.7B` converted to BF16 GGUF, 3.78 GiB |
| Task 3 items 1–3 | Running via `scripts/baseline_bench.sh`; throughput at 0/2k/8k, memory and bandwidth measured Nov 6 |
| Task 3 items 4–5 | Not started:  need the harness and BFCL tasks |

## Pi hardware and OS setup

The board is a Raspberry Pi 5 Model B Rev 1.0 with 8 GB of RAM and a 117 GB SD card. The OS was flashed with Raspberry Pi Imager v2.0.12.

**Imager settings** (Device: Raspberry Pi 5, OS: Raspberry Pi OS (64-bit), Debian Trixie, released 2026-10-06):

| Setting | Value |
| --- | --- |
| Hostname | `edge-9` |
| Username | `edge_maria` (password shared privately, not in this doc) |
| WiFi | Phone hotspot (`Iphone123`); country PK |
| SSH | Enabled, password authentication |

Go through every Customisation page in Imager. A mistyped WiFi password here is why the Pi first failed to join any network.

**Desktop turned off** for clean measurements (frees CPU and a few hundred MB of RAM):

```bash
sudo systemctl set-default multi-user.target   # undo: graphical.target
sudo reboot
```

**Fan fix.** The fan spun for about 2 seconds at power-on, then never again. The firmware had left the fan controller disabled (`/proc/device-tree/cooling_fan/status` = `disabled`). Fix, added to `/boot/firmware/config.txt`:

```bash
printf '\n[all]\ndtparam=cooling_fan=on\n' | sudo tee -a /boot/firmware/config.txt
sudo reboot
```

Check after the reboot: `ls /sys/class/thermal/` shows `cooling_device0`, and the status reads `okay`. Fan RPM (`fan1_input`) reads 0 even while it is cooling, so log temperature and `vcgencmd get_throttled` instead. Force full speed if needed: `sudo pinctrl FAN_PWM op dl` (back to automatic: `a0`).

## Networking and access

The Pi knows three WiFi networks and joins the available one with the highest priority. Tailscale makes it reachable from any network, including uni WiFi, which blocks devices from reaching each other.

| Saved connection | Priority | Used when |
| --- | --- | --- |
| `netplan-wlan0-Iphone123` (phone hotspot) | 10 | Hotspot is on (always wins) |
| `home` | 5 | Hotspot off, at home |
| `uni` (`wireless-2G`, username/password login) | 1 | Hotspot off, at uni |

**How to connect:**

```bash
ssh edge_maria@edge-9           # via Tailscale, from anywhere (Tailscale app on your laptop)
ssh edge_maria@edge-9.local     # same WiFi or hotspot, no Tailscale needed
```

**Commands used** (type straight quotes, one line each; curly quotes from copy-paste break them):

```bash
# priorities
sudo nmcli connection modify "netplan-wlan0-Iphone123" connection.autoconnect-priority 10
# save a network without switching to it
sudo nmcli connection add type wifi con-name home ifname wlan0 ssid "HOME_WIFI" wifi-sec.key-mgmt wpa-psk wifi-sec.psk "PASSWORD" connection.autoconnect-priority 5
sudo nmcli connection add type wifi con-name uni ifname wlan0 ssid "wireless-2G" wifi-sec.key-mgmt wpa-eap 802-1x.eap peap 802-1x.phase2-auth mschapv2 802-1x.identity "ROLL_NO" 802-1x.password "PASSWORD" connection.autoconnect-priority 1
# check
nmcli -f NAME,AUTOCONNECT-PRIORITY connection show
nmcli -t -f NAME,DEVICE connection show --active
# Tailscale
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up               # open the printed link, sign in
tailscale ip -4
```

- On an iPhone hotspot, turn on Maximize Compatibility.
- Never switch networks by hand over SSH (`nmcli connection up …`): the session drops.
- To give a teammate access: share the Pi from the Tailscale admin console to their account, and share the SSH password privately.

## llama.cpp build

The baseline build is llama.cpp commit `efbf6e80a` (build 12115) from our fork, built with default settings in `~/llama.cpp/build-baseline/`. Keep that folder untouched; tuned builds (Task 4) go in separate folders.

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y git cmake build-essential libcurl4-openssl-dev python3-venv python3-pip tmux sysbench
cd ~ && git clone https://github.com/<fork-owner>/llama.cpp && cd llama.cpp
git rev-parse HEAD > ~/baseline_commit.txt
cmake -B build-baseline
cmake --build build-baseline --config Release -j4     # 15-30 min; rerun resumes if interrupted
cmake -B build-baseline -LA | grep -E "GGML_NATIVE|CMAKE_BUILD_TYPE" > ~/baseline_flags.txt
./build-baseline/bin/llama-server --version > ~/baseline_version.txt 2>&1
```

- Binaries are in `build-baseline/bin/` (`llama-server`, `llama-cli`, `llama-bench`, `llama-quantize`).
- `CMAKE_BUILD_TYPE=Release` is confirmed in `build-baseline/CMakeCache.txt`.
- The default build turns on native CPU optimisation (`GGML_NATIVE=ON`). Whether the baseline keeps it is an open decision.

## Baseline model

The baseline model is `Qwen/Qwen3-1.7B` (Qwen3, not the banned Qwen3.5), converted by us to `~/models/qwen3-1.7b-bf16.gguf`: 3.78 GiB, 2.03B parameters. The original weights are BF16, so the conversion is lossless. This file is the master copy every quantized version will be made from.

```bash
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install -U "huggingface_hub[cli]"
pip install -r ~/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
hf download Qwen/Qwen3-1.7B --local-dir ~/models/Qwen3-1.7B
python ~/llama.cpp/convert_hf_to_gguf.py ~/models/Qwen3-1.7B --outtype bf16 --outfile ~/models/qwen3-1.7b-bf16.gguf
sha256sum ~/models/qwen3-1.7b-bf16.gguf > ~/baseline_model_sha.txt
```

- Do the downloads on home or uni WiFi, not the hotspot (about 5 GB including PyTorch).
- Keep `~/models/Qwen3-1.7B` (the original weights): pruning or LoRA in Task 6 needs it.
- Why not 4B: Qwen3-4B in BF16 needs about 7.5 GiB for its weights, plus 2.25 GiB of KV cache at 16k tokens, which is more than 8 GB of RAM. 4B and 8B at low precision belong in the Task 5 size experiment.

## Baseline measurements so far

The BF16 baseline runs at 1.40 tokens/s for prefill and 1.11 tokens/s for decode. At that speed, reading one 16k-token context takes about 3.25 hours. Measurements run through `scripts/baseline_bench.sh` in our repo (`long-context-tool-calling-agent`), so every later configuration is measured the same way.

```bash
cd ~/long-context-tool-calling-agent
OUT=$PWD/results/baseline_$(date +%Y%m%d_%H%M) nohup bash scripts/baseline_bench.sh > results/bench_run.log 2>&1 &
tail -1 results/bench_run.log            # finished when it prints "done -> ..."
# a later configuration, same method:
MODEL=~/models/qwen3-1.7b-q8_0.gguf bash scripts/baseline_bench.sh
```

The script writes `metadata.txt` (commit, flags, model checksum, config.txt, throttling at start and end), `memory.txt`, `bandwidth.txt`, `speed.md` (throughput at depths 0, 2048 and 8192) and `thermal_power.csv` (temperature, throttling and power every 5 s).

| Measurement | Value | Notes |
| --- | --- | --- |
| Prefill, depth 0 (pp64; also pp32) | 1.40 tokens/s | Same result in 3 runs |
| Decode, depth 0 (tg16; also tg8) | 1.11 tokens/s | 1.11 × 4.06 GB ≈ 4.5 GB/s actually read |
| Peak process memory, 16k context | 5.60 GiB (VmHWM 5,876,960 kB) | Weights 3.78 GiB + KV cache about 1.75 GiB (112 KiB per token) |
| System memory with server loaded | 2,380 MB used, 4,155 MB buff/cache | Memory-mapped weights show as cache, not "used" |
| Memory bandwidth | 12.6 GB/s (12,010 MiB/s) | sysbench read, 4 threads, 512M block. The earlier 1M-block result (68 GB/s) measured CPU cache and is discarded |
| Depths 2048 and 8192 | Prefill 1.44 / 1.39 tokens/s; decode 1.00 / 0.71 tokens/s | Run 15:13–17:18 Nov 6, no throttling, peak 75.7°C, mean 7.7 W (PMIC rails), 57.6 kJ |
| Temperature, run of Nov 6 | About 71°C, `throttled=0x0` | Earlier run at 90°C (`0xe0006`) discarded: throttled |

- **Finding 1:** prefill is barely faster than decode. The A76 cores have no BF16 arithmetic, so BF16 gets no benefit from processing tokens in batches.
- **Finding 2 (confirmed):** decode moves 4.5 / 4.3 / 3.6 GB/s at depths 0 / 2k / 8k, which is 36% / 34% / 28% of the measured 12.6 GB/s. BF16 decode is compute-bound, not bandwidth-bound. A bandwidth-bound decode would reach about 3.1 tokens/s at depth 0.
- htop's "used" figure understates memory because the weights are memory-mapped. Use the process's VmHWM (peak) instead.
- `llama-server` defaults to `n_slots = 4` with one shared (unified) KV cache. The harness should use `-np 1`.
- For a quick check, use `llama-bench ... -p 32 -n 8 -r 1` (a few minutes). A full `-p 512 -n 128` run takes about 50 minutes in BF16.

**Scale of the BF16 baseline sessions** (harness replay of the ground truth, about 4 characters per token; real counts get logged on the Pi): the 30 tool schemas are about 3,200 tokens, so about 38 minutes before the first call. The largest context per session has a median of 13,700 tokens (2 of 33 tasks exceed 16k), and a stateless session sends a median of 79,000 tokens. The 13 test tasks need about 230 hours of prefill at 1.4 tokens/s, so measure latency on the Pi with short runs and accuracy off the Pi.

## Next steps

The next step is to finish Task 3 items 1–3, then build the harness, which items 4–5 and every later task depend on. The Nov 29 milestone needs the tool subset, the long tasks, a working harness and the characterised baseline.

**Picking this up as a teammate:**

1. `ssh edge_maria@edge-9`, then `cd ~/long-context-tool-calling-agent && git pull`.
2. Before any measurement: `vcgencmd get_throttled` should be `0x0`, nothing else heavy should be running (`pgrep -af llama`), and the desktop should be off.
3. Run long jobs with `nohup ... &` or in tmux. Never unplug the Pi without `sudo shutdown -h now`.

## Full command reference

Every command needed to go from a fresh SD card to the baseline results, in order. Run on the Pi unless marked **Mac**. Replace CAPS placeholders, and type straight quotes. Commands already listed in the sections above (networking, llama.cpp build, model conversion) are not repeated here.

**1. First connection (Mac)**, about 5 minutes after the first power-on:

```bash
ssh edge_maria@edge-9.local
ssh-keygen -R edge-9.local        # only if SSH warns "host key changed" (after a re-flash)
```

If the name does not resolve, find the Pi on the network (192.168.18.x is the home router; adjust for other networks):

```bash
for i in {1..254}; do ping -c1 -W 200 192.168.18.$i >/dev/null & done; wait
arp -a | grep -iE "2c:cf:67|d8:3a:dd|dc:a6:32|e4:5f:01|b8:27:eb|28:cd:c1"
nc -vz -G 3 IP_ADDRESS 22          # "succeeded" means SSH is up
```

**2. System packages and console-only mode**

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y git cmake build-essential libcurl4-openssl-dev python3-venv python3-pip tmux sysbench htop gh
systemctl get-default                          # graphical.target means the desktop is on
sudo systemctl set-default multi-user.target   # turn it off for measurements
sudo reboot
```

**3. Fan check, fix and monitoring**

```bash
cat /proc/device-tree/cooling_fan/status; echo   # should say: okay
ls /sys/class/thermal/                           # should list cooling_device0
# only if the status says disabled:
printf '\n[all]\ndtparam=cooling_fan=on\n' | sudo tee -a /boot/firmware/config.txt
sudo reboot
# test the fan: force full speed, then back to automatic
sudo pinctrl FAN_PWM op dl
sudo pinctrl FAN_PWM a0
# watch temperature and throttling (0x0 = fine); Ctrl+C to stop
watch -n 2 'vcgencmd measure_temp; vcgencmd get_throttled'
htop                                             # CPU load; q to quit
```

**4. Tailscale on the Mac:** install the Tailscale app, allow its VPN configuration, and sign in with the same account as the Pi. Then `ssh edge_maria@edge-9` works from any network.

**5. Project repo on the Pi**

```bash
gh auth login        # GitHub.com > HTTPS > Login with a web browser; enter the code on the Mac
git clone https://github.com/REPO_OWNER/long-context-tool-calling-agent.git ~/long-context-tool-calling-agent
cd ~/long-context-tool-calling-agent
printf '.DS_Store\nthird_party/\n*.gguf\n' >> .gitignore
# Mac: copy a file onto the Pi
scp ~/Downloads/FILE edge_maria@edge-9:~/long-context-tool-calling-agent/scripts/
```

**6. Baseline measurements (Task 3 items 1–3)**

```bash
cd ~/long-context-tool-calling-agent
grep memory-block-size scripts/baseline_bench.sh    # must say 512M (fixed Nov 7)
# quick check, a few minutes
~/llama.cpp/build-baseline/bin/llama-bench -m ~/models/qwen3-1.7b-bf16.gguf -t 4 -p 32 -n 8 -r 1
# full run, about 2 hours in BF16; survives SSH drops
OUT=$PWD/results/baseline_$(date +%Y%m%d_%H%M) nohup bash scripts/baseline_bench.sh > results/bench_run.log 2>&1 &
pgrep -af "baseline_bench|llama-bench"     # still running?
tail -3 results/bench_run.log              # finished when it prints "done -> ..."
# same method for a later configuration
MODEL=~/models/OTHER_MODEL.gguf OUT=$PWD/results/OTHER_NAME nohup bash scripts/baseline_bench.sh > results/OTHER_NAME.log 2>&1 &
# save results
git add results/ && git commit -m "MESSAGE" && git push
```

**7. Harness (Task 3 items 4–5)**, from `harness/README.md`:

```bash
cd ~/long-context-tool-calling-agent
unzip ~/a1_harness.zip     # harness/, tests/, splits/, scripts/
git clone --filter=blob:none --sparse https://github.com/ShishirPatil/gorilla.git third_party/gorilla
git -C third_party/gorilla sparse-checkout set berkeley-function-call-leaderboard
git -C third_party/gorilla checkout 6ea57973c7a6097fd7c5915698c54c17c5b1b6c8
B=third_party/gorilla/berkeley-function-call-leaderboard
bash tests/run_mock_tests.sh         # self-test, about 1 minute, no model needed
# baseline server (leave running in its own tmux window)
~/llama.cpp/build-baseline/bin/llama-server -m ~/models/qwen3-1.7b-bf16.gguf -c 16384 -t 4 --jinja -np 1 --host 127.0.0.1 --port 8080
# short Pi run: 1 test task, first 2 turns
nohup python3 -m harness.run --bfcl-root $B --split splits/tasks_TradingBot_MessageAPI.json --which test --limit 1 --max-turns 2 --out results/harness_baseline_pi_short > results/harness_short.log 2>&1 &
python3 -m harness.summarize results/harness_baseline_pi_short
```

**8. Cold start:** clear the file cache first, so the model is read from the SD card: `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`. Then pass the server command to the harness with `--launch-server "..."`.

**9. Shut down:** `sudo shutdown -h now`. Unplug only after the light turns red.
