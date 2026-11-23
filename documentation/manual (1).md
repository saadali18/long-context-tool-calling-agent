# Raspberry Pi 5 --- BFCL Long-Context Agent Experiment Manual

This manual explains what each command does, why it is needed, and what
to check before moving to the next step.

## 1. Go to the project and set the BFCL path

``` bash
cd ~/long-context-tool-calling-agent
export B=~/long-context-tool-calling-agent/third_party/gorilla/berkeley-function-call-leaderboard
```

### What these commands do

-   `cd ~/long-context-tool-calling-agent` changes the terminal's
    current working directory to the project root. Run project commands
    from here so relative paths such as `scripts/...` and `results/...`
    resolve correctly.
-   `~` is shorthand for your home directory.
-   `export B=...` creates an environment variable named `B` containing
    the path to the BFCL benchmark repository.
-   `$B` means "the value stored in the `B` environment variable." For
    example, `--bfcl-root $B` passes the BFCL directory to the harness.
-   The variable is available to the current shell and programs started
    from it. If you open a new terminal session, export it again.

## 2. Check or apply the long-context caps

Check the current configuration:

``` bash
python3 scripts/cap_long_context.py --status
```

Expected values for this experiment:

``` text
MA_5=500
MA_10=500
MA_20=500
TRANSACTION_HISTORY=100
```

If the status is not correct, apply the caps:

``` bash
python3 scripts/cap_long_context.py
```

### What these commands do

-   `python3` runs the script using Python 3.
-   `scripts/cap_long_context.py` is the project script that configures
    limits on the relevant long-context tool outputs.
-   `--status` asks the script to display the current values. It checks
    the configuration; it does not apply changes.
-   Running the script without `--status` applies the configured caps.
-   The caps limit the size of selected backend/tool outputs. This helps
    keep tool responses from growing excessively and consuming the
    model's context window.
-   The four values are specific to the project configuration. Do not
    assume they are token counts unless the script defines them that
    way; they are configuration limits for the named fields.

## 3. Start the local model server

First, make sure the `results` directory exists:

``` bash
mkdir -p results
```

Then start the server:

``` bash
nohup ~/llama.cpp/build-baseline/bin/llama-server \
  -m ~/models/qwen3-1.7b-bf16.gguf \
  -c 16384 -t 4 --jinja -np 1 --port 8080 \
  > results/server.log 2>&1 &
```

### What this command does

-   `llama-server` starts a local inference server using llama.cpp.
-   `-m ~/models/qwen3-1.7b-bf16.gguf` tells it which model file to
    load: Qwen3-1.7B in BF16 GGUF format.
-   `-c 16384` sets the context window to 16,384 tokens.
-   `-t 4` uses four CPU threads.
-   `--jinja` enables the model's Jinja chat template handling.
-   `-np 1` configures one parallel decoding slot.
-   `--port 8080` makes the server listen on port 8080.
-   `nohup` allows the process to continue running if the terminal
    session is disconnected.
-   `> results/server.log` sends standard output to the log file. The
    `>` operator overwrites that file when the command starts.
-   `2>&1` sends standard error to the same place as standard output, so
    both are captured in the log.
-   `&` runs the server in the background and returns control to the
    terminal.

### Check that the server is responding

``` bash
curl -s localhost:8080/health; echo
```

-   `curl` sends an HTTP request to the local server.
-   `localhost` means this same Raspberry Pi.
-   `:8080` is the port used above.
-   `/health` is the server's health endpoint.
-   `-s` makes curl quiet, suppressing its progress display.
-   `; echo` prints a newline after the response so the terminal prompt
    does not run into it.

A typical healthy response is:

``` json
{"status":"ok"}
```

This confirms that the health endpoint responds. It does not, by itself,
prove that a complete model inference or benchmark task will succeed.

To inspect server logs:

``` bash
tail -f results/server.log
```

`tail -f` displays the end of the log and keeps following new lines as
they are written. Press `Ctrl+C` to stop following the log; this does
**not** stop the server process.

## 4. Check whether a benchmark harness is already running

``` bash
pgrep -af harness.run || echo IDLE
```

-   `pgrep` searches the process list for processes matching a pattern.
-   `-a` prints the full command line, not just the process ID.
-   `-f` matches against the full command line.
-   `harness.run` is the pattern for the benchmark harness module.
-   `|| echo IDLE` means: if `pgrep` finds no matching process (returns
    a non-zero exit status), print `IDLE`.

Use this to avoid accidentally starting a second copy of the same
benchmark while another one is still running.

## 5. Run Task 132 using an already-running server

Before starting, confirm that the server is healthy and that the caps
are correct:

``` bash
python3 scripts/cap_long_context.py --status
curl -s localhost:8080/health; echo
```

Then launch the benchmark:

``` bash
nohup python3 -m harness.run \
  --bfcl-root $B \
  --ids multi_turn_long_context_132 \
  --out results/pi_bf16_think_132_cap500 \
  > results/pi_bf16_think_132_cap500.log 2>&1 &
```

### What each part does

-   `nohup` lets the benchmark continue if the terminal disconnects.
-   `python3 -m harness.run` runs the Python module `harness.run` as a
    program.
-   `--bfcl-root $B` tells the harness where the BFCL benchmark files
    are located. `$B` expands to the path exported earlier.
-   `--ids multi_turn_long_context_132` selects the benchmark task with
    ID `multi_turn_long_context_132`.
-   `--out results/pi_bf16_think_132_cap500` sets the output directory
    for this run. The harness writes its run artifacts there.
-   `> results/pi_bf16_think_132_cap500.log 2>&1` captures both normal
    output and errors in one log file.
-   `&` runs the harness in the background.

This command does **not** include `--launch-server`, so it expects a
compatible llama-server to be running already at the
endpoint/configuration expected by the harness.

Watch the benchmark log:

``` bash
tail -f results/pi_bf16_think_132_cap500.log
```

When the run has completed, inspect the scoring output in its run
directory:

``` bash
cat results/pi_bf16_think_132_cap500/bfcl_scores.json
```

-   `cat` prints the contents of the named file.
-   `bfcl_scores.json` contains the benchmark scoring summary, if the
    harness produced it.
-   Replace the example run name with the actual output directory if you
    used a different one.

## 6. Run a cold-page-cache / server-start experiment for Task 143

### Stop an existing llama-server

``` bash
pkill -f llama-server
sleep 2
pgrep -af llama-server
```

-   `pkill -f llama-server` sends a termination signal to processes
    whose full command line matches `llama-server`.
-   Be careful: this can stop every matching llama-server process, not
    just one particular instance. Check what is running before using it.
-   `sleep 2` waits for two seconds, giving the process time to exit.
-   `pgrep -af llama-server` checks whether a matching server process
    remains. Ideally, it prints nothing if the server has stopped.

### Drop Linux page, directory-entry and inode caches

``` bash
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
```

-   `sync` asks Linux to flush pending filesystem writes.
-   `&&` runs the next command only if `sync` succeeds.
-   `echo 3` prints the value `3`.
-   `|` pipes that value into the next command.
-   `sudo` runs the command with administrator privileges.
-   `tee /proc/sys/vm/drop_caches` writes the value into the Linux
    kernel setting that drops filesystem caches.
-   The value `3` requests dropping page cache, dentries and inode
    caches.

This is a **cold-page-cache** preparation step. It is not a full
hardware cold boot and does not clear CPU caches or reset the Raspberry
Pi.

### Launch Task 143 and let the harness start the server

``` bash
nohup python3 -m harness.run \
  --bfcl-root $B \
  --ids multi_turn_long_context_143 \
  --max-turns 1 \
  --launch-server "$HOME/llama.cpp/build-baseline/bin/llama-server -m $HOME/models/qwen3-1.7b-bf16.gguf -c 16384 -t 4 --jinja -np 1 --port 8080" \
  --out results/pi_bf16_think_cold_143_t1 \
  > results/pi_bf16_think_cold.log 2>&1 &
```

### What each part does

-   `--ids multi_turn_long_context_143` selects Task 143.
-   `--max-turns 1` limits the harness run to one benchmark turn. A
    single benchmark turn can still involve more than one model request,
    for example a tool-call request followed by a final-answer request.
-   `--launch-server "..."` tells the harness to launch the model server
    itself. The quoted string is the server command, including the model
    path and server settings.
-   `$HOME` is another way to refer to the current user's home
    directory.
-   `-c 16384`, `-t 4`, `--jinja`, `-np 1` and `--port 8080` have the
    same meanings as in the server-start command above.
-   `--out results/pi_bf16_think_cold_143_t1` sets the output directory.
-   `> results/pi_bf16_think_cold.log 2>&1` captures standard output and
    errors in the log.
-   `nohup` and `&` allow the benchmark process to continue in the
    background.

The command shown does not explicitly specify thinking mode,
temperature, top-p, top-k or maximum output tokens. Confirm those
settings from the run metadata/configuration rather than assuming them
from the output directory name.

## 7. Follow a run and inspect its results

Follow a log while a run is in progress:

``` bash
tail -f results/<run>.log
```

Replace `<run>` with the actual log filename. Press `Ctrl+C` to stop
viewing the log; the benchmark continues running.

Inspect the score file after completion:

``` bash
cat results/<run>/bfcl_scores.json
```

Replace `<run>` with the actual output directory.

If the output directory or file does not exist, check the harness log
for the actual output path and any errors.

## 8. Save experiment results with Git

``` bash
git add results/<run>
git commit -m "Pi BF16 <task>"
git push
```

-   `git add results/<run>` stages the specified results directory so
    Git includes it in the next commit.
-   `git commit -m "..."` creates a local commit with the supplied
    message. Use a descriptive message, replacing `<task>` with the task
    or experiment name.
-   `git push` uploads the commit to the configured remote repository.

These commands save the experiment artifacts in version control. Check
that the intended files are staged before committing, especially if the
results directory contains large logs or generated files.

## 9. Important distinction: frozen baseline vs exploratory runs

The frozen baseline configuration recorded for this project is:

  -----------------------------------------------------------------------
  Setting                             Frozen baseline
  ----------------------------------- -----------------------------------
  Model                               Qwen3-1.7B BF16 GGUF

  llama.cpp                           Commit `efbf6e80a`

  Context                             16,384 tokens

  Threads                             4

  Temperature                         0

  Maximum output                      512 tokens

  Thinking                            Off

  Grammar / constrained decoding      Off

  Prefix caching                      Off

  Tool schemas                        All 30 included in the prompt

  Conversation history                Full history re-sent on every turn
                                      (stateless client)
  -----------------------------------------------------------------------

Some Pi runs discussed in the project used **thinking ON**, temperature
`0.6`, top-p `0.95`, top-k `20`, maximum output `4096` and other
exploratory settings. Those runs are useful for investigation, but they
are **not the frozen baseline**. Keep their results labelled separately
and verify each run's metadata before comparing numbers.

## 10. Interpretation notes and cautions

-   A healthy `/health` response only confirms that the endpoint
    responds; it does not guarantee successful tool calling or benchmark
    completion.
-   `--status` checks the configured caps; it does not apply them.
-   Task 132's command uses an existing server. Task 143's command uses
    `--launch-server` so the harness starts one.
-   `--max-turns 1` means one benchmark turn, not necessarily one model
    generation.
-   Dropping Linux caches is not the same as rebooting the Pi or
    clearing every cache.
-   Do not infer thinking/sampling settings from a folder name. Check
    run metadata.
-   Keep functional evaluation results from a Mac separate from
    performance measurements on the Raspberry Pi. Only compare them as
    the same functional baseline if the model, software, tools, context
    and decoding settings match.
-   Report full-task success and per-turn accuracy separately. One
    failed turn can make an entire multi-turn task fail.
-   Treat measured runtime, memory, power and energy as results for the
    exact run conditions. Do not generalize one run's measurements to
    every task or configuration.
