"""Background sampler: temperature, throttling, PMIC power and llama-server memory.

On a machine without vcgencmd (a laptop running accuracy evals) the Pi fields are
simply left empty, so the same harness runs everywhere.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time


def _run(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return ""


def read_power_w() -> float | None:
    out = _run(["vcgencmd", "pmic_read_adc"])
    if not out:
        return None
    cur, volt = {}, {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2 or "=" not in parts[1]:
            continue
        rail, val = parts[0][:-2], parts[1].split("=")[1].rstrip("AV")
        try:
            (cur if "current" in parts[1] else volt)[rail] = float(val)
        except ValueError:
            pass
    return sum(cur[r] * volt[r] for r in cur if r in volt)


def read_temp_c() -> float | None:
    out = _run(["vcgencmd", "measure_temp"])
    try:
        return float(out.split("=")[1].split("'")[0])
    except Exception:
        return None


def read_throttled() -> int | None:
    out = _run(["vcgencmd", "get_throttled"])
    try:
        return int(out.strip().split("=")[1], 16)
    except Exception:
        return None


def read_rss_kb(pid: int | None) -> int | None:
    if not pid:
        return None
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except Exception:
        return None
    return None


def find_server_pid() -> int | None:
    out = _run(["pgrep", "-n", "llama-server"])
    return int(out.strip()) if out.strip().isdigit() else None


class Monitor:
    """Samples every `period_s`; call mark() to get a snapshot index, summary(a, b) for a window."""

    def __init__(self, server_pid: int | None = None, period_s: float = 1.0):
        self.pid = server_pid or find_server_pid()
        self.period = period_s
        self.on_pi = shutil.which("vcgencmd") is not None
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._t.start()
        return self

    def stop(self):
        self._stop.set()
        self._t.join(timeout=10)

    def _loop(self):
        while not self._stop.is_set():
            s = {"t": time.monotonic(), "rss_kb": read_rss_kb(self.pid)}
            if self.on_pi:
                s.update(temp_c=read_temp_c(), throttled=read_throttled(), power_w=read_power_w())
            with self._lock:
                self.samples.append(s)
            self._stop.wait(self.period)

    def mark(self) -> float:
        return time.monotonic()

    def summary(self, t0: float, t1: float) -> dict:
        with self._lock:
            w = [s for s in self.samples if t0 <= s["t"] <= t1]
        out = {"n_samples": len(w)}
        rss = [s["rss_kb"] for s in w if s.get("rss_kb")]
        if rss:
            out["peak_server_rss_mib"] = round(max(rss) / 1024, 1)
        if self.on_pi and w:
            temps = [s["temp_c"] for s in w if s.get("temp_c") is not None]
            thr = [s["throttled"] for s in w if s.get("throttled") is not None]
            pw = [(s["t"], s["power_w"]) for s in w if s.get("power_w") is not None]
            if temps:
                out["max_temp_c"] = max(temps)
            if thr:
                # bits 0-3 = happening now (under-voltage, freq cap, throttled, soft temp limit)
                out["throttled_during"] = any(x & 0xF for x in thr)
            if len(pw) >= 2:
                energy = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(pw, pw[1:]))
                out["energy_j_pmic"] = round(energy, 1)
                out["mean_power_w_pmic"] = round(energy / (pw[-1][0] - pw[0][0]), 3)
        return out
