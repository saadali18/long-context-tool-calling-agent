"""Summarise a harness run into the spec's metrics.

  python -m harness.summarize results/harness_baseline [--json]
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()] if path.exists() else []


def summarize(run_dir: str) -> dict:
    d = Path(run_dir)
    ses, req = load(d / "sessions.jsonl"), load(d / "requests.jsonl")
    if not ses:
        return {"error": f"no sessions in {d}"}
    turns = [t for s in ses for t in s["turns"]]
    n_calls = sum(s["n_calls"] for s in ses)

    def ratio(a, b):
        return round(a / b, 3) if b else None

    def med(xs):
        xs = [x for x in xs if x is not None]
        return round(st.median(xs), 1) if xs else None

    out = {
        "sessions": len(ses),
        "session_success": ratio(sum(s["session_success"] for s in ses), len(ses)),
        "per_turn_accuracy": ratio(sum(t["passed"] for t in turns), len(turns)),
        "call_validity": ratio(sum(s["n_calls_valid"] for s in ses), n_calls),
        "calls_well_formed": ratio(sum(s["n_calls_well_formed"] for s in ses), n_calls),
        "calls_existing_tool": ratio(sum(s["n_calls_existing_tool"] for s in ses), n_calls),
        "calls_schema_valid": ratio(sum(s["n_calls_schema_valid"] for s in ses), n_calls),
        "fail_reasons": {},
        "median_session_wall_s": med([s["session_wall_ms"] / 1000 for s in ses]),
        "median_turn_wall_s": med([t["wall_ms"] / 1000 for t in turns]),
        "median_request_prefill_s": med([r["prefill_ms"] / 1000 for r in req if r.get("prefill_ms")]),
        "median_request_decode_s": med([r["decode_ms"] / 1000 for r in req if r.get("decode_ms")]),
        "median_request_tool_ms": med([r.get("tool_ms") for r in req]),
        "median_warm_time_to_first_call_s": med([s["warm_time_to_first_call_ms"] / 1000
                                                  for s in ses if s.get("warm_time_to_first_call_ms")]),
        "cold_time_to_first_call_s": next((round(s["cold_time_to_first_call_ms"] / 1000, 1)
                                           for s in ses if s.get("cold_time_to_first_call_ms")), None),
        "total_tokens_sent": sum(s["total_prompt_tokens"] for s in ses),
        "total_tokens_reprocessed": sum(s["total_reprocessed_tokens"] for s in ses),
        "max_context_tokens": max(s["max_context_tokens"] for s in ses),
        "peak_server_rss_mib": max((s.get("peak_server_rss_mib") or 0) for s in ses) or None,
        "median_energy_j_pmic": med([s.get("energy_j_pmic") for s in ses]),
        "max_temp_c": max((s.get("max_temp_c") or 0) for s in ses) or None,
        "sessions_throttled": sum(bool(s.get("throttled_during")) for s in ses),
        "aborted": sum(bool(s.get("aborted")) for s in ses),
        "truncated": sum(bool(s.get("truncated")) for s in ses),
    }
    for s in ses:
        if s["fail_reason"]:
            out["fail_reasons"][s["fail_reason"]] = out["fail_reasons"].get(s["fail_reason"], 0) + 1
    # accuracy by turn index: where do sessions start failing?
    by_idx = {}
    for t in turns:
        by_idx.setdefault(t["turn"], []).append(t["passed"])
    out["turn_accuracy_by_index"] = {k: ratio(sum(v), len(v)) for k, v in sorted(by_idx.items())}
    return out


def main():
    res = summarize(sys.argv[1])
    if "--json" in sys.argv:
        print(json.dumps(res, indent=2))
        return
    for k, v in res.items():
        print(f"{k:34s} {v}")


if __name__ == "__main__":
    main()
