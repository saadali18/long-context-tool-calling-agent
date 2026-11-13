"""One agent session: the A1 agent loop, tool execution, per-request logging and scoring."""

from __future__ import annotations

import json
import time

from . import bfcl_env as env
from .llm_client import LlamaServer, ServerError

BFCL_MULTITURN_HINT = (
    "At each turn, you should try your best to complete the tasks requested by the user within "
    "the current turn. Continue to output functions to call until you have fulfilled the user's "
    "request to the best of your ability. Once you have no more functions to call, the system "
    "will consider the current turn complete and proceed to the next turn or task."
)


def run_session(task: env.Task, bfcl: env.BFCL, server: LlamaServer, classes: list[str], cfg: dict,
                log_request, monitor=None) -> dict:
    """Run one task end to end. `log_request(dict)` is called after every model request."""
    tools = bfcl.tools_for(classes, task.excluded_functions)
    tools_by_name = {t["function"]["name"]: t for t in tools}
    model_inst = bfcl.new_instances(classes, task.initial_config)
    gt_inst = bfcl.new_instances(classes, task.initial_config)
    model_owner, gt_owner = env.method_owner(model_inst), env.method_owner(gt_inst)

    messages = []
    if cfg.get("system_prompt") == "bfcl":
        messages.append({"role": "system", "content": BFCL_MULTITURN_HINT})

    n_turns = len(task.turns) if not cfg.get("max_turns") else min(cfg["max_turns"], len(task.turns))
    model_results_all: list[str] = []
    turns, calls_log = [], []
    t_session = time.monotonic()
    ttfc_ms = None
    aborted = None
    n_req = 0
    totals = dict(prompt_tokens=0, reprocessed_tokens=0, cached_tokens=0, prefill_ms=0.0,
                  decode_tokens=0, decode_ms=0.0, tool_ms=0.0, request_wall_ms=0.0)
    max_context = 0

    for ti in range(n_turns):
        messages.extend(task.turns[ti])
        t_turn = time.monotonic()
        turn_calls = 0
        for step in range(env.MAX_STEPS_PER_TURN):
            try:
                out = server.chat(messages, tools, temperature=cfg["temperature"],
                                  max_tokens=cfg["max_tokens"], cache_prompt=cfg["cache_prompt"],
                                  enable_thinking=cfg["thinking"], seed=cfg.get("seed", 0))
            except ServerError as e:
                aborted = f"server_error: {e}"[:500]
                break
            n_req += 1
            msg = out["choices"][0]["message"]
            finish = out["choices"][0].get("finish_reason")
            tool_calls = msg.get("tool_calls") or []
            content = msg.get("content") or ""
            tim = out.get("timings", {}) or {}
            usage = out.get("usage", {}) or {}

            prompt_tokens = usage.get("prompt_tokens", tim.get("prompt_n", 0) + tim.get("cache_n", 0))
            rec = {
                "task": task.id, "turn": ti, "step": step, "n_messages": len(messages),
                "prompt_tokens": prompt_tokens,                 # tokens sent this request
                "reprocessed_tokens": tim.get("prompt_n"),      # tokens actually prefilled
                "cached_tokens": tim.get("cache_n", 0),         # reused from the prompt cache
                "prefill_ms": tim.get("prompt_ms"),
                "decode_tokens": tim.get("predicted_n", usage.get("completion_tokens")),
                "decode_ms": tim.get("predicted_ms"),
                "request_wall_ms": round(out["_client_wall_ms"], 1),
                "finish_reason": finish,
                "content": content[:2000],
                "calls": [],
            }
            max_context = max(max_context, prompt_tokens or 0)

            assistant = {"role": "assistant", "content": content}
            if tool_calls:
                assistant["tool_calls"] = tool_calls
            messages.append(assistant)

            if not tool_calls and "<tool_call>" in content:
                # the model tried to call a tool but the server could not parse it
                rec["calls"].append({"name": None, "well_formed": False, "exists": False,
                                     "schema_valid": False, "problems": ["unparsed <tool_call> text"]})

            tool_ms = 0.0
            for k, tc in enumerate(tool_calls):
                fn = tc.get("function", {})
                v = env.check_call(tools_by_name, fn.get("name"), fn.get("arguments", "{}"))
                t0 = time.monotonic()
                if v["well_formed"]:
                    result = env.execute(model_owner, fn.get("name"), v["args"])
                else:
                    result = "Error during execution: could not parse the call's arguments as JSON"
                dt = (time.monotonic() - t0) * 1000
                tool_ms += dt
                if v["well_formed"] and v["exists"]:
                    model_results_all.append(result)
                turn_calls += 1
                v.update(tool_ms=round(dt, 2), result_chars=len(result))
                rec["calls"].append(v)
                messages.append({"role": "tool", "tool_call_id": tc.get("id") or f"call_{ti}_{step}_{k}",
                                 "content": result})
            rec["tool_ms"] = round(tool_ms, 2)

            if ttfc_ms is None and tool_calls:
                ttfc_ms = (time.monotonic() - t_session) * 1000

            for key, val in (("prompt_tokens", rec["prompt_tokens"]),
                             ("reprocessed_tokens", rec["reprocessed_tokens"]),
                             ("cached_tokens", rec["cached_tokens"]), ("prefill_ms", rec["prefill_ms"]),
                             ("decode_tokens", rec["decode_tokens"]), ("decode_ms", rec["decode_ms"]),
                             ("tool_ms", tool_ms), ("request_wall_ms", rec["request_wall_ms"])):
                totals[key] += val or 0
            calls_log.extend(rec["calls"])
            log_request(rec)

            if not tool_calls:
                break  # the model ended its turn

        # ---- score this turn (BFCL checker logic) ----
        gt_results = []
        for call in task.ground_truth[ti]:
            name, kwargs = env.parse_call_string(call, gt_owner)
            gt_results.append(env.execute(gt_owner, name, kwargs))
        turn = {"turn": ti, "model_calls": turn_calls, "gt_calls": len(task.ground_truth[ti]),
                "wall_ms": round((time.monotonic() - t_turn) * 1000, 1)}
        if aborted:
            turn.update(passed=False, reason="aborted")
        elif task.ground_truth[ti] and turn_calls == 0:
            turn.update(passed=False, reason="empty_turn_model_response")
        else:
            bad = {}
            for c in task.involved_classes:
                ok, diffs = env.state_matches(model_inst[c], gt_inst[c])
                if not ok:
                    bad[c] = sorted(diffs)
            missing = env.responses_contained(gt_results, model_results_all)
            if bad:
                turn.update(passed=False, reason="instance_state_mismatch", state_diff_attrs=bad)
            elif missing:
                turn.update(passed=False, reason="execution_response_mismatch",
                            missing=[m[:200] for m in missing])
            else:
                turn.update(passed=True)
        turns.append(turn)
        if aborted:
            break

    wall_ms = (time.monotonic() - t_session) * 1000
    first_fail = next((t["turn"] for t in turns if not t["passed"]), None)
    session = {
        "task": task.id, "source": task.source, "involved_classes": task.involved_classes,
        "n_turns_total": len(task.turns), "n_turns_run": len(turns),
        "truncated": n_turns < len(task.turns),
        "session_success": (not aborted) and len(turns) == len(task.turns) and first_fail is None,
        "turns_passed": sum(t["passed"] for t in turns),
        "first_failed_turn": first_fail,
        "fail_reason": next((t.get("reason") for t in turns if not t["passed"]), None),
        "aborted": aborted,
        "turns": turns,
        "n_requests": n_req,
        "n_calls": len(calls_log),
        "n_calls_valid": sum(c["well_formed"] and c["exists"] and c["schema_valid"] for c in calls_log),
        "n_calls_well_formed": sum(c["well_formed"] for c in calls_log),
        "n_calls_existing_tool": sum(c["exists"] for c in calls_log),
        "n_calls_schema_valid": sum(c["schema_valid"] for c in calls_log),
        "max_context_tokens": max_context,
        "warm_time_to_first_call_ms": round(ttfc_ms, 1) if ttfc_ms is not None else None,
        "session_wall_ms": round(wall_ms, 1),
        **{f"total_{k}": round(v, 1) for k, v in totals.items()},
    }
    if monitor is not None:
        session.update(monitor.summary(t_session, time.monotonic()))
    return session


def jsonl_writer(path):
    f = open(path, "a", buffering=1)

    def write(rec):
        f.write(json.dumps(rec, default=str) + "\n")
    return write, f
