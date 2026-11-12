"""Fake llama-server for testing the harness without a model.

Replays each task's ground-truth calls as OpenAI tool calls, so a correct harness
must score every session as a success. --mode wrong corrupts string arguments in
turn 1 (state/response mismatch). --mode noisy also injects a malformed call,
an unknown tool and a schema error, to exercise the validity metrics.

  python tests/mock_server.py --bfcl-root <path> --port 8099 [--mode gt|noisy|silent|wrong]
"""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import bfcl_env as env  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--bfcl-root", required=True)
ap.add_argument("--port", type=int, default=8099)
ap.add_argument("--mode", default="gt", choices=["gt", "noisy", "silent", "wrong"])
A = ap.parse_args()

BF = env.BFCL(A.bfcl_root)
TASKS = BF.load_tasks()
BY_FIRST = {t.turns[0][0]["content"]: t for t in TASKS.values() if t.turns and t.turns[0]}


def plan_for(task):
    inst = BF.new_instances(task.involved_classes, task.initial_config)
    owner = env.method_owner(inst)
    return [[env.parse_call_string(c, owner) for c in turn] for turn in task.ground_truth]


def reply(messages):
    first_user = next(m["content"] for m in messages if m["role"] == "user")
    task = BY_FIRST[first_user]
    turn = sum(m["role"] == "user" for m in messages) - 1
    last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
    done = sum(m["role"] == "tool" and not m.get("tool_call_id", "").startswith("bad")
               for m in messages[last_user:])
    plan = plan_for(task)[turn]
    if A.mode == "silent" or done >= len(plan):
        return {"role": "assistant", "content": "Done."}, []
    name, kwargs = plan[done]
    if A.mode == "wrong" and turn == 1:
        kwargs = {k: (v + "X" if isinstance(v, str) else v) for k, v in kwargs.items()}
    calls = [{"id": f"c{turn}_{done}", "type": "function",
              "function": {"name": name, "arguments": json.dumps(kwargs)}}]
    if A.mode == "noisy" and done == 0 and turn == 0:
        # three invalid calls, then the correct one
        calls = [
            {"id": "bad1", "type": "function", "function": {"name": name, "arguments": "{not json"}},
            {"id": "bad2", "type": "function", "function": {"name": "no_such_tool", "arguments": "{}"}},
            {"id": "bad3", "type": "function", "function": {"name": name, "arguments": json.dumps({"zzz": 1})}},
        ] + calls
    return {"role": "assistant", "content": "", "tool_calls": calls}, calls


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            self._send({"status": "ok"})
        elif self.path == "/props":
            self._send({"mock": True, "mode": A.mode})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["cache_prompt"] is False or body["cache_prompt"] is True
        msg, calls = reply(body["messages"])
        n_prompt = len(json.dumps(body["messages"]) + json.dumps(body["tools"])) // 4
        self._send({
            "choices": [{"message": msg, "finish_reason": "tool_calls" if calls else "stop"}],
            "usage": {"prompt_tokens": n_prompt, "completion_tokens": 20},
            "timings": {"prompt_n": 0 if body["cache_prompt"] else n_prompt, "cache_n": 0,
                        "prompt_ms": n_prompt / 1.4 * 1000, "predicted_n": 20, "predicted_ms": 20 / 1.11 * 1000},
        })


ThreadingHTTPServer(("127.0.0.1", A.port), H).serve_forever()
