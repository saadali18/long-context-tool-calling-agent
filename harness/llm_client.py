"""Minimal client for llama-server's OpenAI-compatible chat endpoint (stdlib only)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


class ServerError(RuntimeError):
    pass


class LlamaServer:
    def __init__(self, url: str = "http://127.0.0.1:8080", timeout_s: float = 6 * 3600):
        # BF16 on the Pi prefills ~1.4 tokens/s: a 16k prompt is ~3 h, so the timeout is long.
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    def wait_ready(self, max_wait_s: float = 600) -> float:
        t0 = time.monotonic()
        while time.monotonic() - t0 < max_wait_s:
            if self.health():
                return time.monotonic() - t0
            time.sleep(0.5)
        raise ServerError("llama-server did not become ready")

    def chat(self, messages: list, tools: list, *, temperature: float, max_tokens: int,
             cache_prompt: bool, enable_thinking: bool, seed: int = 0) -> dict:
        body = {
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            # llama-server extensions
            "cache_prompt": cache_prompt,                       # baseline: False (no prefix reuse)
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        req = urllib.request.Request(
            self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                out = json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:2000]
            raise ServerError(f"HTTP {e.code}: {detail}") from e
        out["_client_wall_ms"] = (time.monotonic() - t0) * 1000
        return out
