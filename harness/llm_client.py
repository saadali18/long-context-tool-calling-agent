"""Minimal client for llama-server's OpenAI-compatible chat endpoint (stdlib only)."""

from __future__ import annotations

import logging
from .debug_log import trace

import json
import time
import urllib.error
import urllib.request


log = logging.getLogger(__name__)


class ServerError(RuntimeError):
    pass


class LlamaServer:
    @trace
    def __init__(self, url: str = "http://127.0.0.1:8080", timeout_s: float = 6 * 3600):
        # BF16 on the Pi prefills ~1.4 tokens/s: a 16k prompt is ~3 h, so the timeout is long.
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s

    @trace
    def health(self) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    @trace
    def wait_ready(self, max_wait_s: float = 600) -> float:
        log.info('server_wait_ready timeout_s=%.1f',max_wait_s)
        t0 = time.monotonic()
        while time.monotonic() - t0 < max_wait_s:
            if self.health():
                elapsed=time.monotonic()-t0
                log.info('server_ready elapsed_s=%.1f',elapsed)
                return elapsed
            time.sleep(0.5)
        log.error("server_timeout")
        raise ServerError("llama-server did not become ready")

    @trace
    def chat(self, messages: list, tools: list, *, temperature: float, max_tokens: int,
             cache_prompt: bool, enable_thinking: bool, seed: int = 0,
             top_p: float | None = None, top_k: int | None = None, min_p: float | None = None) -> dict:
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
        # Qwen3's recommended sampling (sent only when set; otherwise llama-server's defaults apply)
        for k, v in (("top_p", top_p), ("top_k", top_k), ("min_p", min_p)):
            if v is not None:
                body[k] = v
        req = urllib.request.Request(
            self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        log.debug("http_request_start messages=%d tools=%d cache=%s",len(messages),len(tools),cache_prompt)
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                out = json.loads(r.read())
        except urllib.error.HTTPError as e:
            log.error("http_error status=%d elapsed_s=%.1f",e.code,time.monotonic()-t0)
            detail = e.read().decode(errors="replace")[:2000]
            raise ServerError(f"HTTP {e.code}: {detail}") from e
        out["_client_wall_ms"] = (time.monotonic() - t0) * 1000
        log.debug("http_request_finish wall_ms=%.1f",out["_client_wall_ms"])
        return out
