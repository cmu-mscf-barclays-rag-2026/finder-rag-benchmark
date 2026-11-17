"""Small, cached, budget-capped OpenAI chat client for the evaluations.

Every response is cached by model, parameters, and exact messages, so reruns
cost nothing. Each run refuses to make more than ``max_new_calls`` uncached
calls, and token usage is taken from the API's own ``usage`` field. The client
is thread-safe, so evaluations can send calls in parallel; rate-limit (429)
and server errors are retried with the server's Retry-After delay.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


URL = "https://api.openai.com/v1/chat/completions"
# USD per million tokens (gpt-4o-mini list price); used only to report cost.
PRICES = {"gpt-4o-mini": (0.15, 0.60)}
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


def api_key(env_file: str | Path = ".env") -> str:
    load_dotenv(dotenv_path=env_file)
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set; add it to .env or the environment")
    return key


class BudgetExceeded(RuntimeError):
    pass


class CachedOpenAIChat:
    def __init__(self, model: str, cache_path: Path, *, max_new_calls: int,
                 max_tokens: int = 96, timeout: float = 60.0, max_attempts: int = 8) -> None:
        self.model, self.cache_path, self.max_tokens, self.timeout = model, cache_path, max_tokens, timeout
        self.max_new_calls, self.max_attempts = max_new_calls, max_attempts
        self.entries: dict[str, dict[str, Any]] = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.entries[row["key"]] = row["output"]
        self.new_calls = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.rate_limits: dict[str, str] = {}
        self._key = None
        self._lock = threading.Lock()
        self._inflight = 0

    def _cache_key(self, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> str:
        parts: list[Any] = [self.model, max_tokens, messages]
        if temperature:  # temperature-0 keys keep the original format, so old caches stay valid
            parts.append(temperature)
        payload = json.dumps(parts, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def is_cached(self, messages: list[dict[str, str]], *, max_tokens: int | None = None,
                  temperature: float = 0.0) -> bool:
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        return self._cache_key(messages, max_tokens, temperature) in self.entries

    def __call__(self, messages: list[dict[str, str]], label: str = "", *,
                 max_tokens: int | None = None, temperature: float = 0.0) -> dict[str, Any]:
        """Returns {"text", "prompt_tokens", "completion_tokens", "finish_reason", "seconds", "model"}."""
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        key = self._cache_key(messages, max_tokens, temperature)
        with self._lock:
            if key in self.entries:
                return self.entries[key]
            if self.new_calls + self._inflight >= self.max_new_calls:
                raise BudgetExceeded(f"refusing more than {self.max_new_calls} new OpenAI calls this run")
            self._inflight += 1
            self._key = self._key or api_key()
        try:
            output = self._request(messages, max_tokens, temperature)
        finally:
            with self._lock:
                self._inflight -= 1
        with self._lock:
            self.new_calls += 1
            self.usage["prompt_tokens"] += output["prompt_tokens"]
            self.usage["completion_tokens"] += output["completion_tokens"]
            if key not in self.entries:
                self.entries[key] = output
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"key": key, "label": label, "output": output},
                                            ensure_ascii=False) + "\n")
            return self.entries[key]

    def _request(self, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> dict[str, Any]:
        body = json.dumps({"model": self.model, "messages": messages, "temperature": temperature,
                           "max_completion_tokens": max_tokens}).encode("utf-8")
        request = Request(URL, data=body, headers={"Content-Type": "application/json",
                                                    "Authorization": f"Bearer {self._key}"})
        started = time.perf_counter()
        for attempt in range(self.max_attempts):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                    self.rate_limits = {k.lower(): v for k, v in response.headers.items()
                                        if k.lower().startswith("x-ratelimit")}
                break
            except HTTPError as error:
                detail = error.read().decode("utf-8", "replace")
                if (error.code not in RETRYABLE_STATUS or "insufficient_quota" in detail
                        or attempt == self.max_attempts - 1):
                    raise RuntimeError(f"OpenAI HTTP {error.code}: {detail[:300]}") from error
                wait = float(error.headers.get("retry-after") or 2 ** attempt)
            except OSError:
                if attempt == self.max_attempts - 1:
                    raise
                wait = 2 ** attempt
            time.sleep(min(wait, 60) + random.random())
        usage = payload.get("usage") or {}
        choice = payload["choices"][0]
        return {"text": (choice["message"]["content"] or "").strip(),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "finish_reason": choice.get("finish_reason"),
                "seconds": time.perf_counter() - started, "model": payload.get("model", self.model)}


def cost_usd(model: str, prompt_tokens: float, completion_tokens: float) -> float | None:
    for name, (prompt_price, completion_price) in PRICES.items():
        if model.startswith(name):
            return (prompt_tokens * prompt_price + completion_tokens * completion_price) / 1e6
    return None
