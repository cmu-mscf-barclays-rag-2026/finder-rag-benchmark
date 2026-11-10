"""Small, cached, budget-capped OpenAI chat client for the evaluations.

Every response is cached by model, parameters, and exact messages, so reruns
cost nothing. Each run refuses to make more than ``max_new_calls`` uncached
calls, and token usage is taken from the API's own ``usage`` field.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from dotenv import load_dotenv


URL = "https://api.openai.com/v1/chat/completions"
# USD per million tokens (gpt-4o-mini list price); used only to report cost.
PRICES = {"gpt-4o-mini": (0.15, 0.60)}


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
                 max_tokens: int = 96, timeout: float = 60.0) -> None:
        self.model, self.cache_path, self.max_tokens, self.timeout = model, cache_path, max_tokens, timeout
        self.max_new_calls = max_new_calls
        self.entries: dict[str, dict[str, Any]] = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.entries[row["key"]] = row["output"]
        self.new_calls = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self._key = None

    def _cache_key(self, messages: list[dict[str, str]]) -> str:
        payload = json.dumps([self.model, self.max_tokens, messages], ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def __call__(self, messages: list[dict[str, str]], label: str = "") -> dict[str, Any]:
        """Returns {"text", "prompt_tokens", "completion_tokens", "seconds", "model"}."""
        key = self._cache_key(messages)
        if key not in self.entries:
            if self.new_calls >= self.max_new_calls:
                raise BudgetExceeded(f"refusing more than {self.max_new_calls} new OpenAI calls this run")
            self._key = self._key or api_key()
            body = json.dumps({"model": self.model, "messages": messages, "temperature": 0,
                               "max_completion_tokens": self.max_tokens}).encode("utf-8")
            request = Request(URL, data=body, headers={"Content-Type": "application/json",
                                                        "Authorization": f"Bearer {self._key}"})
            started = time.perf_counter()
            for attempt in range(4):
                try:
                    with urlopen(request, timeout=self.timeout) as response:
                        payload = json.load(response)
                    break
                except OSError:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
            usage = payload.get("usage") or {}
            output = {"text": payload["choices"][0]["message"]["content"].strip(),
                      "prompt_tokens": usage.get("prompt_tokens", 0),
                      "completion_tokens": usage.get("completion_tokens", 0),
                      "seconds": time.perf_counter() - started, "model": payload.get("model", self.model)}
            self.entries[key] = output
            self.new_calls += 1
            self.usage["prompt_tokens"] += output["prompt_tokens"]
            self.usage["completion_tokens"] += output["completion_tokens"]
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "label": label, "output": output},
                                        ensure_ascii=False) + "\n")
        return self.entries[key]


def cost_usd(model: str, prompt_tokens: float, completion_tokens: float) -> float | None:
    for name, (prompt_price, completion_price) in PRICES.items():
        if model.startswith(name):
            return (prompt_tokens * prompt_price + completion_tokens * completion_price) / 1e6
    return None
