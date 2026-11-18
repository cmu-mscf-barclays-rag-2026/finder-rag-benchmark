"""Credential loading and checkpointed API requests without secret-bearing logs."""

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import threading
from time import perf_counter, sleep

import requests


def load_key():
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    path = Path.home() / "AppData/Local/BarclaysRAG/openai-key.xml"
    if not path.is_file():
        raise RuntimeError("Configure OPENAI_API_KEY or the local Windows-encrypted credential.")
    escaped = str(path).replace("'", "''")
    command = (
        "$s = Import-Clixml -LiteralPath '" + escaped + "'; "
        "[Console]::Write(([pscredential]::new('api', $s)).GetNetworkCredential().Password)"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True, text=True,
    )
    key = result.stdout.strip()
    if result.returncode or not key.startswith("sk-"):
        raise RuntimeError("Unable to decrypt the local credential under this Windows account.")
    return key


class API:
    """Persist complete responses once and expose sanitized usage records."""

    def __init__(self, cache, model):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.key = load_key()
        self.lock = threading.Lock()
        self.request_locks = {}

    def complete(self, messages, temperature=0, max_tokens=1024, **kwargs):
        payload = {
            "model": self.model, "messages": messages, "temperature": temperature,
            "max_completion_tokens": max_tokens, "store": False,
        }
        for name in ("response_format", "seed", "n"):
            if name in kwargs:
                payload[name] = kwargs[name]
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        digest = sha256(encoded).hexdigest()
        path = self.cache / f"{digest}.json"
        with self.lock:
            request_lock = self.request_locks.setdefault(digest, threading.Lock())
        with request_lock:
            if path.exists():
                saved = json.loads(path.read_text(encoding="utf-8"))
                if any(choice.get("finish_reason") != "stop" for choice in saved["response"]["choices"]):
                    raise RuntimeError("A cached generation is incomplete; inspect the saved request.")
                return saved["response"], saved
            for attempt in range(3):
                start = perf_counter()
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": "Bearer " + self.key},
                    json=payload, timeout=90,
                )
                seconds = perf_counter() - start
                if response.status_code in (429, 500, 502, 503) and attempt < 2:
                    sleep(2 ** (attempt + 1))
                    continue
                if response.status_code != 200:
                    error = response.json().get("error", {})
                    raise RuntimeError(
                        f"API HTTP {response.status_code}; code={error.get('code')}; type={error.get('type')}"
                    )
                result = response.json()
                usage = result.get("usage", {})
                saved = {
                    "request_hash": digest, "request": payload, "seconds": seconds,
                    "attempts": attempt + 1, "response": result,
                    "estimated_usd": (
                        usage.get("prompt_tokens", 0) * 0.15
                        + usage.get("completion_tokens", 0) * 0.60
                    ) / 1_000_000,
                }
                partial = path.with_suffix(".tmp")
                partial.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
                partial.replace(path)
                if any(choice.get("finish_reason") != "stop" for choice in result["choices"]):
                    raise RuntimeError("Generation hit a limit or returned an incomplete response; inspect its saved record.")
                return result, saved

    def usage(self):
        rows = []
        for path in sorted(self.cache.glob("*.json")):
            item = json.loads(path.read_text(encoding="utf-8"))
            response = item["response"]
            usage = response.get("usage", {})
            rows.append({
                "request_hash": item["request_hash"], "model": response["model"],
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "seconds": item["seconds"], "attempts": item["attempts"],
                "estimated_usd": item["estimated_usd"],
            })
        return rows
