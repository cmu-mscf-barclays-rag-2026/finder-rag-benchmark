"""Loopback Chat Completions proxy with a durable, fail-closed cost ledger.

LangChain and DSPy use a dummy API key and the URL returned by ``start()``.
The real credential is read only by this proxy. Successful exact requests are
cached; a pending/unknown request is never sent upstream a second time.
"""
from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import sqlite3
import stat
import threading
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL = "gpt-4o-mini"
PRIOR_USD = 4.570655
CUMULATIVE_GUARD_USD = 7.0
NEW_GUARD_USD = 1.0
MAX_UNIQUE_ATTEMPTS = 600
INPUT_USD_PER_MILLION = 0.15
OUTPUT_USD_PER_MILLION = 0.60

# These framework/cache annotations do not affect model generation and are not
# sent to OpenAI. Identity is computed AFTER their removal.
IGNORED_FIELDS = frozenset({
    "metadata", "tags", "cache", "cache_key", "cache_control", "rollout_id",
    "_cache_key", "_cache_metadata", "_framework_metadata",
})
ALLOWED_FIELDS = frozenset({
    "model", "messages", "max_tokens", "max_completion_tokens", "temperature",
    "top_p", "n", "stream", "stop", "frequency_penalty", "presence_penalty",
    "logit_bias", "seed", "response_format", "tools", "tool_choice",
    "parallel_tool_calls", "logprobs", "top_logprobs", "service_tier", "store",
    "prediction", "modalities", "audio", "reasoning_effort", "verbosity",
    "safety_identifier", "prompt_cache_key", "prompt_cache_retention",
    "web_search_options", "function_call", "functions", "user",
})


class ProxyError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


class BudgetProxy:
    """Task-local, thread-safe ledger and loopback-only model gateway.

    ``client`` may be an injected SDK-shaped fake for tests. No upstream client
    or credential is loaded until ``access_check`` or a new paid request.
    Guards count retained reservations for unknown attempts as spent. The byte
    allowance is deliberately conservative and ignores cached-input discounts.
    """

    def __init__(self, out: Path, *, client: Any = None,
                 prior_usd: float = PRIOR_USD,
                 cumulative_guard_usd: float = CUMULATIVE_GUARD_USD,
                 new_guard_usd: float = NEW_GUARD_USD,
                 max_unique_attempts: int = MAX_UNIQUE_ATTEMPTS,
                 input_price: float = INPUT_USD_PER_MILLION,
                 output_price: float = OUTPUT_USD_PER_MILLION):
        self.out = Path(out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        for value in [prior_usd, cumulative_guard_usd, new_guard_usd,
                      input_price, output_price]:
            if not math.isfinite(value) or value < 0:
                raise ValueError("Prices and budget amounts must be finite and nonnegative")
        if isinstance(max_unique_attempts, bool) or max_unique_attempts < 1:
            raise ValueError("max_unique_attempts must be positive")
        self.prior_usd = float(prior_usd)
        self.cumulative_guard_usd = float(cumulative_guard_usd)
        self.new_guard_usd = float(new_guard_usd)
        self.max_unique_attempts = int(max_unique_attempts)
        self.input_price, self.output_price = float(input_price), float(output_price)
        self._client = client
        self._client_lock = threading.Lock()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._active = 0
        self._http_active = 0
        self._stage = "unspecified"
        self._closing = False
        self._closed = False
        self._server = None
        self._thread = None
        self.base_url = None
        self.path = self.out / "api_checkpoint.sqlite"
        if self.path.is_symlink():
            raise ValueError("Refusing a symlink ledger")
        if not self.path.exists():
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
        mode = self.path.stat()
        if not stat.S_ISREG(mode.st_mode) or stat.S_IMODE(mode.st_mode) & 0o077:
            raise ValueError("Ledger must be a regular local file with permissions600")
        self._db = sqlite3.connect(self.path, check_same_thread=False,
                                   isolation_level=None, timeout=30)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("""CREATE TABLE IF NOT EXISTS calls(
            key TEXT PRIMARY KEY, state TEXT NOT NULL, reserve REAL NOT NULL,
            cost REAL, request_json TEXT NOT NULL, response_json TEXT,
            usage_json TEXT, latency_seconds REAL, created_at REAL NOT NULL,
            completed_at REAL, error_type TEXT, created_stage TEXT NOT NULL DEFAULT 'unspecified')""")
        if "created_stage" not in {r[1] for r in self._db.execute("PRAGMA table_info(calls)")}:
            self._db.execute("ALTER TABLE calls ADD COLUMN created_stage TEXT NOT NULL DEFAULT 'unspecified'")
        self._db.execute("""CREATE TABLE IF NOT EXISTS events(
            event_id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL NOT NULL,
            stage TEXT NOT NULL, identity_key TEXT, cache_hit INTEGER NOT NULL,
            http_status INTEGER NOT NULL, retained_upstream_cost_usd REAL,
            retained_upstream_reserve_usd REAL, retained_upstream_latency_seconds REAL,
            request_elapsed_seconds REAL NOT NULL)""")
        self._db.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        settings = _canonical(dict(model=MODEL, prior_usd=self.prior_usd,
             cumulative_guard_usd=self.cumulative_guard_usd,
             new_guard_usd=self.new_guard_usd,
             max_unique_attempts=self.max_unique_attempts,
             input_price=self.input_price, output_price=self.output_price))
        row = self._db.execute("SELECT value FROM meta WHERE key='settings'").fetchone()
        if row is not None and row[0] != settings:
            self._db.close()
            raise ValueError("Existing ledger settings differ; use its frozen settings or a new output directory")
        self._db.execute("INSERT OR IGNORE INTO meta VALUES('settings',?)", (settings,))
        self._db.execute("INSERT OR IGNORE INTO meta VALUES('cache_hits','0')")
        self.summary()

    def _upstream(self):
        with self._client_lock:
            if self._client is None:
                path = ROOT / ".env.local"
                if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
                    raise RuntimeError("Local credential file is missing or has unsafe permissions")
                values = {}
                for line in path.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        name, value = line.split("=", 1)
                        values[name.strip()] = value.strip().strip("\"'")
                key = values.get("OPENAI_API_KEY", "")
                if not key.startswith("sk-") or any(c.isspace() for c in key):
                    raise RuntimeError("Local credential is not configured")
                from openai import OpenAI
                organization = values.get("OPENAI_ORG_ID") or os.environ.get(
                    "OPENAI_ORG_ID", "org-ya6BfJ99A55qzohO1pvOUaXO")
                self._client = OpenAI(api_key=key, organization=organization,
                                      max_retries=0, timeout=120)
            return self._client

    def access_check(self) -> dict:
        """Read-only model lookup; never starts a Chat Completions request."""
        try:
            result = self._upstream().models.retrieve(MODEL)
            identifier = getattr(result, "id", None)
            if identifier is None and isinstance(result, dict):
                identifier = result.get("id")
            if identifier != MODEL:
                raise ValueError("Unexpected model identity")
        except Exception:
            raise RuntimeError("Read-only access check failed; no paid request was sent") from None
        receipt = dict(model=MODEL, read_only_access="passed")
        with self._lock:
            self._write_json(self.out / "model_access.json", receipt)
        return receipt

    def set_stage(self, name: str):
        """Set the phase label before issuing a sequential batch of requests."""
        if not isinstance(name, str) or not name or len(name) > 128 or any(c in name for c in "\r\n"):
            raise ValueError("Stage must be a short, nonempty single-line name")
        with self._lock:
            if self._closed or self._closing:
                raise RuntimeError("Proxy is closed")
            if self._active:
                raise RuntimeError("Wait for active upstream requests before changing stage")
            self._stage = name

    def events(self) -> list[dict]:
        """Return safe per-HTTP-request events; request/response bodies excluded."""
        with self._lock:
            if self._closed:
                return [json.loads(s) for s in (self.out / "request_events.jsonl").read_text().splitlines() if s.strip()] if (self.out / "request_events.jsonl").exists() else []
            cursor = self._db.execute("SELECT * FROM events ORDER BY event_id")
            names = [item[0] for item in cursor.description]
            return [dict(zip(names, row)) for row in cursor]

    def _event(self, *, stage, body, cached, status, timestamp, elapsed):
        try:
            actual = self._validate(body)
            key = hashlib.sha256(_canonical(actual).encode("utf-8")).hexdigest()
        except Exception:
            key = None
        with self._lock:
            retained = self._db.execute("SELECT cost,reserve,latency_seconds FROM calls WHERE key=?", (key,)).fetchone() if key else None
            cost, reserve, latency = retained if retained else (None, None, None)
            cursor = self._db.execute("""INSERT INTO events(timestamp,stage,identity_key,cache_hit,
                http_status,retained_upstream_cost_usd,retained_upstream_reserve_usd,
                retained_upstream_latency_seconds,request_elapsed_seconds) VALUES(?,?,?,?,?,?,?,?,?)""",
                (timestamp, stage, key, int(cached), status, cost, reserve, latency, elapsed))
            event = dict(event_id=cursor.lastrowid, timestamp=timestamp, stage=stage,
                identity_key=key, cache_hit=bool(cached), http_status=status,
                retained_upstream_cost_usd=cost, retained_upstream_reserve_usd=reserve,
                retained_upstream_latency_seconds=latency, request_elapsed_seconds=elapsed)
            fd = os.open(self.out / "request_events.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "w") as handle:
                handle.write(_canonical(event) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.summary()

    def _write_json(self, path: Path, value: dict):
        # Callers hold the ledger lock when this can be invoked concurrently.
        tmp = path.with_suffix(path.suffix + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _summary_locked(self) -> dict:
        amount, cost, reserve, attempts = self._db.execute("""SELECT
            coalesce(sum(coalesce(cost,reserve)),0), coalesce(sum(cost),0),
            coalesce(sum(CASE WHEN cost IS NULL THEN reserve ELSE 0 END),0),
            count(*) FROM calls""").fetchone()
        by_stage = {}
        for row in self._db.execute("""SELECT created_stage,count(*),coalesce(sum(cost),0),
                coalesce(sum(coalesce(cost,reserve)),0) FROM calls GROUP BY created_stage"""):
            name, count, actual, retained = row
            by_stage[name] = dict(actual_unique_attempts=count, actual_new_spend_usd=actual,
                new_spend_plus_reserves_usd=retained)
        for row in self._db.execute("""SELECT stage,count(*),sum(cache_hit),
                coalesce(sum(CASE WHEN http_status=200 THEN retained_upstream_cost_usd ELSE 0 END),0),
                coalesce(sum(CASE WHEN http_status=200 THEN retained_upstream_latency_seconds ELSE 0 END),0),
                sum(request_elapsed_seconds),
                coalesce(sum(CASE WHEN cache_hit=1 THEN request_elapsed_seconds ELSE 0 END),0)
                FROM events GROUP BY stage"""):
            name, requests, hits, logical_cost, fresh_latency, wall, cache_wall = row
            entry = by_stage.setdefault(name, dict(actual_unique_attempts=0,
                actual_new_spend_usd=0., new_spend_plus_reserves_usd=0.))
            entry.update(http_requests=requests, cache_hits=hits,
                logical_retained_cost_usd=logical_cost,
                modeled_fresh_upstream_latency_seconds=fresh_latency,
                current_request_wall_seconds=wall, current_cache_hit_wall_seconds=cache_wall)
        return dict(model=MODEL, prior_cumulative_estimate_usd=self.prior_usd,
            new_cost_and_reserves_usd=amount, known_actual_cost_usd=cost,
            unresolved_reserved_usd=reserve,
            cumulative_estimate_including_reserves_usd=self.prior_usd + amount,
            new_guard_usd=self.new_guard_usd,
            cumulative_guard_usd=self.cumulative_guard_usd,
            unique_attempts=attempts, max_unique_attempts=self.max_unique_attempts,
            cache_hits=int(self._db.execute("SELECT value FROM meta WHERE key='cache_hits'").fetchone()[0]),
            states=dict(self._db.execute("SELECT state,count(*) FROM calls GROUP BY state")),
            prices_per_million=dict(input=self.input_price, output=self.output_price),
            cached_discount_ignored=True, billing_statement=False,
            automatic_upstream_retries=0, by_stage=by_stage,
            latency_note="Logical fresh latency reuses saved upstream durations per successful HTTP request; current wall includes local cache/proxy time. Not a measured end-to-end RAG benchmark.")

    def summary(self) -> dict:
        with self._lock:
            if self._closed:
                return json.loads((self.out / "cost_summary.json").read_text())
            value = self._summary_locked()
            self._write_json(self.out / "cost_summary.json", value)
            return value

    @staticmethod
    def _validate(body: Any) -> dict:
        if not isinstance(body, dict):
            raise ProxyError(400, "invalid_request", "Request body must be a JSON object")
        actual = {k: v for k, v in body.items() if k not in IGNORED_FIELDS}
        if set(actual) - ALLOWED_FIELDS:
            raise ProxyError(400, "unsupported_fields", "Request contains unsupported fields")
        if actual.get("model") != MODEL:
            raise ProxyError(400, "model_not_allowed", "Only gpt-4o-mini is allowed")
        if actual.get("n", 1) != 1 or isinstance(actual.get("n", 1), bool):
            raise ProxyError(400, "n_not_allowed", "Exactly one completion is required")
        if actual.get("stream", False) is not False:
            raise ProxyError(400, "stream_not_allowed", "Streaming is disabled")
        if not isinstance(actual.get("messages"), list) or not actual["messages"]:
            raise ProxyError(400, "invalid_messages", "A nonempty messages list is required")
        caps = [key for key in ("max_tokens", "max_completion_tokens") if key in actual]
        if len(caps) != 1:
            raise ProxyError(400, "missing_output_cap", "Specify exactly one explicit output-token cap")
        cap = actual[caps[0]]
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            raise ProxyError(400, "invalid_output_cap", "Output-token cap must be a positive integer")
        try:
            _canonical(actual)
        except (ValueError, TypeError):
            raise ProxyError(400, "invalid_json", "Request must contain finite JSON values") from None
        return actual

    def _complete(self, body: Any, stage: str | None = None) -> tuple[dict, bool]:
        actual = self._validate(body)
        request_json = _canonical(actual)
        key = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
        cap = actual.get("max_tokens", actual.get("max_completion_tokens"))
        reservation = ((len(request_json.encode("utf-8")) + 256) * self.input_price
                       + cap * self.output_price) / 1e6
        # Load credential / construct SDK before creating a paid-attempt record.
        client = None
        with self._condition:
            if self._closing:
                raise ProxyError(503, "proxy_closing", "Proxy is closing")
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute("SELECT state,response_json FROM calls WHERE key=?", (key,)).fetchone()
                if row is not None:
                    if row[0] == "saved":
                        self._db.execute("UPDATE meta SET value=CAST(value AS INTEGER)+1 WHERE key='cache_hits'")
                        self._db.execute("COMMIT")
                        payload = json.loads(row[1])
                        self.summary()
                        return payload, True
                    raise ProxyError(409, "attempt_not_repeatable", "A pending or unresolved attempt already exists; no upstream retry is permitted")
                state = self._summary_locked()
                if state["unique_attempts"] >= self.max_unique_attempts:
                    raise ProxyError(429, "attempt_guard", "Unique-attempt limit reached")
                if (state["new_cost_and_reserves_usd"] + reservation > self.new_guard_usd + 1e-12
                    or state["cumulative_estimate_including_reserves_usd"] + reservation > self.cumulative_guard_usd + 1e-12):
                    raise ProxyError(429, "budget_guard", "Budget reservation would exceed a fixed guard")
                client = self._upstream()
                self._db.execute("INSERT INTO calls(key,state,reserve,request_json,created_at,created_stage) VALUES(?,?,?,?,?,?)",
                                 (key, "pending", reservation, request_json, time.time(), stage or self._stage))
                self._db.execute("COMMIT")
                self.summary()
                self._active += 1
            except Exception:
                if self._db.in_transaction:
                    self._db.execute("ROLLBACK")
                raise
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(**actual)
            if isinstance(response, dict):
                payload = response
            else:
                payload = response.model_dump(mode="json")
            usage = payload.get("usage") or {}
            prompt_tokens, completion_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
            if (not isinstance(payload.get("choices"), list) or not payload["choices"]
                or isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int) or prompt_tokens < 0
                or isinstance(completion_tokens, bool) or not isinstance(completion_tokens, int) or completion_tokens < 0):
                raise ValueError("Invalid upstream response or missing usage")
            response_json = _canonical(payload)
            cost = (prompt_tokens * self.input_price + completion_tokens * self.output_price) / 1e6
            with self._condition:
                self._db.execute("BEGIN IMMEDIATE")
                self._db.execute("""UPDATE calls SET state='saved',cost=?,response_json=?,usage_json=?,
                    latency_seconds=?,completed_at=? WHERE key=?""",
                    (cost, response_json, _canonical(usage), time.perf_counter()-started, time.time(), key))
                self._db.execute("COMMIT")
                self.summary()
            return payload, False
        except Exception as exc:
            with self._condition:
                if self._db.in_transaction:
                    self._db.execute("ROLLBACK")
                # A response committed successfully stays saved even if a later
                # summary-file write failed. It must never be sent again.
                self._db.execute("""UPDATE calls SET state='unknown_reserved',error_type=?,
                    latency_seconds=?,completed_at=? WHERE key=? AND state='pending'""",
                    (type(exc).__name__, time.perf_counter()-started, time.time(), key))
                self.summary()
            raise ProxyError(502, "upstream_failed_reserved", "Upstream request failed or could not be validated; reservation retained and no retry permitted") from None
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def start(self) -> str:
        """Start a random-port IPv4 loopback server and return its /v1 URL."""
        with self._lock:
            if self._closed or self._closing:
                raise RuntimeError("Closed proxy cannot be restarted")
            if self._server is not None:
                return self.base_url
            owner = self

            class Handler(BaseHTTPRequestHandler):
                protocol_version = "HTTP/1.1"

                def log_message(self, *_args):
                    pass

                def setup(self):
                    super().setup()
                    self.connection.settimeout(150)

                def send_json(self, status, value, cached=None):
                    encoded = _canonical(value).encode("utf-8")
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.send_header("Connection", "close")
                    if cached is not None:
                        self.send_header("X-Local-Cache", "hit" if cached else "miss")
                    self.end_headers()
                    self.wfile.write(encoded)
                    self.close_connection = True

                def do_GET(self):
                    started, timestamp = time.perf_counter(), time.time()
                    with owner._lock:
                        stage = owner._stage
                        owner._http_active += 1
                    status = 500
                    try:
                        if self.path.rstrip("/") == "/v1/models":
                            status = 200
                            self.send_json(200, dict(object="list", data=[dict(id=MODEL, object="model", created=0, owned_by="local-proxy")]))
                        else:
                            status = 404
                            self.send_json(404, dict(error=dict(message="Unknown endpoint", type="invalid_request_error")))
                    finally:
                        try:
                            owner._event(stage=stage, body=None, cached=False, status=status,
                                         timestamp=timestamp, elapsed=time.perf_counter()-started)
                        finally:
                            with owner._condition:
                                owner._http_active -= 1
                                owner._condition.notify_all()

                def do_POST(self):
                    started, timestamp = time.perf_counter(), time.time()
                    body, cached, status = None, False, 500
                    with owner._lock:
                        stage = owner._stage
                        owner._http_active += 1
                    try:
                        if self.path.rstrip("/") != "/v1/chat/completions":
                            raise ProxyError(404, "unknown_endpoint", "Unknown endpoint")
                        if self.headers.get_content_type() != "application/json":
                            raise ProxyError(415, "invalid_content_type", "application/json is required")
                        size = int(self.headers.get("Content-Length", "0"))
                        if size < 1 or size > 4 * 1024 * 1024:
                            raise ProxyError(413, "invalid_body_size", "Request body size is outside allowed limits")
                        body = json.loads(self.rfile.read(size))
                        value, cached = owner._complete(body, stage)
                        status = 200
                        self.send_json(200, value, cached)
                    except ProxyError as exc:
                        status = exc.status
                        self.send_json(exc.status, dict(error=dict(message=exc.message, type="local_proxy_error", code=exc.code)))
                    except Exception:
                        self.send_json(500, dict(error=dict(message="Local proxy failed; inspect the local ledger summary", type="local_proxy_error", code="local_failure")))
                    finally:
                        try:
                            owner._event(stage=stage, body=body, cached=cached, status=status,
                                         timestamp=timestamp, elapsed=time.perf_counter()-started)
                        finally:
                            with owner._condition:
                                owner._http_active -= 1
                                owner._condition.notify_all()

            self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            self._server.daemon_threads = True
            self.base_url = f"http://127.0.0.1:{self._server.server_port}/v1"
            self._thread = threading.Thread(target=self._server.serve_forever,
                                            name="meeting10-budget-proxy", daemon=True)
            self._thread.start()
            return self.base_url

    def close(self):
        with self._condition:
            if self._closed:
                return
            self._closing = True
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join(timeout=5)
        with self._condition:
            while self._active or self._http_active:
                self._condition.wait(timeout=1)
            self.summary()
            self._db.close()
            self._closed = True

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_args):
        self.close()
