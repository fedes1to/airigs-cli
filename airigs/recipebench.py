"""RecipeBenchmark v1: measure an already-running OpenAI-compatible endpoint."""

from __future__ import annotations

import http.client
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import urlparse

from .bench.common import Notes, import_optional, now_iso
from .bench.nvidia import NvmlSession
from .bench.power import summarize_power_samples
from .http import AirigsError, USER_AGENT, is_loopback_url, open_no_redirect

SCHEMA_VERSION = 1
MAX_TOKENS = 128
PROMPT_WORD = "benchmark "
CHARS_PER_TOKEN = 4
TIMEOUT_S = 300.0
#: The models probe should fail fast; a whole stream may legitimately take minutes.
MODELS_TIMEOUT_S = 30.0

#: Bounds on an untrusted endpoint's responses. A hostile or broken inference
#: server must be able to make the CLI fail, never exhaust its memory.
MAX_MODELS_BYTES = 1024 * 1024
MAX_SSE_EVENT_BYTES = 256 * 1024
MAX_STREAM_BYTES = 16 * 1024 * 1024
READ_CHUNK_BYTES = 64 * 1024
_READ_ERRORS = (TimeoutError, OSError, http.client.HTTPException)


class EndpointError(AirigsError):
    """The endpoint did not behave like an OpenAI-compatible server."""


class RecipeBenchmarkIncomplete(Exception):
    def __init__(self, missing: list[dict[str, str]]) -> None:
        self.missing = missing
        lines = ["the recipe benchmark is missing measurements the server requires:"]
        lines.extend(f"  - {item['field']}: {item['remediation']}" for item in missing)
        lines.append("Nothing was uploaded.")
        super().__init__("\n".join(lines))


def normalize_base(endpoint: str) -> str:
    endpoint = endpoint.strip().rstrip("/")
    if not endpoint:
        raise EndpointError("--endpoint requires a URL")
    parsed = urlparse(endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or not parsed.hostname:
        raise EndpointError(f"--endpoint must be an http(s) URL, got {endpoint!r}")
    if parsed.username or parsed.password:
        raise EndpointError(
            "--endpoint must not contain credentials in the URL; use --api-key or the endpoint's own auth"
        )
    path = parsed.path
    if path in ("", "/"):
        endpoint += "/v1"
    elif path.endswith("/chat/completions"):
        endpoint = endpoint[: -len("/chat/completions")]
    elif not path.endswith("/v1") and "/v1/" not in path + "/":
        endpoint += "/v1"
    return endpoint.rstrip("/")


def _assert_endpoint_allowed(url: str, api_key: str | None) -> None:
    """Reject unsafe schemes/URLs and never send a key over plaintext HTTP.

    Local inference servers are normally plain HTTP, so keyless plaintext is
    allowed (with a warning from :func:`check_endpoint_security`).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise EndpointError(f"endpoint must be an http(s) URL, got {url!r}")
    if parsed.username or parsed.password:
        raise EndpointError("refusing an endpoint URL containing userinfo credentials")
    if api_key and parsed.scheme != "https" and not is_loopback_url(url):
        raise EndpointError(
            f"refusing to send --api-key over plaintext HTTP to {parsed.hostname}; "
            "use an https endpoint or a localhost/127.0.0.1 address"
        )


def check_endpoint_security(base: str, api_key: str | None, notes: Notes) -> None:
    """Validate the endpoint and warn about unencrypted remote traffic."""
    _assert_endpoint_allowed(base, api_key)
    parsed = urlparse(base)
    if not api_key and parsed.scheme == "http" and not is_loopback_url(base):
        notes.add(
            f"endpoint {parsed.hostname or base} uses plaintext HTTP; benchmark traffic is "
            "unencrypted (no API key is sent)"
        )


def _read_some(response: Any, size: int) -> bytes:
    """Read whatever is available, not a full buffer.

    ``HTTPResponse.read(n)`` blocks until it has ``n`` bytes or EOF, so a
    slow-drip server defeats a deadline checked between reads. ``read1``
    returns after one underlying read, letting the caller enforce wall time.
    """
    reader = getattr(response, "read1", None)
    return reader(size) if reader is not None else response.read(size)


def _check_deadline(deadline: float, timeout_s: float, phase: str, url: str) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise EndpointError(
            f"{phase} for {url} did not finish within {timeout_s:g}s; aborting the benchmark"
        )


def _read_capped(
    response: Any,
    limit: int,
    what: str,
    url: str,
    *,
    deadline: float | None = None,
    timeout_s: float | None = None,
) -> bytes:
    """Read at most ``limit`` bytes, aborting once the wall-clock deadline passes.

    A slow-drip server can keep every individual socket read under its timeout
    while never finishing; byte caps bound memory, so time needs its own bound.
    """
    chunks: list[bytes] = []
    size = 0
    while True:
        if deadline is not None:
            _check_deadline(
                deadline,
                timeout_s if timeout_s is not None else 0.0,
                f"{what} read",
                url,
            )
        chunk = _read_some(response, READ_CHUNK_BYTES)
        if not chunk:
            break
        if deadline is not None:
            _check_deadline(
                deadline,
                timeout_s if timeout_s is not None else 0.0,
                f"{what} read",
                url,
            )
        size += len(chunk)
        if size > limit:
            raise EndpointError(
                f"{what} response from {url} exceeded the {limit}-byte limit; refusing to read further"
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream", "User-Agent": USER_AGENT}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _get_json(url: str, api_key: str | None, timeout: float) -> Any:
    _assert_endpoint_allowed(url, api_key)
    deadline = time.monotonic() + timeout
    _check_deadline(deadline, timeout, "models probe", url)
    request = urllib.request.Request(url, headers=_headers(api_key), method="GET")
    try:
        with open_no_redirect(request, timeout=timeout) as response:
            raw = _read_capped(
                response,
                MAX_MODELS_BYTES,
                "models JSON",
                url,
                deadline=deadline,
                timeout_s=timeout,
            )
            return json.loads(raw.decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        body = exc.read(4096).decode("utf-8", "replace") if exc.fp else ""
        raise EndpointError(f"GET {url} failed with HTTP {exc.code}: {body[:300]}") from exc
    except urllib.error.URLError as exc:
        raise EndpointError(f"cannot reach {url}: {exc.reason}") from exc
    except _READ_ERRORS as exc:
        raise EndpointError(f"failed while reading {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise EndpointError(f"GET {url} did not return JSON") from exc


def served_models(base: str, api_key: str | None = None, timeout: float = MODELS_TIMEOUT_S) -> list[str]:
    payload = _get_json(f"{base}/models", api_key, timeout)
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise EndpointError("GET /v1/models did not return a data array")
    models = [entry.get("id") for entry in data if isinstance(entry, dict) and entry.get("id")]
    return [str(model) for model in models]


def _prompt(context_tokens: int) -> str:
    words = max(1, context_tokens * CHARS_PER_TOKEN // len(PROMPT_WORD))
    return (
        "Read the following benchmark padding and answer with the single word OK.\n\n"
        + PROMPT_WORD * words
    )


def _iter_sse(
    response: Any,
    *,
    url: str = "endpoint",
    max_event_bytes: int | None = None,
    max_total_bytes: int | None = None,
    deadline: float | None = None,
    timeout_s: float | None = None,
):
    """Yield SSE data payloads with hard bounds on memory use.

    ``http.client``'s line iteration has no length limit, so this reads fixed
    chunks and fails once a line, an accumulated event, or the whole stream
    exceeds its cap. The reader never buffers more than ``max_event_bytes``.
    """
    event_limit = MAX_SSE_EVENT_BYTES if max_event_bytes is None else max_event_bytes
    total_limit = MAX_STREAM_BYTES if max_total_bytes is None else max_total_bytes
    data_lines: list[str] = []
    event_bytes = 0
    total_bytes = 0
    buffer = b""

    def feed(raw_line: bytes) -> str | None:
        nonlocal event_bytes
        if len(raw_line) > event_limit:
            raise EndpointError(
                f"SSE line from {url} exceeded {event_limit} bytes; aborting the benchmark"
            )
        line = raw_line.decode("utf-8", "replace")
        if line.endswith("\r"):
            line = line[:-1]
        if not line:
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines.clear()
                event_bytes = 0
                return payload
            return None
        if line.startswith(":"):
            return None
        if line.startswith("data:"):
            value = line[5:].lstrip()
            event_bytes += len(value.encode("utf-8"))
            if event_bytes > event_limit:
                raise EndpointError(
                    f"SSE event from {url} exceeded {event_limit} bytes; aborting the benchmark"
                )
            data_lines.append(value)
        return None

    while True:
        if deadline is not None:
            _check_deadline(
                deadline,
                timeout_s if timeout_s is not None else 0.0,
                "SSE stream",
                url,
            )
        chunk = _read_some(response, READ_CHUNK_BYTES)
        if not chunk:
            break
        if deadline is not None:
            _check_deadline(
                deadline,
                timeout_s if timeout_s is not None else 0.0,
                "SSE stream",
                url,
            )
        total_bytes += len(chunk)
        if total_bytes > total_limit:
            raise EndpointError(
                f"SSE stream from {url} exceeded {total_limit} bytes; aborting the benchmark"
            )
        buffer += chunk
        if b"\n" in buffer:
            parts = buffer.split(b"\n")
            buffer = parts.pop()
            for raw_line in parts:
                payload = feed(raw_line)
                if payload is not None:
                    yield payload
        if len(buffer) > event_limit:
            raise EndpointError(
                f"SSE line from {url} exceeded {event_limit} bytes; aborting the benchmark"
            )
    if buffer:
        payload = feed(buffer)
        if payload is not None:
            yield payload
    if data_lines:
        yield "\n".join(data_lines)


def run_request(
    base: str,
    *,
    model: str,
    prompt: str,
    api_key: str | None,
    timeout: float = TIMEOUT_S,
) -> dict[str, Any]:
    """One streaming completion. Times TTFT on a monotonic clock."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    }
    url = f"{base}/chat/completions"
    _assert_endpoint_allowed(url, api_key)
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=_headers(api_key),
        method="POST",
    )
    deadline = time.monotonic() + timeout
    _check_deadline(deadline, timeout, "chat completion", url)
    started = time.monotonic()
    first_token: float | None = None
    finished: float | None = None
    content_chunks = 0
    prompt_tokens = None
    completion_tokens = None
    served_model = None
    try:
        with open_no_redirect(request, timeout=timeout) as response:
            for data in _iter_sse(
                response, url=url, deadline=deadline, timeout_s=timeout
            ):
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(chunk, dict) and chunk.get("model"):
                    served_model = served_model or chunk["model"]
                usage = chunk.get("usage") if isinstance(chunk, dict) else None
                if isinstance(usage, dict):
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
                choices = chunk.get("choices") if isinstance(chunk, dict) else None
                if not choices:
                    continue
                for choice in choices:
                    delta = (choice or {}).get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        content_chunks += 1
                        if first_token is None:
                            first_token = time.monotonic()
            finished = time.monotonic()
    except urllib.error.HTTPError as exc:
        body_text = exc.read(4096).decode("utf-8", "replace") if exc.fp else ""
        raise EndpointError(f"chat completion for {model} failed with HTTP {exc.code}: {body_text[:300]}") from exc
    except urllib.error.URLError as exc:
        raise EndpointError(f"cannot reach {url}: {exc.reason}") from exc
    except _READ_ERRORS as exc:
        raise EndpointError(f"failed while reading {url}: {exc}") from exc

    if first_token is None:
        raise EndpointError(f"model {model} returned no streamed content (is streaming enabled?)")
    finished = finished or time.monotonic()
    if served_model and served_model != model:
        raise EndpointError(f"endpoint served model {served_model!r} but {model!r} was requested")
    return {
        "model": model,
        "ttftS": first_token - started,
        "durationS": finished - started,
        "promptTokens": prompt_tokens,
        "completionTokens": completion_tokens,
        "contentChunks": content_chunks,
        "servedModel": served_model,
    }


class _GpuSampler:
    """Samples VRAM/util/RAM and per-GPU power during the benchmark run.

    Power is aggregated with :func:`airigs.bench.power.summarize_power_samples`:
    a single driver outlier (the observed 119 W spike from a 25 W-capable card)
    must not become a published number.
    """

    def __init__(self, session: Any | None = None) -> None:
        self.session = session if session is not None else NvmlSession()
        self.count = int(getattr(self.session, "count", 0) or 0)
        self.limits: dict[int, float | None] = {
            index: self.session.power_limit_w(index) for index in range(self.count)
        }
        self.gpu_power: dict[int, list[float]] = {index: [] for index in range(self.count)}
        self.samples: list[dict[str, float | int]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.psutil = import_optional("psutil")

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            vram = 0
            util_values: list[float] = []
            any_gpu = False
            for index in range(self.count):
                used = self.session.memory_used(index)
                watts = self.session.power_w(index)
                util = self.session.utilization(index)
                if used is not None:
                    vram += used
                    any_gpu = True
                if watts is not None:
                    self.gpu_power.setdefault(index, []).append(float(watts))
                if util is not None:
                    util_values.append(util)
            ram = None
            if self.psutil is not None:
                ram = float(self.psutil.virtual_memory().used)
            self.samples.append(
                {
                    "vram": vram if any_gpu else None,
                    "util": round(statistics.mean(util_values), 1) if util_values else None,
                    "ram": ram,
                }
            )
            self._stop.wait(0.25)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self.session.close()
        vram = [s["vram"] for s in self.samples if s["vram"] is not None]
        util = [s["util"] for s in self.samples if s["util"] is not None]
        ram = [s["ram"] for s in self.samples if s["ram"] is not None]

        summaries = {
            index: summarize_power_samples(
                values, limit_w=self.limits.get(index), label=f"GPU {index}"
            )
            for index, values in self.gpu_power.items()
            if values
        }
        missing = [index for index in range(self.count) if index not in summaries]
        power_avg = None
        power_p95 = None
        power_note = None
        if summaries and not missing and all(s["medianW"] is not None for s in summaries.values()):
            power_avg = round(sum(s["medianW"] for s in summaries.values()), 2)
            power_p95 = round(sum(s["p95W"] for s in summaries.values()), 2)
            power_note = "GPU power: " + " ".join(s["note"] for s in summaries.values())
        elif summaries or missing:
            parts = [s["note"] for s in summaries.values()]
            parts.extend(f"GPU {index}: no power samples." for index in missing)
            power_note = "GPU power not reported: " + " ".join(parts)
        return {
            "vramPeakBytes": int(max(vram)) if vram else None,
            "powerAvgW": power_avg,
            "powerP95W": power_p95,
            "powerSamples": sum(s["samples"] for s in summaries.values()),
            "powerDiscarded": sum(s["discarded"] for s in summaries.values()),
            "powerLimitW": sum(
                limit for limit in self.limits.values() if limit is not None
            )
            or None,
            "powerNote": power_note,
            "gpuUtilPct": round(statistics.mean(util), 1) if util else None,
            "ramPeakBytes": int(max(ram)) if ram else None,
            "nvml": bool(vram or summaries),
        }

def _is_local(endpoint: str) -> bool:
    host = (urlparse(endpoint).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal")


def run(
    endpoint: str,
    *,
    engine: str,
    context_tokens: int = 2048,
    concurrency: int = 4,
    requests: int = 8,
    model_ids: list[str] | None = None,
    rig_id: str | None = None,
    api_key: str | None = None,
    quick: bool = False,
    notes: Notes | None = None,
) -> dict[str, Any]:
    """Benchmark the endpoint and return a RecipeBenchmark v1 payload."""
    notes = notes if notes is not None else Notes()
    if concurrency < 1:
        raise EndpointError("--concurrency must be at least 1")
    if requests < 1:
        raise EndpointError("--requests must be at least 1")
    if context_tokens < 1:
        raise EndpointError("--context must be at least 1 token")
    base = normalize_base(endpoint)
    check_endpoint_security(base, api_key, notes)
    if not _is_local(endpoint):
        notes.add(
            "endpoint is remote: VRAM/power/RAM are sampled on the machine running the CLI, "
            "so run `airigs recipe bench` on the inference host for meaningful numbers."
        )

    served = served_models(base, api_key)
    requested = list(model_ids or [])
    if not requested:
        if not served:
            raise EndpointError("the endpoint reports no models at /v1/models")
        requested = [served[0]]
        notes.add(f"no --model given; benchmarking the first served model ({served[0]}).")
    missing = [model for model in requested if model not in served]
    if missing:
        raise EndpointError(
            "the endpoint does not serve the requested model(s): "
            + ", ".join(missing)
            + f" (served: {', '.join(served) or 'none'})"
        )

    prompt = _prompt(context_tokens)
    if quick:
        requests = min(requests, max(1, len(requested)))
        concurrency = min(concurrency, requests)

    sampler = _GpuSampler()
    sampler.start()
    started_at = now_iso()
    wall_start = time.monotonic()
    outcomes: list[dict[str, Any]] = []
    errors: list[str] = []

    def job(index: int) -> dict[str, Any]:
        model = requested[index % len(requested)]
        return run_request(base, model=model, prompt=prompt, api_key=api_key)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(job, index) for index in range(requests)]
        for future in futures:
            try:
                outcomes.append(future.result())
            except EndpointError as exc:
                errors.append(str(exc))
            except Exception as exc:  # network/thread failure
                errors.append(f"{exc.__class__.__name__}: {exc}")
    wall = time.monotonic() - wall_start
    samples = sampler.stop()
    finished_at = now_iso()

    if errors:
        raise EndpointError(
            f"{len(errors)} of {requests} requests failed; first error: {errors[0]}"
        )

    ttft_values = [outcome["ttftS"] for outcome in outcomes]
    output_tokens = 0
    prompt_tokens = 0
    used_usage = True
    for outcome in outcomes:
        if outcome["completionTokens"] is not None and outcome["promptTokens"] is not None:
            output_tokens += int(outcome["completionTokens"])
            prompt_tokens += int(outcome["promptTokens"])
        else:
            used_usage = False
            output_tokens += int(outcome["contentChunks"])
            prompt_tokens += max(1, context_tokens)
    if not used_usage:
        notes.add(
            "the endpoint did not return token usage; token counts fall back to streamed chunk "
            "counts, so tokens/s is approximate."
        )
    if samples.get("powerNote"):
        notes.add(samples["powerNote"])
    if not samples["nvml"]:
        notes.add(
            "VRAM peak and GPU power were not measured: NVML is unavailable on this host "
            "(install the NVIDIA driver + `nvidia-ml-py`, or run on the inference host)."
        )
    if samples["ramPeakBytes"] is None:
        notes.add("system RAM peak not measured: psutil is unavailable.")
    notes.add(
        "outputTokensPerSec and totalTokensPerSec are aggregate: total tokens divided by the "
        f"wall-clock run ({wall:.2f}s) at concurrency {concurrency}."
    )

    prompt_seconds = sum(ttft_values)
    benchmark = {
        "schemaVersion": SCHEMA_VERSION,
        "rigId": rig_id,
        "engine": engine,
        "startedAt": started_at,
        "finishedAt": finished_at,
        "modelIds": requested,
        "contextTokens": int(context_tokens),
        "concurrency": int(concurrency),
        "requests": len(outcomes),
        "ttftMs": round(statistics.median(ttft_values) * 1000.0, 1),
        "promptTokensPerSec": round(prompt_tokens / prompt_seconds, 2) if prompt_seconds else None,
        "outputTokensPerSec": round(output_tokens / wall, 2) if wall else None,
        "totalTokensPerSec": round((output_tokens + prompt_tokens) / wall, 2) if wall else None,
        "vramPeakBytes": samples["vramPeakBytes"],
        "ramPeakBytes": samples["ramPeakBytes"],
        "powerAvgW": samples["powerAvgW"],
        "gpuUtilPct": samples["gpuUtilPct"],
        "notes": list(notes),
    }
    return benchmark


RECIPE_REQUIRED = (
    ("schemaVersion", "internal error: wrong schema version"),
    ("rigId", "pass --rig-id r_... (opaque string), or register a rig with `airigs rig submit` so the CLI can pick it"),
    ("engine", "pass --engine vllm (or another engine name)"),
    ("startedAt", "internal error: start time missing"),
    ("finishedAt", "internal error: finish time missing"),
    ("modelIds", "pass --model ID for every model under test"),
    ("concurrency", "internal error: concurrency missing"),
    ("ttftMs", "no request produced a first token; check the endpoint"),
    ("outputTokensPerSec", "no output tokens were produced"),
    ("vramPeakBytes", "run on the inference host with NVML available (`pip install nvidia-ml-py`, NVIDIA driver)"),
    ("powerAvgW", "run on the inference host with a GPU power sensor (NVML / rocm-smi)"),
)


def missing_required(benchmark: dict[str, Any]) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    for field, remediation in RECIPE_REQUIRED:
        value = benchmark.get(field)
        if value is None or value == "" or (isinstance(value, list) and not value):
            missing.append({"field": field, "remediation": remediation})
    if benchmark.get("schemaVersion") != SCHEMA_VERSION:
        missing.append({"field": "schemaVersion", "remediation": RECIPE_REQUIRED[0][1]})
    return missing


def require_complete(benchmark: dict[str, Any]) -> None:
    missing = missing_required(benchmark)
    if missing:
        raise RecipeBenchmarkIncomplete(missing)


def check_shape(benchmark: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if not isinstance(benchmark, dict):
        return ["benchmark is not a JSON object"]
    numeric = (
        "schemaVersion",
        "contextTokens",
        "concurrency",
        "requests",
        "ttftMs",
        "promptTokensPerSec",
        "outputTokensPerSec",
        "totalTokensPerSec",
        "vramPeakBytes",
        "ramPeakBytes",
        "powerAvgW",
        "gpuUtilPct",
    )
    for field in numeric:
        if field in benchmark and benchmark[field] is not None and not isinstance(benchmark[field], (int, float)):
            problems.append(f"{field}: expected number|null")
    if not isinstance(benchmark.get("modelIds"), list):
        problems.append("modelIds: expected array")
    return problems
