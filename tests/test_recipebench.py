from __future__ import annotations

import pytest

from airigs import recipebench
from airigs.http import AirigsError


class FakeSampler:
    def __init__(self) -> None:
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> dict:
        return {
            "vramPeakBytes": 123456789,
            "powerAvgW": 45.5,
            "gpuUtilPct": 90.0,
            "ramPeakBytes": 987654321,
            "nvml": True,
        }


class NoGpuSampler:
    """A host without NVML: no VRAM, no power, RAM still sampled."""

    def start(self) -> None:
        return None

    def stop(self) -> dict:
        return {
            "vramPeakBytes": None,
            "powerAvgW": None,
            "gpuUtilPct": None,
            "ramPeakBytes": None,
            "nvml": False,
        }


@pytest.fixture
def fake_gpu(monkeypatch):
    monkeypatch.setattr(recipebench, "_GpuSampler", FakeSampler)


def test_normalize_base():
    assert recipebench.normalize_base("http://host:8000") == "http://host:8000/v1"
    assert recipebench.normalize_base("http://host:8000/") == "http://host:8000/v1"
    assert recipebench.normalize_base("http://host:8000/v1/") == "http://host:8000/v1"
    assert recipebench.normalize_base("http://host:8000/v1/chat/completions") == "http://host:8000/v1"
    with pytest.raises(AirigsError):
        recipebench.normalize_base("not-a-url")


def test_benchmark_from_synthetic_sse(fake_server, fake_gpu):
    benchmark = recipebench.run(
        fake_server.base_url,
        engine="vllm",
        context_tokens=8,
        concurrency=1,
        requests=2,
        model_ids=["test-model"],
        rig_id="r_3",
    )
    assert benchmark["schemaVersion"] == 1
    assert benchmark["rigId"] == "r_3"
    assert benchmark["engine"] == "vllm"
    assert benchmark["modelIds"] == ["test-model"]
    assert benchmark["requests"] == 2
    assert benchmark["concurrency"] == 1
    assert benchmark["ttftMs"] > 0
    assert benchmark["outputTokensPerSec"] > 0
    assert benchmark["totalTokensPerSec"] > 0
    assert benchmark["promptTokensPerSec"] > 0
    assert benchmark["vramPeakBytes"] == 123456789
    assert benchmark["powerAvgW"] == 45.5
    assert benchmark["gpuUtilPct"] == 90.0
    assert benchmark["contextTokens"] == 8
    assert any("aggregate" in note for note in benchmark["notes"])
    recipebench.require_complete(benchmark)
    assert recipebench.check_shape(benchmark) == []


def test_benchmark_rejects_model_the_endpoint_did_not_serve(fake_server, fake_gpu):
    with pytest.raises(AirigsError) as info:
        recipebench.run(
            fake_server.base_url,
            engine="vllm",
            requests=1,
            concurrency=1,
            model_ids=["not-served"],
            rig_id="r_1",
        )
    assert "does not serve" in str(info.value)
    assert "test-model" in str(info.value)


def test_benchmark_rejects_mismatched_served_model(fake_server, fake_gpu):
    fake_server.state.served_model = "some-other-model"
    with pytest.raises(AirigsError) as info:
        recipebench.run(
            fake_server.base_url,
            engine="vllm",
            requests=1,
            concurrency=1,
            model_ids=["test-model"],
            rig_id="r_1",
        )
    assert "served model" in str(info.value)


def test_benchmark_defaults_to_first_served_model(fake_server, fake_gpu):
    benchmark = recipebench.run(
        fake_server.base_url,
        engine="vllm",
        requests=1,
        concurrency=1,
        model_ids=None,
        rig_id="r_1",
    )
    assert benchmark["modelIds"] == ["test-model"]
    assert any("no --model given" in note for note in benchmark["notes"])


def test_benchmark_without_usage_falls_back_and_says_so(fake_server, fake_gpu):
    fake_server.state.usage = {}
    benchmark = recipebench.run(
        fake_server.base_url,
        engine="vllm",
        requests=1,
        concurrency=1,
        model_ids=["test-model"],
        rig_id="r_1",
    )
    assert benchmark["outputTokensPerSec"] > 0
    assert any("did not return token usage" in note for note in benchmark["notes"])


def test_missing_nvml_leaves_null_and_requires_completion(fake_server, monkeypatch):
    # Pin the no-NVML path: a developer machine with a GPU driver must not turn this test red.
    monkeypatch.setattr(recipebench, "_GpuSampler", NoGpuSampler)
    benchmark = recipebench.run(
        fake_server.base_url,
        engine="vllm",
        requests=1,
        concurrency=1,
        model_ids=["test-model"],
        rig_id="r_1",
    )
    # This host has no NVML: the harness reports nulls instead of made-up numbers.
    assert benchmark["vramPeakBytes"] is None
    assert benchmark["powerAvgW"] is None
    assert any("NVML" in note for note in benchmark["notes"])
    with pytest.raises(recipebench.RecipeBenchmarkIncomplete) as info:
        recipebench.require_complete(benchmark)
    fields = {item["field"] for item in info.value.missing}
    assert {"vramPeakBytes", "powerAvgW"} <= fields
    assert "NVML" in str(info.value)


def test_missing_required_fields():
    benchmark = {
        "schemaVersion": 1,
        "rigId": None,
        "engine": "vllm",
        "startedAt": "a",
        "finishedAt": "b",
        "modelIds": [],
        "concurrency": 1,
        "ttftMs": None,
        "outputTokensPerSec": None,
        "vramPeakBytes": None,
        "powerAvgW": None,
    }
    fields = {item["field"] for item in recipebench.missing_required(benchmark)}
    assert {"rigId", "modelIds", "ttftMs", "outputTokensPerSec", "vramPeakBytes", "powerAvgW"} <= fields
    with pytest.raises(recipebench.RecipeBenchmarkIncomplete):
        recipebench.require_complete(benchmark)


def test_concurrency_validation(fake_server, fake_gpu):
    with pytest.raises(AirigsError):
        recipebench.run(fake_server.base_url, engine="vllm", concurrency=0, requests=1, rig_id="r_1")
    with pytest.raises(AirigsError):
        recipebench.run(fake_server.base_url, engine="vllm", concurrency=1, requests=0, rig_id="r_1")


# ---------------------------------------------------------------------------
# Endpoint hardening (N6): redirects, response caps, plaintext keys


import contextlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from airigs.bench.common import Notes


@contextlib.contextmanager
def _server(handler_factory):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_factory)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:%d" % httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


class _Recorder:
    def __init__(self) -> None:
        self.requests: list[dict] = []


def _recording_handler(recorder: _Recorder, body: bytes = b'{"data": []}'):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def _record(self):
            recorder.requests.append(
                {"method": self.command, "path": self.path, "headers": dict(self.headers)}
            )

        def do_GET(self):
            self._record()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            self._record()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def test_recipe_bench_never_forwards_api_key_across_redirects():
    target = _Recorder()
    with _server(_recording_handler(target)) as target_url:
        class Redirect(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def do_GET(self):
                self.send_response(302)
                self.send_header("Location", target_url + "/stolen")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_POST(self):
                self.do_GET()

        with _server(Redirect) as source_url:
            with pytest.raises(recipebench.EndpointError) as info:
                recipebench.served_models(source_url + "/v1", api_key="secret-api-key")
            assert "302" in str(info.value)
            assert target.requests == []

            with pytest.raises(recipebench.EndpointError):
                recipebench.run_request(
                    source_url + "/v1",
                    model="test-model",
                    prompt="hi",
                    api_key="secret-api-key",
                    timeout=5,
                )
            assert target.requests == []


def test_oversized_models_response_is_rejected():
    huge = b"x" * (recipebench.MAX_MODELS_BYTES + 1)
    with _server(_recording_handler(_Recorder(), huge)) as url:
        with pytest.raises(recipebench.EndpointError) as info:
            recipebench.served_models(url + "/v1", timeout=5)
    assert "exceeded" in str(info.value)
    assert str(recipebench.MAX_MODELS_BYTES) in str(info.value)


def test_oversized_sse_stream_aborts_cleanly(monkeypatch):
    monkeypatch.setattr(recipebench, "MAX_STREAM_BYTES", 4096)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = b"data: " + b"x" * 8192 + b"\n\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

    with _server(Handler) as url:
        with pytest.raises(recipebench.EndpointError) as info:
            recipebench.run_request(url + "/v1", model="m", prompt="hi", api_key=None, timeout=5)
    assert "SSE stream" in str(info.value)
    assert "4096" in str(info.value)


def test_oversized_sse_event_aborts_cleanly(monkeypatch):
    monkeypatch.setattr(recipebench, "MAX_SSE_EVENT_BYTES", 1024)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = b"data: " + b"y" * 2048 + b"\n\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    with _server(Handler) as url:
        with pytest.raises(recipebench.EndpointError) as info:
            recipebench.run_request(url + "/v1", model="m", prompt="hi", api_key=None, timeout=5)
    assert "SSE" in str(info.value)


def test_api_key_over_plaintext_remote_is_refused_before_any_request():
    with pytest.raises(recipebench.EndpointError) as info:
        recipebench.run(
            "http://example.com",
            engine="vllm",
            api_key="secret",
            rig_id="r_1",
            requests=1,
            concurrency=1,
            model_ids=["m"],
        )
    assert "plaintext HTTP" in str(info.value)
    assert "secret" not in str(info.value)


def test_keyless_plaintext_remote_warns_but_is_allowed():
    notes = Notes()
    recipebench.check_endpoint_security("http://example.com/v1", None, notes)
    assert any("plaintext HTTP" in note for note in notes)


def test_endpoint_url_with_userinfo_is_rejected():
    with pytest.raises(recipebench.EndpointError):
        recipebench.normalize_base("http://user:password@example.com:8000")


def test_slow_drip_sse_aborts_at_wall_clock_deadline():
    """One byte every 200 ms never trips the 1 s socket timeout; the deadline must."""
    import time as _time

    class Drip(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def do_POST(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                for _ in range(50):
                    self.wfile.write(b"x")
                    self.wfile.flush()
                    _time.sleep(0.2)
            except (BrokenPipeError, ConnectionResetError):
                pass

    with _server(Drip) as url:
        started = _time.monotonic()
        with pytest.raises(recipebench.EndpointError) as info:
            recipebench.run_request(url + "/v1", model="m", prompt="hi", api_key=None, timeout=1.0)
        elapsed = _time.monotonic() - started
    assert "did not finish within 1s" in str(info.value)
    assert "SSE" in str(info.value)
    assert elapsed < 2.5, f"aborted only after {elapsed:.2f}s"


def test_slow_drip_models_aborts_at_wall_clock_deadline():
    import time as _time

    class Drip(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "1000")
            self.end_headers()
            self.close_connection = True
            try:
                for _ in range(20):
                    self.wfile.write(b" ")
                    self.wfile.flush()
                    _time.sleep(0.2)
            except (BrokenPipeError, ConnectionResetError):
                pass

    with _server(Drip) as url:
        started = _time.monotonic()
        with pytest.raises(recipebench.EndpointError) as info:
            recipebench.served_models(url + "/v1", timeout=1.0)
        elapsed = _time.monotonic() - started
    assert "did not finish within 1s" in str(info.value)
    assert "models" in str(info.value)
    assert elapsed < 2.5, f"aborted only after {elapsed:.2f}s"


def test_huge_response_header_is_rejected():
    """http.client caps the header line; that error must surface as EndpointError."""

    class HugeHeader(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def do_GET(self):
            try:
                self.send_response(200)
                self.send_header("X-Padding", "a" * 200_000)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")
            except (BrokenPipeError, ConnectionResetError):
                pass

    with _server(HugeHeader) as url:
        with pytest.raises(recipebench.EndpointError) as info:
            recipebench.served_models(url + "/v1", timeout=5)
    assert "header" in str(info.value)


class FakeNvml:
    """Minimal NVML surface for exercising the real _GpuSampler aggregation."""

    def __init__(self, limit: float | None = 25.0) -> None:
        self.count = 1
        self.available = True
        self.limit = limit
        self.closed = False

    def power_limit_w(self, index: int):
        return self.limit

    def memory_used(self, index: int):
        return None

    def power_w(self, index: int):
        return None

    def utilization(self, index: int):
        return None

    def close(self) -> None:
        self.closed = True


def test_recipe_sampler_reports_median_and_flags_power_outlier():
    sampler = recipebench._GpuSampler(session=FakeNvml(limit=25.0))
    sampler.gpu_power[0] = [1.6, 119.0, 1.7]
    result = sampler.stop()
    assert result["powerAvgW"] == 1.7  # not the 40.77 mean
    assert result["powerDiscarded"] == 1
    assert result["powerSamples"] == 3
    assert "discarded" in result["powerNote"]
    assert "25" in result["powerNote"]


def test_recipe_sampler_all_absurd_power_is_null_and_incomplete():
    sampler = recipebench._GpuSampler(session=FakeNvml(limit=25.0))
    sampler.gpu_power[0] = [119.0, 130.0, 140.0]
    result = sampler.stop()
    assert result["powerAvgW"] is None
    assert "not reported" in result["powerNote"]
    assert "25" in result["powerNote"]
    benchmark = {
        "schemaVersion": 1,
        "rigId": "r_1",
        "engine": "vllm",
        "startedAt": "a",
        "finishedAt": "b",
        "modelIds": ["test-model"],
        "concurrency": 1,
        "ttftMs": 1.0,
        "outputTokensPerSec": 1.0,
        "vramPeakBytes": 1,
        "powerAvgW": result["powerAvgW"],
    }
    with pytest.raises(recipebench.RecipeBenchmarkIncomplete) as info:
        recipebench.require_complete(benchmark)
    assert "powerAvgW" in {item["field"] for item in info.value.missing}


def test_recipe_sampler_notes_are_in_the_payload(fake_server, monkeypatch):
    class NotedSampler:
        def start(self) -> None:
            pass

        def stop(self) -> dict:
            summary = recipebench.summarize_power_samples([1.6, 119.0, 1.7], limit_w=25.0)
            return {
                "vramPeakBytes": 1024,
                "powerAvgW": summary["medianW"],
                "gpuUtilPct": 90.0,
                "ramPeakBytes": 2048,
                "powerNote": summary["note"],
                "nvml": True,
            }

    monkeypatch.setattr(recipebench, "_GpuSampler", NotedSampler)
    benchmark = recipebench.run(
        fake_server.base_url,
        engine="vllm",
        requests=1,
        concurrency=1,
        model_ids=["test-model"],
        rig_id="r_1",
    )
    assert benchmark["powerAvgW"] == 1.7
    assert any("median" in note and "discarded" in note for note in benchmark["notes"])
