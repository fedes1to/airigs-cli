"""A local fake of AIRigs, the Recipes platform and an OpenAI-compatible endpoint.

No network access and no GPU are required. Every request is recorded on
``state.requests`` so tests can assert on payloads.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

DEVICE = {
    "device_code": "device-123",
    "user_code": "ABCD-EFGH",
    "verification_uri": "http://localhost:4567/oauth/device",
    "verification_uri_complete": "http://localhost:4567/oauth/device?user_code=ABCD-EFGH",
    "expires_in": 60,
    "interval": 1,
}

TOKEN_RESPONSE = {
    "access_token": "tok_test_123",
    "token_type": "Bearer",
    "expires_in": 3600,
    "refresh_token": "ref_test_123",
    "scope": "user:read rig:read rig:write recipe:read recipe:write",
}

USER = {
    "uid": 7,
    "username": "tester",
    "userslug": "tester",
    "avatar": "",
    "profileUrl": "http://localhost:4567/user/tester",
    "groups": [{"name": "registered-users"}],
    "joindate": "2026-01-01T00:00:00.000Z",
}


class State:
    def __init__(self) -> None:
        self.device = dict(DEVICE)
        self.poll_script: list[dict] = [{"error": "authorization_pending"}, dict(TOKEN_RESPONSE)]
        self.token = dict(TOKEN_RESPONSE)
        self.user = dict(USER)
        self.rigs: list[dict] = []
        self.next_rig_id = 1
        self.verified = True
        self.benchmark_errors: list[dict] | None = None
        self.used_nonces: set[str] = set()
        self.nonce_counter = 0
        self.challenge_expired = False
        self.expire_first_upload = False
        self.token_delay = 0.0
        self.revoked: list[dict] = []
        self.openapi_engines = ["vllm", "sglang", "llamacpp", "strata", "omlx", "ollama", "custom"]
        self.requests: list[dict] = []
        self.slug = "test-recipe"
        self.models = ["test-model"]
        self.served_model = "test-model"
        self.chunks = ["Hel", "lo ", "world"]
        self.chunk_delay = 0.01
        self.usage = {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
        self.chat_status: tuple[int, dict] | None = None
        self.script = "#!/bin/sh\necho recipe-script\n"
        self.script_status = 200


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args) -> None:  # silence
            pass

        # -- helpers ----------------------------------------------------
        def _body(self) -> bytes:
            length = int(self.headers.get("content-length") or 0)
            return self.rfile.read(length) if length else b""

        def _parsed(self, raw: bytes):
            content_type = (self.headers.get("content-type") or "").lower()
            if "json" in content_type:
                return json.loads(raw.decode() or "{}")
            return {key: values[0] for key, values in parse_qs(raw.decode()).items()}

        def _send_json(self, status: int, payload, extra_headers: dict | None = None) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for key, value in (extra_headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def _send_text(self, status: int, text: str, content_type: str = "text/plain") -> None:
            data = text.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self) -> bool:
            header = self.headers.get("Authorization") or ""
            return header == f"Bearer {state.token['access_token']}"

        def _unauthorized(self) -> None:
            self._send_json(401, {"error": "unauthorized", "message": "invalid token"})

        # -- dispatch ---------------------------------------------------
        def do_GET(self) -> None:
            self._handle("GET")

        def do_POST(self) -> None:
            self._handle("POST")

        def do_PUT(self) -> None:
            self._handle("PUT")

        def do_PATCH(self) -> None:
            self._handle("PATCH")

        def do_DELETE(self) -> None:
            self._handle("DELETE")

        def _handle(self, method: str) -> None:
            raw = self._body()
            try:
                body = self._parsed(raw) if raw else None
            except Exception:
                body = None
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            state.requests.append(
                {
                    "method": method,
                    "path": path,
                    "body": body,
                    "headers": dict(self.headers),
                }
            )

            # -- OAuth --------------------------------------------------
            if path == "/oauth/device_authorization" and method == "POST":
                self._send_json(200, state.device)
                return
            if path == "/oauth/device_token" and method == "POST":
                step = state.poll_script.pop(0) if state.poll_script else dict(state.token)
                if "error" in step:
                    self._send_json(400, step)
                else:
                    self._send_json(200, step)
                return
            if path == "/oauth/token" and method == "POST":
                if state.token_delay:
                    time.sleep(state.token_delay)
                self._send_json(200, state.token)
                return
            if path == "/oauth/revoke" and method == "POST":
                state.revoked.append(body or {})
                self._send_json(200, {"revoked": True})
                return
            if path == "/oauth/userinfo" and method == "GET":
                if not self._authorized():
                    self._unauthorized()
                    return
                self._send_json(200, state.user)
                return

            # -- rig API ------------------------------------------------
            if path.startswith("/api/v3/plugins/airigs-rig"):
                if not self._authorized():
                    self._unauthorized()
                    return
                self._rigs(method, path, body)
                return

            # -- recipes API --------------------------------------------
            if path.startswith("/api/v1"):
                if not self._authorized():
                    self._unauthorized()
                    return
                self._recipes(method, path, body)
                return

            # -- OpenAI-compatible endpoint -----------------------------
            if path == "/v1/models" and method == "GET":
                self._send_json(200, {"data": [{"id": model} for model in state.models]})
                return
            if path == "/v1/chat/completions" and method == "POST":
                self._chat(body or {})
                return

            self._send_json(404, {"error": "not_found", "path": path})

        # -- rigs -------------------------------------------------------
        def _rigs(self, method: str, path: str, body) -> None:
            suffix = path[len("/api/v3/plugins/airigs-rig") :].strip("/")
            segments = [segment for segment in suffix.split("/") if segment]

            if not segments:
                if method == "GET":
                    self._send_json(200, {"rigs": state.rigs})
                    return
                if method == "POST":
                    rig = {
                        "id": f"r_{state.next_rig_id}",
                        "revision": 1,
                        "name": (body or {}).get("name", ""),
                        "verified": False,
                        "hardware": (body or {}).get("hardware", []),
                        "models": (body or {}).get("models", []),
                        "notes": (body or {}).get("notes", ""),
                    }
                    state.next_rig_id += 1
                    state.rigs.append(rig)
                    self._send_json(200, rig)
                    return

            if segments == ["benchmark", "schema"] and method == "GET":
                self._send_json(200, {"schemaVersion": 1})
                return

            rig_id = segments[0]
            rig = next((item for item in state.rigs if str(item["id"]) == rig_id), None)
            if rig is None and method == "GET":
                rig = next((item for item in state.rigs if item.get("userslug") == rig_id), None)
            if rig is None:
                self._send_json(404, {"error": "not_found", "message": "no such rig"})
                return

            if len(segments) == 1:
                if method == "GET":
                    self._send_json(200, rig)
                    return
                if method == "PUT":
                    rig.update(body or {})
                    self._send_json(200, rig)
                    return
                if method == "DELETE":
                    state.rigs.remove(rig)
                    self._send_json(200, {"deleted": True})
                    return
            if len(segments) == 3 and segments[1:] == ["benchmark", "challenge"] and method == "POST":
                state.nonce_counter += 1
                self._send_json(
                    200,
                    {
                        "nonce": f"nonce-{state.nonce_counter}",
                        "expiresAt": "2099-01-01T00:00:00Z",
                        "rigId": rig["id"],
                    },
                )
                return
            if len(segments) == 2 and segments[1] == "benchmark" and method == "POST":
                if state.benchmark_errors is not None:
                    self._send_json(422, {"errors": state.benchmark_errors})
                    return
                benchmark = (body or {}).get("benchmark") or {}
                challenge = benchmark.get("challenge") if isinstance(benchmark, dict) else None
                nonce = challenge.get("nonce") if isinstance(challenge, dict) else None
                if not nonce:
                    self._send_json(422, {"errors": [{"path": "challenge", "message": "required"}]})
                    return
                if state.challenge_expired or state.expire_first_upload:
                    state.expire_first_upload = False
                    self._send_json(410, {"error": "challenge_expired"})
                    return
                if nonce in state.used_nonces:
                    self._send_json(409, {"error": "challenge_replayed"})
                    return
                if not str(nonce).startswith("nonce-"):
                    self._send_json(409, {"error": "unknown_challenge"})
                    return
                state.used_nonces.add(nonce)
                rig["verified"] = state.verified
                rig["verifiedAt"] = "2026-01-01T00:03:00Z" if state.verified else None
                rig["benchmark"] = (body or {}).get("benchmark")
                self._send_json(200, rig)
                return
            self._send_json(405, {"error": "method_not_allowed"})

        # -- recipes ----------------------------------------------------
        def _recipes(self, method: str, path: str, body) -> None:
            suffix = path[len("/api/v1") :].strip("/")
            segments = [segment for segment in suffix.split("/") if segment]
            if segments == ["me"] and method == "GET":
                self._send_json(200, dict(state.user))
                return
            if segments == ["rigs"] and method == "GET":
                self._send_json(200, {"rigs": state.rigs})
                return
            if segments == ["openapi.json"] and method == "GET":
                self._send_json(
                    200,
                    {
                        "components": {
                            "schemas": {
                                "RecipeInput": {
                                    "properties": {
                                        "engine": {
                                            "type": "string",
                                            "enum": state.openapi_engines,
                                        }
                                    }
                                }
                            }
                        }
                    },
                )
                return
            if segments == ["recipes"] and method == "POST":
                self._send_json(200, {"slug": state.slug, "recipe": {"slug": state.slug, **(body or {})}})
                return
            if len(segments) >= 2 and segments[0] == "recipes":
                slug = segments[1]
                if len(segments) == 3 and segments[2] == "benchmark" and method == "POST":
                    self._send_json(200, {"slug": slug, "ok": True})
                    return
                if len(segments) == 3 and segments[2] == "publish" and method == "POST":
                    self._send_json(200, {"slug": slug, "published": True})
                    return
                if len(segments) == 3 and segments[2] == "script" and method == "GET":
                    if state.script_status != 200:
                        self._send_json(state.script_status, {"error": "no_script"})
                    else:
                        self._send_text(200, state.script)
                    return
                if len(segments) == 2 and method == "PATCH":
                    self._send_json(200, {"slug": slug, **(body or {})})
                    return
                if len(segments) == 2 and method == "DELETE":
                    self._send_json(200, {"deleted": True})
                    return
            self._send_json(404, {"error": "not_found", "path": path})

        # -- OpenAI streaming ------------------------------------------
        def _chat(self, body) -> None:
            if state.chat_status is not None:
                status, payload = state.chat_status
                self._send_json(status, payload)
                return
            model = body.get("model")
            if model not in state.models:
                self._send_json(404, {"error": {"message": f"model {model!r} not found"}})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                for piece in state.chunks:
                    chunk = {
                        "id": "chatcmpl-1",
                        "model": state.served_model,
                        "choices": [{"index": 0, "delta": {"content": piece}}],
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(state.chunk_delay)
                final = {
                    "id": "chatcmpl-1",
                    "model": state.served_model,
                    "choices": [],
                    "usage": state.usage,
                }
                self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


class FakeServer:
    def __init__(self) -> None:
        self.state = State()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.state))
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "FakeServer":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    def __enter__(self) -> "FakeServer":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()


def make_benchmark(**overrides) -> dict:
    """A minimal payload satisfying every required field."""
    benchmark = {
        "schemaVersion": 1,
        "runner": {"name": "airigs-cli", "version": "0.1.0", "platform": "linux", "arch": "x86_64", "python": "3.12"},
        "startedAt": "2026-01-01T00:00:00Z",
        "finishedAt": "2026-01-01T00:03:00Z",
        "durationS": 180.0,
        "system": {
            "hostname": "box",
            "os": {"name": "Linux", "version": "6.10", "kernel": "6.10.1"},
            "cpu": {
                "model": "AMD Ryzen 9 7950X",
                "vendor": "AMD",
                "physicalCores": 16,
                "logicalCores": 32,
                "baseClockMhz": 4500,
                "maxClockMhz": 5700,
                "cacheL3Mb": 64,
            },
            "memory": {"totalBytes": 68719476736, "type": "DDR5", "speedMhz": 6000, "channels": 2},
            "drivers": {"nvidia": {"version": "550.54", "cuda": "12.4"}, "amd": {"version": "", "rocm": ""}, "metal": {"version": ""}},
        },
        "cpuBench": {
            "singleThread": {"sha256Mbps": 190.2},
            "multiThread": {"sha256Mbps": 2900.5, "threads": 32},
            "fp32Gflops": 120.0,
            "fp64Gflops": 60.0,
            "memory": {"readGBs": 64.2, "writeGBs": 52.1, "copyGBs": 48.3, "triadGBs": 50.0},
        },
        "gpus": [
            {
                "index": 0,
                "name": "NVIDIA GeForce RTX 4090",
                "uuid": "GPU-abc",
                "vramBytes": 25757220864,
                "driverVersion": "550.54",
                "pcie": {"version": 4, "width": 16, "currentVersion": 4, "currentWidth": 16, "maxVersion": 4, "maxWidth": 16},
                "clocks": {"graphicsMhz": 2520, "memoryMhz": 10501},
                "power": {"limitW": 450, "idleW": 20.1, "loadW": 395.2, "measured": True, "sensor": "nvml"},
                "memoryBandwidth": {"measuredGBs": 981.4, "theoreticalGBs": 1008.0, "method": "device-to-device copy"},
                "tflops": {"fp64": 1.2, "fp32": 78.9, "tf32": 157.0, "bf16": 318.0, "fp16": 312.0, "fp8": 640.0, "int8": 620.0},
                "tflopsMethod": "torch.matmul",
                "thermal": {"maxTempC": 71},
            }
        ],
        "interconnect": {
            "p2p": {"supported": True, "enabled": True, "topologyMatrix": ["SYS", "SYS"], "pairs": [{"a": 0, "b": 1, "p2p": True, "bandwidthGBs": 11.8}]},
            "nvlink": {"present": False},
        },
        "power": {"idleW": 65.0, "loadW": 505.0, "measurement": "sensor", "psuWatts": 1000},
        "notes": [],
    }
    benchmark.update(overrides)
    return benchmark


def make_recipe(**overrides) -> dict:
    recipe = {
        "title": "70B on one GPU",
        "summary": "A test recipe",
        "engine": "vllm",
        "models": [{"source": "huggingface", "id": "meta-llama/Llama-3.1-8B", "name": "Llama 3.1 8B", "role": "base"}],
        "args": [{"flag": "--max-model-len", "value": "32768"}],
        "launchCommand": "",
        "customInstructions": "",
        "repos": [],
        "hardwareNotes": "",
        "rigId": "r_1",
        "publish": False,
    }
    recipe.update(overrides)
    return recipe
