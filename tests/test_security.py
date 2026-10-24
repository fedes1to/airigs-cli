from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from airigs import auth, config, http, recipes, scripts
from airigs.http import AirigsError, ApiClient, HttpError


class _Recorder:
    def __init__(self) -> None:
        self.headers: list[dict] = []
        self.paths: list[str] = []


def _serve(handler_factory):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_factory)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread


def test_authorization_is_never_forwarded_across_a_redirect():
    target = _Recorder()

    class TargetHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def do_GET(self):
            target.headers.append(dict(self.headers))
            target.paths.append(self.path)
            body = b"{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    target_server, target_thread = _serve(TargetHandler)
    target_url = "http://127.0.0.1:%d/steal" % target_server.server_address[1]

    class RedirectHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", target_url)
            self.send_header("Content-Length", "0")
            self.end_headers()

    redirect_server, redirect_thread = _serve(RedirectHandler)
    source_url = "http://127.0.0.1:%d" % redirect_server.server_address[1]
    try:
        client = ApiClient(source_url, token="super-secret-token", timeout=5)
        with pytest.raises(HttpError) as info:
            client.get("/api/v3/plugins/airigs-rig")
        assert info.value.status == 302
        time.sleep(0.05)
        assert target.paths == []
        assert target.headers == []
    finally:
        redirect_server.shutdown()
        redirect_server.server_close()
        redirect_thread.join(timeout=5)
        target_server.shutdown()
        target_server.server_close()
        target_thread.join(timeout=5)


def test_join_url_refuses_absolute_paths():
    with pytest.raises(AirigsError):
        http.join_url("http://localhost:4567", "https://attacker.example/collect")
    with pytest.raises(AirigsError):
        http.join_url("http://localhost:4567", "//attacker.example/collect")


def test_token_is_bound_to_its_issuer_origin(isolated_config):
    config.save_tokens({"access_token": "tok", "expires_in": 3600}, airigs_url="http://localhost:1111")
    settings = config.resolve_settings(airigs_url="http://localhost:2222")
    assert settings.token_origin == "http://localhost:1111"
    with pytest.raises(auth.NotLoggedInError) as info:
        auth.ensure_token(settings)
    assert "issued for" in str(info.value)
    assert "localhost:2222" in str(info.value)


def test_plaintext_remote_issuer_is_refused(isolated_config, monkeypatch):
    monkeypatch.setenv("AIRIGS_TOKEN", "tok")
    settings = config.resolve_settings(airigs_url="http://forum.example.com")
    with pytest.raises(AirigsError) as info:
        auth.ensure_token(settings)
    assert "plaintext HTTP" in str(info.value)

    # https is accepted by the guard itself (no network call is made here).
    http.assert_secure_url("https://forum.example.com")


def test_concurrent_refresh_is_single_flight(fake_server, isolated_config):
    config.save_tokens(
        {"access_token": "stale", "refresh_token": "refresh-1", "expires_in": 3600},
        airigs_url=fake_server.base_url,
    )
    stored = config.load_config()
    stored["expires_at"] = 1
    config.save_config(stored)
    fake_server.state.token = {
        "access_token": "fresh",
        "token_type": "Bearer",
        "expires_in": 3600,
        "refresh_token": "refresh-2",
    }
    fake_server.state.token_delay = 0.15

    settings = config.resolve_settings()
    results: list[str] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            results.append(auth.ensure_token(settings))
        except BaseException as exc:  # pragma: no cover - surfaced in the assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert errors == []
    assert set(results) == {"fresh"}
    refresh_calls = [r for r in fake_server.state.requests if r["path"] == "/oauth/token"]
    assert len(refresh_calls) == 1, f"expected one refresh, saw {len(refresh_calls)}"
    assert config.load_config()["token"] == "fresh"


def test_rig_id_must_be_an_opaque_string():
    def recipe(rig_id):
        return {
            "title": "t",
            "engine": "vllm",
            "models": [{"source": "huggingface", "id": "org/m", "name": "m"}],
            "rigId": rig_id,
        }

    for bad in (7, "7", "abc", "r_", "r_" + "a" * 93):
        problems = recipes.validate_recipe(recipe(bad))
        assert any("rigId" in problem for problem in problems), bad
    assert recipes.validate_recipe(recipe("r_abc123")) == []
    assert recipes.validate_recipe(recipe("r_" + "a" * 92)) == []
    assert recipes.normalize_recipe(recipe("r_abc123"))["rigId"] == "r_abc123"


def test_every_engine_in_the_shared_enum_round_trips():
    base = {
        "title": "t",
        "models": [{"source": "huggingface", "id": "org/m", "name": "m"}],
    }
    for engine in recipes.ENGINES:
        payload = recipes.require_valid({**base, "engine": engine})
        assert payload["engine"] == engine
    assert any("engine" in problem for problem in recipes.validate_recipe({**base, "engine": "llama.cpp"}))
    assert any("engine" in problem for problem in recipes.validate_recipe({**base, "engine": "lmdeploy"}))


def test_model_role_enum_and_skeleton_default():
    base = {
        "title": "t",
        "engine": "vllm",
        "models": [{"source": "huggingface", "id": "org/m", "name": "m"}],
    }
    assert recipes.skeleton()["models"][0]["role"] == "base"
    for role in recipes.MODEL_ROLES:
        assert recipes.validate_recipe({**base, "models": [{**base["models"][0], "role": role}]}) == []
    problems = recipes.validate_recipe(
        {**base, "models": [{**base["models"][0], "role": "target"}]}
    )
    assert any("role" in problem for problem in problems)
    # A missing role is normalised to the shared default.
    assert recipes.require_valid(base)["models"][0]["role"] == "base"


def test_engines_come_from_openapi_when_available(fake_server):
    client = ApiClient(fake_server.base_url, token="tok_test_123")
    engines, source = recipes.engines_for(client)
    assert source == "openapi"
    assert engines == fake_server.state.openapi_engines


def test_engines_fall_back_to_builtin(fake_server, monkeypatch):
    client = ApiClient(fake_server.base_url, token="tok_test_123")

    def unavailable(*_args, **_kwargs):
        raise AirigsError("offline")

    monkeypatch.setattr(client, "get", unavailable)
    engines, source = recipes.engines_for(client)
    assert source == "builtin"
    assert engines == list(recipes.ENGINES)


def test_sanitize_environment_removes_credentials():
    env = scripts.sanitize_environment(
        {
            "PATH": "/bin",
            "AIRIGS_TOKEN": "a",
            "RECIPES_TOKEN": "b",
            "AIRIGS_CLIENT_SECRET": "c",
            "HF_TOKEN": "d",
            "MY_API_KEY": "e",
            "NORMAL": "ok",
        }
    )
    assert env == {"PATH": "/bin", "NORMAL": "ok"}


def test_recipe_run_does_not_expose_tokens(monkeypatch, capfd):
    monkeypatch.setenv("AIRIGS_TOKEN", "airigs-secret-value")
    monkeypatch.setenv("RECIPES_TOKEN", "recipes-secret-value")
    code = scripts.execute(
        'echo "a=${AIRIGS_TOKEN:-missing} r=${RECIPES_TOKEN:-missing}"\n',
        yes=True,
        printer=lambda _line: None,
    )
    assert code == 0
    output = capfd.readouterr().out
    assert "airigs-secret-value" not in output
    assert "recipes-secret-value" not in output
    assert "a=missing r=missing" in output


def test_airigs_yes_env_confirms(monkeypatch):
    monkeypatch.setenv("AIRIGS_YES", "1")
    calls = []
    code = scripts.execute(
        "echo hi\n",
        yes=False,
        tty=False,
        printer=lambda _line: None,
        runner=lambda *a, **k: calls.append((a, k)) or type("C", (), {"returncode": 0})(),
    )
    assert code == 0
    assert calls


def test_challenge_is_attached_and_expiry_is_retried(fake_server, isolated_config, monkeypatch, tmp_path, capsys):
    from fake_server import make_benchmark

    monkeypatch.setenv("AIRIGS_URL", fake_server.base_url)
    monkeypatch.setenv("AIRIGS_TOKEN", fake_server.state.token["access_token"])
    bench_file = tmp_path / "bench.json"
    bench_file.write_text(json.dumps(make_benchmark()))
    fake_server.state.expire_first_upload = True

    from airigs import cli

    code = cli.main(
        ["rig", "submit", "--name", "Box", "--benchmark", str(bench_file), "--json",
         "--airigs-url", fake_server.base_url]
    )
    assert code == 0, capsys.readouterr().err
    challenges = [r for r in fake_server.state.requests if r["path"].endswith("/benchmark/challenge")]
    uploads = [r for r in fake_server.state.requests if r["path"].endswith("/benchmark")]
    assert len(challenges) == 2  # one expired, one fresh
    assert len(uploads) == 2
    payload = uploads[-1]["body"]["benchmark"]
    assert payload["challenge"]["nonce"].startswith("nonce-")
    assert isinstance(payload["challenge"]["issuedAt"], str)


def test_benchmark_without_challenge_is_rejected_by_the_server(fake_server):
    client = ApiClient(fake_server.base_url, token=fake_server.state.token["access_token"])
    client.post("/api/v3/plugins/airigs-rig", {"name": "Box"})
    from airigs.http import ValidationError

    with pytest.raises(ValidationError) as info:
        client.post(
            "/api/v3/plugins/airigs-rig/r_1/benchmark",
            {"benchmark": {"schemaVersion": 1}},
        )
    assert "challenge" in str(info.value)


def test_logout_revokes_refresh_token(fake_server, isolated_config):
    config.save_tokens(
        {"access_token": "tok", "refresh_token": "refresh-1", "expires_in": 3600},
        airigs_url=fake_server.base_url,
    )
    settings = config.resolve_settings()
    result = auth.logout(settings)
    assert result["hadRefreshToken"] is True
    assert result["revoked"] is True
    assert result["removedLocalState"] is True
    assert fake_server.state.revoked[0]["token"] == "refresh-1"
    assert "token" not in config.load_config()


def test_logout_removes_local_state_even_when_revocation_fails(isolated_config):
    config.save_tokens(
        {"access_token": "tok", "refresh_token": "refresh-1", "expires_in": 3600},
        airigs_url="http://127.0.0.1:1",
    )
    result = auth.logout(config.resolve_settings())
    assert result["revoked"] is False
    assert result["removedLocalState"] is True
    assert "error" in result
    assert "refresh_token" not in config.load_config()


def test_self_check_requires_server_fields(monkeypatch):
    from airigs.bench import report, runner
    from fake_server import make_benchmark

    complete = make_benchmark()
    monkeypatch.setattr(runner, "run_benchmark", lambda **_kw: complete)
    result = runner.self_check()
    assert result["ok"] is True and result["uploadable"] is True

    incomplete = make_benchmark(gpus=[])
    monkeypatch.setattr(runner, "run_benchmark", lambda **_kw: incomplete)
    result = runner.self_check()
    assert result["ok"] is False
    assert result["uploadable"] is False
    assert result["missingRequired"]
