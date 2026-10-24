from __future__ import annotations

import json

import pytest

from airigs import auth, config
from airigs.http import AuthError

from fake_server import DEVICE, TOKEN_RESPONSE


class Clock:
    def __init__(self) -> None:
        self.now_s = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.now_s

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now_s += seconds


def scripted(*responses):
    queue = list(responses)
    calls: list[tuple[str, dict]] = []

    def transport(url, form):
        calls.append((url, form))
        if len(queue) > 1:
            return queue.pop(0)
        return queue[0]

    transport.calls = calls  # type: ignore[attr-defined]
    return transport


def device(**overrides):
    payload = dict(DEVICE)
    payload.update(overrides)
    return payload


def test_start_device_authorization_shapes_request():
    transport = scripted((200, device()))
    payload = auth.start_device_authorization("http://127.0.0.1:4567/", transport=transport)
    assert payload["device_code"] == "device-123"
    url, form = transport.calls[0]  # type: ignore[attr-defined]
    assert url == "http://127.0.0.1:4567/oauth/device_authorization"
    assert form["client_id"] == "airigs-cli"
    assert "rig:write" in form["scope"]


def test_start_device_authorization_rejects_missing_fields():
    transport = scripted((200, {"device_code": "x"}))
    with pytest.raises(Exception) as info:
        auth.start_device_authorization("http://127.0.0.1:4567", transport=transport)
    assert "user_code" in str(info.value)


def test_device_code_becomes_required():
    transport = scripted((200, {"user_code": "x"}))
    with pytest.raises(Exception):
        auth.start_device_authorization("http://127.0.0.1:4567", transport=transport)


def test_poll_pending_then_slow_down_then_token():
    clock = Clock()
    transport = scripted(
        (400, {"error": "authorization_pending"}),
        (400, {"error": "slow_down"}),
        (200, dict(TOKEN_RESPONSE)),
    )
    payload = auth.poll_device_token(
        "http://127.0.0.1:4567",
        device(interval=2, expires_in=60),
        transport=transport,
        sleep=clock.sleep,
        clock=clock.now,
    )
    assert payload["access_token"] == TOKEN_RESPONSE["access_token"]
    # slow_down increases the interval by exactly 5 seconds (RFC 8628 §3.5).
    assert clock.sleeps == [2, 7]
    assert transport.calls[0][0] == "http://127.0.0.1:4567/oauth/device_token"  # type: ignore[attr-defined]
    assert transport.calls[0][1]["grant_type"] == auth.DEVICE_CODE_GRANT  # type: ignore[attr-defined]


def test_poll_expired_token():
    clock = Clock()
    transport = scripted((400, {"error": "authorization_pending"}), (400, {"error": "expired_token"}))
    with pytest.raises(auth.DeviceFlowError) as info:
        auth.poll_device_token(
            "http://127.0.0.1:4567", device(interval=1, expires_in=30), transport=transport, sleep=clock.sleep, clock=clock.now
        )
    assert info.value.code == "expired_token"


def test_poll_access_denied():
    transport = scripted((400, {"error": "access_denied"}))
    with pytest.raises(auth.DeviceFlowError) as info:
        auth.poll_device_token("http://127.0.0.1:4567", device(), transport=transport, sleep=lambda _s: None, clock=lambda: 0.0)
    assert info.value.code == "access_denied"


def test_poll_stops_at_expiry():
    clock = Clock()
    transport = scripted((400, {"error": "authorization_pending"}))
    with pytest.raises(auth.DeviceFlowError) as info:
        auth.poll_device_token(
            "http://127.0.0.1:4567", device(interval=5, expires_in=10), transport=transport, sleep=clock.sleep, clock=clock.now
        )
    assert info.value.code == "expired_token"
    assert clock.sleeps == [5, 5]


def test_poll_unknown_error_is_reported():
    transport = scripted((400, {"error": "invalid_scope", "error_description": "no"}))
    with pytest.raises(auth.DeviceFlowError) as info:
        auth.poll_device_token("http://127.0.0.1:4567", device(), transport=transport, sleep=lambda _s: None, clock=lambda: 0.0)
    assert info.value.code == "invalid_scope"


def _point_env(monkeypatch, server):
    monkeypatch.setenv("AIRIGS_URL", server.base_url)


def test_device_login_against_fake_server(fake_server, isolated_config, monkeypatch):
    _point_env(monkeypatch, fake_server)
    fake_server.state.device["interval"] = 0.01
    fake_server.state.poll_script = [{"error": "authorization_pending"}, dict(TOKEN_RESPONSE)]
    printed: list[str] = []
    opened: list[str] = []

    payload = auth.login_with_device_flow(
        fake_server.base_url,
        printer=printed.append,
        opener=lambda url: opened.append(url) or True,
    )
    assert payload["access_token"] == TOKEN_RESPONSE["access_token"]
    assert any("ABCD-EFGH" in line for line in printed)
    assert opened == [DEVICE["verification_uri_complete"]]
    stored = config.load_config()
    assert stored["token"] == TOKEN_RESPONSE["access_token"]
    assert stored["refresh_token"] == TOKEN_RESPONSE["refresh_token"]
    assert config.config_mode() == 0o600


def test_device_login_no_browser(fake_server, isolated_config, monkeypatch):
    _point_env(monkeypatch, fake_server)
    fake_server.state.device["interval"] = 0.01
    fake_server.state.poll_script = [dict(TOKEN_RESPONSE)]
    opened: list[str] = []
    auth.login_with_device_flow(
        fake_server.base_url,
        no_browser=True,
        printer=lambda _line: None,
        opener=lambda url: opened.append(url) or True,
    )
    assert opened == []


def test_login_with_token_verifies_and_stores(fake_server, isolated_config):
    info = auth.login_with_token(fake_server.base_url, TOKEN_RESPONSE["access_token"])
    assert info["username"] == "tester"
    stored = config.load_config()
    assert stored["token"] == TOKEN_RESPONSE["access_token"]
    assert stored["airigs_url"] == fake_server.base_url
    assert config.config_mode() == 0o600


def test_login_with_bad_token_does_not_store(fake_server, isolated_config):
    with pytest.raises(AuthError):
        auth.login_with_token(fake_server.base_url, "definitely-wrong")
    assert not config.config_path().exists()


def test_refresh_when_expired(fake_server, isolated_config):
    config.save_tokens(
        {"access_token": "old", "refresh_token": "refresh-1", "expires_in": 3600},
        airigs_url=fake_server.base_url,
    )
    stored = config.load_config()
    stored["expires_at"] = 1  # long expired
    config.save_config(stored)
    fake_server.state.token = {**TOKEN_RESPONSE, "access_token": "fresh-token"}

    settings = config.resolve_settings()
    token = auth.ensure_token(settings)
    assert token == "fresh-token"
    assert config.load_config()["token"] == "fresh-token"
    refresh_request = [r for r in fake_server.state.requests if r["path"] == "/oauth/token"]
    assert refresh_request and refresh_request[0]["body"]["grant_type"] == "refresh_token"


def test_ensure_token_without_login():
    with pytest.raises(auth.NotLoggedInError):
        auth.ensure_token(config.resolve_settings(config={}))


def test_ensure_token_env_is_used_without_refresh(isolated_config, monkeypatch):
    monkeypatch.setenv("AIRIGS_TOKEN", "env-token")
    settings = config.resolve_settings(config={})
    assert auth.ensure_token(settings) == "env-token"
    assert settings.refresh_token is None


def test_describe_token_redacts(isolated_config):
    config.save_tokens({"access_token": "secret-token-value", "expires_in": 3600})
    description = auth.describe_token(config.resolve_settings())
    assert description["state"] == "valid"
    assert description["token"] == "***"
    assert "secret-token-value" not in json.dumps(description)
    assert description["expires_in"] > 0
