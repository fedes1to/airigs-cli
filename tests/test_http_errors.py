from __future__ import annotations

import json

import pytest

from airigs import http
from airigs.http import (
    ApiClient,
    AuthError,
    ForbiddenError,
    HttpError,
    NetworkError,
    NotFoundError,
    RateLimitError,
    Response,
    ValidationError,
    raise_for_response,
)


def response(status: int, payload, headers: dict | None = None) -> Response:
    return Response(
        status=status,
        data=payload,
        text=json.dumps(payload),
        headers=headers or {},
        url="http://example.test/x",
    )


def test_401_maps_to_auth_error():
    with pytest.raises(AuthError) as info:
        raise_for_response(response(401, {"error": "unauthorized", "message": "bad token"}))
    assert "bad token" in str(info.value)
    assert info.value.exit_code == 3


def test_403_maps_to_forbidden():
    with pytest.raises(ForbiddenError):
        raise_for_response(response(403, {"message": "no"}))


def test_404_maps_to_not_found():
    with pytest.raises(NotFoundError):
        raise_for_response(response(404, {"message": "missing"}))


def test_422_lists_field_errors():
    payload = {
        "errors": [
            {"path": "title", "message": "is required"},
            {"path": "models.0.name", "message": "must not be empty"},
        ]
    }
    with pytest.raises(ValidationError) as info:
        raise_for_response(response(422, payload), action="create recipe")
    message = str(info.value)
    assert "title: is required" in message
    assert "models.0.name: must not be empty" in message
    assert message.startswith("create recipe:")
    assert info.value.errors == payload["errors"]
    assert info.value.exit_code == 5


def test_429_reads_retry_after():
    with pytest.raises(RateLimitError) as info:
        raise_for_response(response(429, {"message": "slow down"}, {"retry-after": "7"}))
    assert info.value.retry_after == 7.0


def test_500_is_generic_http_error():
    with pytest.raises(HttpError) as info:
        raise_for_response(response(500, {"message": "boom"}))
    assert not isinstance(info.value, (AuthError, ValidationError))
    assert info.value.status == 500


def test_network_error_on_closed_port():
    with pytest.raises(NetworkError):
        http.send("GET", "http://127.0.0.1:1/", timeout=0.5)


def test_api_client_401_and_422_against_fake_server(fake_server):
    client = ApiClient(fake_server.base_url, token="wrong")
    with pytest.raises(AuthError):
        client.get("/api/v3/plugins/airigs-rig", action="list rigs")

    fake_server.state.rigs = [{"id": "r_1", "name": "Box", "verified": False}]
    fake_server.state.benchmark_errors = [{"path": "benchmark.gpus", "message": "at least one GPU"}]
    client = ApiClient(fake_server.base_url, token=fake_server.state.token["access_token"])
    with pytest.raises(ValidationError) as info:
        client.post("/api/v3/plugins/airigs-rig/r_1/benchmark", {"benchmark": {}})
    assert "at least one GPU" in str(info.value)
    assert info.value.errors[0]["path"] == "benchmark.gpus"


def test_verbose_logging_redacts_token(fake_server):
    logs: list[str] = []
    token = fake_server.state.token["access_token"]
    client = ApiClient(
        fake_server.base_url,
        token=token,
        verbose=True,
        log=logs.append,
        secrets={token},
    )
    client.get("/api/v3/plugins/airigs-rig")
    assert logs
    joined = "\n".join(logs)
    assert token not in joined
    assert "GET" in joined


def test_join_url():
    assert http.join_url("http://h:1/", "/api/v1/me") == "http://h:1/api/v1/me"
    assert http.join_url("http://h:1", "api/v1/me") == "http://h:1/api/v1/me"


def test_unwrap_envelope_handles_nodebb_and_plain_bodies():
    envelope = {"status": {"code": "ok", "message": "OK"}, "response": {"rigs": [{"id": "r_abc"}]}}
    assert http.unwrap_envelope(envelope) == {"rigs": [{"id": "r_abc"}]}
    plain = {"rigs": [{"id": "r_abc"}]}
    assert http.unwrap_envelope(plain) == plain
    error = {"status": {"code": 422, "message": "no"}, "errors": [{"path": "x", "message": "y"}]}
    assert http.unwrap_envelope(error) == error
    assert http.unwrap_envelope(None) is None
    assert http.unwrap_envelope([1, 2]) == [1, 2]
