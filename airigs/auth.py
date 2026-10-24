"""Authentication: RFC 8628 device flow, refresh, and token lifecycle.

The polling state machine is split from the transport so it can be tested
without a network or a clock: pass ``transport``, ``sleep`` and ``clock``.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from .config import (
    CLIENT_ID,
    SCOPE,
    Settings,
    clear_tokens,
    config_dir,
    load_config,
    redact,
    resolve_settings,
    save_tokens,
    token_expired,
    url_origin,
)
from .http import AirigsError, ApiClient, AuthError, Response, assert_secure_url, join_url, send

DEVICE_CODE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
SLOW_DOWN_INCREMENT = 5.0

#: One refresh at a time per process: concurrent callers re-read the store after
#: the winner lands and reuse its token instead of racing a second refresh.
_REFRESH_LOCK = threading.Lock()
#: Serialise config writes so two threads cannot interleave token state.
_WRITE_LOCK = threading.Lock()


def _persist_tokens(payload: dict[str, Any], **kwargs: Any) -> Any:
    with _WRITE_LOCK:
        return save_tokens(payload, **kwargs)


def check_token_origin(settings: Settings) -> None:
    """Refuse a stored token that belongs to a different issuer origin."""
    if not settings.token_origin:
        return
    current = url_origin(settings.airigs_url)
    if settings.token_origin != current:
        raise NotLoggedInError(
            f"the stored token was issued for {settings.token_origin}, but AIRigs is now {current}; "
            "run `airigs login` to authenticate against this origin"
        )


class DeviceFlowError(AirigsError):
    """The device authorization grant ended without a token."""

    def __init__(self, code: str, description: str = "") -> None:
        self.code = code
        self.description = description
        message = description or {
            "expired_token": "the device code expired before it was approved",
            "access_denied": "the authorization request was denied",
            "authorization_pending": "the authorization request is still pending",
        }.get(code, f"device authorization failed: {code}")
        super().__init__(message)


class NotLoggedInError(AuthError):
    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            401,
            message
            or "not logged in: run `airigs login` (or set AIRIGS_TOKEN for headless use)",
        )


def _post_form(url: str, form: dict[str, Any]) -> tuple[int, Any]:
    response: Response = send("POST", url, form=form, timeout=30.0)
    return response.status, response.data


def _token_url(airigs_url: str) -> str:
    return airigs_url.rstrip("/") + "/oauth/device_token"


def start_device_authorization(
    airigs_url: str,
    *,
    client_id: str = CLIENT_ID,
    scope: str = SCOPE,
    transport: Callable[[str, dict[str, Any]], tuple[int, Any]] = _post_form,
) -> dict[str, Any]:
    """Ask AIRigs for a device + user code (RFC 8628 section 3.2)."""
    assert_secure_url(airigs_url, what="AIRigs")
    url = airigs_url.rstrip("/") + "/oauth/device_authorization"
    status, payload = transport(url, {"client_id": client_id, "scope": scope})
    if status != 200 or not isinstance(payload, dict):
        detail = ""
        if isinstance(payload, dict):
            detail = str(payload.get("error_description") or payload.get("error") or "")
        raise AirigsError(f"device authorization failed (HTTP {status}){': ' + detail if detail else ''}")
    missing = [key for key in ("device_code", "user_code", "verification_uri", "expires_in", "interval") if key not in payload]
    if missing:
        raise AirigsError(f"device authorization response is missing {', '.join(missing)}")
    if not isinstance(payload.get("interval"), (int, float)) or payload["interval"] <= 0:
        payload["interval"] = 5
    if not isinstance(payload.get("expires_in"), (int, float)) or payload["expires_in"] <= 0:
        raise AirigsError("device authorization response has an invalid expires_in")
    return payload


def poll_device_token(
    airigs_url: str,
    device: dict[str, Any],
    *,
    client_id: str = CLIENT_ID,
    transport: Callable[[str, dict[str, Any]], tuple[int, Any]] = _post_form,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    on_pending: Callable[[dict[str, Any], float], None] | None = None,
) -> dict[str, Any]:
    """Poll the token endpoint until it yields a token or fails.

    Implements ``authorization_pending`` (wait ``interval``), ``slow_down``
    (interval += 5s), ``expired_token`` and ``access_denied`` exactly as RFC 8628
    section 3.5 requires.
    """
    device_code = device.get("device_code")
    if not device_code:
        raise DeviceFlowError("invalid_response", "device authorization has no device_code")
    interval = float(device.get("interval") or 5)
    expires_in = float(device.get("expires_in") or 0)
    deadline = clock() + expires_in
    url = _token_url(airigs_url)

    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            raise DeviceFlowError("expired_token")
        status, payload = transport(
            url,
            {
                "grant_type": DEVICE_CODE_GRANT,
                "device_code": device_code,
                "client_id": client_id,
            },
        )
        if status == 200 and isinstance(payload, dict) and payload.get("access_token"):
            return payload

        error = payload.get("error") if isinstance(payload, dict) else None
        description = payload.get("error_description") if isinstance(payload, dict) else None
        if error == "authorization_pending":
            if on_pending:
                on_pending(payload, interval)
            wait = min(interval, max(0.0, deadline - clock()))
            if wait <= 0:
                raise DeviceFlowError("expired_token")
            sleep(wait)
        elif error == "slow_down":
            interval += SLOW_DOWN_INCREMENT
            if on_pending:
                on_pending(payload, interval)
            wait = min(interval, max(0.0, deadline - clock()))
            if wait <= 0:
                raise DeviceFlowError("expired_token")
            sleep(wait)
        elif error == "expired_token":
            raise DeviceFlowError("expired_token", description or "")
        elif error == "access_denied":
            raise DeviceFlowError("access_denied", description or "")
        elif error:
            raise DeviceFlowError(str(error), str(description or ""))
        else:
            raise DeviceFlowError(
                "invalid_response",
                f"unexpected token endpoint response (HTTP {status})",
            )


def fetch_userinfo(airigs_url: str, token: str, *, timeout: float = 30.0) -> dict[str, Any]:
    assert_secure_url(airigs_url, what="AIRigs")
    client = ApiClient(airigs_url, token, timeout=timeout)
    data = client.get("/oauth/userinfo", action="userinfo")
    if not isinstance(data, dict) or "uid" not in data:
        raise AirigsError("userinfo response is missing uid")
    return data


def refresh_access_token(settings: Settings, *, verbose: bool = False) -> str:
    """Use the stored refresh token. Single-flight within the process."""
    if not settings.refresh_token:
        raise NotLoggedInError(
            "access token expired and there is no refresh token; run `airigs login` again"
        )
    assert_secure_url(settings.airigs_url, what="AIRigs")
    check_token_origin(settings)
    with _REFRESH_LOCK:
        stored = load_config()
        current = stored.get("token")
        if (
            isinstance(current, str)
            and current
            and current != settings.airigs_token
            and not token_expired(stored.get("expires_at") if isinstance(stored.get("expires_at"), (int, float)) else None)
        ):
            # Another thread refreshed while this one waited for the lock.
            return current
        refresh_token = stored.get("refresh_token") if isinstance(stored.get("refresh_token"), str) else settings.refresh_token
        client = ApiClient(
            settings.airigs_url,
            verbose=verbose,
            secrets={refresh_token, settings.airigs_token or ""} - {""},
        )
        payload = client.request_json(
            "POST",
            "/oauth/token",
            form={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CLIENT_ID,
            },
            token=None,
            action="token refresh",
        )
        if not isinstance(payload, dict) or not payload.get("access_token"):
            raise AuthError(401, "token refresh returned no access token")
        _persist_tokens(payload, previous=stored)
        return payload["access_token"]


def ensure_token(settings: Settings | None = None, *, verbose: bool = False) -> str:
    """Return a usable access token, refreshing when it is about to expire."""
    settings = settings or resolve_settings()
    if not settings.airigs_token:
        raise NotLoggedInError()
    assert_secure_url(settings.airigs_url, what="AIRigs")
    if settings.token_source == "env:AIRIGS_TOKEN":
        return settings.airigs_token
    check_token_origin(settings)
    if token_expired(settings.expires_at):
        return refresh_access_token(settings, verbose=verbose)
    return settings.airigs_token


def login_with_device_flow(
    airigs_url: str,
    *,
    no_browser: bool = False,
    printer: Callable[[str], None] = print,
    opener: Callable[[str], bool] | None = None,
    transport: Callable[[str, dict[str, Any]], tuple[int, Any]] = _post_form,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Full interactive login. Returns the token payload."""
    assert_secure_url(airigs_url, what="AIRigs")
    device = start_device_authorization(airigs_url, transport=transport)
    verification_uri = device.get("verification_uri")
    complete = device.get("verification_uri_complete") or verification_uri
    printer(f"Open {verification_uri} and enter the code: {device['user_code']}")
    if complete and complete != verification_uri:
        printer(f"Or open directly: {complete}")
    if not no_browser:
        open_url = opener
        if open_url is None:
            import webbrowser

            open_url = webbrowser.open
        try:
            opened = open_url(str(complete))
        except Exception:  # a missing browser is never fatal
            opened = False
        if not opened:
            printer("Could not open a browser; open the URL above manually.")
    printer("Waiting for approval...")
    payload = poll_device_token(
        airigs_url,
        device,
        transport=transport,
        sleep=sleep,
        clock=clock,
        on_pending=lambda _p, interval: printer(f"Still waiting (next check in {interval:g}s)..."),
    )
    _persist_tokens(payload, airigs_url=airigs_url)
    return payload


def login_with_token(airigs_url: str, token: str) -> dict[str, Any]:
    """Headless login: verify the token against userinfo, then store it."""
    token = token.strip()
    if not token:
        raise AirigsError("a token is required on stdin")
    assert_secure_url(airigs_url, what="AIRigs")
    info = fetch_userinfo(airigs_url, token)
    previous = load_config()
    previous.pop("refresh_token", None)
    previous.pop("expires_at", None)
    _persist_tokens(
        {"access_token": token, "token_type": "Bearer", "scope": SCOPE},
        airigs_url=airigs_url,
        previous=previous,
    )
    return info


def logout(settings: Settings | None = None, *, verbose: bool = False) -> dict[str, Any]:
    """Revoke the refresh token when possible, then delete local state.

    The provider's revocation is best-effort: local credentials are always
    removed, and the result says which of the two happened.
    """
    settings = settings or resolve_settings()
    stored = load_config()
    refresh_token = stored.get("refresh_token") if isinstance(stored.get("refresh_token"), str) else None
    result: dict[str, Any] = {
        "hadRefreshToken": bool(refresh_token),
        "revoked": None,
        "status": None,
    }
    if refresh_token:
        current_origin = url_origin(settings.airigs_url)
        if settings.token_origin and settings.token_origin != current_origin:
            # Never hand a refresh token to a different origin.
            result["revoked"] = False
            result["error"] = (
                f"not sent: the refresh token belongs to {settings.token_origin}, "
                f"not {current_origin}"
            )
        else:
            try:
                assert_secure_url(settings.airigs_url, what="AIRigs")
                response: Response = send(
                    "POST",
                    join_url(settings.airigs_url, "/oauth/revoke"),
                    form={"token": refresh_token, "client_id": CLIENT_ID},
                    timeout=15.0,
                )
                result["status"] = response.status
                result["revoked"] = response.ok
                if not response.ok:
                    result["error"] = f"revocation returned HTTP {response.status}"
            except AirigsError as exc:
                result["revoked"] = False
                result["error"] = str(exc)
    clear_tokens()
    result["removedLocalState"] = True
    return result


def config_location() -> str:
    return str(config_dir() / "config.json")


def describe_token(settings: Settings) -> dict[str, Any]:
    from .config import token_ttl

    current_origin = url_origin(settings.airigs_url)
    if not settings.airigs_token:
        state = "missing"
    elif settings.token_source == "env:AIRIGS_TOKEN":
        state = "env"
    elif settings.token_origin and settings.token_origin != current_origin:
        state = "wrong-origin"
    elif token_expired(settings.expires_at):
        state = "expired"
    else:
        state = "valid"
    return {
        "state": state,
        "source": settings.token_source,
        "expires_in": token_ttl(settings.expires_at),
        "has_refresh_token": bool(settings.refresh_token),
        "config_path": config_location(),
        "tokenOrigin": settings.token_origin,
        "currentOrigin": current_origin,
        "token": redact(settings.airigs_token, {settings.airigs_token}) if settings.airigs_token else None,
    }
