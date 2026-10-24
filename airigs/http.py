"""A deliberately small HTTP layer over :mod:`urllib`.

No third-party HTTP dependency: the CLI needs JSON, bearer auth, form posts, and
clear error mapping. Everything else is a wrapper around the standard library.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from . import __version__
from .config import redact

USER_AGENT = f"airigs-cli/{__version__}"
_UNSET = object()

LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect.

    urllib forwards ``Authorization`` across hosts when it follows a 302. An
    authenticated API call has no legitimate redirect, so refusing them is both
    simpler and safer than trying to decide which hop is same-origin.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


_OPENER = urllib.request.build_opener(_RefuseRedirects)


def open_no_redirect(request: urllib.request.Request, *, timeout: float):
    """Open a request with redirects refused.

    Exposed so other modules (the recipe benchmark client) cannot accidentally
    reintroduce the urllib default of following a 302 with its headers intact.
    """
    return _OPENER.open(request, timeout=timeout)


class AirigsError(Exception):
    """Base class for every CLI-facing failure. ``exit_code`` drives the shell status."""

    exit_code = 1


class NetworkError(AirigsError):
    """The request never produced an HTTP response."""


class HttpError(AirigsError):
    def __init__(
        self,
        status: int,
        message: str,
        *,
        url: str | None = None,
        errors: list[dict[str, Any]] | None = None,
        body: Any = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.url = url
        self.errors = errors or []
        self.body = body


class AuthError(HttpError):
    exit_code = 3


class ForbiddenError(HttpError):
    exit_code = 3


class NotFoundError(HttpError):
    exit_code = 4


class ConflictError(HttpError):
    exit_code = 4


class ValidationError(HttpError):
    exit_code = 5


class RateLimitError(HttpError):
    exit_code = 6

    def __init__(self, *args: Any, retry_after: float | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.retry_after = retry_after


@dataclass
class Response:
    status: int
    data: Any
    text: str
    headers: dict[str, str]
    url: str

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


def join_url(base: str, path: str) -> str:
    """Join a configured base URL with a *relative* path.

    Absolute or scheme-relative paths are rejected: a server-supplied URL must
    never be able to move a bearer token to another origin.
    """
    if not path:
        return base
    if "://" in path or path.startswith("//"):
        raise AirigsError(f"refusing to request an absolute URL from a relative path: {path!r}")
    return base.rstrip("/") + "/" + path.lstrip("/")


def is_loopback_url(url: str) -> bool:
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return host in LOOPBACK_HOSTS or host.endswith(".localhost")


def assert_secure_url(url: str, *, what: str = "service") -> None:
    """Refuse to send a bearer token over plaintext HTTP to a remote host."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise AirigsError(f"{what} URL must be an absolute http(s) URL, got {url!r}")
    if parsed.scheme != "https" and not is_loopback_url(url):
        raise AirigsError(
            f"refusing to send a bearer token to {url} over plaintext HTTP; "
            "use an https URL, or localhost/127.0.0.1 for development"
        )


def _decode(resp_body: bytes, charset: str | None) -> str:
    try:
        return resp_body.decode(charset or "utf-8", errors="replace")
    except (LookupError, AttributeError):
        return resp_body.decode("utf-8", errors="replace")


def _parse_body(text: str) -> Any:
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def send(
    method: str,
    url: str,
    *,
    json_body: Any = None,
    form: dict[str, Any] | None = None,
    token: str | None = None,
    timeout: float = 30.0,
    headers: dict[str, str] | None = None,
    accept: str = "application/json",
) -> Response:
    """Perform one request. HTTP error statuses are returned, not raised."""
    data: bytes | None = None
    request_headers = {"Accept": accept, "User-Agent": USER_AGENT}
    if headers:
        request_headers.update(headers)
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    elif form is not None:
        data = urllib.parse.urlencode(form).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
    if token:
        request_headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url, data=data, method=method.upper(), headers=request_headers)
    try:
        with _OPENER.open(request, timeout=timeout) as handle:
            raw = handle.read()
            status = getattr(handle, "status", 200) or 200
            response_headers = {k.lower(): v for k, v in handle.headers.items()}
            charset = handle.headers.get_content_charset()
    except urllib.error.HTTPError as exc:
        raw = exc.read() if exc.fp is not None else b""
        status = exc.code
        response_headers = {k.lower(): v for k, v in (exc.headers or {}).items()}
        charset = exc.headers.get_content_charset() if exc.headers else None
    except urllib.error.URLError as exc:
        raise NetworkError(f"cannot reach {url}: {exc.reason}") from exc
    except (socket.timeout, TimeoutError) as exc:
        raise NetworkError(f"timed out after {timeout:g}s talking to {url}") from exc
    except OSError as exc:
        raise NetworkError(f"cannot reach {url}: {exc}") from exc

    text = _decode(raw or b"", charset)
    return Response(status=status, data=_parse_body(text), text=text, headers=response_headers, url=url)


def _describe(data: Any, text: str) -> str:
    if isinstance(data, dict):
        errors = data.get("errors")
        if isinstance(errors, list) and errors:
            parts = []
            for item in errors:
                if isinstance(item, dict):
                    path = item.get("path") or item.get("field") or ""
                    message = item.get("message") or item.get("msg") or ""
                    parts.append(f"{path}: {message}" if path else str(message))
                else:
                    parts.append(str(item))
            return "; ".join(part for part in parts if part)
        for key in ("error_description", "message", "error", "detail"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
    snippet = " ".join((text or "").split())
    return snippet[:400] if snippet else "request failed"


def unwrap_envelope(data: Any) -> Any:
    """Unwrap NodeBB's ``{status, response}`` API envelope.

    The AIRigs rig API is a NodeBB v3 API, so every success body is wrapped;
    the Recipes API is plain Express JSON and is returned untouched. Only a
    body that really is the envelope is unwrapped, so an error document
    (``{status, errors}`` with no ``response``) is never mistaken for one.
    """
    if (
        isinstance(data, dict)
        and isinstance(data.get("status"), dict)
        and "response" in data
    ):
        return data["response"]
    return data


def raise_for_response(response: Response, *, action: str = "") -> None:
    """Map an error status onto the exception hierarchy."""
    if response.ok:
        return
    message = _describe(response.data, response.text)
    if message == "request failed":
        message = f"HTTP {response.status} from {response.url}"
    if action:
        message = f"{action}: {message}"
    errors = response.data.get("errors") if isinstance(response.data, dict) else None
    if not isinstance(errors, list):
        errors = None
    kwargs: dict[str, Any] = {
        "url": response.url,
        "errors": errors,
        "body": response.data,
    }
    status = response.status
    if status == 401:
        raise AuthError(status, message, **kwargs)
    if status == 403:
        raise ForbiddenError(status, message, **kwargs)
    if status == 404:
        raise NotFoundError(status, message, **kwargs)
    if status == 409:
        raise ConflictError(status, message, **kwargs)
    if status == 422:
        raise ValidationError(status, message, **kwargs)
    if status == 429:
        retry_after: float | None = None
        raw = response.headers.get("retry-after")
        if raw:
            try:
                retry_after = float(raw)
            except ValueError:
                retry_after = None
        raise RateLimitError(status, message, retry_after=retry_after, **kwargs)
    raise HttpError(status, message, **kwargs)


class ApiClient:
    """A bearer-authenticated JSON client for one service base URL."""

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        *,
        timeout: float = 30.0,
        verbose: bool = False,
        log: Callable[[str], None] | None = None,
        secrets: set[str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.verbose = verbose
        self._log = log
        self.secrets = secrets or set()

    # -- plumbing ---------------------------------------------------------
    def _verbose(self, message: str) -> None:
        if self.verbose and self._log:
            self._log(redact(message, self.secrets))

    def _send(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        form: dict[str, Any] | None = None,
        token: Any = _UNSET,
        accept: str = "application/json",
    ) -> Response:
        url = join_url(self.base_url, path)
        used_token = self.token if token is _UNSET else token
        self._verbose(f"-> {method.upper()} {url}")
        response = send(
            method,
            url,
            json_body=json_body,
            form=form,
            token=used_token,
            timeout=self.timeout,
            accept=accept,
            headers={"X-Airigs-Client": USER_AGENT},
        )
        self._verbose(f"<- {response.status} {url}")
        return response

    def request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        form: dict[str, Any] | None = None,
        token: Any = _UNSET,
        action: str = "",
        expect: Any = None,
    ) -> Any:
        response = self._send(method, path, json_body=json_body, form=form, token=token)
        raise_for_response(response, action=action)
        data = unwrap_envelope(response.data)
        if expect is not None and not isinstance(data, expect):
            raise HttpError(
                response.status,
                f"{method.upper()} {path}: unexpected response shape "
                f"(expected {getattr(expect, '__name__', expect)})",
                url=response.url,
                body=data,
            )
        return data

    def request_text(self, method: str, path: str, *, action: str = "") -> tuple[str, Response]:
        response = self._send(method, path, accept="text/plain, text/x-shellscript, */*")
        raise_for_response(response, action=action)
        return response.text, response

    # -- verbs ------------------------------------------------------------
    def get(self, path: str, *, action: str = "", expect: Any = dict) -> Any:
        return self.request_json("GET", path, action=action, expect=expect)

    def post(self, path: str, payload: Any, *, action: str = "", expect: Any = dict) -> Any:
        return self.request_json("POST", path, json_body=payload, action=action, expect=expect)

    def put(self, path: str, payload: Any, *, action: str = "", expect: Any = dict) -> Any:
        return self.request_json("PUT", path, json_body=payload, action=action, expect=expect)

    def patch(self, path: str, payload: Any, *, action: str = "", expect: Any = dict) -> Any:
        return self.request_json("PATCH", path, json_body=payload, action=action, expect=expect)

    def delete(self, path: str, *, action: str = "", expect: Any = (dict, type(None))) -> Any:
        return self.request_json("DELETE", path, action=action, expect=expect)
