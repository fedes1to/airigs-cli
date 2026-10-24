"""Configuration discovery, secret storage and redaction.

The config file is ``$XDG_CONFIG_HOME/airigs/config.json`` (default
``~/.config/airigs/config.json``) and is always written mode ``0600`` inside a
``0700`` directory. Environment variables override the stored values at read
time and are never written back.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_AIRIGS_URL = "http://localhost:4567"
DEFAULT_RECIPES_URL = "http://localhost:4568"

#: The OAuth client the CLI authenticates as. AIRigs generates the id when the
#: administrator registers the client (``aic_...``), so it is deployment-specific
#: and cannot be a compile-time constant: override it with ``AIRIGS_CLIENT_ID``.
CLIENT_ID = os.environ.get("AIRIGS_CLIENT_ID") or "airigs-cli"
SCOPE = "user:read rig:read rig:write recipe:read recipe:write"

#: Refresh this many seconds before the access token actually expires.
EXPIRY_SKEW_S = 60

CONFIG_MODE = 0o600
DIR_MODE = 0o700

SECRET_KEYS = (
    "token",
    "access_token",
    "refresh_token",
    "client_secret",
    "password",
    "authorization",
)


class ConfigError(Exception):
    """The configuration cannot be used (corrupt, unreadable, unsafe)."""


def url_origin(url: str) -> str:
    """``scheme://host[:port]`` with default ports omitted, lower-cased."""
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    host = (parsed.hostname or "").lower()
    port = parsed.port
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        return f"{scheme}://{host}:{port}"
    return f"{scheme}://{host}"


def config_dir() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "airigs"


def config_path() -> Path:
    return config_dir() / "config.json"


def load_config() -> dict[str, Any]:
    """Read the config file. A missing file is an empty config, not an error."""
    path = config_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a JSON object")
    return data


def save_config(config: dict[str, Any]) -> Path:
    """Atomically write the config with mode 0600 (directory 0700)."""
    directory = config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(directory, DIR_MODE)
    except OSError:
        pass  # a shared XDG dir may be owned elsewhere; the file mode is the guarantee
    path = directory / "config.json"
    payload = json.dumps(config, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=".config-", dir=str(directory))
    try:
        os.fchmod(fd, CONFIG_MODE)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    os.chmod(path, CONFIG_MODE)
    return path


def config_mode() -> int | None:
    try:
        return config_path().stat().st_mode & 0o777
    except OSError:
        return None


@dataclass(frozen=True)
class Settings:
    """Effective connection settings after env/config resolution."""

    airigs_url: str
    recipes_url: str
    airigs_token: str | None
    recipes_token: str | None
    token_source: str
    expires_at: float | None
    refresh_token: str | None
    scope: str | None
    token_origin: str | None = None

    @property
    def token(self) -> str | None:
        return self.airigs_token


def _clean_url(value: str | None, default: str) -> str:
    if not value:
        return default
    return value.rstrip("/")


def resolve_settings(
    *,
    airigs_url: str | None = None,
    recipes_url: str | None = None,
    config: dict[str, Any] | None = None,
) -> Settings:
    """Environment overrides config, explicit CLI flags override both."""
    cfg = load_config() if config is None else config
    env_airigs = os.environ.get("AIRIGS_URL")
    env_recipes = os.environ.get("RECIPES_URL")
    env_token = os.environ.get("AIRIGS_TOKEN")
    env_recipes_token = os.environ.get("RECIPES_TOKEN")

    cfg_token = cfg.get("token") if isinstance(cfg.get("token"), str) else None
    if env_token:
        token, source = env_token, "env:AIRIGS_TOKEN"
    elif cfg_token:
        token, source = cfg_token, "config"
    else:
        token, source = None, "none"

    recipes_token = env_recipes_token or token
    if not recipes_token and isinstance(cfg.get("recipes_token"), str):
        recipes_token = cfg["recipes_token"]

    expires_at = cfg.get("expires_at")
    if not isinstance(expires_at, (int, float)):
        expires_at = None

    refresh = cfg.get("refresh_token")
    if env_token:
        refresh = None  # a headless token has no refresh chain attached

    token_origin = cfg.get("tokenOrigin") if isinstance(cfg.get("tokenOrigin"), str) else None
    if not token_origin and cfg_token and cfg.get("airigs_url"):
        token_origin = url_origin(str(cfg["airigs_url"]))

    return Settings(
        airigs_url=_clean_url(airigs_url or env_airigs or cfg.get("airigs_url"), DEFAULT_AIRIGS_URL),
        recipes_url=_clean_url(recipes_url or env_recipes or cfg.get("recipes_url"), DEFAULT_RECIPES_URL),
        airigs_token=token,
        recipes_token=recipes_token,
        token_source=source,
        expires_at=float(expires_at) if expires_at is not None else None,
        refresh_token=refresh if isinstance(refresh, str) and refresh else None,
        scope=cfg.get("scope") if isinstance(cfg.get("scope"), str) else None,
        token_origin=token_origin,
    )


def token_expired(expires_at: float | None, *, skew: int = EXPIRY_SKEW_S) -> bool:
    return expires_at is not None and time.time() >= expires_at - skew


def token_ttl(expires_at: float | None) -> int | None:
    if expires_at is None:
        return None
    return int(expires_at - time.time())


def save_tokens(
    payload: dict[str, Any],
    *,
    airigs_url: str | None = None,
    recipes_url: str | None = None,
    previous: dict[str, Any] | None = None,
) -> Path:
    """Persist an OAuth token response (and reuse its refresh token when absent)."""
    cfg = dict(load_config() if previous is None else previous)
    access = payload.get("access_token")
    if not isinstance(access, str) or not access:
        raise ConfigError("token response did not contain an access_token")
    cfg["token"] = access
    cfg["token_type"] = payload.get("token_type") or "Bearer"
    if isinstance(payload.get("refresh_token"), str) and payload["refresh_token"]:
        cfg["refresh_token"] = payload["refresh_token"]
    expires_in = payload.get("expires_in")
    if isinstance(expires_in, (int, float)):
        cfg["expires_at"] = int(time.time() + float(expires_in))
        cfg["expires_in"] = int(expires_in)
    else:
        cfg.pop("expires_at", None)
        cfg.pop("expires_in", None)
    if isinstance(payload.get("scope"), str):
        cfg["scope"] = payload["scope"]
    if airigs_url:
        cfg["airigs_url"] = airigs_url.rstrip("/")
        cfg["tokenOrigin"] = url_origin(airigs_url)
    elif cfg.get("airigs_url"):
        cfg.setdefault("tokenOrigin", url_origin(str(cfg["airigs_url"])))
    if recipes_url:
        cfg["recipes_url"] = recipes_url.rstrip("/")
    return save_config(cfg)


def clear_tokens(*, keep_urls: bool = True) -> Path:
    cfg = load_config() if keep_urls else {}
    for key in (
        "token",
        "refresh_token",
        "expires_at",
        "expires_in",
        "scope",
        "token_type",
        "tokenOrigin",
        "recipes_token",
    ):
        cfg.pop(key, None)
    return save_config(cfg)


# --------------------------------------------------------------------------
# Redaction

_BEARER_RE = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/\-=]+")
_FORM_RE = re.compile(r"(?i)\b(access_token|refresh_token|device_code|client_secret)=([^&\s\"']+)")


def collect_secrets(*configs: dict[str, Any] | None) -> set[str]:
    secrets: set[str] = set()
    for cfg in configs:
        if not cfg:
            continue
        for key in SECRET_KEYS:
            value = cfg.get(key)
            if isinstance(value, str) and len(value) >= 4:
                secrets.add(value)
    for name in ("AIRIGS_TOKEN", "RECIPES_TOKEN"):
        value = os.environ.get(name)
        if value:
            secrets.add(value)
    return secrets


def redact(text: str, secrets: set[str] | None = None) -> str:
    """Replace known secrets and bearer/form token patterns with ``***``.

    Never returns a known token even when it is embedded in a longer string.
    """
    if not isinstance(text, str):
        text = str(text)
    for value in sorted(secrets or (), key=len, reverse=True):
        if value:
            text = text.replace(value, "***")
    text = _BEARER_RE.sub(r"\1***", text)
    text = _FORM_RE.sub(r"\1=***", text)
    return text


def redacted_config(config: dict[str, Any]) -> dict[str, Any]:
    """A copy of the config safe to print."""
    out: dict[str, Any] = {}
    for key, value in config.items():
        if key in SECRET_KEYS and value:
            out[key] = "***"
        else:
            out[key] = value
    return out
