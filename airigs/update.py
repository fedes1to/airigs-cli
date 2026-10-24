"""Update notices: check PyPI once a day and say how to upgrade.

The check is deliberately boring: it runs after a command has done its work,
only for interactive terminals, at most once per day, with a short timeout, and
it never raises. Offline hosts therefore pay nothing and see nothing. Set
``AIRIGS_NO_UPDATE_CHECK=1`` to opt out entirely.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, TextIO

from . import __version__, config, http

PYPI_URL = "https://pypi.org/pypi/airigs/json"
CACHE_NAME = "update-check.json"
CHECK_INTERVAL_S = 24 * 60 * 60
REQUEST_TIMEOUT_S = 2.0
DISABLE_ENV = "AIRIGS_NO_UPDATE_CHECK"


def cache_path() -> Path:
    return config.config_dir() / CACHE_NAME


def _version_tuple(value: str) -> tuple[int, ...] | None:
    """A comparable numeric version, or ``None`` when it is not version-like."""
    parts = re.findall(r"\d+", value.split("+", 1)[0].split("-", 1)[0])
    if not parts:
        return None
    return tuple(int(part) for part in parts)


def is_newer(latest: str, current: str = __version__) -> bool:
    newer = _version_tuple(latest)
    ours = _version_tuple(current)
    if newer is None or ours is None:
        return False
    width = max(len(newer), len(ours))
    return newer + (0,) * (width - len(newer)) > ours + (0,) * (width - len(ours))


def _fetch_latest() -> str | None:
    response = http.send("GET", PYPI_URL, timeout=REQUEST_TIMEOUT_S)
    if not response.ok or not isinstance(response.data, dict):
        return None
    info = response.data.get("info")
    version = info.get("version") if isinstance(info, dict) else None
    return version if isinstance(version, str) and version else None


def _read_cache() -> dict[str, Any]:
    try:
        data = json.loads(cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_cache(latest: str | None, *, now: float | None = None) -> None:
    directory = config.config_dir()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        os.chmod(directory, config.DIR_MODE)
    except OSError:
        pass
    payload = json.dumps({"checkedAt": int(time.time() if now is None else now), "latest": latest}) + "\n"
    try:
        fd, tmp_name = tempfile.mkstemp(prefix=".update-", dir=str(directory))
        try:
            os.fchmod(fd, config.CONFIG_MODE)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(tmp_name, cache_path())
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    except OSError:
        pass


def latest_known(*, now: float | None = None, force: bool = False) -> str | None:
    """Return the latest published version, refreshing the daily cache if stale.

    Network failures are cached as "no answer" for the same interval, so an
    offline machine does not retry on every command.
    """
    current_time = time.time() if now is None else now
    cache = _read_cache()
    checked_at = cache.get("checkedAt")
    if (
        not force
        and isinstance(checked_at, (int, float))
        and current_time - checked_at < CHECK_INTERVAL_S
    ):
        cached = cache.get("latest")
        return cached if isinstance(cached, str) else None
    latest = _fetch_latest()
    _write_cache(latest, now=current_time)
    return latest


def upgrade_command() -> str:
    """Best guess at how this installation can upgrade itself."""
    executable = sys.executable.replace("\\", "/").lower()
    if "pipx" in executable:
        return "pipx upgrade airigs"
    if "/uv/" in executable or "uv-tools" in executable:
        return "uv tool upgrade airigs"
    return "python -m pip install --upgrade airigs"


def maybe_notify(stream: TextIO | None = None, *, now: float | None = None) -> bool:
    """Print the update line when there is one. Returns whether it printed."""
    target: TextIO = sys.stderr if stream is None else stream
    if os.environ.get(DISABLE_ENV):
        return False
    try:
        if not target.isatty():
            return False
    except (AttributeError, ValueError):
        return False
    try:
        latest = latest_known(now=now)
        if latest is None or not is_newer(latest):
            return False
        print(
            f"airigs {latest} is available (installed: {__version__}); "
            f"update with: {upgrade_command()}",
            file=target,
            flush=True,
        )
        return True
    except Exception:
        # An update notice must never break the command that just succeeded.
        return False
