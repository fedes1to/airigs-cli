from __future__ import annotations

import io
import json

from airigs import update


class FakeStream(io.StringIO):
    def __init__(self, *, tty: bool = True) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def test_version_comparison_handles_padding_and_metadata():
    assert update.is_newer("0.2.0", "0.1.0")
    assert update.is_newer("0.1.1", "0.1")
    assert not update.is_newer("0.1.0", "0.1.0")
    assert not update.is_newer("0.0.9", "0.1.0")
    assert not update.is_newer("not-a-version", "0.1.0")


def test_fetch_latest_reads_the_repository_pyproject(monkeypatch):
    class Response:
        ok = True
        text = '[project]\nname = "airigs"\nversion = "1.2.3"\n'

    monkeypatch.setattr(update.http, "send", lambda *args, **kwargs: Response())
    assert update._fetch_latest() == "1.2.3"


def test_upgrade_command_uses_the_source_pipx_recorded(tmp_path, monkeypatch):
    venv = tmp_path / "pipx" / "venvs" / "airigs"
    (venv / "bin").mkdir(parents=True)
    (venv / "pipx_metadata.json").write_text(
        json.dumps({"main_package": {"package_or_url": update.REPO_SOURCE}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(update.sys, "executable", str(venv / "bin" / "python"))
    assert update.upgrade_command() == f"pipx install --force '{update.REPO_SOURCE}'"


def test_upgrade_command_falls_back_to_the_repository(tmp_path, monkeypatch):
    monkeypatch.setattr(update.sys, "executable", str(tmp_path / "python"))
    assert update.upgrade_command() == f"python -m pip install --upgrade '{update.REPO_SOURCE}'"


def test_latest_known_caches_the_answer_for_a_day(isolated_config, monkeypatch):
    calls = []

    def fake_fetch():
        calls.append(1)
        return "0.2.0"

    monkeypatch.setattr(update, "_fetch_latest", fake_fetch)
    now = 1_000_000.0
    assert update.latest_known(now=now) == "0.2.0"
    assert update.latest_known(now=now + 60) == "0.2.0"
    assert calls == [1]
    assert update.latest_known(now=now + update.CHECK_INTERVAL_S + 1) == "0.2.0"
    assert calls == [1, 1]


def test_maybe_notify_prints_the_upgrade_command(isolated_config, monkeypatch):
    monkeypatch.delenv(update.DISABLE_ENV, raising=False)
    monkeypatch.setattr(update, "_fetch_latest", lambda: "9.9.9")
    stream = FakeStream()
    assert update.maybe_notify(stream) is True
    line = stream.getvalue()
    assert "airigs 9.9.9 is available" in line
    assert "update with:" in line


def test_maybe_notify_is_silent_when_not_a_tty(isolated_config, monkeypatch):
    monkeypatch.delenv(update.DISABLE_ENV, raising=False)
    monkeypatch.setattr(update, "_fetch_latest", lambda: "9.9.9")
    stream = FakeStream(tty=False)
    assert update.maybe_notify(stream) is False
    assert stream.getvalue() == ""


def test_maybe_notify_is_silent_when_disabled(isolated_config, monkeypatch):
    monkeypatch.setenv(update.DISABLE_ENV, "1")
    monkeypatch.setattr(update, "_fetch_latest", lambda: "9.9.9")
    stream = FakeStream()
    assert update.maybe_notify(stream) is False
    assert stream.getvalue() == ""


def test_maybe_notify_never_raises_on_network_failure(isolated_config, monkeypatch):
    monkeypatch.delenv(update.DISABLE_ENV, raising=False)

    def boom():
        raise RuntimeError("offline")

    monkeypatch.setattr(update, "_fetch_latest", boom)
    stream = FakeStream()
    assert update.maybe_notify(stream) is False
    assert stream.getvalue() == ""
