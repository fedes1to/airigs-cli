from __future__ import annotations

import os
import time

import pytest

from airigs import config


def test_missing_config_is_empty(isolated_config):
    assert config.load_config() == {}
    assert config.config_mode() is None


def test_save_and_load_with_0600(isolated_config):
    path = config.save_config({"token": "abc", "airigs_url": "http://x"})
    assert path == config.config_path()
    assert path.exists()
    assert config.config_mode() == 0o600
    assert config.config_dir().stat().st_mode & 0o777 == 0o700
    assert config.load_config()["token"] == "abc"


def test_corrupt_config_raises(isolated_config):
    config.config_dir().mkdir(parents=True)
    config.config_path().write_text("{not json", encoding="utf-8")
    with pytest.raises(config.ConfigError):
        config.load_config()


def test_defaults_are_the_public_deployment(isolated_config):
    """A fresh install with no env or config must target production."""
    settings = config.resolve_settings()
    assert settings.airigs_url == "https://airigs.net"
    assert settings.recipes_url == "https://recipes.airigs.net"
    assert settings.token_origin is None
    assert "localhost" not in settings.airigs_url + settings.recipes_url


def test_env_overrides_config(isolated_config, monkeypatch):
    config.save_config({"airigs_url": "http://config-airigs", "token": "config-token"})
    monkeypatch.setenv("AIRIGS_URL", "http://env-airigs/")
    monkeypatch.setenv("RECIPES_URL", "http://env-recipes/")
    monkeypatch.setenv("AIRIGS_TOKEN", "env-token")
    settings = config.resolve_settings()
    assert settings.airigs_url == "http://env-airigs"
    assert settings.recipes_url == "http://env-recipes"
    assert settings.airigs_token == "env-token"
    assert settings.token_source == "env:AIRIGS_TOKEN"
    assert settings.recipes_token == "env-token"
    assert settings.refresh_token is None

    # CLI flags win over the environment.
    settings = config.resolve_settings(airigs_url="http://flag/")
    assert settings.airigs_url == "http://flag"


def test_recipes_token_falls_back_to_airigs_token(isolated_config, monkeypatch):
    config.save_config({"token": "shared"})
    settings = config.resolve_settings(config=config.load_config())
    assert settings.recipes_token == "shared"
    monkeypatch.setenv("RECIPES_TOKEN", "recipes-only")
    settings = config.resolve_settings(config=config.load_config())
    assert settings.recipes_token == "recipes-only"
    assert settings.airigs_token == "shared"


def test_redaction_removes_tokens_everywhere():
    secrets = {"tok_secret_123456", "refresh_secret_99"}
    text = (
        "Authorization: Bearer tok_secret_123456\n"
        "POST /oauth/token?access_token=tok_secret_123456&refresh_token=refresh_secret_99"
    )
    redacted = config.redact(text, secrets)
    assert "tok_secret" not in redacted
    assert "refresh_secret" not in redacted
    assert "Bearer ***" in redacted
    assert "access_token=***" in redacted


def test_collect_secrets_includes_env(isolated_config, monkeypatch):
    config.save_config({"token": "configtoken", "refresh_token": "refreshvalue"})
    monkeypatch.setenv("RECIPES_TOKEN", "envrecipes")
    secrets = config.collect_secrets(config.load_config())
    assert {"configtoken", "refreshvalue", "envrecipes"} <= secrets


def test_save_tokens_preserves_refresh_token(isolated_config):
    first = config.save_tokens(
        {
            "access_token": "access-1",
            "refresh_token": "refresh-1",
            "expires_in": 3600,
            "scope": "user:read",
        },
        airigs_url="http://localhost:4567/",
    )
    assert first.exists()
    assert config.config_mode() == 0o600
    original = config.load_config()
    assert original["expires_at"] > time.time()
    assert original["airigs_url"] == "http://localhost:4567"

    config.save_tokens({"access_token": "access-2", "expires_in": 60}, previous=original)
    updated = config.load_config()
    assert updated["token"] == "access-2"
    assert updated["refresh_token"] == "refresh-1"  # carried over
    assert updated["scope"] == "user:read"


def test_clear_tokens_keeps_urls(isolated_config):
    config.save_config({"token": "abc", "refresh_token": "ref", "airigs_url": "http://x"})
    config.clear_tokens()
    stored = config.load_config()
    assert "token" not in stored and "refresh_token" not in stored
    assert stored["airigs_url"] == "http://x"


def test_redacted_config():
    safe = config.redacted_config({"token": "abc", "airigs_url": "http://x"})
    assert safe == {"token": "***", "airigs_url": "http://x"}
