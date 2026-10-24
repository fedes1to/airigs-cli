from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fake_server import FakeServer  # noqa: E402


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Point XDG_CONFIG_HOME at a temp dir and clear token/URL env overrides."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    for name in ("AIRIGS_URL", "RECIPES_URL", "AIRIGS_TOKEN", "RECIPES_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    return Path(tmp_path / "config")


@pytest.fixture
def fake_server():
    server = FakeServer().start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def no_optional_imports(monkeypatch):
    """Make torch/pynvml imports fail inside every harness module."""
    modules = [
        "airigs.bench.common",
        "airigs.bench.nvidia",
        "airigs.bench.amd",
        "airigs.bench.p2p",
        "airigs.bench.runner",
        "airigs.bench.power",
        "airigs.bench.apple",
        "airigs.recipebench",
    ]
    originals = {}
    for name in modules:
        module = importlib.import_module(name)
        if hasattr(module, "import_optional"):
            originals[name] = module.import_optional

    def factory(original):
        def fake_import(target):
            if target in ("torch", "pynvml"):
                return None
            return original(target)

        return fake_import

    for name, original in originals.items():
        monkeypatch.setattr(importlib.import_module(name), "import_optional", factory(original))
    return monkeypatch
