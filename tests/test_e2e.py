from __future__ import annotations

import json

import pytest

from airigs import cli, config, recipebench
from fake_server import TOKEN_RESPONSE, make_benchmark


class FakeSampler:
    """Stand-in for NVML so the recipe benchmark can complete off-GPU."""

    def start(self) -> None:
        pass

    def stop(self) -> dict:
        return {
            "vramPeakBytes": 8 * 1024**3,
            "powerAvgW": 250.0,
            "gpuUtilPct": 88.0,
            "ramPeakBytes": 16 * 1024**3,
            "nvml": True,
        }


def test_full_flow_login_rig_recipe_publish_script(fake_server, isolated_config, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(recipebench, "_GpuSampler", FakeSampler)
    monkeypatch.setenv("AIRIGS_URL", fake_server.base_url)
    monkeypatch.setenv("RECIPES_URL", fake_server.base_url)

    # 1. Device-flow login (no browser), storing a refresh token.
    fake_server.state.device["interval"] = 0.01
    fake_server.state.poll_script = [{"error": "authorization_pending"}, dict(TOKEN_RESPONSE)]
    assert cli.main(["login", "--no-browser", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["loggedIn"] is True
    assert config.load_config()["refresh_token"] == TOKEN_RESPONSE["refresh_token"]

    # 2. Submit a rig with a saved benchmark: challenge -> verify.
    benchmark_file = tmp_path / "bench.json"
    benchmark_file.write_text(json.dumps(make_benchmark()))
    assert cli.main(
        ["rig", "submit", "--name", "Box", "--benchmark", str(benchmark_file), "--json"]
    ) == 0
    rig = json.loads(capsys.readouterr().out)["rig"]
    rig_id = rig["id"]
    assert rig_id.startswith("r_")
    assert rig["verified"] is True

    # 3. A recipe pinned to that rig.
    recipe = {
        "title": "Full-flow recipe",
        "summary": "e2e",
        "engine": "vllm",
        "models": [{"source": "huggingface", "id": "org/model", "name": "Model", "role": "base"}],
        "args": [{"flag": "--max-model-len", "value": "8192"}],
        "launchCommand": "",
        "repos": [{"url": "https://example.com/repo", "note": "weights"}],
        "hardwareNotes": "one GPU",
        "rigId": rig_id,
        "rigRevision": rig.get("revision", 1),
    }
    recipe_file = tmp_path / "recipe.json"
    recipe_file.write_text(json.dumps(recipe))
    assert cli.main(["recipe", "validate", str(recipe_file)]) == 0
    capsys.readouterr()  # discard the human-readable validation line

    # 4. Submit the draft.
    assert cli.main(["recipe", "submit", str(recipe_file), "--json"]) == 0
    submitted = json.loads(capsys.readouterr().out)["recipe"]
    slug = submitted["slug"]
    assert slug == fake_server.state.slug
    create_request = next(r for r in fake_server.state.requests if r["path"] == "/api/v1/recipes")
    assert create_request["body"]["rigId"] == rig_id
    assert create_request["body"]["rigRevision"] == rig.get("revision", 1)
    assert create_request["body"]["engine"] == "vllm"

    # 5. Benchmark the running endpoint and publish.
    assert cli.main(
        [
            "recipe",
            "bench",
            slug,
            "--endpoint",
            fake_server.base_url,
            "--engine",
            "vllm",
            "--model",
            "test-model",
            "--rig-id",
            rig_id,
            "--requests",
            "2",
            "--concurrency",
            "1",
            "--publish",
            "--json",
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["benchmark"]["rigId"] == rig_id
    assert result["benchmark"]["outputTokensPerSec"] > 0
    bench_request = next(r for r in fake_server.state.requests if r["path"].endswith("/benchmark") and r["path"].startswith("/api/v1"))
    assert bench_request["body"]["publish"] is True
    assert bench_request["body"]["benchmark"]["vramPeakBytes"] == 8 * 1024**3

    # 6. Fetch the launch script.
    script_path = tmp_path / "run.sh"
    assert cli.main(["recipe", "script", slug, "--out", str(script_path)]) == 0
    assert "recipe-script" in script_path.read_text()
    capsys.readouterr()

    # 7. Logout revokes the refresh token.
    assert cli.main(["logout", "--json"]) == 0
    logout = json.loads(capsys.readouterr().out)
    assert logout["loggedOut"] is True
    assert logout["removedLocalState"] is True
    assert fake_server.state.revoked
    assert "token" not in config.load_config()

    paths = [request["path"] for request in fake_server.state.requests]
    for expected in (
        "/oauth/userinfo",
        "/api/v1/openapi.json",
        f"/api/v3/plugins/airigs-rig/{rig_id}/benchmark/challenge",
        f"/api/v3/plugins/airigs-rig/{rig_id}/benchmark",
        "/api/v1/recipes",
        f"/api/v1/recipes/{slug}/benchmark",
        f"/api/v1/recipes/{slug}/script",
        "/oauth/revoke",
    ):
        assert expected in paths, expected


def test_rig_benchmark_measures_after_requesting_challenge(fake_server, isolated_config, monkeypatch, capsys):
    """`rig benchmark` must obtain the nonce before it measures, then upload it."""
    monkeypatch.setenv("AIRIGS_URL", fake_server.base_url)
    monkeypatch.setenv("AIRIGS_TOKEN", fake_server.state.token["access_token"])
    fake_server.state.rigs = [
        {"id": "r_1", "revision": 1, "name": "Box", "verified": False, "hardware": [], "models": []}
    ]
    measured: list[float] = []

    def fake_run_benchmark(**_kwargs):
        measured.append(1)
        return make_benchmark()

    monkeypatch.setattr(cli.bench_runner, "run_benchmark", fake_run_benchmark)
    code = cli.main(["rig", "benchmark", "r_1", "--json"])
    assert code == 0, capsys.readouterr().err
    assert json.loads(capsys.readouterr().out)["verified"] is True
    assert measured == [1]

    paths = [request["path"] for request in fake_server.state.requests]
    challenge_index = paths.index("/api/v3/plugins/airigs-rig/r_1/benchmark/challenge")
    upload_index = paths.index("/api/v3/plugins/airigs-rig/r_1/benchmark")
    assert challenge_index < upload_index
    upload = next(
        request for request in fake_server.state.requests
        if request["path"] == "/api/v3/plugins/airigs-rig/r_1/benchmark"
    )
    challenge = upload["body"]["benchmark"]["challenge"]
    assert challenge["nonce"].startswith("nonce-")
    assert "T" in challenge["issuedAt"]
