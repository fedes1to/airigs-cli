from __future__ import annotations

import json

import pytest

from airigs import cli, config
from fake_server import TOKEN_RESPONSE, make_benchmark, make_recipe


def run(monkeypatch, server, argv):
    monkeypatch.setenv("AIRIGS_URL", server.base_url)
    monkeypatch.setenv("RECIPES_URL", server.base_url)
    monkeypatch.setenv("AIRIGS_TOKEN", server.state.token["access_token"])
    return cli.main(argv)


def test_parser_accepts_flags_before_and_after_subcommand():
    parser = cli.build_parser()
    assert parser.parse_args(["--json", "rig", "list"]).json is True
    assert parser.parse_args(["rig", "list", "--json"]).json is True
    args = parser.parse_args(["bench", "--quick", "--gpu", "0", "--no-power"])
    assert args.quick and args.gpu == "0" and args.no_power
    assert args.func is cli.cmd_bench


def test_rig_list_json(fake_server, isolated_config, monkeypatch, capsys):
    fake_server.state.rigs = [
        {"id": 1, "name": "Box", "verified": True, "hardware": [{"kind": "gpu", "name": "GPU"}], "models": []}
    ]
    code = run(monkeypatch, fake_server, ["rig", "list", "--json", "--airigs-url", fake_server.base_url])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rigs"][0]["name"] == "Box"


def test_rig_list_human_output(fake_server, isolated_config, monkeypatch, capsys):
    fake_server.state.rigs = [{"id": 1, "name": "Box", "verified": False, "hardware": [], "models": []}]
    code = run(monkeypatch, fake_server, ["rig", "list", "--airigs-url", fake_server.base_url])
    assert code == 0
    out = capsys.readouterr().out
    assert "id" in out and "Box" in out


def test_rig_show_by_name(fake_server, isolated_config, monkeypatch, capsys):
    fake_server.state.rigs = [{"id": "r_5", "name": "Box", "verified": True, "hardware": [], "models": []}]
    code = run(monkeypatch, fake_server, ["rig", "show", "Box", "--json", "--airigs-url", fake_server.base_url])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["id"] == "r_5"


def test_whoami_json(fake_server, isolated_config, monkeypatch, capsys):
    code = run(monkeypatch, fake_server, ["whoami", "--json", "--airigs-url", fake_server.base_url])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["username"] == "tester"
    assert payload["uid"] == 7


def test_401_exits_nonzero_and_reports_json(fake_server, isolated_config, monkeypatch, capsys):
    monkeypatch.setenv("AIRIGS_URL", fake_server.base_url)
    monkeypatch.setenv("AIRIGS_TOKEN", "wrong-token")
    code = cli.main(["rig", "list", "--json", "--airigs-url", fake_server.base_url])
    assert code == 3
    captured = capsys.readouterr()
    assert "error" in json.loads(captured.err)


def test_login_with_token_on_argv_is_refused(fake_server, isolated_config, capsys):
    code = cli.main(
        ["login", "--token", TOKEN_RESPONSE["access_token"], "--airigs-url", fake_server.base_url, "--json"]
    )
    assert code == 1
    assert "leaks it" in capsys.readouterr().err
    assert not config.config_path().exists()


def test_login_with_token_on_stdin_stores_0600(fake_server, isolated_config, monkeypatch, capsys):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO(TOKEN_RESPONSE["access_token"] + "\n"))
    code = cli.main(
        ["login", "--token", "-", "--airigs-url", fake_server.base_url, "--json"]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["loggedIn"] is True
    assert payload["user"]["username"] == "tester"
    assert config.load_config()["token"] == TOKEN_RESPONSE["access_token"]
    assert config.config_mode() == 0o600


def test_bench_json_uses_harness(monkeypatch, isolated_config, capsys):
    expected = make_benchmark()
    monkeypatch.setattr(cli.bench_runner, "run_benchmark", lambda **kwargs: expected)
    code = cli.main(["bench", "--quick", "--json"])
    assert code == 0
    assert json.loads(capsys.readouterr().out) == expected


def test_bench_writes_output_file(monkeypatch, isolated_config, tmp_path, capsys):
    expected = make_benchmark()
    monkeypatch.setattr(cli.bench_runner, "run_benchmark", lambda **kwargs: expected)
    target = tmp_path / "out.json"
    code = cli.main(["bench", "--quick", "--output", str(target)])
    assert code == 0
    assert json.loads(target.read_text()) == expected


def test_rig_submit_requires_verified_true(fake_server, isolated_config, monkeypatch, tmp_path, capsys):
    benchmark_file = tmp_path / "bench.json"
    benchmark_file.write_text(json.dumps(make_benchmark()))

    code = run(
        monkeypatch,
        fake_server,
        ["rig", "submit", "--name", "Box", "--benchmark", str(benchmark_file), "--json",
         "--airigs-url", fake_server.base_url],
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verified"] is True

    fake_server.state.verified = False
    code = run(
        monkeypatch,
        fake_server,
        ["rig", "submit", "--name", "Box", "--benchmark", str(benchmark_file), "--json",
         "--airigs-url", fake_server.base_url],
    )
    assert code == 1
    error = json.loads(capsys.readouterr().err)
    assert "did not mark the rig verified" in error["error"]
    # The update used PUT rather than creating a duplicate.
    assert any(request["method"] == "PUT" for request in fake_server.state.requests)


def test_recipe_validate_reports_problems(tmp_path, isolated_config, capsys):
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps({"engine": "vllm"}))
    code = cli.main(["recipe", "validate", str(path), "--json"])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is False
    assert any("title" in problem for problem in payload["problems"])


def test_recipe_validate_accepts_valid_document(tmp_path, isolated_config, capsys):
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(make_recipe()))
    code = cli.main(["recipe", "validate", str(path), "--json"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True


def test_recipe_submit_posts_contract_fields(fake_server, isolated_config, monkeypatch, tmp_path, capsys):
    path = tmp_path / "recipe.json"
    document = make_recipe()
    path.write_text(json.dumps(document))
    code = run(monkeypatch, fake_server, ["recipe", "submit", str(path), "--json",
                                          "--recipes-url", fake_server.base_url])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["recipe"]["slug"] == "test-recipe"
    posted = next(request for request in fake_server.state.requests if request["path"] == "/api/v1/recipes")
    assert posted["body"]["title"] == document["title"]
    assert posted["body"]["models"][0]["source"] == "huggingface"
    assert posted["body"]["args"] == document["args"]


def test_recipe_script_writes_file(fake_server, isolated_config, monkeypatch, tmp_path, capsys):
    target = tmp_path / "run.sh"
    code = run(
        monkeypatch,
        fake_server,
        ["recipe", "script", "test-recipe", "--out", str(target), "--recipes-url", fake_server.base_url],
    )
    assert code == 0
    assert target.read_text() == fake_server.state.script


def test_recipe_run_requires_yes_when_interactive_or_refuses(fake_server, isolated_config, monkeypatch, capsys):
    # Under pytest stdin is not a TTY, so without --yes the CLI must refuse.
    code = run(
        monkeypatch,
        fake_server,
        ["recipe", "run", "test-recipe", "--recipes-url", fake_server.base_url],
    )
    assert code == 1
    assert "refusing to run a script without --yes" in capsys.readouterr().err


def test_recipe_script_falls_back_to_local_recipe(fake_server, isolated_config, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    document = make_recipe()
    document["launchCommand"] = "python serve.py"
    (tmp_path / "recipe.json").write_text(json.dumps(document))
    fake_server.state.script_status = 404
    code = run(
        monkeypatch,
        fake_server,
        ["recipe", "script", "test-recipe", "--json", "--recipes-url", fake_server.base_url],
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "local recipe.json"
    assert payload["script"] == "python serve.py\n"


def test_recipe_run_with_yes_executes(fake_server, isolated_config, monkeypatch, capsys):
    code = run(
        monkeypatch,
        fake_server,
        ["recipe", "run", "test-recipe", "--yes", "--json", "--recipes-url", fake_server.base_url],
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "recipe-script" in captured.err  # script echoed to stderr in --json mode
    assert json.loads(captured.out)["exitCode"] == 0


def test_doctor_reports_checks(fake_server, isolated_config, monkeypatch, capsys):
    code = run(monkeypatch, fake_server, ["doctor", "--json", "--airigs-url", fake_server.base_url,
                                          "--recipes-url", fake_server.base_url])
    payload = json.loads(capsys.readouterr().out)
    by_name = {item["check"]: item for item in payload["checks"]}
    assert by_name["python"]["status"] == "ok"
    assert by_name["AIRigs reachable"]["status"] == "ok"
    assert by_name["Recipes reachable"]["status"] == "ok"
    assert by_name["token validity"]["status"] == "ok"
    assert by_name["mongodb"]["status"] == "skip"
    assert code == 0


def test_bench_self_check_mode(monkeypatch, isolated_config, capsys):
    monkeypatch.setattr(
        cli.bench_runner,
        "self_check",
        lambda: {"ok": True, "schemaProblems": [], "missingRequired": [], "benchmark": make_benchmark()},
    )
    code = cli.main(["--self-check", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["benchmark"]["schemaVersion"] == 1


def test_rig_show_by_opaque_id(fake_server, isolated_config, monkeypatch, capsys):
    """Rig ids are `r_...` strings, so id lookup must not be gated on `isdigit()`."""
    fake_server.state.rigs = [{"id": "r_9bd202caf0768f52", "name": "Box", "verified": True, "hardware": [], "models": []}]
    code = run(monkeypatch, fake_server, ["rig", "show", "r_9bd202caf0768f52", "--json", "--airigs-url", fake_server.base_url])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["name"] == "Box"


def test_rig_delete_by_opaque_id(fake_server, isolated_config, monkeypatch, capsys):
    fake_server.state.rigs = [{"id": "r_9bd202caf0768f52", "name": "Box", "verified": False, "hardware": [], "models": []}]
    code = run(monkeypatch, fake_server, ["rig", "delete", "r_9bd202caf0768f52", "--yes", "--airigs-url", fake_server.base_url])
    assert code == 0
