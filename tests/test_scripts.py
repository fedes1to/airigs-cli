from __future__ import annotations

import pytest

from airigs import scripts


def test_generate_vllm_script():
    recipe = {
        "engine": "vllm",
        "models": [{"source": "huggingface", "id": "org/model", "name": "Model"}],
        "args": [
            {"flag": "--max-model-len", "value": "32768"},
            {"flag": "--gpu-memory-utilization", "value": 0.9},
            {"flag": "--trust-remote-code", "value": None},
        ],
    }
    script = scripts.generate_script(recipe)
    assert "vllm serve org/model" in script
    assert "--max-model-len 32768" in script
    assert "--gpu-memory-utilization 0.9" in script
    assert "--trust-remote-code" in script


def test_launch_command_wins():
    assert scripts.generate_script({"launchCommand": "python serve.py", "engine": "custom"}) == "python serve.py\n"


def test_unknown_engine_requires_launch_command():
    with pytest.raises(scripts.ScriptError) as info:
        scripts.generate_script({"engine": "mystery", "models": [{"name": "x"}]})
    assert "launchCommand" in str(info.value)


def test_execute_refuses_non_interactive_without_yes():
    calls = []
    with pytest.raises(scripts.ScriptError) as info:
        scripts.execute("echo hi\n", yes=False, tty=False, printer=lambda _l: None, runner=lambda *a, **k: calls.append(a))
    assert "--yes" in str(info.value)
    assert calls == []


def test_execute_prompts_in_a_tty_and_declines():
    calls = []
    printed = []
    code = scripts.execute(
        "echo hi\n",
        yes=False,
        tty=True,
        prompt=lambda _q: "n",
        printer=printed.append,
        runner=lambda *a, **k: calls.append(a),
    )
    assert code == 1
    assert calls == []
    assert any("arbitrary shell commands" in line for line in printed)


def test_execute_runs_with_yes_and_reports_exit_code():
    captured = {}

    def runner(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return type("Completed", (), {"returncode": 7})()

    code = scripts.execute("echo hi\n", yes=True, printer=lambda _l: None, runner=runner)
    assert code == 7
    assert captured["kwargs"]["shell"] is True
    assert captured["kwargs"]["executable"] == "/bin/sh"


def test_execute_passes_script_arguments_without_a_shell_string():
    captured = {}

    def runner(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return type("Completed", (), {"returncode": 0})()

    scripts.execute(
        "echo \"$1\"\n",
        yes=True,
        script_args=["hello world"],
        printer=lambda _l: None,
        runner=runner,
    )
    assert captured["argv"][:3] == ["/bin/sh", "-c", "echo \"$1\"\n"]
    assert captured["argv"][3:] == ["airigs-recipe", "hello world"]
    assert "shell" not in captured["kwargs"]


def test_empty_script_rejected():
    with pytest.raises(scripts.ScriptError):
        scripts.execute("   \n", yes=True, printer=lambda _l: None, runner=lambda *a, **k: None)
