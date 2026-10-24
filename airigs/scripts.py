"""Recipe launch scripts: generate, fetch, display and execute safely."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from typing import Any, Callable

from .http import AirigsError, NotFoundError
from .recipes import Recipes

WARNING = (
    "This script runs arbitrary shell commands (fetched from the Recipes platform or generated "
    "from recipe.json). Review it before continuing."
)

#: Environment variables a recipe script must never see. Tokens and secrets are
#: removed by name, not just the two the CLI itself uses, so a recipe cannot
#: exfiltrate an unrelated credential either.
_SECRET_ENV_RE = re.compile(r"(TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY|APIKEY)", re.I)
_EXTRA_SECRET_ENV = {"AIRIGS_TOKEN", "RECIPES_TOKEN", "AIRIGS_CLIENT_SECRET"}


def sanitize_environment(environment: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if environment is None else environment)
    for key in list(env):
        if key.upper() in _EXTRA_SECRET_ENV or _SECRET_ENV_RE.search(key):
            env.pop(key, None)
    return env


class ScriptError(AirigsError):
    """A script cannot be generated or execution was refused."""


def _quote(value: Any) -> str:
    return shlex.quote(str(value))


def _flag_value(arg: dict[str, Any]) -> str:
    flag = str(arg.get("flag") or "").strip()
    value = arg.get("value")
    if value is None or value == "":
        return _quote(flag)
    return f"{_quote(flag)} {_quote(value)}"


def generate_script(recipe: dict[str, Any]) -> str:
    """Build a shell script from a recipe document, or use its launchCommand."""
    launch = (recipe.get("launchCommand") or "").strip()
    if launch:
        return launch if launch.endswith("\n") else launch + "\n"

    engine = str(recipe.get("engine") or "").strip()
    models = recipe.get("models") or []
    model = ""
    if models:
        first = models[0]
        model = str(first.get("id") or first.get("name") or "")
    args = " ".join(_flag_value(arg) for arg in recipe.get("args") or [])
    model_arg = _quote(model) if model else ""

    if engine == "vllm":
        command = f"vllm serve {model_arg} {args}".strip()
    elif engine == "llamacpp":
        command = f"llama-server -m {model_arg} {args}".strip()
    elif engine == "sglang":
        command = f"python -m sglang.launch_server --model-path {model_arg} {args}".strip()
    elif engine == "ollama":
        command = f"ollama run {model_arg} {args}".strip()
    else:
        raise ScriptError(
            f"cannot generate a launch script for engine {engine!r}; "
            "set launchCommand in recipe.json or request the script from the platform."
        )
    if not model and engine != "custom":
        raise ScriptError("recipe has no model id; cannot generate a launch script.")
    return "#!/bin/sh\nset -eu\n" + command + "\n"


def fetch_script(client: Recipes, slug: str) -> str:
    return client.script(slug)


def resolve_script(
    client: Recipes | None,
    slug: str,
    *,
    recipe: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Return ``(script, source)``: fetched from the platform, or generated locally."""
    if client is not None and not recipe:
        try:
            return fetch_script(client, slug), "recipes:/api/v1/recipes/%s/script" % slug
        except NotFoundError:
            raise ScriptError(f"recipe {slug!r} has no script (HTTP 404).")
    if recipe is None:
        raise ScriptError("no script source: pass a recipe file or a published slug")
    return generate_script(recipe), "local recipe"


def execute(
    script: str,
    *,
    yes: bool = False,
    slug: str = "",
    script_args: list[str] | None = None,
    printer: Callable[[str], None] = print,
    prompt: Callable[[str], str] = input,
    tty: bool | None = None,
    runner: Callable[..., Any] = subprocess.run,
    warn: Callable[[str], None] | None = None,
) -> int:
    """Show the script, confirm, then run it with ``/bin/sh``.

    Refuses non-interactively unless ``yes`` is set: an unattended process must
    opt in explicitly.
    """
    if not script.strip():
        raise ScriptError("the script is empty; nothing to run")
    if not yes and os.environ.get("AIRIGS_YES") == "1":
        yes = True
    printer(script if script.endswith("\n") else script + "\n")
    warn = warn or printer
    warn(WARNING if not slug else f"{WARNING} Target recipe: {slug}.")
    if not yes:
        interactive = tty if tty is not None else sys.stdin.isatty()
        if not interactive:
            raise ScriptError(
                "refusing to run a script without --yes in a non-interactive session"
            )
        answer = prompt("Run this script? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            printer("Aborted.")
            return 1
    env = sanitize_environment()
    if script_args:
        # Run through `sh -c` so the script can read $1..$n without string mangling.
        completed = runner(
            ["/bin/sh", "-c", script, "airigs-recipe", *script_args], check=False, env=env
        )
    else:
        completed = runner(script, shell=True, executable="/bin/sh", check=False, env=env)
    return int(getattr(completed, "returncode", 0))
