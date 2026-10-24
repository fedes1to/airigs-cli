"""Recipe document handling and the Recipes platform API surface."""

from __future__ import annotations

import re
from typing import Any

from .http import AirigsError, ApiClient

#: The shared engine vocabulary (contract v1.1). ``engines_for`` prefers the
#: platform's published OpenAPI enum and falls back to this list.
ENGINES = ("vllm", "sglang", "llamacpp", "strata", "omlx", "ollama", "custom")
MODEL_SOURCES = ("huggingface", "modelscope", "custom")
MODEL_ROLES = ("base", "drafter", "adapter", "vision", "embedding")
RECIPE_FILENAME = "recipe.json"

RIG_ID_RE = re.compile(r"^r_[A-Za-z0-9_-]{1,92}$")
_URL_RE = re.compile(r"^https?://[^\s]+$")


def is_rig_id(value: Any) -> bool:
    return isinstance(value, str) and bool(RIG_ID_RE.match(value))


def _find_engine_enum(value: Any) -> list[str] | None:
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        if "vllm" in value and "sglang" in value and "llamacpp" in value:
            return list(value)
        return None
    if isinstance(value, dict):
        for child in value.values():
            found = _find_engine_enum(child)
            if found:
                return found
    return None


def engines_for(client: Any) -> tuple[list[str], str]:
    """Read the engine enum from the platform, falling back to the constant."""
    api = getattr(client, "client", client)
    try:
        payload = api.get("/api/v1/openapi.json", action="read OpenAPI schema")
        if isinstance(payload, dict):
            found = _find_engine_enum(payload)
            if found:
                return found, "openapi"
    except AirigsError:
        pass
    return list(ENGINES), "builtin"


class RecipeError(Exception):
    """A recipe document does not satisfy the platform contract."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("recipe is invalid:\n" + "\n".join(f"  - {problem}" for problem in problems))


def skeleton(name: str = "my-recipe") -> dict[str, Any]:
    return {
        "title": name,
        "summary": "",
        "engine": "vllm",
        "models": [{"source": "huggingface", "id": "", "name": "", "role": "base"}],
        "args": [{"flag": "--max-model-len", "value": "32768"}],
        "launchCommand": "",
        "customInstructions": "",
        "repos": [],
        "hardwareNotes": "",
        "rigId": None,
        "publish": False,
    }


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_recipe(data: Any, engines: list[str] | None = None) -> list[str]:
    """Return a list of human-readable contract violations (empty = valid)."""
    problems: list[str] = []
    if not isinstance(data, dict):
        return ["recipe must be a JSON object"]

    allowed = list(engines or ENGINES)
    if not _is_text(data.get("title")):
        problems.append("title: required non-empty string")
    if not _is_text(data.get("engine")):
        problems.append(f"engine: required non-empty string (one of {', '.join(allowed)})")
    elif data.get("engine") not in allowed:
        problems.append(f"engine: {data.get('engine')!r} is not supported (use one of {', '.join(allowed)})")

    models = data.get("models")
    if not isinstance(models, list) or not models:
        problems.append("models: at least one model is required")
    else:
        for index, model in enumerate(models):
            if not isinstance(model, dict):
                problems.append(f"models[{index}]: must be an object")
                continue
            source = model.get("source")
            if source not in MODEL_SOURCES:
                problems.append(f"models[{index}].source: must be one of {', '.join(MODEL_SOURCES)}")
            if not _is_text(model.get("name")):
                problems.append(f"models[{index}].name: required non-empty string")
            if source in ("huggingface", "modelscope") and not _is_text(model.get("id")):
                problems.append(f"models[{index}].id: required for source {source}")
            role = model.get("role")
            if role is not None and role not in MODEL_ROLES:
                problems.append(
                    f"models[{index}].role: must be one of {', '.join(MODEL_ROLES)} (got {role!r})"
                )

    args = data.get("args")
    if args is not None:
        if not isinstance(args, list):
            problems.append("args: must be an array of {flag, value}")
        else:
            for index, arg in enumerate(args):
                if not isinstance(arg, dict):
                    problems.append(f"args[{index}]: must be an object with flag/value")
                    continue
                if not _is_text(arg.get("flag")):
                    problems.append(f"args[{index}].flag: required non-empty string")
                value = arg.get("value")
                if value is not None and not isinstance(value, (str, int, float, bool)):
                    problems.append(f"args[{index}].value: must be a scalar")
                if isinstance(arg.get("flag"), str) and not re.match(r"^-{1,2}[\w.:-]+$", arg["flag"]):
                    problems.append(f"args[{index}].flag: flags must start with '-' or '--'")

    for field in ("summary", "launchCommand", "customInstructions", "hardwareNotes"):
        value = data.get(field)
        if value is not None and not isinstance(value, str):
            problems.append(f"{field}: must be a string")

    repos = data.get("repos")
    if repos is not None:
        if not isinstance(repos, list):
            problems.append("repos: must be an array of {url, note}")
        else:
            for index, repo in enumerate(repos):
                if not isinstance(repo, dict):
                    problems.append(f"repos[{index}]: must be an object with url")
                    continue
                url = repo.get("url")
                if not isinstance(url, str) or not _URL_RE.match(url):
                    problems.append(f"repos[{index}].url: must be an http(s) URL")
                note = repo.get("note")
                if note is not None and not isinstance(note, str):
                    problems.append(f"repos[{index}].note: must be a string")

    rig_id = data.get("rigId")
    if rig_id is not None and not is_rig_id(rig_id):
        problems.append(
            "rigId: must be an opaque rig id matching r_[A-Za-z0-9_-]{1,92} (not an integer)"
        )

    revision = data.get("rigRevision")
    if revision is not None and not isinstance(revision, int):
        problems.append("rigRevision: must be an integer or null")

    publish = data.get("publish")
    if publish is not None and not isinstance(publish, bool):
        problems.append("publish: must be a boolean")

    return problems


def normalize_recipe(data: dict[str, Any]) -> dict[str, Any]:
    """Whitelist the contract fields and drop empties the server does not need."""
    payload: dict[str, Any] = {
        "title": str(data.get("title", "")).strip(),
        "engine": str(data.get("engine", "")).strip(),
        "models": [],
        "args": [],
        "repos": [],
    }
    for model in data.get("models") or []:
        entry = {
            "source": model.get("source"),
            "name": str(model.get("name") or "").strip(),
            "role": model.get("role") or "base",
        }
        if model.get("id"):
            entry["id"] = str(model["id"]).strip()
        payload["models"].append({key: value for key, value in entry.items() if value not in (None, "")})
    for arg in data.get("args") or []:
        payload["args"].append({"flag": arg.get("flag"), "value": arg.get("value")})
    for repo in data.get("repos") or []:
        entry = {"url": repo.get("url")}
        if repo.get("note"):
            entry["note"] = repo["note"]
        payload["repos"].append(entry)
    for field in ("summary", "launchCommand", "customInstructions", "hardwareNotes"):
        value = data.get(field)
        if isinstance(value, str) and value.strip():
            payload[field] = value.strip()
    if is_rig_id(data.get("rigId")):
        payload["rigId"] = data["rigId"]
    if isinstance(data.get("rigRevision"), int):
        payload["rigRevision"] = data["rigRevision"]
    if isinstance(data.get("publish"), bool):
        payload["publish"] = data["publish"]
    return payload


def require_valid(data: Any, *, engines: list[str] | None = None) -> dict[str, Any]:
    problems = validate_recipe(data, engines=engines)
    if problems:
        raise RecipeError(problems)
    return normalize_recipe(data)  # type: ignore[arg-type]


class Recipes:
    """Thin wrapper around the Recipes API v1."""

    def __init__(self, client: ApiClient) -> None:
        self.client = client

    def me(self) -> dict[str, Any]:
        return self.client.get("/api/v1/me", action="recipes me")

    def rigs(self) -> dict[str, Any]:
        return self.client.get("/api/v1/rigs", action="list rigs")

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.client.post("/api/v1/recipes", payload, action="create recipe")

    def benchmark(self, slug: str, benchmark: dict[str, Any], *, publish: bool = False) -> dict[str, Any]:
        body: dict[str, Any] = {"benchmark": benchmark}
        if publish:
            body["publish"] = True
        return self.client.post(f"/api/v1/recipes/{_q(slug)}/benchmark", body, action="upload recipe benchmark")

    def publish(self, slug: str) -> dict[str, Any]:
        return self.client.post(f"/api/v1/recipes/{_q(slug)}/publish", {}, action="publish recipe")

    def patch(self, slug: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.client.patch(f"/api/v1/recipes/{_q(slug)}", payload, action="update recipe")

    def delete(self, slug: str) -> Any:
        return self.client.delete(f"/api/v1/recipes/{_q(slug)}", action="delete recipe")

    def script(self, slug: str) -> str:
        text, _response = self.client.request_text("GET", f"/api/v1/recipes/{_q(slug)}/script", action="fetch recipe script")
        return text


def _q(slug: str) -> str:
    from urllib.parse import quote

    return quote(slug, safe="")
