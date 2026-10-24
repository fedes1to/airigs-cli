"""The ``airigs`` command-line interface.

Every command supports ``--json`` for machine-readable output and exits
non-zero on failure. Human output is deliberately compact.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import __version__, auth, config, http, recipebench, recipes
from .bench import report as bench_report
from .bench import runner as bench_runner
from .bench.common import Notes, now_iso
from .config import ConfigError, Settings, collect_secrets, load_config, resolve_settings
from .http import (
    AirigsError,
    ApiClient,
    HttpError,
    NetworkError,
    NotFoundError,
    assert_secure_url,
    is_loopback_url,
)
from .recipes import RecipeError, Recipes
from .scripts import ScriptError, execute, generate_script, resolve_script

RIG_API = "/api/v3/plugins/airigs-rig"


# ---------------------------------------------------------------------------
# Output helpers


def render_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    rows = [["" if cell is None else str(cell) for cell in row] for row in rows]
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            if index < len(widths):
                widths[index] = max(widths[index], len(cell))
    lines = ["  ".join(str(header).ljust(widths[i]) for i, header in enumerate(headers)).rstrip()]
    for row in rows:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
    return "\n".join(lines)


def emit(args: argparse.Namespace, data: Any, *, headers: Sequence[str] | None = None, rows: Iterable[Sequence[Any]] | None = None, text: str | None = None) -> None:
    if args.json:
        print(json.dumps(data, indent=2))
    elif text is not None:
        print(text)
    elif headers is not None and rows is not None:
        print(render_table(headers, rows))
    else:
        print(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Context / clients


class Context:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.settings: Settings = resolve_settings(
            airigs_url=args.airigs_url, recipes_url=args.recipes_url
        )
        self.secrets = collect_secrets(load_config())
        self._token: str | None = None

    def _log(self, message: str) -> None:
        print(message, file=sys.stderr)

    def airigs(self, *, auth_required: bool = True) -> ApiClient:
        token = None
        if auth_required:
            token = self.token()
        return ApiClient(
            self.settings.airigs_url,
            token,
            verbose=self.args.verbose,
            log=self._log,
            secrets=self.secrets | ({token} if token else set()),
        )

    def token(self) -> str:
        if self._token is None:
            self._token = auth.ensure_token(self.settings, verbose=self.args.verbose)
        return self._token

    def recipes(self) -> Recipes:
        assert_secure_url(self.settings.recipes_url, what="Recipes")
        token = self.settings.recipes_token
        if not token:
            raise auth.NotLoggedInError(
                "not logged in: run `airigs login` (or set RECIPES_TOKEN/AIRIGS_TOKEN)"
            )
        return Recipes(
            ApiClient(
                self.settings.recipes_url,
                token,
                verbose=self.args.verbose,
                log=self._log,
                secrets=self.secrets | {token},
            )
        )


# ---------------------------------------------------------------------------
# Shared helpers


def _rigs(client: ApiClient) -> list[dict[str, Any]]:
    data = client.get(RIG_API, action="list rigs")
    if isinstance(data, dict):
        rigs = data.get("rigs")
    else:
        rigs = data
    if not isinstance(rigs, list):
        raise AirigsError("the rig API returned no rigs array")
    return [rig for rig in rigs if isinstance(rig, dict)]


def _rig_name(rig: dict[str, Any]) -> str:
    return str(rig.get("name") or "")


def _validate_rig_id(value: Any) -> str:
    if not recipes.is_rig_id(value):
        raise AirigsError(
            f"rig id must be an opaque string such as r_abc123 (got {value!r}); "
            "integers are not valid rig ids"
        )
    return str(value)


def _resolve_rig(client: ApiClient, ident: str) -> dict[str, Any]:
    rigs = _rigs(client)
    # Rig ids are opaque strings (`r_` + up to 92 URL-safe characters), never integers, so
    # matching on `isdigit()` made `rig show r_...`/`rig delete r_...` fall through to a name
    # lookup and fail for every real id.
    if recipes.is_rig_id(ident):
        for rig in rigs:
            if str(rig.get("id")) == ident:
                return rig
    lowered = ident.lower()
    exact = [rig for rig in rigs if _rig_name(rig).lower() == lowered]
    if len(exact) == 1:
        return exact[0]
    partial = [rig for rig in rigs if lowered in _rig_name(rig).lower()]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        names = ", ".join(f"{_rig_name(rig)!r} ({rig.get('id')})" for rig in partial)
        raise AirigsError(f"{ident!r} matches several rigs: {names}")
    # Fall back to the userslug endpoint.
    from urllib.parse import quote

    try:
        rig = client.get(f"{RIG_API}/{quote(ident, safe='')}", action="get rig")
    except NotFoundError as exc:
        raise AirigsError(f"no rig matches {ident!r}") from exc
    if isinstance(rig, dict) and "rig" in rig and isinstance(rig["rig"], dict):
        return rig["rig"]
    return rig


def _rig_payload(data: dict[str, Any], *, name: str | None = None) -> dict[str, Any]:
    payload = {
        "name": (name or data.get("name") or "").strip(),
        "hardware": data.get("hardware") if isinstance(data.get("hardware"), list) else [],
        "models": data.get("models") if isinstance(data.get("models"), list) else [],
        "notes": data.get("notes") if isinstance(data.get("notes"), str) else "",
    }
    if not payload["name"]:
        raise AirigsError("a rig needs a name (--name NAME or \"name\" in the file)")
    return payload


def _load_json(path: str) -> Any:
    try:
        return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AirigsError(f"{path}: file not found") from exc
    except OSError as exc:
        raise AirigsError(f"{path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AirigsError(f"{path} is not valid JSON: {exc}") from exc


def _confirm(args: argparse.Namespace, question: str) -> None:
    if getattr(args, "yes", False):
        return
    if not sys.stdin.isatty():
        raise AirigsError("refusing without --yes in a non-interactive session")
    answer = input(f"{question} [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        raise AirigsError("aborted")


def _hardware_from_system() -> dict[str, Any]:
    """Detect CPU/RAM/GPU for a rig draft without running any benchmark."""
    from .bench import amd as bench_amd
    from .bench import apple as bench_apple
    from .bench import nvidia as bench_nvidia

    notes = Notes()
    system = bench_runner.collect_system(notes=notes)
    session = bench_nvidia.NvmlSession()
    try:
        gpus = bench_nvidia.enumerate_gpus(notes=notes, session=session)
        vendor = "nvidia" if gpus else None
        if not gpus:
            gpus = bench_amd.enumerate_gpus(notes=notes)
            vendor = "amd" if gpus else None
        if not gpus:
            gpus = bench_apple.run(quick=True, notes=notes)
            vendor = "apple" if gpus else None
    finally:
        session.close()
    hardware: list[dict[str, Any]] = []
    cpu = system.get("cpu") or {}
    if cpu.get("model"):
        hardware.append(
            {
                "kind": "cpu",
                "name": cpu["model"],
                "manufacturer": cpu.get("vendor"),
                "count": 1,
                "specs": {
                    "physicalCores": str(cpu.get("physicalCores") or ""),
                    "logicalCores": str(cpu.get("logicalCores") or ""),
                    "maxClockMhz": str(cpu.get("maxClockMhz") or ""),
                },
            }
        )
    memory = system.get("memory") or {}
    if memory.get("totalBytes"):
        specs = {"totalBytes": str(memory["totalBytes"])}
        if memory.get("type"):
            specs["type"] = memory["type"]
        if memory.get("speedMhz"):
            specs["speedMhz"] = str(memory["speedMhz"])
        hardware.append({"kind": "ram", "name": "System RAM", "count": 1, "specs": specs})
    for gpu in gpus:
        hardware.append(
            {
                "kind": "gpu",
                "name": gpu.get("name") or "GPU",
                "manufacturer": {
                    "nvidia": "NVIDIA",
                    "amd": "AMD",
                    "apple": "Apple",
                }.get(vendor or ""),
                "count": 1,
                "specs": {
                    "vramBytes": str(gpu.get("vramBytes") or ""),
                    "driver": str(gpu.get("driverVersion") or ""),
                },
            }
        )
    # Drop empty spec values so the server's sanitiser keeps the entry tidy.
    for item in hardware:
        item["specs"] = {key: value for key, value in item["specs"].items() if value}
        for key in list(item):
            if item[key] is None:
                del item[key]
    return {"hardware": hardware, "models": [], "notes": notes[-1] if notes else ""}


def _enrich_rig_revision(ctx: Context, payload: dict[str, Any]) -> dict[str, Any]:
    """Attach the current rig revision when a recipe pins a rig but omits it."""
    rig_id = payload.get("rigId")
    if not recipes.is_rig_id(rig_id) or isinstance(payload.get("rigRevision"), int):
        return payload
    try:
        rig = next(
            (item for item in _rigs(ctx.airigs()) if str(item.get("id")) == rig_id), None
        )
    except AirigsError:
        return payload
    if rig is not None and isinstance(rig.get("revision"), int):
        payload["rigRevision"] = rig["revision"]
    return payload


def _default_rig_id(ctx: Context) -> str | None:
    try:
        rigs = _rigs(ctx.airigs())
    except AirigsError:
        return None
    if len(rigs) == 1 and recipes.is_rig_id(rigs[0].get("id")):
        return rigs[0]["id"]
    verified = [rig for rig in rigs if rig.get("verified") and recipes.is_rig_id(rig.get("id"))]
    if len(verified) == 1:
        return verified[0]["id"]
    return None


# ---------------------------------------------------------------------------
# login / logout / whoami / doctor


def cmd_login(args: argparse.Namespace) -> int:
    ctx = Context(args)
    airigs_url = ctx.settings.airigs_url
    if args.token is not None:
        if args.token != "-":
            raise AirigsError(
                "passing a token as an argument leaks it through `ps` and shell history; "
                "pipe it instead: `airigs login --token -` (or set AIRIGS_TOKEN)"
            )
        token = sys.stdin.read().strip()
        if not token:
            raise AirigsError("no token was provided on stdin")
        info = auth.login_with_token(airigs_url, token)
        message = f"Token verified and stored for {info.get('username')} (uid {info.get('uid')})."
    else:
        auth.login_with_device_flow(
            airigs_url,
            no_browser=args.no_browser,
            printer=lambda line: print(line, file=sys.stderr if args.json else sys.stdout),
        )
        settings = resolve_settings(airigs_url=airigs_url, config=load_config())
        token = auth.ensure_token(settings, verbose=args.verbose)
        info = auth.fetch_userinfo(airigs_url, token)
        message = f"Logged in as {info.get('username')} (uid {info.get('uid')})."
    emit(
        args,
        {
            "loggedIn": True,
            "user": info,
            "configPath": auth.config_location(),
            "airigsUrl": airigs_url,
        },
        text=f"{message}\nTokens stored in {auth.config_location()} (mode 600).",
    )
    return 0


def cmd_logout(args: argparse.Namespace) -> int:
    ctx = Context(args)
    result = auth.logout(ctx.settings, verbose=args.verbose)
    if result.get("hadRefreshToken") and not result.get("revoked"):
        print(
            "warning: the server did not confirm revocation; local tokens were removed anyway"
            + (f" ({result['error']})" if result.get("error") else ""),
            file=sys.stderr,
        )
    emit(
        args,
        {"loggedOut": True, "configPath": auth.config_location(), **result},
        text="Logged out; local tokens removed."
        + (" Refresh token revoked." if result.get("revoked") else ""),
    )
    return 0


def cmd_whoami(args: argparse.Namespace) -> int:
    ctx = Context(args)
    token = ctx.token()
    info = auth.fetch_userinfo(ctx.settings.airigs_url, token)
    groups = info.get("groups") or []
    if isinstance(groups, list):
        groups_text = ", ".join(
            str(group.get("name") if isinstance(group, dict) else group) for group in groups
        )
    else:
        groups_text = str(groups)
    emit(
        args,
        info,
        headers=("field", "value"),
        rows=[
            ("uid", info.get("uid")),
            ("username", info.get("username")),
            ("userslug", info.get("userslug")),
            ("groups", groups_text),
            ("joined", info.get("joindate")),
            ("profile", info.get("profileUrl")),
            ("avatar", info.get("avatar")),
        ],
    )
    return 0


def _probe(url: str) -> tuple[bool, str]:
    try:
        response = http.send("GET", url, timeout=5.0)
    except NetworkError as exc:
        return False, str(exc)
    return response.status < 500, f"HTTP {response.status}"


def cmd_doctor(args: argparse.Namespace) -> int:
    ctx = Context(args)
    checks: list[dict[str, Any]] = []

    def check(name: str, status: str, detail: str, remediation: str | None = None) -> None:
        checks.append({"check": name, "status": status, "detail": detail, "remediation": remediation})

    version = sys.version_info
    check(
        "python",
        "ok" if version >= (3, 11) else "fail",
        f"{version.major}.{version.minor}.{version.micro}",
        None if version >= (3, 11) else "install Python 3.11 or newer",
    )
    for module, remediation in (
        ("numpy", "pip install numpy"),
        ("psutil", "pip install psutil"),
    ):
        try:
            imported = __import__(module)
            check(module, "ok", getattr(imported, "__version__", "installed"))
        except Exception:
            check(module, "fail", "not importable", remediation)

    try:
        import pynvml

        check("pynvml", "ok", getattr(pynvml, "__version__", "installed"))
    except Exception:
        check("pynvml", "warn", "not installed", "pip install pynvml (needed for NVIDIA power/VRAM)")

    try:
        import torch

        check("torch", "ok", getattr(torch, "__version__", "installed"))
        try:
            cuda = bool(torch.cuda.is_available())
        except Exception:
            cuda = False
        if cuda:
            check("torch.cuda", "ok", f"{torch.cuda.device_count()} device(s)")
        else:
            mps = False
            try:
                mps = bool(torch.backends.mps.is_available())
            except Exception:
                pass
            check(
                "torch.cuda",
                "warn" if not mps else "ok",
                "MPS available" if mps else "CUDA not available",
                None if mps else "install a GPU build of torch: pip install 'airigs[gpu]'",
            )
    except Exception:
        check(
            "torch",
            "warn",
            "not installed",
            "install `airigs[gpu]` for TFLOPS / GPU bandwidth / P2P measurements",
        )

    from .bench import amd as bench_amd
    from .bench import nvidia as bench_nvidia

    check(
        "nvidia-smi",
        "ok" if bench_nvidia.nvidia_smi_available() else "warn",
        shutil.which("nvidia-smi") or "not found",
        None if bench_nvidia.nvidia_smi_available() else "install the NVIDIA driver",
    )
    session = bench_nvidia.NvmlSession()
    try:
        if session.available:
            check("nvml", "ok", f"{session.count} GPU(s) visible")
        else:
            check("nvml", "warn", "no NVML device", "install/repair the NVIDIA driver (nvidia-smi)")
    finally:
        session.close()
    check(
        "rocm",
        "ok" if bench_amd.available() else "skip",
        bench_amd.rocm_version() or ("AMD device present" if bench_amd.available() else "not present"),
    )

    mode = config.config_mode()
    if mode is None:
        check("config file", "warn", "not created yet", "run `airigs login`")
    elif mode & 0o077:
        check("config file", "fail", f"mode {mode:o}", f"chmod 600 {config.config_path()}")
    else:
        check("config file", "ok", f"{config.config_path()} (mode {mode:o})")

    token_state = auth.describe_token(ctx.settings)
    state = token_state["state"]
    check(
        "token",
        "ok" if state in ("valid", "env") else ("fail" if state == "wrong-origin" else "warn"),
        f"{state} (source: {token_state['source']}, issuer: {token_state.get('tokenOrigin') or 'unknown'}, "
        f"expires in {token_state.get('expires_in') if token_state.get('expires_in') is not None else 'n/a'}s)",
        None
        if state in ("valid", "env")
        else (
            f"the token belongs to {token_state.get('tokenOrigin')}; run `airigs login` for "
            f"{token_state.get('currentOrigin')}"
            if state == "wrong-origin"
            else "run `airigs login`"
        ),
    )
    issuer_ok = ctx.settings.airigs_url.startswith("https://") or is_loopback_url(ctx.settings.airigs_url)
    check(
        "issuer transport",
        "ok" if issuer_ok else "fail",
        ctx.settings.airigs_url,
        None if issuer_ok else "bearer tokens are refused over plaintext HTTP to a remote host; use https",
    )
    if ctx.settings.airigs_token:
        try:
            info = auth.fetch_userinfo(ctx.settings.airigs_url, ctx.settings.airigs_token, timeout=10.0)
            check("token validity", "ok", f"userinfo OK for {info.get('username')}")
        except AirigsError as exc:
            check("token validity", "fail", str(exc), "run `airigs login` to refresh")

    reachable, detail = _probe(ctx.settings.airigs_url)
    check(
        "AIRigs reachable",
        "ok" if reachable else "fail",
        f"{ctx.settings.airigs_url} ({detail})",
        None if reachable else "start AIRigs or set AIRIGS_URL",
    )
    reachable, detail = _probe(ctx.settings.recipes_url)
    check(
        "Recipes reachable",
        "ok" if reachable else "fail",
        f"{ctx.settings.recipes_url} ({detail})",
        None if reachable else "start the Recipes platform or set RECIPES_URL",
    )
    check("mongodb", "skip", "not used by the CLI; the forum owns its own database")

    failed = [item for item in checks if item["status"] == "fail"]
    if args.json:
        print(json.dumps({"ok": not failed, "checks": checks}, indent=2))
    else:
        rows = []
        icons = {"ok": "ok", "warn": "warn", "fail": "FAIL", "skip": "skip"}
        for item in checks:
            row = [icons[item["status"]], item["check"], item["detail"]]
            if item["remediation"]:
                row.append(f"-> {item['remediation']}")
            rows.append(row)
        print(render_table(("state", "check", "detail", "action"), rows))
        if failed:
            print(f"\n{len(failed)} check(s) failed.", file=sys.stderr)
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# bench


def cmd_bench(args: argparse.Namespace) -> int:
    bench = bench_runner.run_benchmark(
        quick=args.quick,
        gpu=args.gpu,
        no_power=args.no_power,
        hold_load=args.hold_load,
    )
    written = None
    if args.output:
        written = bench_report.write_json(args.output, bench)
    missing = bench_report.missing_required(bench)
    if args.json:
        print(json.dumps(bench, indent=2))
    else:
        print(render_table(("field", "value"), bench_report.summary(bench)))
        if missing:
            print("\nThe server requires measurements that are missing:", file=sys.stderr)
            for item in missing:
                print(f"  - {item['field']}: {item['remediation']}", file=sys.stderr)
            print("Use `airigs bench --output FILE` to inspect, then fix and re-run.", file=sys.stderr)
        if written:
            print(f"\nWrote {written}", file=sys.stderr)
        elif not args.output:
            print("\nPass --output FILE to save this payload.", file=sys.stderr)
    return 0


def cmd_self_check(args: argparse.Namespace) -> int:
    result = bench_runner.self_check()
    bench = result.pop("benchmark")
    if args.json:
        print(json.dumps({**result, "benchmark": bench}, indent=2))
    else:
        print(render_table(("field", "value"), bench_report.summary(bench)))
        print()
        print(
            render_table(
                ("item", "result"),
                [
                    ("schema", "ok" if not result["schemaProblems"] else "FAILED"),
                    ("uploadable", "yes" if result.get("uploadable") else "no"),
                    ("schema problems", len(result["schemaProblems"])),
                    ("missing required (server)", len(result["missingRequired"])),
                    ("notes", len(bench.get("notes") or [])),
                ],
            )
        )
        for problem in result["schemaProblems"]:
            print(f"  schema: {problem}", file=sys.stderr)
        for item in result["missingRequired"]:
            print(f"  missing: {item['field']} -> {item['remediation']}", file=sys.stderr)
    return 0 if result["ok"] else 1


# ---------------------------------------------------------------------------
# rig


def cmd_rig_list(args: argparse.Namespace) -> int:
    ctx = Context(args)
    rigs = _rigs(ctx.airigs())
    rows = [
        (
            rig.get("id", ""),
            rig.get("revision", ""),
            _rig_name(rig),
            "yes" if rig.get("verified") else "no",
            len(rig.get("hardware") or []),
            len(rig.get("models") or []),
        )
        for rig in rigs
    ]
    emit(
        args,
        {"rigs": rigs},
        headers=("id", "revision", "name", "verified", "hardware", "models"),
        rows=rows,
        text=None if rigs else "No rigs yet. Create one with `airigs rig create --name NAME`.",
    )
    return 0


def cmd_rig_show(args: argparse.Namespace) -> int:
    ctx = Context(args)
    rig = _resolve_rig(ctx.airigs(), args.rig)
    if args.json:
        print(json.dumps(rig, indent=2))
        return 0
    print(
        render_table(
            ("field", "value"),
            [
                ("id", rig.get("id")),
                ("revision", rig.get("revision") or ""),
                ("name", _rig_name(rig)),
                ("verified", "yes" if rig.get("verified") else "no"),
                ("verifiedAt", rig.get("verifiedAt") or ""),
                ("notes", rig.get("notes") or ""),
            ],
        )
    )
    hardware = rig.get("hardware") or []
    if hardware:
        print()
        print(render_table(("kind", "count", "name", "specs"), [(item.get("kind"), item.get("count", 1), item.get("name"), json.dumps(item.get("specs") or {})) for item in hardware]))
    models = rig.get("models") or []
    if models:
        print()
        print(render_table(("source", "id", "name", "context", "quantization"), [(model.get("source"), model.get("id") or "", model.get("name"), model.get("context") or "", model.get("quantization") or "") for model in models]))
    benchmark = rig.get("benchmark")
    if isinstance(benchmark, dict):
        print()
        print(render_table(("benchmark", "value"), bench_report.summary(benchmark)))
    return 0


def cmd_rig_create(args: argparse.Namespace) -> int:
    ctx = Context(args)
    data: dict[str, Any] = {}
    if args.file:
        loaded = _load_json(args.file)
        if not isinstance(loaded, dict):
            raise AirigsError(f"{args.file} must contain a JSON object")
        data = loaded
    if args.notes and not data.get("notes"):
        data["notes"] = args.notes
    payload = _rig_payload(data, name=args.name)
    rig = ctx.airigs().post(RIG_API, payload, action="create rig")
    rig = rig.get("rig", rig) if isinstance(rig, dict) else rig
    rig_id = rig.get("id") if isinstance(rig, dict) else None
    emit(
        args,
        {"rig": rig},
        text=f"Created draft rig {rig_id} ({payload['name']}). Upload a benchmark with `airigs rig benchmark {rig_id}`.",
    )
    return 0


def cmd_rig_import(args: argparse.Namespace) -> int:
    ctx = Context(args)
    loaded = _load_json(args.file)
    if not isinstance(loaded, dict):
        raise AirigsError(f"{args.file} must contain a JSON object")
    payload = _rig_payload(loaded, name=args.name)
    rig = ctx.airigs().post(RIG_API, payload, action="import rig")
    rig = rig.get("rig", rig) if isinstance(rig, dict) else rig
    rig_id = rig.get("id") if isinstance(rig, dict) else None
    emit(
        args,
        {"rig": rig},
        text=f"Imported rig {rig_id} ({payload['name']}) with "
        f"{len(payload['hardware'])} hardware item(s) and {len(payload['models'])} model(s).",
    )
    return 0


def _request_challenge(client: ApiClient, rig_id: str) -> dict[str, str]:
    """Fetch a single-use benchmark nonce before measuring."""
    data = client.post(
        f"{RIG_API}/{rig_id}/benchmark/challenge", {}, action="request benchmark challenge"
    )
    if isinstance(data, dict) and isinstance(data.get("challenge"), dict):
        data = data["challenge"]
    nonce = data.get("nonce") if isinstance(data, dict) else None
    if not isinstance(nonce, str) or not nonce:
        raise AirigsError("the server did not return a benchmark challenge nonce")
    # issuedAt is when the CLI received the challenge; the server owns expiry.
    return {"nonce": nonce, "issuedAt": now_iso()}


def _challenge_rejected(exc: HttpError) -> bool:
    return getattr(exc, "status", None) in (409, 410)


def _require_verified(uploaded: Any) -> dict[str, Any]:
    rig = uploaded.get("rig", uploaded) if isinstance(uploaded, dict) else uploaded
    if not isinstance(rig, dict) or rig.get("verified") is not True:
        verified = rig.get("verified") if isinstance(rig, dict) else None
        raise AirigsError(
            "the server did not mark the rig verified "
            f"(verified={verified!r}); the rig was not accepted. Nothing else was changed."
        )
    return rig


def _upload_benchmark(client: ApiClient, rig_id: str, bench: dict[str, Any]) -> dict[str, Any]:
    """Attach a fresh challenge and upload. A 409/410 gets one fresh challenge."""
    bench_report.require_complete(bench)
    last_error: HttpError | None = None
    for attempt in (1, 2):
        payload = dict(bench)
        payload["challenge"] = _request_challenge(client, rig_id)
        try:
            uploaded = client.post(
                f"{RIG_API}/{rig_id}/benchmark", {"benchmark": payload}, action="upload benchmark"
            )
        except HttpError as exc:
            if attempt == 1 and _challenge_rejected(exc):
                last_error = exc
                continue
            raise AirigsError(
                f"benchmark upload was rejected (HTTP {exc.status}): {exc}. "
                "Re-run `airigs rig benchmark` to measure again."
            ) from exc
        return _require_verified(uploaded)
    raise AirigsError(
        "the server rejected two fresh benchmark challenges (expired or replayed); "
        "re-run `airigs rig benchmark` to measure again"
    ) from last_error


def _run_and_upload_benchmark(
    ctx: Context,
    rig_id: str,
    *,
    output: str | None = None,
    quick: bool = False,
    gpu: Any = "all",
    no_power: bool = False,
    hold_load: float | None = None,
) -> dict[str, Any]:
    client = ctx.airigs()
    last_error: HttpError | None = None
    for attempt in (1, 2):
        # The server issues a single-use challenge that must precede measurement.
        challenge = _request_challenge(client, rig_id)
        bench = bench_runner.run_benchmark(quick=quick, gpu=gpu, no_power=no_power, hold_load=hold_load)
        bench["challenge"] = challenge
        bench_report.require_complete(bench)
        if output:
            bench_report.write_json(output, bench)
        try:
            return _require_verified(
                client.post(
                    f"{RIG_API}/{rig_id}/benchmark", {"benchmark": bench}, action="upload benchmark"
                )
            )
        except HttpError as exc:
            if attempt == 1 and _challenge_rejected(exc):
                last_error = exc
                continue
            raise AirigsError(
                f"benchmark upload was rejected (HTTP {exc.status}): {exc}. "
                "Re-run `airigs rig benchmark` to measure again."
            ) from exc
    raise AirigsError(
        "the server rejected two fresh benchmark challenges (expired or replayed); "
        "re-run `airigs rig benchmark` to measure again"
    ) from last_error


def cmd_rig_benchmark(args: argparse.Namespace) -> int:
    ctx = Context(args)
    rig = _resolve_rig(ctx.airigs(), args.rig)
    rig_id = rig.get("id")
    result = _run_and_upload_benchmark(
        ctx,
        rig_id,
        output=args.output,
        quick=args.quick,
        gpu=args.gpu,
        no_power=args.no_power,
        hold_load=args.hold_load,
    )
    emit(
        args,
        {"rig": result, "verified": True},
        text=f"Rig {rig_id} is verified ({result.get('verifiedAt') or 'now'}).",
    )
    return 0


def cmd_rig_submit(args: argparse.Namespace) -> int:
    ctx = Context(args)
    data: dict[str, Any] = {}
    source = args.file
    if source is None and Path("rig.json").exists():
        source = "rig.json"
    if source:
        loaded = _load_json(source)
        if not isinstance(loaded, dict):
            raise AirigsError(f"{source} must contain a JSON object")
        data = loaded
    elif args.name:
        data = {"name": args.name, **_hardware_from_system()}
    payload = _rig_payload(data, name=args.name)

    client = ctx.airigs()
    rig_id = _validate_rig_id(data["id"]) if data.get("id") is not None else None
    if rig_id is None:
        existing = next(
            (rig for rig in _rigs(client) if _rig_name(rig).lower() == payload["name"].lower()),
            None,
        )
        if existing is not None:
            rig_id = existing.get("id")
    if rig_id is not None:
        rig = client.put(f"{RIG_API}/{rig_id}", payload, action="update rig")
    else:
        rig = client.post(RIG_API, payload, action="create rig")
    rig = rig.get("rig", rig) if isinstance(rig, dict) else rig
    if not isinstance(rig, dict) or rig.get("id") is None:
        raise AirigsError("the rig API did not return a rig id")
    rig_id = _validate_rig_id(rig["id"])

    if args.benchmark:
        bench = _load_json(args.benchmark)
        if not isinstance(bench, dict):
            raise AirigsError(f"{args.benchmark} must contain a JSON object")
        rig = _upload_benchmark(client, rig_id, bench)
    else:
        rig = _run_and_upload_benchmark(
            ctx,
            rig_id,
            output=args.output,
            quick=args.quick,
            gpu=args.gpu,
            no_power=args.no_power,
            hold_load=args.hold_load,
        )
    emit(
        args,
        {"rig": rig, "verified": rig.get("verified") is True},
        text=f"Rig {rig_id} submitted and verified ({rig.get('verifiedAt') or 'now'}).",
    )
    return 0


def cmd_rig_delete(args: argparse.Namespace) -> int:
    ctx = Context(args)
    rig = _resolve_rig(ctx.airigs(), args.rig)
    rig_id = rig.get("id")
    _confirm(args, f"Delete rig {rig_id} ({_rig_name(rig)})?")
    ctx.airigs().delete(f"{RIG_API}/{rig_id}", action="delete rig")
    emit(args, {"deleted": True, "id": rig_id}, text=f"Deleted rig {rig_id}.")
    return 0


# ---------------------------------------------------------------------------
# recipe


def cmd_recipe_init(args: argparse.Namespace) -> int:
    path = Path(args.file or recipes.RECIPE_FILENAME)
    if path.exists() and not args.force:
        raise AirigsError(f"{path} already exists; pass --force to overwrite")
    document = recipes.skeleton(args.name)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    emit(
        args,
        {"path": str(path), "recipe": document},
        text=f"Wrote {path}. Edit it, then run `airigs recipe validate`.",
    )
    return 0


def _load_recipe_file(path: str) -> dict[str, Any]:
    document = _load_json(path)
    if not isinstance(document, dict):
        raise AirigsError(f"{path} must contain a JSON object")
    return document


def cmd_recipe_validate(args: argparse.Namespace) -> int:
    path = args.file or recipes.RECIPE_FILENAME
    document = _load_recipe_file(path)
    problems = recipes.validate_recipe(document)
    if args.json:
        print(json.dumps({"valid": not problems, "problems": problems, "path": path}, indent=2))
    else:
        if problems:
            print(f"{path}: invalid", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
        else:
            print(f"{path}: valid")
    return 1 if problems else 0


def cmd_recipe_submit(args: argparse.Namespace) -> int:
    ctx = Context(args)
    path = args.file or recipes.RECIPE_FILENAME
    document = _load_recipe_file(path)
    client = ctx.recipes()
    engines, _source = recipes.engines_for(client)
    payload = recipes.require_valid(document, engines=engines)
    payload = _enrich_rig_revision(ctx, payload)
    if args.publish:
        payload["publish"] = True
    created = client.create(payload)
    recipe = created.get("recipe", created) if isinstance(created, dict) else created
    slug = recipe.get("slug") if isinstance(recipe, dict) else None
    emit(
        args,
        {"recipe": recipe},
        text=f"Submitted recipe {slug or '(no slug returned)'}"
        + (" and requested publication." if args.publish else " as a draft."),
    )
    return 0


def cmd_recipe_bench(args: argparse.Namespace) -> int:
    ctx = Context(args)
    target = args.target
    document: dict[str, Any] | None = None
    client = ctx.recipes()
    engines, _source = recipes.engines_for(client)
    if os.path.isfile(target):
        document = _load_recipe_file(target)
        recipes.require_valid(document, engines=engines)
        slug = document.get("slug")
        if not slug:
            draft = _enrich_rig_revision(ctx, recipes.require_valid(document, engines=engines))
            created = client.create(draft)
            recipe = created.get("recipe", created) if isinstance(created, dict) else created
            slug = recipe.get("slug") if isinstance(recipe, dict) else None
            if not slug:
                raise AirigsError("the Recipes platform did not return a slug for the draft recipe")
    else:
        slug = target

    rig_id = args.rig_id or (document or {}).get("rigId")
    if rig_id is not None:
        rig_id = _validate_rig_id(rig_id)
    if rig_id is None:
        rig_id = _default_rig_id(ctx)
        if rig_id is not None:
            print(f"Using rig {rig_id} (pass --rig-id to override).", file=sys.stderr)
    if rig_id is None:
        raise AirigsError(
            "no rig id: pass --rig-id r_..., set rigId in the recipe file, or create exactly one "
            "rig with `airigs rig submit`"
        )

    engine = args.engine or (document or {}).get("engine") or "vllm"
    if engine not in engines:
        raise AirigsError(
            f"engine {engine!r} is not supported by the platform (use one of {', '.join(engines)})"
        )
    benchmark = recipebench.run(
        args.endpoint,
        engine=engine,
        context_tokens=args.context,
        concurrency=args.concurrency,
        requests=args.requests,
        model_ids=args.model or None,
        rig_id=rig_id,
        api_key=args.api_key,
    )
    if args.output:
        bench_report.write_json(args.output, benchmark)
    recipebench.require_complete(benchmark)
    uploaded = client.benchmark(slug, benchmark, publish=args.publish)
    if args.json:
        print(json.dumps({"slug": slug, "benchmark": benchmark, "response": uploaded}, indent=2))
    else:
        print(
            render_table(
                ("metric", "value"),
                [
                    ("models", ", ".join(benchmark["modelIds"])),
                    ("requests", f"{benchmark['requests']} at concurrency {benchmark['concurrency']}"),
                    ("TTFT", f"{benchmark['ttftMs']} ms (median)"),
                    ("output", f"{benchmark['outputTokensPerSec']} tok/s (aggregate)"),
                    ("total", f"{benchmark['totalTokensPerSec']} tok/s"),
                    ("prompt", f"{benchmark['promptTokensPerSec']} tok/s prefill"),
                    ("VRAM peak", f"{benchmark['vramPeakBytes']} bytes"),
                    ("GPU power", f"{benchmark['powerAvgW']} W avg"),
                ],
            )
        )
        print(f"\nUploaded benchmark for {slug}." + (" Published." if args.publish else ""))
    return 0


def _resolve_script_with_fallback(
    ctx: Context, slug: str, file_path: str | None
) -> tuple[str, str]:
    """Fetch the platform script; if it is absent, generate from recipe.json."""
    document = _load_recipe_file(file_path) if file_path else None
    client = ctx.recipes() if not document else None
    try:
        return resolve_script(client, slug, recipe=document)
    except ScriptError:
        local = Path(recipes.RECIPE_FILENAME)
        if document or not local.exists():
            raise
        generated = generate_script(_load_recipe_file(str(local)))
        print(
            "warning: the platform has no script for this recipe; generated one from recipe.json",
            file=sys.stderr,
        )
        return generated, "local recipe.json"


def cmd_recipe_script(args: argparse.Namespace) -> int:
    ctx = Context(args)
    script, source = _resolve_script_with_fallback(ctx, args.slug, args.file)
    if args.out:
        Path(args.out).expanduser().write_text(script, encoding="utf-8")
        emit(args, {"path": args.out, "source": source}, text=f"Wrote {args.out} ({source}).")
    else:
        if args.json:
            print(json.dumps({"slug": args.slug, "source": source, "script": script}, indent=2))
        else:
            print(script, end="" if script.endswith("\n") else "\n")
    return 0


def _strip_separator(extra: list[str] | None) -> list[str] | None:
    values = list(extra or [])
    if values and values[0] == "--":
        values = values[1:]
    return values or None


def cmd_recipe_run(args: argparse.Namespace) -> int:
    ctx = Context(args)
    script, source = _resolve_script_with_fallback(ctx, args.slug, args.file)
    code = execute(
        script,
        yes=args.yes,
        slug=args.slug,
        script_args=_strip_separator(args.extra),
        printer=lambda line: print(line, file=sys.stderr if args.json else sys.stdout),
        warn=lambda line: print(f"warning: {line}", file=sys.stderr),
    )
    emit(args, {"slug": args.slug, "source": source, "exitCode": code})
    return code


# ---------------------------------------------------------------------------
# Parser


def add_common(parser: argparse.ArgumentParser, *, suppress: bool = True) -> None:
    """Add the global flags.

    ``suppress=True`` (the shared parent for subcommands) leaves values untouched
    when a flag is absent, so a ``--json`` given before the subcommand survives
    argparse's subparser namespace copy. The root parser uses real defaults and
    owns separate Action objects, so ``set_defaults`` cannot mutate shared ones.
    """
    default = argparse.SUPPRESS if suppress else None
    parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS if suppress else False, help="machine-readable JSON output")
    parser.add_argument("--verbose", action="store_true", default=argparse.SUPPRESS if suppress else False, help="log HTTP requests (tokens redacted)")
    parser.add_argument("--airigs-url", metavar="URL", default=default, help="AIRigs base URL (env AIRIGS_URL)")
    parser.add_argument("--recipes-url", metavar="URL", default=default, help="Recipes base URL (env RECIPES_URL)")


def _bench_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--quick", action="store_true", help="fewer iterations (CI / smoke runs)")
    parser.add_argument("--gpu", default="all", metavar="N|all", help="restrict GPU measurements to one index")
    parser.add_argument("--no-power", action="store_true", help="skip power sampling (fields stay null)")
    parser.add_argument("--hold-load", type=float, default=None, metavar="SECONDS", help="duration of the load phase")


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    add_common(common)

    parser = argparse.ArgumentParser(
        prog="airigs",
        description="AIRigs official CLI: authenticate, benchmark hardware, submit rigs and recipes.",
    )
    add_common(parser, suppress=False)
    parser.set_defaults(self_check=False)
    parser.add_argument("--version", action="version", version=f"airigs {__version__}")
    parser.add_argument("--self-check", action="store_true", help="run a quick benchmark and validate its own output")

    sub = parser.add_subparsers(dest="command")

    login = sub.add_parser("login", parents=[common], help="authenticate with AIRigs")
    login.add_argument("--no-browser", action="store_true", help="do not try to open a browser")
    login.add_argument(
        "--token",
        metavar="-",
        help="read a token from stdin (only '-' is accepted; or set AIRIGS_TOKEN)",
    )
    login.set_defaults(func=cmd_login)

    logout = sub.add_parser("logout", parents=[common], help="remove stored tokens")
    logout.set_defaults(func=cmd_logout)

    whoami = sub.add_parser("whoami", parents=[common], help="show the authenticated AIRigs user")
    whoami.set_defaults(func=cmd_whoami)

    doctor = sub.add_parser("doctor", parents=[common], help="check the environment and connectivity")
    doctor.set_defaults(func=cmd_doctor)

    bench = sub.add_parser("bench", parents=[common], help="run the hardware benchmark")
    _bench_options(bench)
    bench.add_argument("--output", metavar="FILE", help="write the payload JSON to FILE")
    bench.add_argument("--self-check", action="store_true", default=argparse.SUPPRESS, dest="self_check")
    bench.set_defaults(func=cmd_bench)

    rig = sub.add_parser("rig", parents=[common], help="manage rigs on AIRigs")
    rig_sub = rig.add_subparsers(dest="rig_command", required=True)

    rig_list = rig_sub.add_parser("list", parents=[common], help="list your rigs")
    rig_list.set_defaults(func=cmd_rig_list)

    rig_show = rig_sub.add_parser("show", parents=[common], help="show one rig by id or name")
    rig_show.add_argument("rig")
    rig_show.set_defaults(func=cmd_rig_show)

    rig_create = rig_sub.add_parser("create", parents=[common], help="create a draft rig")
    rig_create.add_argument("--name", required=True, help="rig name")
    rig_create.add_argument("--file", metavar="FILE", help="hardware/models JSON (same shape as rig import)")
    rig_create.add_argument("--notes", help="free-form notes")
    rig_create.set_defaults(func=cmd_rig_create)

    rig_import = rig_sub.add_parser("import", parents=[common], help="create a rig from a JSON file")
    rig_import.add_argument("file")
    rig_import.add_argument("--name", help="override the name in the file")
    rig_import.set_defaults(func=cmd_rig_import)

    rig_bench = rig_sub.add_parser("benchmark", parents=[common], help="benchmark this machine and upload")
    rig_bench.add_argument("rig")
    rig_bench.add_argument("--output", metavar="FILE", help="also write the payload JSON")
    _bench_options(rig_bench)
    rig_bench.set_defaults(func=cmd_rig_benchmark)

    rig_submit = rig_sub.add_parser("submit", parents=[common], help="create/update a rig and upload a benchmark")
    rig_submit.add_argument("--file", metavar="FILE", help="rig JSON (hardware/models/name)")
    rig_submit.add_argument("--name", help="rig name (auto-detects hardware when no file is given)")
    rig_submit.add_argument("--benchmark", metavar="FILE", help="upload a previously saved benchmark instead of running one")
    rig_submit.add_argument("--output", metavar="FILE", help="also write the payload JSON")
    _bench_options(rig_submit)
    rig_submit.set_defaults(func=cmd_rig_submit)

    rig_delete = rig_sub.add_parser("delete", parents=[common], help="delete a rig")
    rig_delete.add_argument("rig")
    rig_delete.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    rig_delete.set_defaults(func=cmd_rig_delete)

    recipe = sub.add_parser("recipe", parents=[common], help="manage recipes and recipe benchmarks")
    recipe_sub = recipe.add_subparsers(dest="recipe_command", required=True)

    r_init = recipe_sub.add_parser("init", parents=[common], help="write a recipe.json skeleton")
    r_init.add_argument("--name", default="my-recipe", help="recipe title")
    r_init.add_argument("--file", default=None, help="path to write (default recipe.json)")
    r_init.add_argument("--force", action="store_true", help="overwrite an existing file")
    r_init.set_defaults(func=cmd_recipe_init)

    r_validate = recipe_sub.add_parser("validate", parents=[common], help="validate a recipe file")
    r_validate.add_argument("file", nargs="?", default=None)
    r_validate.set_defaults(func=cmd_recipe_validate)

    r_submit = recipe_sub.add_parser("submit", parents=[common], help="submit a recipe draft")
    r_submit.add_argument("file", nargs="?", default=None)
    r_submit.add_argument("--publish", action="store_true", help="publish after submission")
    r_submit.set_defaults(func=cmd_recipe_submit)

    r_bench = recipe_sub.add_parser("bench", parents=[common], help="benchmark an OpenAI-compatible endpoint")
    r_bench.add_argument("target", metavar="slug|file")
    r_bench.add_argument("--endpoint", required=True, metavar="URL", help="OpenAI-compatible base URL")
    r_bench.add_argument("--engine", help="engine name recorded in the payload")
    r_bench.add_argument("--context", type=int, default=2048, metavar="N", help="prompt size in tokens")
    r_bench.add_argument("--concurrency", type=int, default=4)
    r_bench.add_argument("--requests", type=int, default=8)
    r_bench.add_argument("--model", action="append", metavar="ID", help="model id (repeatable)")
    r_bench.add_argument("--rig-id", metavar="r_...", help="opaque rig id (defaults to the recipe / your single rig)")
    r_bench.add_argument("--api-key", help="endpoint API key (usually not needed locally)")
    r_bench.add_argument("--publish", action="store_true", help="publish the recipe after uploading")
    r_bench.add_argument("--output", metavar="FILE", help="also write the benchmark JSON")
    r_bench.set_defaults(func=cmd_recipe_bench)

    r_script = recipe_sub.add_parser("script", parents=[common], help="fetch or generate a launch script")
    r_script.add_argument("slug")
    r_script.add_argument("--out", metavar="FILE", help="write to FILE instead of stdout")
    r_script.add_argument("--file", metavar="FILE", help="generate from a local recipe file")
    r_script.set_defaults(func=cmd_recipe_script)

    r_run = recipe_sub.add_parser("run", parents=[common], help="run a recipe's launch script")
    r_run.add_argument("slug")
    r_run.add_argument("--file", metavar="FILE", help="use a local recipe file instead of fetching")
    r_run.add_argument("--yes", action="store_true", help="run without the interactive confirmation")
    r_run.set_defaults(func=cmd_recipe_run)

    return parser


GLOBAL_VALUE_FLAGS = ("--airigs-url", "--recipes-url")


def hoist_global_flags(argv: Sequence[str]) -> list[str]:
    """Move root-level flags in front of the subcommand.

    argparse's subparser action copies a fresh namespace over the parent, so a
    ``--json`` placed before ``rig`` is lost. Hoisting keeps both orders working
    and stops at a bare ``--`` (recipe passthrough arguments).
    """
    hoisted: list[str] = []
    rest: list[str] = []
    index = 0
    values = list(argv)
    while index < len(values):
        token = values[index]
        if token == "--":
            rest.extend(values[index:])
            break
        if token in ("--json", "--verbose"):
            hoisted.append(token)
        elif token in GLOBAL_VALUE_FLAGS and index + 1 < len(values):
            hoisted.extend((token, values[index + 1]))
            index += 1
        elif token.startswith(tuple(flag + "=" for flag in GLOBAL_VALUE_FLAGS)):
            hoisted.append(token)
        else:
            rest.append(token)
        index += 1
    return hoisted + rest


def split_passthrough(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split ``recipe run slug -- a b`` into parser args and script arguments."""
    values = list(argv)
    try:
        index = values.index("--")
    except ValueError:
        return values, []
    return values[:index], values[index + 1 :]


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    raw, extra = split_passthrough(raw)
    args = parser.parse_args(hoist_global_flags(raw))
    args.extra = extra
    if getattr(args, "self_check", False):
        args.command = "self-check"
        args.output = None
        return cmd_self_check(args)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    try:
        return int(args.func(args))
    except (ConfigError, AirigsError, RecipeError, bench_report.IncompleteBenchmark,
            recipebench.RecipeBenchmarkIncomplete, ScriptError) as exc:
        status = getattr(exc, "status", None)
        if getattr(args, "json", False):
            print(json.dumps({"error": str(exc), "status": status}, indent=2), file=sys.stderr)
        else:
            print(f"error: {exc}", file=sys.stderr)
        return getattr(exc, "exit_code", 1)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
