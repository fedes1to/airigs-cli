"""Benchmark payload construction, required-field validation and summary tables."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from .common import Notes

SCHEMA_VERSION = 1

TOP_LEVEL_KEYS = (
    "schemaVersion",
    "runner",
    "startedAt",
    "finishedAt",
    "durationS",
    "system",
    "cpuBench",
    "gpus",
    "interconnect",
    "power",
    "notes",
)

REMEDIATION = {
    "gpus": "no GPU was detected; install the vendor driver and `pynvml` (NVIDIA) or run on a machine with a visible GPU",
    "gpus.name": "GPU name was not reported; install NVIDIA/AMD drivers or `pip install pynvml`",
    "gpus.vramBytes": "VRAM size unavailable; install `pynvml` (NVIDIA) or check `rocm-smi` (AMD)",
    "gpus.memoryBandwidth.measuredGBs": "install PyTorch for your GPU: `pip install 'airigs[gpu]'`",
    "gpus.tflops.fp16": "install PyTorch for your GPU: `pip install 'airigs[gpu]'` (fp16 matmul required)",
    "gpus.power.loadW": "GPU power sensor unavailable; install `pynvml`/NVML or rocm-smi, or pass `--no-power` to keep the field null only if you accept an unverified rig",
    "gpus.pcie.version": "PCIe link generation unavailable; install `nvidia-smi` (NVIDIA) or check sysfs (AMD)",
    "gpus.pcie.width": "PCIe link width unavailable; install `nvidia-smi` (NVIDIA) or check sysfs (AMD)",
    "cpuBench.memory.readGBs": "RAM bandwidth was not measured; install numpy (it is a hard dependency)",
    "cpuBench.memory.copyGBs": "RAM bandwidth was not measured; install numpy (it is a hard dependency)",
    "system.cpu.model": "CPU model unreadable; on Linux check /proc/cpuinfo and the cpu MHz sysfs entries",
    "system.memory.totalBytes": "RAM capacity unreadable; install psutil",
    "interconnect.p2p.enabled": "P2P state unknown; install PyTorch so peer access can be queried",
    "runner.name": "internal error: the harness did not identify itself",
    "runner.version": "internal error: the harness did not report its version",
    "startedAt": "internal error: start time missing",
    "finishedAt": "internal error: finish time missing",
}


class IncompleteBenchmark(Exception):
    """One or more server-required measurements are missing."""

    def __init__(self, missing: list[dict[str, str]]) -> None:
        self.missing = missing
        lines = ["the benchmark is missing measurements the server requires:"]
        for item in missing:
            lines.append(f"  - {item['field']}: {item['remediation']}")
        lines.append("Nothing was uploaded.")
        super().__init__("\n".join(lines))


def _absent(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _get(data: Any, path: str) -> Any:
    current = data
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _missing_fields(bench: dict[str, Any]) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []

    def check(path: str, value: Any) -> None:
        if _absent(value):
            missing.append({"field": path, "remediation": REMEDIATION.get(path, "measurement unavailable")})

    if bench.get("schemaVersion") != SCHEMA_VERSION:
        missing.append({"field": "schemaVersion", "remediation": "internal error: wrong schema version"})
    for path in (
        "runner.name",
        "runner.version",
        "startedAt",
        "finishedAt",
        "system.cpu.model",
        "system.memory.totalBytes",
        "cpuBench.memory.readGBs",
        "cpuBench.memory.copyGBs",
    ):
        check(path, _get(bench, path))

    p2p = _get(bench, "interconnect.p2p.enabled")
    if not isinstance(p2p, bool):
        check("interconnect.p2p.enabled", None)

    gpus = _get(bench, "gpus")
    if not isinstance(gpus, list) or not gpus:
        missing.append({"field": "gpus", "remediation": REMEDIATION["gpus"]})
        return missing
    for index, gpu in enumerate(gpus):
        if not isinstance(gpu, dict):
            missing.append({"field": f"gpus[{index}]", "remediation": "malformed GPU entry"})
            continue
        for field in (
            "name",
            "vramBytes",
            "memoryBandwidth.measuredGBs",
            "tflops.fp16",
            "power.loadW",
            "pcie.version",
            "pcie.width",
        ):
            path = f"gpus.{field}"
            value = _get(gpu, field)
            if _absent(value):
                missing.append(
                    {
                        "field": f"gpus[{index}].{field}",
                        "remediation": REMEDIATION.get(path, "measurement unavailable"),
                    }
                )
    return missing


def missing_required(bench: dict[str, Any]) -> list[dict[str, str]]:
    return _missing_fields(bench)


def require_complete(bench: dict[str, Any]) -> None:
    missing = _missing_fields(bench)
    if missing:
        raise IncompleteBenchmark(missing)


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _expect_type(problems: list[str], data: Any, path: str, types: tuple[str, ...]) -> None:
    value = _get(data, path)
    actual = _type_name(value)
    if actual not in types:
        problems.append(f"{path}: expected {'|'.join(types)}, got {actual}")


def check_shape(bench: dict[str, Any]) -> list[str]:
    """Local structural validation of the payload (used by ``--self-check``)."""
    problems: list[str] = []
    if not isinstance(bench, dict):
        return ["payload is not a JSON object"]
    for key in TOP_LEVEL_KEYS:
        if key not in bench:
            problems.append(f"{key}: missing from payload")
    for path, types in (
        ("schemaVersion", ("number",)),
        ("runner", ("object",)),
        ("startedAt", ("string",)),
        ("finishedAt", ("string",)),
        ("durationS", ("number",)),
        ("system", ("object",)),
        ("cpuBench", ("object",)),
        ("gpus", ("array",)),
        ("interconnect", ("object",)),
        ("power", ("object",)),
        ("notes", ("array",)),
        ("runner.name", ("string",)),
        ("runner.version", ("string",)),
        ("system.os", ("object",)),
        ("system.cpu", ("object",)),
        ("system.memory", ("object",)),
        ("system.drivers", ("object",)),
        ("cpuBench.singleThread", ("object",)),
        ("cpuBench.multiThread", ("object",)),
        ("cpuBench.memory", ("object",)),
        ("cpuBench.singleThread.sha256Mbps", ("number", "null")),
        ("cpuBench.multiThread.sha256Mbps", ("number", "null")),
        ("cpuBench.fp32Gflops", ("number", "null")),
        ("cpuBench.fp64Gflops", ("number", "null")),
        ("interconnect.p2p", ("object",)),
        ("interconnect.nvlink", ("object",)),
        ("interconnect.p2p.enabled", ("boolean",)),
        ("power.measurement", ("string", "null")),
    ):
        _expect_type(problems, bench, path, types)
    if not isinstance(bench.get("notes"), list) or not all(isinstance(n, str) for n in bench.get("notes") or []):
        problems.append("notes: must be an array of strings")
    challenge = bench.get("challenge")
    if challenge is not None:
        if not isinstance(challenge, dict) or not isinstance(challenge.get("nonce"), str) or not challenge["nonce"]:
            problems.append("challenge.nonce: expected non-empty string")
        if not isinstance(challenge.get("issuedAt"), str):
            problems.append("challenge.issuedAt: expected ISO timestamp")
    gpus = bench.get("gpus")
    if isinstance(gpus, list):
        for index, gpu in enumerate(gpus):
            if not isinstance(gpu, dict):
                problems.append(f"gpus[{index}]: must be an object")
                continue
            for field in (
                "index",
                "name",
                "vramBytes",
                "pcie",
                "clocks",
                "power",
                "memoryBandwidth",
                "tflops",
                "thermal",
            ):
                if field not in gpu:
                    problems.append(f"gpus[{index}].{field}: missing")
            for field in ("version", "width", "currentVersion", "currentWidth", "maxVersion", "maxWidth"):
                if not isinstance((gpu.get("pcie") or {}).get(field), (int, type(None))):
                    problems.append(f"gpus[{index}].pcie.{field}: expected number|null")
            for field in ("measuredGBs", "theoreticalGBs"):
                value = (gpu.get("memoryBandwidth") or {}).get(field)
                if not isinstance(value, (int, float, type(None))):
                    problems.append(f"gpus[{index}].memoryBandwidth.{field}: expected number|null")
            for field in ("fp64", "fp32", "tf32", "bf16", "fp16", "fp8", "int8"):
                value = (gpu.get("tflops") or {}).get(field)
                if not isinstance(value, (int, float, type(None))):
                    problems.append(f"gpus[{index}].tflops.{field}: expected number|null")
    return problems


def build(
    *,
    started_at: str,
    finished_at: str,
    duration_s: float,
    system: dict[str, Any],
    cpu_bench: dict[str, Any],
    gpus: list[dict[str, Any]],
    interconnect: dict[str, Any],
    power: dict[str, Any],
    notes: Notes | Iterable[str],
    runner: dict[str, Any],
    under_load: dict[str, Any] | None = None,
    challenge: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the exact Benchmark v1 payload, ordered like the contract."""
    system_out = dict(system)
    if under_load:
        # Additive extension (not part of the required schema): the measurements
        # taken while the load phase ran, so a rig page can show real load numbers.
        system_out["underLoad"] = under_load
    payload: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "runner": runner,
        "startedAt": started_at,
        "finishedAt": finished_at,
        "durationS": float(round(duration_s, 3)),
    }
    if challenge:
        payload["challenge"] = challenge
    payload.update({
        "system": system_out,
        "cpuBench": cpu_bench,
        "gpus": gpus,
        "interconnect": interconnect,
        "power": power,
        "notes": list(notes) if not isinstance(notes, Notes) else list(notes),
    })
    return payload


def write_json(path: str | os.PathLike[str], bench: dict[str, Any]) -> Path:
    target = Path(path).expanduser()
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(bench, indent=2) + "\n", encoding="utf-8")
    return target


def read_json(path: str | os.PathLike[str]) -> Any:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def summary(bench: dict[str, Any]) -> list[tuple[str, str]]:
    """Rows for the compact human summary."""
    rows: list[tuple[str, str]] = []
    runner = bench.get("runner") or {}
    rows.append(("runner", f"{runner.get('name')} {runner.get('version')} ({runner.get('platform')}/{runner.get('arch')})"))
    rows.append(("window", f"{bench.get('startedAt')} -> {bench.get('finishedAt')} ({bench.get('durationS')}s)"))
    system = bench.get("system") or {}
    cpu = system.get("cpu") or {}
    cores = f"{cpu.get('physicalCores')}c/{cpu.get('logicalCores')}t" if cpu.get("physicalCores") else f"{cpu.get('logicalCores')}t"
    rows.append(("cpu", f"{cpu.get('model') or 'unknown'} [{cores}]"))
    memory = system.get("memory") or {}
    total = memory.get("totalBytes")
    rows.append(("ram", f"{_gib(total)} {memory.get('type') or ''} {memory.get('speedMhz') or ''}".rstrip()))
    cpu_bench = bench.get("cpuBench") or {}
    single = (cpu_bench.get("singleThread") or {}).get("sha256Mbps")
    multi = (cpu_bench.get("multiThread") or {}).get("sha256Mbps")
    rows.append(("sha256", f"single {_fmt(single)} MB/s | multi {_fmt(multi)} MB/s"))
    rows.append(("fp32/fp64", f"{_fmt(cpu_bench.get('fp32Gflops'))} / {_fmt(cpu_bench.get('fp64Gflops'))} GFLOP/s"))
    ram = cpu_bench.get("memory") or {}
    rows.append(("ram bw", f"read {_fmt(ram.get('readGBs'))} | write {_fmt(ram.get('writeGBs'))} | copy {_fmt(ram.get('copyGBs'))} | triad {_fmt(ram.get('triadGBs'))} GB/s"))
    for gpu in bench.get("gpus") or []:
        bw = gpu.get("memoryBandwidth") or {}
        tf = gpu.get("tflops") or {}
        power_gpu = gpu.get("power") or {}
        rows.append(
            (
                f"gpu{gpu.get('index')}",
                f"{gpu.get('name') or '?'} | {_gib(gpu.get('vramBytes'))} | "
                f"bw {_fmt(bw.get('measuredGBs'))}/{_fmt(bw.get('theoreticalGBs'))} GB/s | "
                f"fp16 {_fmt(tf.get('fp16'))} TFLOPS | {_fmt(power_gpu.get('loadW'))} W",
            )
        )
    inter = (bench.get("interconnect") or {}).get("p2p") or {}
    rows.append(("p2p", f"supported={inter.get('supported')} enabled={inter.get('enabled')}"))
    power = bench.get("power") or {}
    rows.append(("power", f"idle {_fmt(power.get('idleW'))} W | load {_fmt(power.get('loadW'))} W ({power.get('measurement')})"))
    return rows


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _gib(value: Any) -> str:
    if not isinstance(value, (int, float)) or value <= 0:
        return "n/a"
    return f"{value / (1024 ** 3):.1f} GiB"
