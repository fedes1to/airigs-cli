"""Benchmark orchestration: specs, measurements, load phase and the payload."""

from __future__ import annotations

import hashlib
import platform
import statistics
import sys
import threading
import time
from typing import Any

from .. import __version__
from ..http import AirigsError
from . import amd, apple, cpu, memory, nvidia, p2p, power, report
from .common import Notes, cpu_logical_count, elapsed_seconds, import_optional, now_iso, utc_now

DEFAULT_LOAD_SECONDS = 10.0
QUICK_LOAD_SECONDS = 3.0
IDLE_SECONDS = 2.0
QUICK_IDLE_SECONDS = 0.6


def runner_info() -> dict[str, Any]:
    return {
        "name": "airigs-cli",
        "version": __version__,
        "platform": sys.platform,
        "arch": platform.machine() or None,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    }


def collect_system(*, notes: Notes) -> dict[str, Any]:
    cpu_spec = cpu.cpu_info()
    if not cpu_spec.get("model"):
        notes.add("CPU model could not be read.")
    mem = memory.memory_specs(notes)
    drivers = {
        "nvidia": nvidia.driver_info(),
        "amd": amd.driver_info(),
        "metal": apple.driver_info(),
    }
    return {
        "hostname": cpu.hostname() or None,
        "os": cpu.os_info(),
        "cpu": cpu_spec,
        "memory": mem,
        "drivers": drivers,
    }


def collect_gpus(
    *, quick: bool, notes: Notes
) -> tuple[list[dict[str, Any]], nvidia.NvmlSession, str | None]:
    """Detect GPUs and run the matching torch measurements.

    Returns ``(gpus, nvml_session, vendor)``. The session must be closed by the
    caller; it is inert when NVML is unavailable.
    """
    session = nvidia.NvmlSession()
    gpus = nvidia.enumerate_gpus(notes=notes, session=session)
    if gpus:
        nvidia.run_torch_measurements(gpus, quick=quick, notes=notes)
        return gpus, session, "nvidia"

    session.close()
    gpus = amd.run(quick=quick, notes=notes)
    if gpus:
        return gpus, session, "amd"

    gpus = apple.run(quick=quick, notes=notes)
    return gpus, session, "apple" if gpus else None


def _select_gpus(gpus: list[dict[str, Any]], selection: Any) -> list[dict[str, Any]]:
    if not gpus or selection in (None, "", "all"):
        return gpus
    try:
        wanted = int(selection)
    except (TypeError, ValueError) as exc:
        raise AirigsError(f"--gpu expects an index or 'all', got {selection!r}") from exc
    chosen = [gpu for gpu in gpus if gpu.get("index") == wanted]
    if not chosen:
        available = ", ".join(str(gpu.get("index")) for gpu in gpus) or "none"
        raise AirigsError(f"GPU {wanted} not found (visible GPUs: {available})")
    return chosen


def _gpu_power_reader(session: nvidia.NvmlSession, vendor: str | None):
    if vendor == "amd":
        return amd.power_w
    return session.power_w


def _gpu_idle_power(session: nvidia.NvmlSession, gpus: list[dict[str, Any]], vendor: str | None, duration: float) -> None:
    reader = _gpu_power_reader(session, vendor)
    per_gpu: dict[int, list[float]] = {gpu.get("index", 0): [] for gpu in gpus}
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        for gpu in gpus:
            index = gpu.get("index", 0)
            value = reader(index)
            if value is not None:
                per_gpu[index].append(value)
        time.sleep(0.2)
    for gpu in gpus:
        values = per_gpu.get(gpu.get("index", 0), [])
        if values:
            gpu.setdefault("power", {})["idleW"] = round(statistics.median(values), 2)
            gpu["power"]["measured"] = True


class _LoadGenerator:
    """Deterministic CPU + GPU load, stopped by an event."""

    def __init__(self, gpus: list[dict[str, Any]], vendor: str | None, notes: Notes) -> None:
        self.gpus = gpus
        self.vendor = vendor
        self.notes = notes
        self.stop = threading.Event()
        self.threads: list[threading.Thread] = []
        self.torch = import_optional("torch")
        self._cuda = False
        self._mps = False
        if self.torch is not None:
            try:
                self._cuda = bool(self.torch.cuda.is_available())
            except Exception:
                self._cuda = False
            try:
                self._mps = bool(self.torch.backends.mps.is_available())
            except Exception:
                self._mps = False
            try:
                self.torch.manual_seed(1234567)
            except Exception:
                pass

    def _cpu_worker(self) -> None:
        block = hashlib.sha256(b"airigs-load").digest() * 4096
        counter = 0
        while not self.stop.is_set():
            hashlib.sha256(block + counter.to_bytes(8, "little")).digest()
            counter += 1

    def _gpu_worker(self, index: int) -> None:
        torch = self.torch
        if torch is None or not self._cuda:
            return
        size = 2048
        device = torch.device(f"cuda:{index}")
        try:
            dtype = torch.float16
            a = torch.randn((size, size), device=device, dtype=dtype)
            b = torch.randn((size, size), device=device, dtype=dtype)
            while not self.stop.is_set():
                torch.matmul(a, b)
            torch.cuda.synchronize()
        except Exception as exc:
            self.notes.add(f"GPU load generation on device {index} stopped: {exc.__class__.__name__}.")

    def _mps_worker(self) -> None:
        torch = self.torch
        if torch is None or not self._mps:
            return
        try:
            size = 1024
            a = torch.randn((size, size), device="mps", dtype=torch.float16)
            b = torch.randn((size, size), device="mps", dtype=torch.float16)
            while not self.stop.is_set():
                torch.matmul(a, b)
            torch.mps.synchronize()
        except Exception as exc:
            self.notes.add(f"Apple GPU load generation stopped: {exc.__class__.__name__}.")

    def start(self) -> None:
        for _ in range(max(1, cpu_logical_count())):
            thread = threading.Thread(target=self._cpu_worker, daemon=True)
            thread.start()
            self.threads.append(thread)
        if self.vendor == "apple":
            thread = threading.Thread(target=self._mps_worker, daemon=True)
            thread.start()
            self.threads.append(thread)
        elif self._cuda:
            for gpu in self.gpus:
                thread = threading.Thread(target=self._gpu_worker, args=(gpu.get("index", 0),), daemon=True)
                thread.start()
                self.threads.append(thread)

    def shutdown(self) -> None:
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=5.0)
        self.threads.clear()
        if self.torch is not None and self._cuda:
            try:
                self.torch.cuda.synchronize()
                self.torch.cuda.empty_cache()
            except Exception:
                pass


def _run_load_phase(
    *,
    gpus: list[dict[str, Any]],
    session: nvidia.NvmlSession,
    vendor: str | None,
    duration: float,
    notes: Notes,
    measure_power: bool,
) -> tuple[dict[str, Any], float | None]:
    """Generate load while sampling CPU/RAM/GPU. Returns (underLoad, systemLoadW)."""
    samples: dict[str, list[float]] = {"cpu": [], "ram": [], "system": []}
    gpu_samples: dict[int, dict[str, list[float]]] = {
        gpu.get("index", 0): {"power": [], "util": [], "temp": []} for gpu in gpus
    }
    stop = threading.Event()
    monitor = power.SystemPowerMonitor() if measure_power else None
    if monitor is not None:
        monitor.start()
    reader = _gpu_power_reader(session, vendor)

    psutil = import_optional("psutil")
    if psutil is not None:
        psutil.cpu_percent(interval=None)

    def sampler() -> None:
        while not stop.is_set():
            if monitor is not None:
                watts = monitor.sample()
                if watts is not None:
                    samples["system"].append(watts)
            if psutil is not None:
                samples["cpu"].append(psutil.cpu_percent(interval=None))
                samples["ram"].append(float(psutil.virtual_memory().used))
            for gpu in gpus:
                index = gpu.get("index", 0)
                bucket = gpu_samples[index]
                if measure_power:
                    value = reader(index)
                    if value is not None:
                        bucket["power"].append(value)
                if vendor == "nvidia":
                    util = session.utilization(index)
                    if util is not None:
                        bucket["util"].append(util)
                    temp = session.temperature(index)
                    if temp is not None:
                        bucket["temp"].append(temp)
            stop.wait(0.25)

    sampler_thread = threading.Thread(target=sampler, daemon=True)
    sampler_thread.start()
    generator = _LoadGenerator(gpus, vendor, notes)
    generator.start()
    window_start = time.monotonic()
    try:
        while time.monotonic() - window_start < duration:
            time.sleep(0.1)
    finally:
        generator.shutdown()
        stop.set()
        sampler_thread.join(timeout=5.0)
    window = round(time.monotonic() - window_start, 2)

    load_by_gpu = []
    for gpu in gpus:
        index = gpu.get("index", 0)
        bucket = gpu_samples[index]
        limit = (gpu.get("power") or {}).get("limitW")
        if limit is None and vendor == "nvidia":
            limit = session.power_limit_w(index)
        summary = power.summarize_power_samples(
            bucket["power"], limit_w=limit, label=f"GPU {index}"
        )
        if summary["samples"]:
            # Documents the sample count and any implausible samples the driver produced.
            notes.add(summary["note"])
        if summary["medianW"] is not None:
            gpu.setdefault("power", {})["loadW"] = summary["medianW"]
            gpu["power"]["measured"] = True
            gpu["power"]["sensor"] = gpu["power"].get("sensor") or (
                "nvml" if vendor == "nvidia" else "rocm-smi"
            )
        if bucket["temp"]:
            gpu.setdefault("thermal", {})["maxTempC"] = round(max(bucket["temp"]), 1)
        load_by_gpu.append(
            {
                "index": index,
                "powerW": summary["medianW"],
                "utilPct": round(statistics.mean(bucket["util"]), 1) if bucket["util"] else None,
                "tempC": round(max(bucket["temp"]), 1) if bucket["temp"] else None,
            }
        )

    under_load = {
        "durationS": window,
        "cpuUtilPct": round(statistics.mean(samples["cpu"]), 1) if samples["cpu"] else None,
        "ramUsedBytes": int(max(samples["ram"])) if samples["ram"] else None,
        "gpus": load_by_gpu,
        "sampled": bool(samples["cpu"] or samples["system"]),
    }
    system_load = round(statistics.mean(samples["system"]), 2) if samples["system"] else None
    return under_load, system_load


def run_benchmark(
    *,
    quick: bool = False,
    gpu: Any = "all",
    no_power: bool = False,
    hold_load: float | None = None,
) -> dict[str, Any]:
    """Run the hardware benchmark and return the Benchmark v1 payload."""
    notes = Notes()
    started_real = utc_now()
    started_at = now_iso()
    t0 = time.monotonic()

    system = collect_system(notes=notes)

    idle_duration = QUICK_IDLE_SECONDS if quick else IDLE_SECONDS
    if no_power:
        idle = {"idleW": None, "loadW": None, "measurement": "none", "psuWatts": None}
        notes.add("Power measurement skipped (--no-power); idleW/loadW/psuWatts are null.")
    else:
        idle = power.idle_power(idle_duration, notes=notes)

    cpu_bench = cpu.run(quick=quick, notes=notes)
    cpu_bench["memory"] = memory.run(quick=quick, notes=notes)

    gpus, session, vendor = collect_gpus(quick=quick, notes=notes)
    under_load: dict[str, Any]
    system_load = None
    try:
        interconnect = p2p.run(quick=quick, notes=notes)
        gpus = _select_gpus(gpus, gpu)
        if no_power:
            for entry in gpus:
                entry.setdefault("power", {})["measured"] = False
        elif gpus:
            if vendor == "nvidia" and not session.available:
                notes.add("GPU idle/load power not sampled: NVML is unavailable.")
            else:
                _gpu_idle_power(session, gpus, vendor, 0.5 if quick else 1.0)

        load_duration = hold_load if hold_load is not None else (QUICK_LOAD_SECONDS if quick else DEFAULT_LOAD_SECONDS)
        load_duration = max(0.5, float(load_duration))
        under_load, system_load = _run_load_phase(
            gpus=gpus,
            session=session,
            vendor=vendor,
            duration=load_duration,
            notes=notes,
            measure_power=not no_power,
        )
    finally:
        session.close()
    if vendor == "nvidia" and gpus:
        # Clocks read at enumeration are idle clocks; refresh after the load phase.
        nvidia.refresh_clocks(gpus)

    power_block = dict(idle)
    if no_power:
        power_block["loadW"] = None
    else:
        power_block["loadW"] = system_load
        if system_load is not None:
            power_block["measurement"] = "sensor"
    power_block.setdefault("psuWatts", power.psu_watts())

    finished_at = now_iso()
    duration_s = elapsed_seconds(started_real, utc_now())
    return report.build(
        started_at=started_at,
        finished_at=finished_at,
        duration_s=duration_s if duration_s > 0 else round(time.monotonic() - t0, 3),
        system=system,
        cpu_bench=cpu_bench,
        gpus=gpus,
        interconnect=interconnect,
        power=power_block,
        notes=notes,
        runner=runner_info(),
        under_load=under_load,
    )


def self_check() -> dict[str, Any]:
    """Run a quick benchmark and validate the payload against the local schema."""
    bench = run_benchmark(quick=True)
    problems = report.check_shape(bench)
    missing = report.missing_required(bench)
    return {
        # The same required-field list the server enforces: a self-check that
        # ignores it would pass on a host that cannot produce an uploadable rig.
        "ok": not problems and not missing,
        "schemaProblems": problems,
        "missingRequired": missing,
        "uploadable": not missing,
        "benchmark": bench,
    }
