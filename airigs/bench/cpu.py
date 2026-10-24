"""CPU identification and CPU benchmarks: SHA-256 and numpy matmul GFLOP/s."""

from __future__ import annotations

import hashlib
import os
import platform
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .common import Notes, cpu_logical_count, gflops, measure

PROC_CPUINFO = "/proc/cpuinfo"
SYSFS_CPU = "/sys/devices/system/cpu"


def read_text(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return None


def parse_proc_cpuinfo(text: str) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            if current:
                blocks.append(current)
                current = {}
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        current[key.strip().lower()] = value.strip()
    if current:
        blocks.append(current)
    return blocks


def _cache_l3_mb() -> float | None:
    raw = read_text(f"{SYSFS_CPU}/cpu0/cache/index3/size")
    if raw:
        match = re.match(r"\s*(\d+(?:\.\d+)?)\s*([KMG]?)", raw, re.I)
        if match:
            value = float(match.group(1))
            unit = match.group(2).upper()
            if unit == "K":
                value /= 1024
            elif unit == "M":
                pass
            elif unit == "G":
                value *= 1024
            return round(value, 2)
    return None


def _khz_to_mhz(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return round(float(value.strip()) / 1000.0, 1)
    except ValueError:
        return None


def cpu_info() -> dict[str, Any]:
    """CPU model, core counts and clocks from /proc/cpuinfo + sysfs.

    Unknown fields stay ``None`` rather than guessed.
    """
    info: dict[str, Any] = {
        "model": None,
        "vendor": None,
        "physicalCores": None,
        "logicalCores": None,
        "baseClockMhz": None,
        "maxClockMhz": None,
        "cacheL3Mb": _cache_l3_mb(),
    }
    text = read_text(PROC_CPUINFO)
    if text:
        blocks = parse_proc_cpuinfo(text)
        if blocks:
            first = blocks[0]
            info["model"] = first.get("model name") or first.get("processor") or None
            info["vendor"] = first.get("vendor_id") or None
            info["logicalCores"] = len(blocks)
            pairs = {
                (b.get("physical id"), b.get("core id"))
                for b in blocks
                if b.get("physical id") is not None and b.get("core id") is not None
            }
            if pairs:
                info["physicalCores"] = len(pairs)
            else:
                cores = first.get("cpu cores")
                info["physicalCores"] = int(cores) if cores and cores.isdigit() else None
            info["maxClockMhz"] = info["maxClockMhz"] or _khz_to_mhz(
                read_text(f"{SYSFS_CPU}/cpu0/cpufreq/cpuinfo_max_freq")
            )
            info["baseClockMhz"] = _khz_to_mhz(read_text(f"{SYSFS_CPU}/cpu0/cpufreq/base_frequency"))
            if info["cacheL3Mb"] is None:
                cache = first.get("cache size")
                if cache:
                    match = re.search(r"(\d+)\s*KB", cache, re.I)
                    if match:
                        info["cacheL3Mb"] = round(int(match.group(1)) / 1024, 2)

    if info["model"] is None:
        info["model"] = platform.processor() or None
    vendors = {
        "genuineintel": "Intel",
        "authenticamd": "AMD",
        "arm": "ARM",
        "apple": "Apple",
    }
    info["vendor"] = vendors.get((info["vendor"] or "").strip().lower(), info["vendor"])
    if info["vendor"] is None and "intel" in (info["model"] or "").lower():
        info["vendor"] = "Intel"

    try:
        import psutil

        if info["logicalCores"] is None:
            info["logicalCores"] = psutil.cpu_count(logical=True)
        if info["physicalCores"] is None:
            info["physicalCores"] = psutil.cpu_count(logical=False)
        if info["maxClockMhz"] is None:
            freq = psutil.cpu_freq()
            if freq and freq.max:
                info["maxClockMhz"] = round(float(freq.max), 1)
        if info["baseClockMhz"] is None:
            freq = psutil.cpu_freq()
            if freq and freq.min:
                info["baseClockMhz"] = round(float(freq.min), 1)
    except Exception:
        pass

    if info["logicalCores"] is None:
        info["logicalCores"] = cpu_logical_count()
    return info


def sha256_single_mbps(*, quick: bool = False) -> float | None:
    """Single-threaded SHA-256 throughput in MB/s (decimal megabytes)."""
    block = b"\x00" * (4 * 1024 * 1024 if quick else 8 * 1024 * 1024)
    repeats = 2 if quick else 5

    def once() -> None:
        hashlib.sha256(block).digest()

    try:
        seconds = measure(once, warmup=1, repeats=repeats)
        return round(len(block) / 1e6 / seconds, 2)
    except Exception:
        return None


def sha256_multi_mbps(*, quick: bool = False, threads: int | None = None) -> tuple[float | None, int]:
    """Multi-threaded SHA-256 throughput (all bytes hashed / wall time)."""
    threads = threads or cpu_logical_count()
    chunk = b"\x01" * (2 * 1024 * 1024 if quick else 4 * 1024 * 1024)
    repeats = 2 if quick else 3

    def once() -> None:
        def work() -> bytes:
            return hashlib.sha256(chunk).digest()

        with ThreadPoolExecutor(max_workers=threads) as pool:
            list(pool.map(lambda _i: work(), range(threads)))

    try:
        seconds = measure(once, warmup=1, repeats=repeats)
        total = len(chunk) * threads
        return round(total / 1e6 / seconds, 2), threads
    except Exception:
        return None, threads


def matmul_gflops(dtype: str, *, quick: bool = False) -> float | None:
    """numpy matmul throughput for ``float32``/``float64``.

    This is the BLAS path of the installed numpy (typically multi-threaded), which
    is what "fp32 GFLOP/s" means for CPU inference on this machine.
    """
    try:
        import numpy as np
    except Exception:
        return None
    size = 1024 if quick else 2048
    rng = np.random.default_rng(20260101)
    try:
        a = rng.standard_normal((size, size)).astype(np.float32)
        b = rng.standard_normal((size, size)).astype(np.float32)
        if dtype == "float64":
            a = a.astype(np.float64)
            b = b.astype(np.float64)
        flops = 2.0 * size**3

        def once() -> None:
            np.matmul(a, b)

        seconds = measure(once, warmup=1, repeats=2 if quick else 3)
        return gflops(flops, seconds)
    except (MemoryError, ValueError, FloatingPointError):
        if size > 512:
            try:
                return matmul_gflops_small(dtype)
            except Exception:
                return None
        return None
    except Exception:
        return None


def matmul_gflops_small(dtype: str) -> float | None:
    import numpy as np

    size = 512
    rng = np.random.default_rng(7)
    a = rng.standard_normal((size, size)).astype(dtype)
    b = rng.standard_normal((size, size)).astype(dtype)
    flops = 2.0 * size**3

    def once() -> None:
        np.matmul(a, b)

    return gflops(flops, measure(once, warmup=1, repeats=3))


def run(*, quick: bool = False, notes: Notes | None = None) -> dict[str, Any]:
    """The ``cpuBench`` block minus RAM bandwidth (composed by the runner)."""
    notes = notes if notes is not None else Notes()
    single = sha256_single_mbps(quick=quick)
    multi, threads = sha256_multi_mbps(quick=quick)
    if single is None:
        notes.add("SHA-256 single-thread benchmark failed; hashlib unavailable?")
    if multi is None:
        notes.add("SHA-256 multi-thread benchmark failed.")
    fp32 = matmul_gflops("float32", quick=quick)
    fp64 = matmul_gflops("float64", quick=quick)
    if fp32 is None or fp64 is None:
        notes.add("numpy matmul benchmark failed; fp32/fp64 GFLOPs unavailable.")
    return {
        "singleThread": {"sha256Mbps": single},
        "multiThread": {"sha256Mbps": multi, "threads": threads},
        "fp32Gflops": fp32,
        "fp64Gflops": fp64,
    }


def os_info() -> dict[str, Any]:
    kernel = platform.release()
    return {
        "name": platform.system() or None,
        "version": platform.version() or None,
        "kernel": kernel or None,
    }


def hostname() -> str:
    return platform.node() or os.uname().nodename
