"""Apple Silicon (Metal / MPS) GPU reporting.

macOS exposes no NVML-style power sensor and integrated GPUs have no PCIe link,
so those fields stay ``null`` with a note instead of being guessed.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import time
from typing import Any

from .common import Notes, gbps, import_optional, median_or_none, tflops as to_tflops


def available() -> bool:
    return sys.platform == "darwin"


def _sysctl(name: str) -> str | None:
    try:
        completed = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def chip_name() -> str | None:
    return _sysctl("machdep.cpu.brand_string") or _sysctl("hw.chip") or platform.processor() or None


def macos_version() -> str | None:
    return platform.mac_ver()[0] or None


def _mps_bandwidth(torch: Any, *, quick: bool, notes: Notes) -> float | None:
    total_bytes = 64 * 1024 * 1024 if quick else 256 * 1024 * 1024
    elements = total_bytes // 4
    try:
        device = torch.device("mps")
        src = torch.empty(elements, device=device, dtype=torch.float32)
        dst = torch.empty(elements, device=device, dtype=torch.float32)
        torch.mps.synchronize()

        def once() -> None:
            dst.copy_(src)
            torch.mps.synchronize()

        once()
        samples = []
        for _ in range(2 if quick else 4):
            torch.mps.synchronize()
            start = time.perf_counter()
            dst.copy_(src)
            torch.mps.synchronize()
            samples.append(time.perf_counter() - start)
        elapsed = median_or_none(samples)
        return gbps(total_bytes * 2, elapsed) if elapsed else None
    except Exception as exc:
        notes.add(f"Apple GPU memory bandwidth not measured ({exc.__class__.__name__}).")
        return None


def _mps_tflops(torch: Any, dtype_name: str, *, quick: bool, notes: Notes) -> float | None:
    size = 1024 if quick else 2048
    device = torch.device("mps")
    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        return None
    try:
        a = torch.randn((size, size), device=device, dtype=dtype)
        b = torch.randn((size, size), device=device, dtype=dtype)
        torch.matmul(a, b)
        torch.mps.synchronize()
        samples = []
        for _ in range(2 if quick else 3):
            torch.mps.synchronize()
            start = time.perf_counter()
            torch.matmul(a, b)
            torch.mps.synchronize()
            samples.append(time.perf_counter() - start)
        elapsed = median_or_none(samples)
        return to_tflops(2 * size**3, elapsed) if elapsed else None
    except Exception:
        notes.add(f"Apple GPU {dtype_name} matmul not supported by this torch/MPS build.")
        return None


def run(*, quick: bool = False, notes: Notes | None = None) -> list[dict[str, Any]]:
    notes = notes if notes is not None else Notes()
    if not available():
        return []
    name = chip_name()
    torch = import_optional("torch")
    mps_ok = False
    unified_bytes = None
    if torch is not None:
        try:
            import psutil

            unified_bytes = int(psutil.virtual_memory().total)
        except Exception:
            pass
        try:
            mps_ok = bool(torch.backends.mps.is_available())
        except Exception:
            mps_ok = False
        if mps_ok:
            try:
                unified_bytes = int(torch.mps.recommended_max_memory())
            except Exception:
                pass

    memory_bandwidth = None
    tflops = {"fp64": None, "fp32": None, "tf32": None, "bf16": None, "fp16": None, "fp8": None, "int8": None}
    if mps_ok:
        memory_bandwidth = _mps_bandwidth(torch, quick=quick, notes=notes)
        tflops["fp32"] = _mps_tflops(torch, "float32", quick=quick, notes=notes)
        tflops["fp16"] = _mps_tflops(torch, "float16", quick=quick, notes=notes)

    notes.add(
        "Apple GPU: power is not exposed by macOS and the integrated GPU has no PCIe link, "
        "so power.* and pcie.* are null; the AIRigs server requires them for a verified upload."
    )

    gpu = {
        "index": 0,
        "name": name,
        "uuid": None,
        "vramBytes": unified_bytes,
        "driverVersion": macos_version(),
        "pcie": {
            "version": None,
            "width": None,
            "currentVersion": None,
            "currentWidth": None,
            "maxVersion": None,
            "maxWidth": None,
        },
        "clocks": {"graphicsMhz": None, "memoryMhz": None},
        "power": {"limitW": None, "idleW": None, "loadW": None, "measured": False, "sensor": None},
        "memoryBandwidth": {
            "measuredGBs": memory_bandwidth,
            "theoreticalGBs": None,
            "method": "mps device-to-device copy",
        },
        "tflops": tflops,
        "tflopsMethod": "torch.matmul (mps)" if mps_ok else None,
        "thermal": {"maxTempC": None},
    }
    return [gpu]


def driver_info() -> dict[str, Any]:
    return {"metal": macos_version()}
