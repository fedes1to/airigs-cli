"""NVIDIA GPU discovery (NVML / nvidia-smi) and torch-based measurements.

torch measurement helpers live here because CUDA is the reference path; the
ROCm path in :mod:`airigs.bench.amd` reuses :func:`measure_torch_gpu` when torch
was built against HIP. Every value that cannot be measured is ``None`` with an
entry in ``notes``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from typing import Any
from xml.etree import ElementTree

from .common import Notes, gbps, import_optional, measure, median_or_none, tflops as to_tflops

SMI_FIELDS = [
    "index",
    "name",
    "uuid",
    "memory.total",
    "driver_version",
    "pcie.link.gen.current",
    "pcie.link.gen.max",
    "pcie.link.width.current",
    "pcie.link.width.max",
    "clocks.current.graphics",
    "clocks.current.memory",
    "power.limit",
    "temperature.gpu",
    "power.draw",
]


def _run(cmd: list[str], timeout: float = 15.0) -> str | None:
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


# ---------------------------------------------------------------------------
# NVML


class NvmlSession:
    """A thin NVML handle. ``available`` is False when the library or driver is absent."""

    def __init__(self) -> None:
        self.lib = import_optional("pynvml")
        self.available = False
        self.count = 0
        if self.lib is None:
            return
        try:
            self.lib.nvmlInit()
            self.count = int(self.lib.nvmlDeviceGetCount())
            self.available = True
        except Exception:
            self.available = False
            self.lib = None

    def __enter__(self) -> "NvmlSession":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def close(self) -> None:
        if self.lib is not None:
            try:
                self.lib.nvmlShutdown()
            except Exception:
                pass
        self.lib = None
        self.available = False

    def _handle(self, index: int) -> Any:
        return self.lib.nvmlDeviceGetHandleByIndex(index)

    def power_w(self, index: int) -> float | None:
        if not self.available:
            return None
        try:
            return round(self.lib.nvmlDeviceGetPowerUsage(self._handle(index)) / 1000.0, 2)
        except Exception:
            return None

    def power_limit_w(self, index: int) -> float | None:
        if not self.available:
            return None
        try:
            return round(self.lib.nvmlDeviceGetEnforcedPowerLimit(self._handle(index)) / 1000.0, 2)
        except Exception:
            return None

    def utilization(self, index: int) -> float | None:
        if not self.available:
            return None
        try:
            rates = self.lib.nvmlDeviceGetUtilizationRates(self._handle(index))
            return float(rates.gpu)
        except Exception:
            return None

    def temperature(self, index: int) -> float | None:
        if not self.available:
            return None
        try:
            return float(self.lib.nvmlDeviceGetTemperature(self._handle(index), self.lib.NVML_TEMPERATURE_GPU))
        except Exception:
            return None

    def memory_bus_width(self, index: int) -> int | None:
        if not self.available or not hasattr(self.lib, "nvmlDeviceGetMemoryBusWidth"):
            return None
        try:
            return int(self.lib.nvmlDeviceGetMemoryBusWidth(self._handle(index)))
        except Exception:
            return None

    def clock_mhz(self, index: int, kind: str) -> float | None:
        if not self.available:
            return None
        attribute = {
            "graphics": "NVML_CLOCK_GRAPHICS",
            "memory": "NVML_CLOCK_MEM",
            "sm": "NVML_CLOCK_SM",
        }.get(kind)
        if attribute is None or not hasattr(self.lib, attribute):
            return None
        try:
            return float(self.lib.nvmlDeviceGetClockInfo(self._handle(index), getattr(self.lib, attribute)))
        except Exception:
            return None

    def memory_used(self, index: int) -> int | None:
        if not self.available:
            return None
        try:
            return int(self.lib.nvmlDeviceGetMemoryInfo(self._handle(index)).used)
        except Exception:
            return None

    def pcie(self, index: int) -> dict[str, int | None] | None:
        if not self.available:
            return None
        handle = self._handle(index)
        values: dict[str, int | None] = {}
        for key, func in (
            ("currentVersion", "nvmlDeviceGetCurrPcieLinkGeneration"),
            ("currentWidth", "nvmlDeviceGetCurrPcieLinkWidth"),
            ("maxVersion", "nvmlDeviceGetMaxPcieLinkGeneration"),
            ("maxWidth", "nvmlDeviceGetMaxPcieLinkWidth"),
        ):
            try:
                values[key] = int(getattr(self.lib, func)(handle))
            except Exception:
                values[key] = None
        return values


# ---------------------------------------------------------------------------
# nvidia-smi parsing


def parse_csv(output: str) -> list[list[str]]:
    rows = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([cell.strip() for cell in line.split(",")])
    return rows


def smi_query() -> list[dict[str, str]]:
    output = _run(["nvidia-smi", f"--query-gpu={','.join(SMI_FIELDS)}", "--format=csv,noheader,nounits"])
    if not output:
        return []
    rows = parse_csv(output)
    result = []
    for row in rows:
        if len(row) != len(SMI_FIELDS):
            continue
        result.append(dict(zip(SMI_FIELDS, row)))
    return result


def _num(value: Any) -> float | None:
    if value is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def _int(value: Any) -> int | None:
    parsed = _num(value)
    return int(parsed) if parsed is not None else None


def smi_xml() -> list[dict[str, Any]]:
    """Per-GPU fields nvidia-smi only exposes in XML: bus width, memory clock."""
    output = _run(["nvidia-smi", "-q", "-x"], timeout=30.0)
    if not output:
        return []
    try:
        root = ElementTree.fromstring(output)
    except ElementTree.ParseError:
        return []
    gpus = []
    for gpu in root.findall("gpu"):
        bus = gpu.findtext("./memory/bus_width")
        mem_clock = gpu.findtext("./clocks/memory_clock")
        gpus.append(
            {
                "busWidthBits": _int(bus),
                "memoryClockMhz": _num(mem_clock),
                "vramTotalBytes": _num(gpu.findtext("./fb_memory_usage/total")),
                "powerLimitW": _num(
                    gpu.findtext("./gpu_power_readings/current_power_limit")
                    or gpu.findtext("./power_readings/current_power_limit")
                ),
            }
        )
    return gpus


def theoretical_bandwidth_gbs(bus_width_bits: int | None, memory_clock_mhz: float | None) -> float | None:
    """Bus width (bits) times effective memory clock (MHz), DDR double-pumped.

    nvidia-smi reports the effective clock (e.g. 10501 MHz on GDDR6X), so the
    formula is ``2 * bytes * MHz / 1000``. Derivation, not measurement.
    """
    if not bus_width_bits or not memory_clock_mhz:
        return None
    bytes_per_transfer = bus_width_bits / 8.0
    return round(2 * bytes_per_transfer * memory_clock_mhz / 1000.0, 1)


def driver_info() -> dict[str, Any]:
    info = {"version": None, "cuda": None}
    output = _run(["nvidia-smi"])
    if output:
        match = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", output)
        if match:
            info["cuda"] = match.group(1)
    rows = smi_query()
    if rows:
        info["version"] = rows[0].get("driver_version")
    if info["version"] is None:
        version = None
        try:
            with open("/proc/driver/nvidia/version", "r", encoding="utf-8") as handle:
                match = re.search(r"Kernel Module\s+([\d.]+)", handle.read())
                version = match.group(1) if match else None
        except OSError:
            pass
        info["version"] = version
    return info


def enumerate_gpus(*, notes: Notes | None = None, session: NvmlSession | None = None) -> list[dict[str, Any]]:
    """Build contract-shaped GPU entries from NVML, nvidia-smi or both."""
    notes = notes if notes is not None else Notes()
    rows = smi_query()
    xml_rows = smi_xml()
    own_session = session is None
    session = session or NvmlSession()
    try:
        if not rows and session.available:
            rows = [
                {
                    "index": str(i),
                    "name": _nvml_name(session, i),
                    "uuid": _nvml_uuid(session, i) or "",
                    "memory.total": "",
                    "driver_version": "",
                }
                for i in range(session.count)
            ]
        if not rows:
            if not nvidia_smi_available() and not session.available:
                notes.add("No NVIDIA GPU detected: NVML is unavailable and nvidia-smi is not installed.")
            else:
                notes.add("nvidia-smi returned no GPUs (driver not loaded or no device visible).")
            return []

        gpus: list[dict[str, Any]] = []
        for position, row in enumerate(rows):
            index = _int(row.get("index")) or position
            xml = xml_rows[index] if index < len(xml_rows) else {}
            bus_width = xml.get("busWidthBits")
            if bus_width is None:
                bus_width = session.memory_bus_width(index)
            mem_clock = xml.get("memoryClockMhz")
            if mem_clock is None:
                mem_clock = _num(row.get("clocks.current.memory"))
            vram = _num(row.get("memory.total"))
            vram_bytes = int(vram * 1024 * 1024) if vram else None
            if vram_bytes is None and xml.get("vramTotalBytes"):
                vram_bytes = int(float(xml["vramTotalBytes"]))
            pcie = session.pcie(index) or {}
            current_version = _int(row.get("pcie.link.gen.current"))
            current_width = _int(row.get("pcie.link.width.current"))
            max_version = _int(row.get("pcie.link.gen.max"))
            max_width = _int(row.get("pcie.link.width.max"))
            if max_version is None:
                max_version = pcie.get("maxVersion")
            if max_width is None:
                max_width = pcie.get("maxWidth")
            if current_version is None:
                current_version = pcie.get("currentVersion")
            if current_width is None:
                current_width = pcie.get("currentWidth")
            if bus_width is None:
                notes.add(f"GPU {index}: theoretical memory bandwidth unknown (bus width not reported).")
            if vram_bytes is None:
                notes.add(f"GPU {index}: VRAM size unknown (nvidia-smi and NVML both silent).")
            limit = _num(row.get("power.limit"))
            if limit is None:
                limit = xml.get("powerLimitW")
            if limit is None:
                limit = session.power_limit_w(index)

            gpus.append(
                {
                    "index": index,
                    "name": row.get("name") or None,
                    "uuid": row.get("uuid") or None,
                    "vramBytes": vram_bytes,
                    "driverVersion": row.get("driver_version") or None,
                    "pcie": {
                        "version": max_version,
                        "width": max_width,
                        "currentVersion": current_version,
                        "currentWidth": current_width,
                        "maxVersion": max_version,
                        "maxWidth": max_width,
                    },
                    "clocks": {
                        "graphicsMhz": _num(row.get("clocks.current.graphics")),
                        "memoryMhz": mem_clock,
                    },
                    "power": {
                        "limitW": limit,
                        "idleW": None,
                        "loadW": None,
                        "measured": bool(session.available),
                        "sensor": "nvml" if session.available else None,
                    },
                    "memoryBandwidth": {
                        "measuredGBs": None,
                        "theoreticalGBs": theoretical_bandwidth_gbs(bus_width, mem_clock),
                        "method": "device-to-device copy",
                    },
                    "tflops": {
                        "fp64": None,
                        "fp32": None,
                        "tf32": None,
                        "bf16": None,
                        "fp16": None,
                        "fp8": None,
                        "int8": None,
                    },
                    "tflopsMethod": None,
                    "thermal": {"maxTempC": _num(row.get("temperature.gpu"))},
                }
            )
        return gpus
    finally:
        if own_session:
            session.close()


def _nvml_name(session: NvmlSession, index: int) -> str | None:
    try:
        return session.lib.nvmlDeviceGetName(session._handle(index))
    except Exception:
        return None


def _nvml_uuid(session: NvmlSession, index: int) -> str | None:
    try:
        return session.lib.nvmlDeviceGetUUID(session._handle(index))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# torch measurements


def _dtype_tflops(
    torch: Any,
    device: Any,
    dtype_name: str,
    *,
    size: int,
    repeats: int,
    notes: Notes,
    label: str,
    int8: bool = False,
) -> float | None:
    """Measure one precision, halving the matrix on OOM."""
    current = size
    while current >= 256:
        a = b = None
        try:
            if int8:
                if not hasattr(torch, "_int_mm"):
                    notes.add(f"{label}: torch._int_mm is unavailable in this torch build; int8 TFLOPS not measured.")
                    return None
                a = torch.randint(-128, 127, (current, current), device=device, dtype=torch.int8)
                b = torch.randint(-128, 127, (current, current), device=device, dtype=torch.int8)
                fn = lambda: torch._int_mm(a, b)  # noqa: E731
                torch._int_mm(a, b)
            else:
                dtype = getattr(torch, dtype_name, None)
                if dtype is None:
                    notes.add(f"{label}: torch has no {dtype_name} dtype; that precision was not measured.")
                    return None
                a = torch.randn((current, current), device=device, dtype=dtype)
                b = torch.randn((current, current), device=device, dtype=dtype)
                fn = lambda: torch.matmul(a, b)  # noqa: E731
                torch.matmul(a, b)
            torch.cuda.synchronize()
            samples = []
            for _ in range(repeats):
                torch.cuda.synchronize()
                start = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                samples.append(time.perf_counter() - start)
            elapsed = median_or_none(samples)
            if elapsed:
                return to_tflops(2 * current**3, elapsed)
            return None
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                current //= 2
                _empty_cache(torch)
                continue
            notes.add(f"{label}: {dtype_name} matmul failed ({str(exc).splitlines()[0][:160]}); not measured.")
            return None
        except Exception as exc:
            notes.add(f"{label}: {dtype_name} matmul failed ({exc.__class__.__name__}); not measured.")
            return None
        finally:
            del a, b
    notes.add(f"{label}: {dtype_name} matmul did not fit in VRAM even at 256x256; not measured.")
    return None


def _empty_cache(torch: Any) -> None:
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def measure_memory_bandwidth(torch: Any, device: Any, *, quick: bool, notes: Notes, label: str) -> float | None:
    """Device-to-device copy bandwidth, median of repeats, GB/s effective."""
    total_bytes = 64 * 1024 * 1024 if quick else 256 * 1024 * 1024
    elements = total_bytes // 4
    src = dst = None
    try:
        src = torch.empty(elements, device=device, dtype=torch.float32)
        dst = torch.empty(elements, device=device, dtype=torch.float32)
        torch.cuda.synchronize()

        def once() -> None:
            dst.copy_(src)
            torch.cuda.synchronize()

        seconds = measure(once, warmup=1, repeats=2 if quick else 4)
        return gbps(total_bytes * 2, seconds)
    except RuntimeError as exc:
        notes.add(f"{label}: GPU memory bandwidth not measured ({str(exc).splitlines()[0][:160]}).")
        return None
    except Exception as exc:
        notes.add(f"{label}: GPU memory bandwidth not measured ({exc.__class__.__name__}).")
        return None
    finally:
        del src, dst
        _empty_cache(torch)


def measure_torch_gpu(
    torch: Any,
    index: int,
    gpu: dict[str, Any],
    *,
    quick: bool = False,
    notes: Notes | None = None,
) -> None:
    """Fill ``memoryBandwidth.measuredGBs`` and ``tflops`` for one GPU in place."""
    notes = notes if notes is not None else Notes()
    label = f"GPU {index} ({(gpu.get('name') or 'unknown')})"
    device = torch.device(f"cuda:{index}")
    try:
        torch.cuda.set_device(index)
    except Exception:
        pass

    bandwidth = measure_memory_bandwidth(torch, device, quick=quick, notes=notes, label=label)
    gpu.setdefault("memoryBandwidth", {})["measuredGBs"] = bandwidth
    gpu["memoryBandwidth"].setdefault("method", "device-to-device copy")

    size = 2048 if quick else 4096
    repeats = 2 if quick else 3
    tflops = gpu.setdefault("tflops", {})

    tflops["fp64"] = _dtype_tflops(torch, device, "float64", size=size, repeats=repeats, notes=notes, label=label)
    tflops["fp32"] = _dtype_tflops(torch, device, "float32", size=size, repeats=repeats, notes=notes, label=label)

    previous_tf32 = None
    try:
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        tflops["tf32"] = _dtype_tflops(torch, device, "float32", size=size, repeats=repeats, notes=notes, label=label + " tf32")
    except Exception:
        tflops["tf32"] = None
    finally:
        if previous_tf32 is not None:
            try:
                torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            except Exception:
                pass

    tflops["bf16"] = _dtype_tflops(torch, device, "bfloat16", size=size, repeats=repeats, notes=notes, label=label)
    tflops["fp16"] = _dtype_tflops(torch, device, "float16", size=size, repeats=repeats, notes=notes, label=label)
    tflops["int8"] = _dtype_tflops(torch, device, "int8", size=size, repeats=repeats, notes=notes, label=label, int8=True)

    capability = None
    try:
        capability = torch.cuda.get_device_capability(index)
    except Exception:
        pass
    if capability is not None and tuple(capability) < (8, 9):
        notes.add(f"{label}: fp8 not supported before compute capability 8.9; fp8 TFLOPS not measured.")
    elif not hasattr(torch, "float8_e4m3fn"):
        notes.add(f"{label}: this torch build has no float8 dtype; fp8 TFLOPS not measured.")
    else:
        tflops["fp8"] = _dtype_tflops(
            torch,
            device,
            "float8_e4m3fn",
            size=size,
            repeats=repeats,
            notes=notes,
            label=label,
        )

    gpu["tflopsMethod"] = "torch.matmul"


def refresh_clocks(gpus: list[dict[str, Any]]) -> None:
    """Re-read clocks after the load phase so the payload shows active clocks."""
    rows = smi_query()
    for gpu in gpus:
        index = gpu.get("index", 0)
        if index >= len(rows):
            continue
        graphics = _num(rows[index].get("clocks.current.graphics"))
        memory = _num(rows[index].get("clocks.current.memory"))
        clocks = gpu.setdefault("clocks", {})
        if graphics is not None:
            clocks["graphicsMhz"] = graphics
        if memory is not None:
            clocks["memoryMhz"] = memory


def torch_available(torch: Any | None) -> bool:
    return torch is not None and bool(getattr(torch, "cuda", None)) and torch.cuda.is_available()


def run_torch_measurements(gpus: list[dict[str, Any]], *, quick: bool = False, notes: Notes | None = None) -> None:
    """Apply torch measurements to CUDA GPUs. No torch -> note, no numbers."""
    notes = notes if notes is not None else Notes()
    if not gpus:
        return
    torch = import_optional("torch")
    if torch is None:
        notes.add(
            "torch is not installed: TFLOPS, GPU memory bandwidth and P2P bandwidth were not measured "
            "(install with `pip install 'airigs[gpu]'`)."
        )
        return
    if not torch_available(torch):
        notes.add("torch is installed but CUDA is not available: GPU compute measurements were not taken.")
        return
    try:
        device_count = torch.cuda.device_count()
    except Exception:
        device_count = 0
    if device_count < len(gpus):
        notes.add(f"torch sees {device_count} CUDA device(s) but nvidia-smi listed {len(gpus)}.")
    for gpu in gpus:
        index = gpu.get("index", 0)
        if index >= device_count:
            notes.add(f"GPU {index}: not visible to torch; compute measurements skipped.")
            continue
        measure_torch_gpu(torch, index, gpu, quick=quick, notes=notes)
