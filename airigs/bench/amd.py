"""AMD GPU discovery via sysfs / rocm-smi, plus torch-ROCm measurements."""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any

from .common import Notes, import_optional

DRM_ROOT = "/sys/class/drm"
PCI_ROOT = "/sys/bus/pci/devices"

_SPEED_TO_GEN = {
    2.5: 1,
    5.0: 2,
    8.0: 3,
    16.0: 4,
    32.0: 5,
    64.0: 6,
}


def read_text(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read().strip()
    except OSError:
        return None


def read_int(path: str) -> int | None:
    raw = read_text(path)
    if raw is None:
        return None
    try:
        return int(raw, 0)
    except ValueError:
        return None


def _pci_gen(speed: str | None) -> int | None:
    if not speed:
        return None
    match = re.search(r"([\d.]+)\s*GT/s", speed)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    for key, gen in _SPEED_TO_GEN.items():
        if abs(value - key) < 0.01:
            return gen
    return None


def _clock_mhz(path: str) -> float | None:
    """Parse the currently selected DPM clock line, e.g. ``2: 2200Mhz *``."""
    raw = read_text(path)
    if not raw:
        return None
    lines = raw.splitlines()
    chosen = next((line for line in lines if line.strip().endswith("*")), lines[-1] if lines else "")
    match = re.search(r"([\d.]+)\s*Mhz", chosen, re.I)
    return round(float(match.group(1)), 1) if match else None


def _hwmon_dir(device_dir: str) -> str | None:
    candidates = sorted(glob.glob(os.path.join(device_dir, "hwmon", "hwmon*")))
    return candidates[0] if candidates else None


def _rocm_json(args: list[str]) -> dict[str, Any] | None:
    if not shutil.which("rocm-smi"):
        return None
    try:
        completed = subprocess.run(
            ["rocm-smi", *args, "--json"], capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _rocm_product_names() -> dict[str, str]:
    data = _rocm_json(["--showproductname"])
    names: dict[str, str] = {}
    if not data:
        return names
    for card, values in data.items():
        if not isinstance(values, dict):
            continue
        for key in ("Card series", "Card model", "Card SKU", "card series", "Card Series"):
            value = values.get(key)
            if isinstance(value, str) and value.strip() and value.strip().upper() != "N/A":
                names[str(card).lower()] = value.strip()
                break
    return names


def _rocm_power_w() -> dict[str, float]:
    data = _rocm_json(["--showpower"])
    powers: dict[str, float] = {}
    if not data:
        return powers
    for card, values in data.items():
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if "power" not in key.lower():
                continue
            match = re.search(r"([\d.]+)", str(value))
            if match:
                try:
                    powers[str(card).lower()] = round(float(match.group(1)), 2)
                except ValueError:
                    pass
                break
    return powers


def rocm_version() -> str | None:
    raw = read_text("/opt/rocm/.info/version")
    if raw:
        return raw.splitlines()[0].strip()
    try:
        completed = subprocess.run(
            ["rocm-smi", "--version"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"ROCm\s+version[: ]+([\d.]+)", completed.stdout, re.I)
    return match.group(1) if match else None


def _amd_card_dirs() -> list[str]:
    """AMD render nodes in the same order :func:`enumerate_gpus` numbers them."""
    cards = []
    for card_path in sorted(glob.glob(os.path.join(DRM_ROOT, "card[0-9]*"))):
        vendor = (read_text(os.path.join(card_path, "device", "vendor")) or "").lower()
        if vendor == "0x1002":
            cards.append(os.path.join(card_path, "device"))
    return cards


def power_w(index: int) -> float | None:
    """Current board power from sysfs (cheap enough to poll during load)."""
    cards = _amd_card_dirs()
    if index < 0 or index >= len(cards):
        return None
    hwmon = _hwmon_dir(cards[index])
    if not hwmon:
        return None
    micro_watts = read_int(os.path.join(hwmon, "power1_average")) or read_int(
        os.path.join(hwmon, "power1_input")
    )
    return round(micro_watts / 1e6, 2) if micro_watts else None


def available() -> bool:
    if sys.platform.startswith("linux"):
        for card in glob.glob(os.path.join(DRM_ROOT, "card[0-9]*", "device", "vendor")):
            if (read_text(card) or "").lower() == "0x1002":
                return True
    return shutil.which("rocm-smi") is not None


def enumerate_gpus(*, notes: Notes | None = None) -> list[dict[str, Any]]:
    notes = notes if notes is not None else Notes()
    if not sys.platform.startswith("linux"):
        return []
    product_names = _rocm_product_names()
    rocm_power = _rocm_power_w()
    gpus: list[dict[str, Any]] = []
    for position, card_path in enumerate(sorted(glob.glob(os.path.join(DRM_ROOT, "card[0-9]*")))):
        device_dir = os.path.join(card_path, "device")
        vendor = (read_text(os.path.join(device_dir, "vendor")) or "").lower()
        if vendor != "0x1002":
            continue
        node = os.path.basename(card_path)
        pci_addr = os.path.basename(os.path.realpath(device_dir))
        pci_dir = os.path.join(PCI_ROOT, pci_addr)
        card_key = node.replace("card", "card")
        name = product_names.get(card_key)
        if not name:
            name = read_text(os.path.join(device_dir, "product_name"))
        if not name:
            name = f"AMD GPU {pci_addr}"

        vram = read_int(os.path.join(device_dir, "mem_info_vram_total"))
        driver = read_text("/sys/module/amdgpu/version")
        hwmon = _hwmon_dir(device_dir)
        temp_c = None
        power_limit = None
        power_now = None
        if hwmon:
            raw_temp = read_int(os.path.join(hwmon, "temp1_input"))
            temp_c = round(raw_temp / 1000.0, 1) if raw_temp is not None else None
            cap = read_int(os.path.join(hwmon, "power1_cap"))
            power_limit = round(cap / 1e6, 2) if cap else None
            avg = read_int(os.path.join(hwmon, "power1_average")) or read_int(
                os.path.join(hwmon, "power1_input")
            )
            power_now = round(avg / 1e6, 2) if avg else None
        if power_now is None:
            power_now = rocm_power.get(card_key)

        current_width = read_int(os.path.join(pci_dir, "current_link_width"))
        max_width = read_int(os.path.join(pci_dir, "max_link_width"))
        current_gen = _pci_gen(read_text(os.path.join(pci_dir, "current_link_speed")))
        max_gen = _pci_gen(read_text(os.path.join(pci_dir, "max_link_speed")))

        gpus.append(
            {
                "index": position,
                "name": name,
                "uuid": None,
                "vramBytes": vram,
                "driverVersion": driver,
                "pcie": {
                    "version": max_gen,
                    "width": max_width,
                    "currentVersion": current_gen,
                    "currentWidth": current_width,
                    "maxVersion": max_gen,
                    "maxWidth": max_width,
                },
                "clocks": {
                    "graphicsMhz": _clock_mhz(os.path.join(device_dir, "pp_dpm_sclk")),
                    "memoryMhz": _clock_mhz(os.path.join(device_dir, "pp_dpm_mclk")),
                },
                "power": {
                    "limitW": power_limit,
                    "idleW": None,
                    "loadW": None,
                    "measured": power_now is not None,
                    "sensor": "rocm-smi" if power_now is not None else None,
                },
                "memoryBandwidth": {
                    "measuredGBs": None,
                    "theoreticalGBs": None,
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
                "thermal": {"maxTempC": temp_c},
            }
        )
    if not gpus and not available():
        return []
    return gpus


def driver_info() -> dict[str, Any]:
    version = read_text("/sys/module/amdgpu/version")
    return {"version": version, "rocm": rocm_version()}


def run(*, quick: bool = False, notes: Notes | None = None) -> list[dict[str, Any]]:
    notes = notes if notes is not None else Notes()
    gpus = enumerate_gpus(notes=notes)
    if not gpus:
        return []
    torch = import_optional("torch")
    hip = getattr(getattr(torch, "version", None), "hip", None) if torch is not None else None
    if torch is not None and hip:
        from .nvidia import measure_torch_gpu

        try:
            cuda_ok = torch.cuda.is_available()
        except Exception:
            cuda_ok = False
        if cuda_ok:
            for gpu in gpus:
                index = gpu.get("index", 0)
                if index >= torch.cuda.device_count():
                    continue
                try:
                    measure_torch_gpu(torch, index, gpu, quick=quick, notes=notes)
                except Exception as exc:  # pragma: no cover - ROCm only
                    notes.add(f"AMD GPU {index}: torch measurement failed ({exc.__class__.__name__}).")
        else:
            notes.add("ROCm torch is installed but no AMD device is available to it.")
    else:
        notes.add(
            "No torch-ROCm build: AMD TFLOPS and memory bandwidth were not measured "
            "(install a ROCm torch wheel)."
        )
    return gpus
