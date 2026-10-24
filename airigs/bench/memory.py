"""System memory: capacity identity from the OS/DMI, and measured RAM bandwidth.

The bandwidth test is a STREAM-like numpy loop (read, write, copy, triad) timed
on a monotonic clock with warmup and a median over several iterations. Nothing
is reported unless it was measured.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from typing import Any

from .common import Notes, gbps, measure

RAM_DETAILS_PROMPT = "Would you like to include RAM type and speed? (Requires root) [Y/n]: "

#: Size of each working array. Four float64 arrays are live during the triad,
#: so the normal path commits about 512 MiB.
ARRAY_BYTES = 128 * 1024 * 1024
QUICK_ARRAY_BYTES = 32 * 1024 * 1024


def read_text(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return None


def _total_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    text = read_text("/proc/meminfo")
    if text:
        match = re.search(r"^MemTotal:\s*(\d+)\s*kB", text, re.M)
        if match:
            return int(match.group(1)) * 1024
    if sys_platform() == "darwin":
        value = _sysctl("hw.memsize")
        if value:
            return int(value)
    return None


def sys_platform() -> str:
    import sys

    return sys.platform


def _sysctl(name: str) -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, timeout=5, check=False
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _dmidecode_text(command: list[str]) -> str | None:
    """Run one dmidecode invocation and return stdout only on success."""
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode == 0 and completed.stdout.strip():
        return completed.stdout
    return None


def _can_prompt() -> bool:
    """Only ask when a person can see the question and answer it."""
    try:
        return bool(sys.stdin.isatty() and sys.stderr.isatty())
    except (AttributeError, ValueError):
        return False


def _confirm_ram_details() -> bool:
    """Default yes, so pressing Enter includes the details."""
    print(RAM_DETAILS_PROMPT, end="", file=sys.stderr, flush=True)
    try:
        answer = input().strip().lower()
    except EOFError:
        return False
    return answer in ("", "y", "yes")


def _elevation_tool() -> str | None:
    """The privilege tool available on this host, preferring sudo."""
    for tool in ("sudo", "doas"):
        if shutil.which(tool):
            return tool
    return None


def _authenticate(tool: str) -> bool:
    """Let the tool prompt for the password itself; the CLI never handles it.

    ``sudo -v`` caches credentials so the ``-n`` call that follows is quiet. ``doas``
    has no credential cache, so its command prompts directly and needs no preflight.
    """
    if tool != "sudo":
        return True
    try:
        return subprocess.run(["sudo", "-v"], check=False).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def parse_dmidecode(text: str) -> dict[str, Any]:
    """Extract type/speed/channels from ``dmidecode -t memory`` output."""
    result: dict[str, Any] = {"type": None, "speedMhz": None, "channels": None}
    devices = re.split(r"\n(?=Memory Device)", text)
    types: list[str] = []
    speeds: list[int] = []
    banks: set[str] = set()
    for device in devices:
        if "Memory Device" not in device:
            continue
        size = re.search(r"^\s*Size:\s*(.+)$", device, re.M)
        if not size or "no module installed" in size.group(1).lower():
            continue
        type_match = re.search(r"^\s*Type:\s*(\S.*)$", device, re.M)
        if type_match:
            value = type_match.group(1).strip()
            if value and value.lower() != "unknown":
                types.append(value)
        configured = re.search(r"^\s*Configured Memory Speed:\s*(\d+)\s*(?:MT/s|MHz)?", device, re.M)
        speed = re.search(r"^\s*Speed:\s*(\d+)\s*(?:MT/s|MHz)?", device, re.M)
        chosen = configured or speed
        if chosen:
            speeds.append(int(chosen.group(1)))
        bank = re.search(r"^\s*Bank Locator:\s*(.+)$", device, re.M)
        if bank and "channel" in bank.group(1).lower():
            banks.add(bank.group(1).strip().lower())
    if types:
        result["type"] = types[0]
    if speeds:
        result["speedMhz"] = max(speeds)
    if banks:
        result["channels"] = len(banks)
    return result


def memory_specs(notes: Notes | None = None) -> dict[str, Any]:
    notes = notes if notes is not None else Notes()
    specs: dict[str, Any] = {"totalBytes": _total_bytes(), "type": None, "speedMhz": None, "channels": None}
    if specs["totalBytes"] is None:
        notes.add("RAM capacity could not be read (psutil / /proc/meminfo unavailable).")

    if sys_platform() == "linux":
        if not shutil.which("dmidecode"):
            notes.add("RAM type/speed unknown: dmidecode is not installed (util-linux).")
        else:
            text = _dmidecode_text(["dmidecode", "-t", "memory"])
            reason = "RAM type/speed unknown: dmidecode needs root (run `sudo dmidecode -t memory`)."
            tool = _elevation_tool()
            if text is None and tool is not None:
                # Cached credentials (or NOPASSWD) need no interaction; use them silently.
                text = _dmidecode_text([tool, "-n", "dmidecode", "-t", "memory"])
                if text is None and _can_prompt():
                    if _confirm_ram_details():
                        if _authenticate(tool):
                            # doas has no cache, so its command must do the prompting.
                            command = (
                                [tool, "dmidecode", "-t", "memory"]
                                if tool == "doas"
                                else [tool, "-n", "dmidecode", "-t", "memory"]
                            )
                            text = _dmidecode_text(command)
                            if text is None:
                                reason = f"RAM type/speed unknown: dmidecode failed after {tool}."
                        else:
                            reason = f"RAM type/speed unknown: {tool} authentication failed."
                    else:
                        reason = "RAM type/speed omitted at your request."
            if text is not None:
                specs.update(parse_dmidecode(text))
            else:
                notes.add(reason)
    elif sys_platform() == "darwin":
        notes.add("RAM type/speed unknown: Apple Silicon reports unified memory without DIMM SPD.")
    else:
        notes.add("RAM type/speed unknown on this platform.")
    return specs


def _bandwidth_pass(size: int, *, repeats: int) -> dict[str, float]:
    import numpy as np

    n = size // 8
    a = np.ones(n, dtype=np.float64)
    b = np.full(n, 2.0, dtype=np.float64)
    c = np.empty(n, dtype=np.float64)
    scratch = np.empty(n, dtype=np.float64)
    scalar = 2.0

    def read() -> None:
        a.sum()

    def write() -> None:
        c.fill(3.0)

    def copy() -> None:
        c[:] = a

    def triad() -> None:
        # Fused into two ufunc calls with a preallocated scratch buffer: numpy
        # cannot fuse otherwise, and allocating a temporary per iteration charges
        # the measurement for page faults rather than bandwidth.
        np.multiply(c, scalar, out=scratch)
        np.add(b, scratch, out=a)

    results = {}
    results["readGBs"] = gbps(n * 8, measure(read, warmup=1, repeats=repeats))
    results["writeGBs"] = gbps(n * 8, measure(write, warmup=1, repeats=repeats))
    results["copyGBs"] = gbps(n * 16, measure(copy, warmup=1, repeats=repeats))
    # multiply reads c/writes scratch (2n), add reads b/scratch and writes a (3n).
    results["triadGBs"] = gbps(n * 8 * 5, measure(triad, warmup=1, repeats=repeats))
    return results


def ram_bandwidth(*, quick: bool = False, notes: Notes | None = None) -> dict[str, Any]:
    notes = notes if notes is not None else Notes()
    size = QUICK_ARRAY_BYTES if quick else ARRAY_BYTES
    repeats = 2 if quick else 5
    try:
        return _bandwidth_pass(size, repeats=repeats)
    except ImportError:
        notes.add("RAM bandwidth not measured: numpy is not installed (`pip install numpy`).")
    except MemoryError:
        notes.add("RAM bandwidth not measured: not enough free RAM for the STREAM arrays.")
    except Exception as exc:  # pragma: no cover - platform specific failures
        notes.add(f"RAM bandwidth not measured: {exc.__class__.__name__}: {exc}")
    return {"readGBs": None, "writeGBs": None, "copyGBs": None, "triadGBs": None}


def run(*, quick: bool = False, notes: Notes | None = None) -> dict[str, Any]:
    notes = notes if notes is not None else Notes()
    return ram_bandwidth(quick=quick, notes=notes)
