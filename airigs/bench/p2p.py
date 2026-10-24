"""GPU interconnect: peer access, topology, NVLink and measured peer bandwidth."""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from typing import Any

from .common import Notes, gbps, import_optional, median_or_none

_NV_LINK = re.compile(r"NV\d+")


def parse_topology(output: str) -> list[list[str]] | None:
    """Parse ``nvidia-smi topo -m`` text into a row-major matrix.

    The header line begins with ``GPU0 GPU1 ...`` too, so it is identified by
    the trailing ``CPU Affinity``/``NUMA Affinity`` columns rather than by shape.
    """
    lines = output.splitlines()
    header_index = next(
        (i for i, line in enumerate(lines) if "Affinity" in line or "NUMA" in line),
        None,
    )
    if header_index is not None:
        header = [cell for cell in lines[header_index].split() if re.fullmatch(r"GPU\d+", cell)]
        data_lines = lines[header_index + 1 :]
    else:
        header = []
        data_lines = lines
    rows: list[list[str]] = []
    for line in data_lines:
        cells = line.split()
        if not cells or not re.fullmatch(r"GPU\d+", cells[0]):
            continue
        rows.append(cells)
    if not rows:
        return None
    size = len(header) or len(rows)
    matrix: list[list[str]] = []
    for cells in rows[:size]:
        values = [cell.upper() for cell in cells[1 : 1 + size]]
        while len(values) < size:
            values.append("X")
        matrix.append(values)
    return matrix if len(matrix) == size else None


def topology_matrix() -> list[list[str]] | None:
    """Run ``nvidia-smi topo -m`` and parse it, or return ``None``."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        completed = subprocess.run(
            ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return parse_topology(completed.stdout)


def nvlink_present(matrix: list[list[str]] | None) -> bool:
    if not matrix:
        return False
    return any(_NV_LINK.fullmatch(cell) for row in matrix for cell in row)


def _measure_pair(torch: Any, a: int, b: int, *, quick: bool, notes: Notes) -> float | None:
    total_bytes = 64 * 1024 * 1024 if quick else 256 * 1024 * 1024
    elements = total_bytes // 4
    src = dst = None
    try:
        src = torch.empty(elements, device=torch.device(f"cuda:{a}"), dtype=torch.float32)
        dst = torch.empty(elements, device=torch.device(f"cuda:{b}"), dtype=torch.float32)
        torch.cuda.synchronize()
        dst.copy_(src)
        torch.cuda.synchronize()
        samples = []
        for _ in range(2 if quick else 3):
            torch.cuda.synchronize()
            start = time.perf_counter()
            dst.copy_(src)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - start)
        elapsed = median_or_none(samples)
        return gbps(total_bytes, elapsed) if elapsed else None
    except Exception as exc:
        notes.add(f"P2P bandwidth {a}->{b} not measured ({exc.__class__.__name__}).")
        return None
    finally:
        del src, dst
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def run(*, quick: bool = False, notes: Notes | None = None) -> dict[str, Any]:
    notes = notes if notes is not None else Notes()
    matrix = topology_matrix()
    result: dict[str, Any] = {
        "p2p": {"supported": False, "enabled": False, "topologyMatrix": matrix, "pairs": []},
        "nvlink": {"present": nvlink_present(matrix)},
    }

    torch = import_optional("torch")
    if torch is None:
        notes.add("torch is not installed: P2P capability and bandwidth were not measured.")
        return result
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False
    if not cuda_available:
        notes.add("CUDA is not available to torch: P2P capability was not measured.")
        return result

    count = int(torch.cuda.device_count())
    p2p = result["p2p"]
    p2p["supported"] = count > 1
    if count < 2:
        notes.add("Only one CUDA device: peer-to-peer is not applicable.")
        return result

    access: dict[tuple[int, int], bool] = {}
    for a in range(count):
        for b in range(count):
            if a == b:
                continue
            try:
                access[(a, b)] = bool(torch.cuda.can_device_access_peer(a, b))
            except Exception:
                access[(a, b)] = False

    pairs: list[dict[str, Any]] = []
    enabled_pairs = 0
    for a in range(count):
        for b in range(count):
            if a == b:
                continue
            allowed = access.get((a, b), False)
            bandwidth = None
            if allowed:
                enabled_pairs += 1
                bandwidth = _measure_pair(torch, a, b, quick=quick, notes=notes)
            pairs.append({"a": a, "b": b, "p2p": allowed, "bandwidthGBs": bandwidth})
    p2p["enabled"] = enabled_pairs > 0
    p2p["pairs"] = pairs
    if not enabled_pairs:
        notes.add(
            "Peer access is disabled for every GPU pair (IOMMU, chipset or driver policy); "
            "P2P bandwidth was not measured."
        )
    return result
