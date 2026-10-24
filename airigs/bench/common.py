"""Shared helpers for the benchmark harness (timing, notes, optional imports)."""

from __future__ import annotations

import importlib
import os
import statistics
import time
from datetime import datetime, timezone
from typing import Any, Callable, Iterable


def now_iso() -> str:
    """UTC timestamp with seconds precision and a Z suffix, as the contract expects."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def elapsed_seconds(start: datetime, end: datetime) -> float:
    return round((end - start).total_seconds(), 3)


class Notes(list):
    """Ordered list of human-readable caveats. Every null gets one."""

    def add(self, message: str) -> None:
        text = " ".join(str(message).split())
        if text and text not in self:
            self.append(text)


def measure(fn: Callable[[], Any], *, warmup: int = 1, repeats: int = 3) -> float:
    """Run ``fn`` after warmup and return the median wall time in seconds.

    Uses ``time.perf_counter`` (a monotonic clock) throughout the harness.
    """
    for _ in range(max(0, warmup)):
        fn()
    samples: list[float] = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def gbps(bytes_moved: float, seconds: float) -> float:
    if not seconds or seconds <= 0:
        raise ValueError("cannot compute bandwidth over a non-positive interval")
    return round(bytes_moved / seconds / 1e9, 2)


def gflops(flops: float, seconds: float) -> float:
    if not seconds or seconds <= 0:
        raise ValueError("cannot compute throughput over a non-positive interval")
    return round(flops / seconds / 1e9, 2)


def tflops(flops: float, seconds: float) -> float:
    """TFLOP/s. ``gflops`` is the same number in GFLOP/s; the payload field is TFLOPS."""
    if not seconds or seconds <= 0:
        raise ValueError("cannot compute throughput over a non-positive interval")
    return round(flops / seconds / 1e12, 2)


def import_optional(name: str) -> Any | None:
    """Import a module, returning ``None`` (never raising) when absent."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def median_or_none(values: Iterable[float]) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def cpu_logical_count() -> int:
    try:
        return len(os.sched_getaffinity(0)) or os.cpu_count() or 1
    except AttributeError:
        return os.cpu_count() or 1


