"""Power measurement that never pretends to be wall power.

Sources, in priority order: RAPL package counters, a battery's ``power_now``,
then the sum of hwmon power inputs. Each yields ``measurement: "sensor"`` and the
sensor that produced it. Nothing here reads a PSU rating, so ``psuWatts`` stays
null rather than being invented.
"""

from __future__ import annotations

import glob
import math
import os
import statistics
import time
from typing import Any

from .common import Notes

RAPL_GLOB = "/sys/class/powercap/intel-rapl:[0-9]*/energy_uj"
HWMON_GLOB = "/sys/class/hwmon/hwmon*/power1_input"
BATTERY_GLOB = "/sys/class/power_supply/BAT*/power_now"


def read_text(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read().strip()
    except OSError:
        return None


def read_int(path: str) -> int | None:
    raw = read_text(path)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _hwmon_name(directory: str) -> str:
    return read_text(os.path.join(directory, "name")) or os.path.basename(directory)


def detect_source() -> tuple[str | None, list[str]]:
    rapl = sorted(glob.glob(RAPL_GLOB))
    if rapl:
        return "rapl", rapl
    batteries = sorted(glob.glob(BATTERY_GLOB))
    if batteries:
        return "battery", batteries
    hwmon = [
        path
        for path in sorted(glob.glob(HWMON_GLOB))
        if "psu" not in _hwmon_name(os.path.dirname(path)).lower()
    ]
    if hwmon:
        return "hwmon", hwmon
    return None, []


class SystemPowerMonitor:
    """Average power from whatever sensor the OS exposes.

    RAPL counters are cumulative energy in microjoules: a sample is the delta
    since the previous sample divided by the monotonic-clock interval, with
    counter wraparound handled. hwmon/battery sensors are already power.
    """

    def __init__(self) -> None:
        self.kind, self.paths = detect_source()
        self.sensor = f"{self.kind}:{len(self.paths)}" if self.kind else None
        self._previous_raw: dict[str, int] = {}
        self._previous_time: float | None = None

    @property
    def available(self) -> bool:
        return self.kind is not None

    def _energy_uj(self) -> int | None:
        total = 0
        for path in self.paths:
            value = read_int(path)
            if value is None:
                return None
            previous = self._previous_raw.get(path)
            if previous is None:
                self._previous_raw[path] = value
                continue
            delta = value - previous
            if delta < 0:  # counter wrapped
                limit = read_int(os.path.join(os.path.dirname(path), "max_energy_range_uj"))
                delta += limit or (previous + 1)
            self._previous_raw[path] = value
            total += delta
        return total

    def _instant_watts(self) -> float | None:
        if not self.paths:
            return None
        if self.kind in ("battery", "hwmon"):
            total = 0
            for path in self.paths:
                value = read_int(path)
                if value is None:
                    return None
                total += value
            return round(total / 1e6, 2)

        now = time.monotonic()
        energy = self._energy_uj()  # microjoules consumed since the previous read
        if energy is None:
            return None
        if self._previous_time is None:
            self._previous_time = now
            return None
        interval = now - self._previous_time
        self._previous_time = now
        if interval <= 0:
            return None
        return round(energy / 1e6 / interval, 2)

    def start(self) -> None:
        if self.kind == "rapl":
            for path in self.paths:
                value = read_int(path)
                if value is not None:
                    self._previous_raw[path] = value
            self._previous_time = time.monotonic()

    def sample(self) -> float | None:
        return self._instant_watts()


def measure_system_power(duration_s: float, *, notes: Notes | None = None) -> dict[str, Any]:
    """Average system draw over a window (idle or load, the caller decides)."""
    notes = notes if notes is not None else Notes()
    monitor = SystemPowerMonitor()
    if not monitor.available:
        notes.add(
            "System power not measured: no RAPL, battery or hwmon power sensor is readable "
            "(on Linux try `sudo` or load the intel_rapl/amd_energy module)."
        )
        return {"idleW": None, "loadW": None, "measurement": "none", "psuWatts": None}
    monitor.start()
    samples: list[float] = []
    start = time.monotonic()
    while time.monotonic() - start < duration_s:
        value = monitor.sample()
        if value is not None:
            samples.append(value)
        time.sleep(0.2)
    if not samples:
        notes.add(f"System power sensor ({monitor.sensor}) produced no readings.")
        return {"idleW": None, "loadW": None, "measurement": "none", "psuWatts": None}
    return {
        "averageW": round(sum(samples) / len(samples), 2),
        "maxW": round(max(samples), 2),
        "sensor": monitor.sensor,
        "samples": len(samples),
        "windowS": round(time.monotonic() - start, 2),
    }


def idle_power(duration_s: float, *, notes: Notes | None = None) -> dict[str, Any]:
    measured = measure_system_power(duration_s, notes=notes)
    return {
        "idleW": measured.get("averageW"),
        "loadW": None,
        "measurement": "sensor" if measured.get("averageW") is not None else "none",
        "psuWatts": psu_watts(),
        "sensor": measured.get("sensor"),
        "idleSamples": measured.get("samples", 0),
    }


def psu_watts() -> int | None:
    """A PSU *rating* is not exposed by any standard sensor, so this is always null."""
    return None


#: A single NVML/rocm reading above this factor of the enforced limit is a
#: driver/sensor outlier, not a real measurement.
POWER_LIMIT_FACTOR = 1.5


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of an empty sequence")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_power_samples(
    values: list[float] | tuple[float, ...],
    *,
    limit_w: float | None = None,
    label: str = "GPU",
) -> dict[str, Any]:
    """Robust power summary for a load window.

    Individual samples that are non-finite, negative, or above 1.5x the
    enforced power limit are discarded. The returned value is the median (upper
    median for an even count) of the remaining samples; if more than half the
    samples were discarded, the value is ``None`` and the note says why. The
    note always documents the sample count so the payload explains itself.
    """
    total = len(values)
    limit_text = f"enforced limit {limit_w:g} W" if limit_w else "enforced limit unavailable"
    invalid = 0
    kept: list[float] = []
    for value in values:
        try:
            watts = float(value)
        except (TypeError, ValueError):
            invalid += 1
            continue
        if (
            not math.isfinite(watts)
            or watts < 0
            or (limit_w is not None and watts > POWER_LIMIT_FACTOR * limit_w)
        ):
            invalid += 1
            continue
        kept.append(watts)

    if total == 0:
        return {
            "medianW": None,
            "p95W": None,
            "samples": 0,
            "kept": 0,
            "discarded": 0,
            "limitW": limit_w,
            "note": f"{label}: no power samples were available.",
        }

    if not kept or invalid > total / 2:
        return {
            "medianW": None,
            "p95W": None,
            "samples": total,
            "kept": len(kept),
            "discarded": invalid,
            "limitW": limit_w,
            "note": (
                f"{label}: power not reported: {invalid} of {total} samples discarded as "
                f"implausible ({limit_text}); more than half the samples were invalid."
            ),
        }

    median = statistics.median_high(kept)
    p95 = _percentile(kept, 95)
    return {
        "medianW": round(median, 2),
        "p95W": round(p95, 2),
        "samples": total,
        "kept": len(kept),
        "discarded": invalid,
        "limitW": limit_w,
        "note": (
            f"{label}: median of {len(kept)} valid power samples (of {total}; "
            f"{invalid} discarded as implausible, {limit_text}); p95 {p95:.2f} W."
        ),
    }
