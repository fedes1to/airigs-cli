from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from airigs.bench import amd, cpu, memory, nvidia, p2p, power, report
from airigs.bench.common import Notes
from fake_server import make_benchmark


class DeadSession:
    available = False
    count = 0

    def close(self) -> None:
        pass

    def power_w(self, index):
        return None

    def memory_bus_width(self, index):
        return None

    def pcie(self, index):
        return None

    def power_limit_w(self, index):
        return None


def test_enumerate_gpus_without_driver_reports_note(monkeypatch):
    monkeypatch.setattr(nvidia, "smi_query", lambda: [])
    monkeypatch.setattr(nvidia, "smi_xml", lambda: [])
    monkeypatch.setattr(nvidia, "nvidia_smi_available", lambda: False)
    notes = Notes()
    gpus = nvidia.enumerate_gpus(notes=notes, session=DeadSession())
    assert gpus == []
    assert any("No NVIDIA GPU" in note for note in notes)


def test_enumerate_gpus_with_smi_but_no_xml_never_fabricates(monkeypatch):
    row = {
        "index": "0",
        "name": "NVIDIA Test",
        "uuid": "GPU-1",
        "memory.total": "1024",
        "driver_version": "1.0",
        "pcie.link.gen.current": "4",
        "pcie.link.gen.max": "4",
        "pcie.link.width.current": "16",
        "pcie.link.width.max": "16",
        "clocks.current.graphics": "100",
        "clocks.current.memory": "500",
        "power.limit": "[N/A]",
        "temperature.gpu": "40",
        "power.draw": "5",
    }
    monkeypatch.setattr(nvidia, "smi_query", lambda: [row])
    monkeypatch.setattr(nvidia, "smi_xml", lambda: [{}])
    notes = Notes()
    [gpu] = nvidia.enumerate_gpus(notes=notes, session=DeadSession())
    assert gpu["vramBytes"] == 1024 * 1024 * 1024
    assert gpu["power"]["limitW"] is None  # never invented
    assert gpu["memoryBandwidth"]["theoreticalGBs"] is None
    assert gpu["tflops"]["fp16"] is None
    assert gpu["power"]["measured"] is False
    assert any("theoretical memory bandwidth unknown" in note for note in notes)


def test_theoretical_bandwidth_math():
    assert nvidia.theoretical_bandwidth_gbs(384, 10501) == 1008.1
    assert nvidia.theoretical_bandwidth_gbs(None, 10501) is None
    assert nvidia.theoretical_bandwidth_gbs(384, None) is None


def test_topology_parser_handles_header_and_nvlink():
    output = """
\tGPU0\tGPU1\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID
GPU0\t X \tNV12\t0-15\t0\t\tN/A
GPU1\tNV12\t X \t0-15\t0\t\tN/A

Legend:

  X    = Self
  NV#  = Connection traversing a bonded set of # NVLinks
"""
    matrix = p2p.parse_topology(output)
    assert matrix == [["X", "NV12"], ["NV12", "X"]]
    assert p2p.nvlink_present(matrix) is True


def test_topology_parser_without_affinity_header():
    matrix = p2p.parse_topology("GPU0\tX\tSYS\nGPU1\tSYS\tX\n")
    assert matrix == [["X", "SYS"], ["SYS", "X"]]


def test_nvlink_detection():
    assert p2p.nvlink_present([["X", "NV12"], ["NV12", "X"]]) is True
    assert p2p.nvlink_present([["X", "SYS"], ["SYS", "X"]]) is False
    assert p2p.nvlink_present(None) is False


def test_run_torch_measurements_without_torch(no_optional_imports):
    gpu = {
        "index": 0,
        "name": "NVIDIA Test",
        "memoryBandwidth": {"measuredGBs": None, "theoreticalGBs": None},
        "tflops": {"fp16": None},
        "tflopsMethod": None,
    }
    notes = Notes()
    nvidia.run_torch_measurements([gpu], quick=True, notes=notes)
    assert gpu["memoryBandwidth"]["measuredGBs"] is None
    assert gpu["tflops"]["fp16"] is None
    assert gpu["tflopsMethod"] is None
    assert any("torch is not installed" in note for note in notes)


def test_p2p_without_torch_is_honest(monkeypatch, no_optional_imports):
    monkeypatch.setattr(p2p, "topology_matrix", lambda: None)
    notes = Notes()
    result = p2p.run(quick=True, notes=notes)
    assert result["p2p"]["supported"] is False
    assert result["p2p"]["enabled"] is False
    assert result["p2p"]["pairs"] == []
    assert result["nvlink"]["present"] is False
    assert any("torch is not installed" in note for note in notes)


def test_ram_bandwidth_failure_returns_nulls(monkeypatch):
    def boom(size, repeats):
        raise MemoryError("no room")

    monkeypatch.setattr(memory, "_bandwidth_pass", boom)
    notes = Notes()
    result = memory.ram_bandwidth(quick=True, notes=notes)
    assert result == {"readGBs": None, "writeGBs": None, "copyGBs": None, "triadGBs": None}
    assert any("not measured" in note for note in notes)


def test_system_power_without_sensor_reports_none(monkeypatch):
    monkeypatch.setattr(power, "detect_source", lambda: (None, []))
    notes = Notes()
    result = power.idle_power(0.05, notes=notes)
    assert result["idleW"] is None
    assert result["loadW"] is None
    assert result["measurement"] == "none"
    assert any("no RAPL" in note for note in notes)


def test_dmidecode_parsing():
    sample = """
Memory Device
        Size: 32 GB
        Type: DDR5
        Speed: 5600 MT/s
        Configured Memory Speed: 6000 MT/s
        Locator: DIMM 0
        Bank Locator: P0 CHANNEL A
Memory Device
        Size: No Module Installed
        Type: Unknown
Memory Device
        Size: 32 GB
        Type: DDR5
        Configured Memory Speed: 6000 MT/s
        Locator: DIMM 1
        Bank Locator: P0 CHANNEL B
"""
    parsed = memory.parse_dmidecode(sample)
    assert parsed == {"type": "DDR5", "speedMhz": 6000, "channels": 2}


def test_proc_cpuinfo_parsing_tmp_path():
    sample = """processor\t: 0
vendor_id\t: GenuineIntel
model name\t: Test CPU
physical id\t: 0
core id\t\t: 0
cpu cores\t: 2
cache size\t: 32768 KB

processor\t: 1
vendor_id\t: GenuineIntel
model name\t: Test CPU
physical id\t: 0
core id\t\t: 1
cpu cores\t: 2
"""
    blocks = cpu.parse_proc_cpuinfo(sample)
    assert len(blocks) == 2
    assert blocks[0]["model name"] == "Test CPU"
    assert blocks[1]["core id"] == "1"


def test_amd_on_other_platforms_is_empty(monkeypatch):
    monkeypatch.setattr(amd, "sys", SimpleNamespace(platform="darwin"))
    assert amd.enumerate_gpus(notes=Notes()) == []


def test_check_shape_flags_wrong_types():
    problems = report.check_shape({"schemaVersion": 1})
    assert any("runner" in problem for problem in problems)


def test_report_upsert_extension_under_load_is_not_a_top_level_key():
    built = report.build(
        started_at="a",
        finished_at="b",
        duration_s=1,
        system={},
        cpu_bench={},
        gpus=[],
        interconnect={},
        power={},
        notes=[],
        runner={},
        under_load={"cpuUtilPct": 1},
    )
    assert "underLoad" not in built
    assert built["system"]["underLoad"] == {"cpuUtilPct": 1}


def test_gpu_tflops_are_reported_in_tflops_not_gflops(monkeypatch):
    """A 1024³ fp16 matmul at 10 ms is 0.21 TFLOP/s, not 215 GFLOP/s mislabelled."""
    from airigs.bench import common

    assert common.tflops(2 * 1024**3, 0.01) == pytest.approx(0.21, rel=1e-3)
    # 2*1024^3 flops / 0.01 s = 2.1e11 FLOP/s = 0.21 TFLOP/s.

    class FakeCuda:
        @staticmethod
        def synchronize():
            return None

        @staticmethod
        def empty_cache():
            return None

    class FakeTensor:
        def __init__(self, shape):
            self.shape = shape

    class FakeTorch:
        float16 = "float16"
        float32 = "float32"
        cuda = FakeCuda

        @staticmethod
        def randn(shape, device=None, dtype=None):
            return FakeTensor(shape)

        @staticmethod
        def matmul(a, b):
            return FakeTensor(a.shape)

    clock = {"now": 0.0}

    def perf_counter():
        clock["now"] += 0.01
        return clock["now"]

    monkeypatch.setattr(nvidia.time, "perf_counter", perf_counter)
    value = nvidia._dtype_tflops(
        FakeTorch, "cuda:0", "float16", size=1024, repeats=3, notes=Notes(), label="GPU 0"
    )
    # 2*1024^3 flops / 0.01 s = 0.21 TFLOP/s (0.21 TFLOP/s, not 214.75).
    assert value == pytest.approx(0.21, rel=1e-6)


def test_power_summary_uses_median_and_discards_outlier():
    summary = power.summarize_power_samples([1.6, 119.0, 1.7], limit_w=25.0)
    assert summary["medianW"] == 1.7  # mean would be 40.77
    assert summary["discarded"] == 1
    assert summary["kept"] == 2
    assert "25" in summary["note"]
    # Negative and NaN readings are discarded too.
    negative = power.summarize_power_samples([-1.0, 2.0], limit_w=25.0)
    assert negative["medianW"] == 2.0
    assert negative["discarded"] == 1
    nan_case = power.summarize_power_samples([float("nan"), 2.0], limit_w=25.0)
    assert nan_case["medianW"] == 2.0
    assert nan_case["discarded"] == 1


def test_all_implausible_power_is_null_and_refuses_upload():
    summary = power.summarize_power_samples([119.0, 130.0, 140.0], limit_w=25.0)
    assert summary["medianW"] is None
    assert summary["p95W"] is None
    assert "discarded" in summary["note"]
    bench = make_benchmark()
    bench["gpus"][0]["power"]["loadW"] = summary["medianW"]
    with pytest.raises(report.IncompleteBenchmark) as info:
        report.require_complete(bench)
    assert "gpus[0].power.loadW" in str(info.value)
