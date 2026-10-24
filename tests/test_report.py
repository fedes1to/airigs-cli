from __future__ import annotations

import json

import pytest

from airigs.bench import report
from fake_server import make_benchmark


def test_complete_payload_passes_validation():
    bench = make_benchmark()
    assert report.missing_required(bench) == []
    assert report.check_shape(bench) == []
    report.require_complete(bench)


def test_schema_version_is_pinned():
    bench = make_benchmark(schemaVersion=2)
    fields = [item["field"] for item in report.missing_required(bench)]
    assert "schemaVersion" in fields


def test_missing_cpu_model_is_reported_with_remediation():
    bench = make_benchmark()
    bench["system"]["cpu"]["model"] = None
    missing = {item["field"]: item for item in report.missing_required(bench)}
    assert "system.cpu.model" in missing
    assert missing["system.cpu.model"]["remediation"]


def test_missing_gpu_measurements_are_listed_individually():
    bench = make_benchmark()
    gpu = bench["gpus"][0]
    gpu["tflops"]["fp16"] = None
    gpu["power"]["loadW"] = None
    gpu["pcie"]["width"] = None
    missing = {item["field"] for item in report.missing_required(bench)}
    assert "gpus[0].tflops.fp16" in missing
    assert "gpus[0].power.loadW" in missing
    assert "gpus[0].pcie.width" in missing
    assert all("pip install" in item["remediation"] or "install" in item["remediation"] for item in report.missing_required(bench))


def test_no_gpu_is_reported():
    bench = make_benchmark(gpus=[])
    missing = report.missing_required(bench)
    assert missing[0]["field"] == "gpus"


def test_require_complete_raises_with_every_field():
    bench = make_benchmark(gpus=[])
    bench["cpuBench"]["memory"]["readGBs"] = None
    with pytest.raises(report.IncompleteBenchmark) as info:
        report.require_complete(bench)
    message = str(info.value)
    assert "gpus" in message
    assert "cpuBench.memory.readGBs" in message
    assert "Nothing was uploaded." in message
    assert len(info.value.missing) == 2


def test_check_shape_catches_missing_top_level_and_bad_types():
    bench = make_benchmark()
    del bench["power"]
    bench["gpus"][0]["pcie"]["width"] = "sixteen"
    problems = report.check_shape(bench)
    assert any("power" in problem for problem in problems)
    assert any("pcie.width" in problem for problem in problems)


def test_build_is_ordered_and_adds_under_load():
    built = report.build(
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:01:00Z",
        duration_s=60.123456,
        system={"cpu": {}, "memory": {}, "os": {}, "drivers": {}},
        cpu_bench={},
        gpus=[],
        interconnect={},
        power={},
        notes=["a note"],
        runner={"name": "airigs-cli", "version": "0.1.0"},
        under_load={"cpuUtilPct": 55.5},
    )
    assert list(built)[:4] == ["schemaVersion", "runner", "startedAt", "finishedAt"]
    assert built["durationS"] == 60.123
    assert built["system"]["underLoad"] == {"cpuUtilPct": 55.5}
    assert built["notes"] == ["a note"]


def test_write_and_read_json(tmp_path):
    bench = make_benchmark()
    target = tmp_path / "nested" / "bench.json"
    written = report.write_json(target, bench)
    assert written.exists()
    assert report.read_json(written) == bench


def test_summary_rows_include_every_section():
    rows = dict(report.summary(make_benchmark()))
    assert "runner" in rows
    assert "gpu0" in rows
    assert "RTX 4090" in rows["gpu0"]
    assert "p2p" in rows
    assert rows["power"].startswith("idle 65")
