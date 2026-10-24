# Frozen payload contracts (v1 / v1.1)

These shapes are enforced by the servers. The CLI must match them byte-for-byte
on the required fields; additive, clearly-named extensions are marked as such.
Source of truth for the service side lives in the AIRigs and Recipes
repositories; this document is the copy the CLI is built and tested against.

## Rig ids and vocabularies

* `rigId` is an opaque string: `^r_[A-Za-z0-9_-]{1,92}$`. It is never an
  integer. JSON `7` and `"7"` are both invalid.
* `engine` ∈ `vllm, sglang, llamacpp, strata, omlx, ollama, custom`.
  `GET {RECIPES_URL}/api/v1/openapi.json` publishes the enum (including
  `x-airigs-rig-id-pattern`); the CLI reads it at submit time and falls back to
  the constant.
* model `role` ∈ `base, drafter, adapter, vision, embedding`. Missing roles are
  normalised to `base`.
* rigs carry an integer `revision`; the CLI displays it and passes it through as
  `rigRevision` when present.

## Benchmark v1 (POST `.../airigs-rig/:rigId/benchmark`)

```jsonc
{
  "schemaVersion": 1,
  "runner": {"name": "airigs-cli", "version": "0.1.0", "platform": "linux",
             "arch": "x86_64", "python": "3.12"},
  "startedAt": "2026-01-01T00:00:00Z",
  "finishedAt": "2026-01-01T00:03:00Z",
  "durationS": 180.0,
  "challenge": {"nonce": "server-issued", "issuedAt": "2026-01-01T00:00:00Z"}, // upload-time, not a measurement
  "system": {
    "hostname": "box",                              // local report only; not public
    "os": {"name": "Linux", "version": "6.10", "kernel": "6.10.1"},
    "cpu": {"model": "...", "vendor": "AMD", "physicalCores": 16,
            "logicalCores": 32, "baseClockMhz": 4500, "maxClockMhz": 5700,
            "cacheL3Mb": 64},
    "memory": {"totalBytes": 68719476736, "type": "DDR5", "speedMhz": 6000,
               "channels": 2},
    "drivers": {"nvidia": {"version": "550.54", "cuda": "12.4"},
                "amd": {"version": "", "rocm": ""},
                "metal": {"version": ""}},
    "underLoad": {                                  // additive extension
      "durationS": 10.0, "cpuUtilPct": 71.2, "ramUsedBytes": 21474836480,
      "gpus": [{"index": 0, "powerW": 395.2, "utilPct": 99.0, "tempC": 71.0}],
      "sampled": true
    }
  },
  "cpuBench": {
    "singleThread": {"sha256Mbps": 190.2},
    "multiThread": {"sha256Mbps": 2900.5, "threads": 32},
    "fp32Gflops": 120.0,
    "fp64Gflops": 60.0,
    "memory": {"readGBs": 64.2, "writeGBs": 52.1, "copyGBs": 48.3,
               "triadGBs": 50.0}
  },
  "gpus": [{
    "index": 0, "name": "NVIDIA GeForce RTX 4090", "uuid": "GPU-...",
    "vramBytes": 25757220864, "driverVersion": "550.54",
    "pcie": {"version": 4, "width": 16, "currentVersion": 4,
             "currentWidth": 16, "maxVersion": 4, "maxWidth": 16},
    "clocks": {"graphicsMhz": 2520, "memoryMhz": 10501},
    "power": {"limitW": 450, "idleW": 20.1, "loadW": 395.2,
              "measured": true, "sensor": "nvml"},
    "memoryBandwidth": {"measuredGBs": 981.4, "theoreticalGBs": 1008.0,
                        "method": "device-to-device copy"},
    "tflops": {"fp64": 1.2, "fp32": 78.9, "tf32": 157.0, "bf16": 318.0,
               "fp16": 312.0, "fp8": 640.0, "int8": 620.0},
    "tflopsMethod": "torch.matmul",
    "thermal": {"maxTempC": 71}
  }],
  "interconnect": {
    "p2p": {"supported": true, "enabled": true,
            "topologyMatrix": [["X", "SYS"], ["SYS", "X"]],
            "pairs": [{"a": 0, "b": 1, "p2p": true, "bandwidthGBs": 11.8}]},
    "nvlink": {"present": false}
  },
  "power": {"idleW": 65.0, "loadW": 505.0, "measurement": "sensor",
            "psuWatts": 1000},
  "notes": []
}
```

### Required by the server

* `schemaVersion` (exactly 1), `runner.name`, `runner.version`,
  `startedAt`, `finishedAt`;
* `system.cpu.model`, `system.memory.totalBytes`;
* `cpuBench.memory.readGBs`, `cpuBench.memory.copyGBs`;
* `interconnect.p2p.enabled` (boolean);
* at least one GPU entry, each with `name`, `vramBytes`,
  `memoryBandwidth.measuredGBs`, `tflops.fp16`, `power.loadW`,
  `pcie.version`, `pcie.width`.

Anything else may be `null`, but the harness records a matching `notes[]` entry
explaining what was not measured and how to enable it. The CLI refuses to upload
when a required field is `null` and prints the list with remediation.

### Notes

* `power.psuWatts` is a PSU *rating* and is not readable from any standard
  sensor; it is normally `null`.
* `power.measurement` is `"sensor"` when read from RAPL/battery/hwmon and
  `"none"` when nothing is available. It is never wall power.
* `system.underLoad` is an **additive** extension; servers that ignore unknown
  keys are unaffected.
* `challenge` is added to the upload copy, not to a locally saved measurement.
  It comes from `POST /api/v3/plugins/airigs-rig/:rigId/benchmark/challenge`.
  The nonce is single-use and expires (10 minutes). Missing → 422, replayed →
  409, expired → 410. The CLI re-requests and re-measures once.
* `benchmarkHash` is computed by the server; the CLI never sends one.

## RecipeBenchmark v1 (POST `/api/v1/recipes/:slug/benchmark`)

Body: `{"benchmark": <payload>, "publish"?: true}`.

```jsonc
{
  "schemaVersion": 1,
  "rigId": "r_abc123",
  "engine": "vllm",
  "startedAt": "2026-01-01T00:00:00Z",
  "finishedAt": "2026-01-01T00:03:00Z",
  "modelIds": ["meta-llama/Llama-3.1-8B"],
  "contextTokens": 8192,
  "concurrency": 4,
  "requests": 16,
  "ttftMs": 120.5,
  "promptTokensPerSec": 1200.0,
  "outputTokensPerSec": 480.2,
  "totalTokensPerSec": 545.1,
  "vramPeakBytes": 21474836480,
  "ramPeakBytes": 68719476736,
  "powerAvgW": 350.5,
  "gpuUtilPct": 97.0,
  "notes": ["..."]
}
```

Required: `schemaVersion`, `rigId`, `engine`, `startedAt`, `finishedAt`,
`modelIds` (≥ 1), `concurrency`, `ttftMs`, `outputTokensPerSec`,
`vramPeakBytes`, `powerAvgW`.

Semantics:

* `ttftMs` is the median time from request send to the first streamed content
  chunk across all requests.
* `outputTokensPerSec` and `totalTokensPerSec` are aggregate: tokens divided by
  the wall-clock run duration at the stated concurrency.
  `promptTokensPerSec` is prompt tokens divided by summed TTFT (prefill rate).
* Token counts come from the endpoint's `usage` object
  (`stream_options.include_usage`). If the server omits usage the CLI counts
  content chunks and says so in `notes`.
* `vramPeakBytes`/`powerAvgW`/`gpuUtilPct` are sampled with NVML on the host
  running the CLI; a remote endpoint produces a note. `ramPeakBytes` is the
  peak system-used RAM during the run.
* `notes` is a list here; the Recipes service joins it for storage.
