# AGENTS.md — orientation for coding agents

This repository is the **standalone** `airigs` CLI. It is shared tooling for two
sites and lives outside both of them:

| service | default | role |
|---|---|---|
| AIRigs | `http://localhost:4567` (`AIRIGS_URL`) | identity provider (OAuth 2.0 device flow), rigs, hardware benchmarks |
| Recipes platform | `http://localhost:4568` (`RECIPES_URL`) | recipes, recipe benchmarks, published scripts; introspects the AIRigs token |

Nothing in this repo may assume it is a subdirectory of either site. The frozen
HTTP contract is documented in `docs/benchmark-v1.md` and quoted in
`airigs/*.py`; treat it as external and do not change payload shapes without a
contract revision.

## Layout

```
airigs/
  cli.py            argparse entry, all commands, human tables, exit codes
  config.py         XDG config path, 0600/0700 storage, env overrides, redaction
  http.py           urllib client: no redirects, relative-only URLs, error mapping
  auth.py           RFC 8628 device flow, refresh (single-flight), revoke/logout
  recipes.py        Recipes API v1, recipe validation, engine enum (OpenAPI)
  recipebench.py    RecipeBenchmark v1: drives an OpenAI-compatible endpoint
  scripts.py        launch-script generation, fetch, confirmation gate, env scrub
  bench/
    common.py       monotonic timing, Notes, optional imports
    cpu.py          procfs/sysfs CPU identity, SHA-256, matmul GFLOP/s
    memory.py       RAM identity + measured STREAM-like bandwidth
    nvidia.py       NVML/nvidia-smi identity, torch TFLOPS/bandwidth, clocks
    amd.py          sysfs/rocm-smi identity, torch-ROCm measurements
    apple.py        Apple Silicon / MPS identity and measurements
    power.py        RAPL / battery / hwmon system power
    p2p.py          peer access, topology matrix, NVLink, peer bandwidth
    report.py       Benchmark v1 assembly + required-field validation
    runner.py       orchestration, load phase, --self-check
tests/              pytest, offline fake AIRigs + Recipes + OpenAI server
docs/benchmark-v1.md frozen payload contract
```

## How the OAuth client works

* `auth.start_device_authorization` POSTs `client_id=airigs-cli` and the scope
  string. `auth.poll_device_token` is the RFC 8628 §3.5 state machine:
  `authorization_pending` waits, `slow_down` adds 5 s to the interval,
  `expired_token`/`access_denied` raise. It takes injectable transport, sleep and
  clock so tests never wait.
* Tokens are saved by `config.save_tokens` to
  `$XDG_CONFIG_HOME/airigs/config.json` (mode 0600, directory 0700) together with
  `tokenOrigin` (scheme/host/port of the issuer) and `expires_at`.
* `auth.ensure_token` refuses a token whose issuer origin differs from the
  configured `AIRIGS_URL`, refuses plaintext HTTP to remote hosts, and refreshes
  under `_REFRESH_LOCK` with a double-check so concurrent callers produce one
  refresh request and one config write.
* `auth.logout` POSTs the refresh token to `/oauth/revoke` (best effort, and
  never to a different origin), then always removes local state and reports
  which happened.

## How HTTP works

`http.ApiClient` is a thin urllib wrapper. Two security properties are
non-negotiable:

1. **Redirects are never followed.** `_RefuseRedirects` turns every 3xx into an
   `HttpError`; urllib would otherwise forward `Authorization` to another host.
2. **`join_url` accepts relative paths only.** Absolute URLs and `//host` are
   rejected so a server response cannot redirect a request's origin.

`recipebench` reuses `open_no_redirect` for `/v1/models` and SSE requests, caps
the models body, each SSE event and the total stream (`MAX_*_BYTES`), rejects
endpoint URLs with userinfo or non-http(s) schemes, and refuses to send
`--api-key` over plaintext HTTP outside loopback (keyless plaintext remote is
allowed with a note).

Error mapping: 401 → `AuthError` (exit 3), 403 → `ForbiddenError`, 404 →
`NotFoundError` (4), 409 → `ConflictError`, 422 → `ValidationError` (5, with
`errors[{path,message}]`), 429 → `RateLimitError` (6, `retry_after`). `--verbose`
logs method/URL/status through `config.redact`, which strips bearer tokens and
form token parameters.

## Benchmark harness rules

* Every measurement uses `time.perf_counter`, warmup, and a median of repeats.
* GPU load power uses `power.summarize_power_samples`: per-sample values above
  1.5x the enforced limit (or negative/NaN) are discarded, the reported number
  is the median of the rest, and >50% discarded yields `null` plus a note. The
  note always states the sample count.
* A missing library, tool, or sensor yields `None` plus an entry in `notes[]`.
  **Never fabricate a number** and never substitute a theoretical value for a
  measured one.
* `report.missing_required` is the server's required-field list. Uploads call
  `report.require_complete` and print the exact missing fields with remediation.
* `--self-check` runs a quick benchmark and sets `ok` only when the shape is
  valid **and** every server-required measurement is present (`uploadable`).
  On a host without torch/NVML it exits non-zero by design.
* Rig benchmarks need a server challenge *before* measuring:
  `POST /api/v3/plugins/airigs-rig/:rigId/benchmark/challenge`, then include
  `challenge:{nonce, issuedAt}` in the payload. 409/410 means expired/replayed:
  fetch a fresh nonce and re-measure (bounded, one retry). Never fabricate
  `benchmarkHash`; only echo it if the server returned it.

## Contracts you must not break

* Rig ids are opaque strings matching `^r_[A-Za-z0-9_-]{1,92}$`. Integers,
  including `"7"`, are invalid.
* Engines are `vllm, sglang, llamacpp, strata, omlx, ollama, custom`. Prefer
  `GET /api/v1/openapi.json`, fall back to `recipes.ENGINES`.
* Model roles are `base, drafter, adapter, vision, embedding`; the skeleton
  defaults to `base`.
* Benchmark v1 and RecipeBenchmark v1 required fields are listed in
  `docs/benchmark-v1.md`. `system.underLoad` is an additive extension; the
  challenge is attached at upload time.
* Tokens are stored `0600` and never printed. They are opaque values that must
  be replayed, so they are stored as issued (the provider hashes its own
  records); do not weaken the file/dir modes, log them, or bind them to fewer
  checks. `--token` accepts only `-` (stdin); env vars
  `AIRIGS_TOKEN`/`RECIPES_TOKEN` are supported.
* `recipe run` shows the script, requires `--yes`/`AIRIGS_YES=1` when
  non-interactive, and scrubs token/secret/password/API-key variables from the
  child environment.

## Tests

```sh
make test          # python3 -m pytest tests -q — offline, no GPU required
make lint          # compileall
make self-check    # quick benchmark + required-field validation
```

`tests/fake_server.py` serves AIRigs OAuth/rig endpoints, Recipes endpoints, an
OpenAPI engine enum, and a streaming OpenAI-compatible endpoint. It enforces the
challenge nonce (missing → 422, replayed → 409, expired → 410). When changing a
payload or a command, extend that server and add an assertion on the recorded
request body rather than mocking at the Python boundary.
