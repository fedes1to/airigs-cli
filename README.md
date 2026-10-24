# `airigs` — the official AIRigs CLI

A standalone repository: this CLI is shared tooling that drives **two services**.
AIRigs (`AIRIGS_URL`) is the identity provider and the rig/benchmark host; the
Recipes platform (`RECIPES_URL`) is a separate resource server that accepts the
same AIRigs access token. Keeping the client in its own repo means neither site
owns it and both can version it independently.

One binary for the whole hardware-to-recipe loop:

1. authenticate against AIRigs (OAuth 2.0 device authorization, RFC 8628);
2. measure this machine (CPU, RAM, GPUs, power, PCIe, NVLink/P2P);
3. submit a **rig** and get it verified;
4. submit a **recipe** to the Recipes platform, benchmark the running endpoint,
   publish it, and fetch/run its launch script.

Everything that can be machine-readable is: `--json` on every command. Nothing
is uploaded unless it passed the server-required field check, and no measurement
is ever fabricated — a missing sensor yields `null` plus a note.

## Install

```sh
# recommended for a CLI (from the repository root)
pipx install .
pipx install '.[gpu]'         # with torch, for TFLOPS / GPU bandwidth / P2P

# or into a virtualenv
python -m venv .venv && . .venv/bin/activate
pip install -e .
pip install -e '.[gpu,dev]'
```

Python 3.11+.

The full install/upgrade notes — `[gpu]`, device-flow login, non-local URLs, `doctor`, config
location and revocation — are in [`docs/installing.md`](docs/installing.md).

| dependency | what it buys |
|---|---|
| `numpy`, `psutil` | required: CPU benchmarks and RAM bandwidth/size |
| `pynvml` | required dependency, but only *used* when an NVIDIA driver is present: VRAM, power, PCIe |
| `[gpu]` → `torch` | TFLOPS per precision, measured GPU memory bandwidth, P2P bandwidth, MPS on Apple |
| `[dev]` → `pytest` | the test suite |

Without torch those fields are `null` with a note, and a rig upload is refused
with the exact list of missing measurements and how to install them.

## The flow

```sh
# 1. Log in. Opens a browser; prints a URL + user code to enter.
airigs login
#    Headless (CI, SSH): token on stdin. A token on argv is refused on purpose
#    because it leaks through `ps` and shell history.
printf '%s' "$AIRIGS_TOKEN" | airigs login --token -

# 2. Measure. Writes nothing unless --output is given.
airigs bench --output bench.json
airigs bench --quick                    # CI-sized iteration counts

# 3. Submit the rig and its benchmark in one step. This asks the server for a
#    single-use benchmark challenge, measures, uploads, and fails loudly unless
#    the server returns verified:true.
airigs rig submit --name "workstation"
#    Reuse a previous measurement, or create the draft separately:
airigs rig submit --name "workstation" --benchmark bench.json
airigs rig create --name "workstation"
airigs rig benchmark r_abc123 --output bench.json

# 4. Recipe: scaffold, validate, submit a draft.
airigs recipe init --name "70B on 2x4090"
airigs recipe validate
airigs recipe submit --publish          # or without --publish for a draft

# 5. Benchmark the already-running OpenAI-compatible server, then publish.
airigs recipe bench my-recipe \
  --endpoint http://localhost:8000/v1 \
  --engine vllm --context 8192 --concurrency 4 --requests 16 \
  --model meta-llama/Llama-3.1-8B --publish

# 6. Fetch and run the launch script. It is always shown first; running it
#    requires `--yes` (or AIRIGS_YES=1) when stdin is not a TTY.
airigs recipe script my-recipe --out run.sh
airigs recipe run my-recipe --yes -- --port 8000
```

Other commands:

```sh
airigs whoami
airigs rig list
airigs rig show r_abc123          # id or name
airigs rig delete r_abc123 --yes
airigs recipe bench ./recipe.json --endpoint http://localhost:8000
airigs doctor                     # environment / URL / token checklist
airigs logout                     # revokes the refresh token, then clears local state
airigs --self-check               # quick benchmark + schema/required-field validation
```

`airigs recipe bench <slug|file>` accepts a local recipe file and submits a
draft automatically (it needs a slug to attach the benchmark to).

### Rig ids and engines

* Rig ids are opaque strings (`r_` + up to 92 characters), never integers.
  `rigId: 7` is rejected by the CLI exactly as the server rejects it.
* Engines come from the shared vocabulary: `vllm`, `sglang`, `llamacpp`,
  `strata`, `omlx`, `ollama`, `custom`. `recipe submit` / `recipe bench`
  prefer the enum published at `GET /api/v1/openapi.json` and fall back to that
  list when the platform does not expose it.
* Rig payloads carry an integer `revision`; it is shown by `rig list` / `rig
  show` and passed through as `rigRevision` when a recipe file contains it.

### Benchmark challenges

Rig benchmarks are bound to a server-issued, single-use challenge. Before
measuring, the CLI calls

```
POST /api/v3/plugins/airigs-rig/:rigId/benchmark/challenge
```

and includes `challenge: {nonce, issuedAt}` in the Benchmark v1 payload. A
`409`/`410` (expired or replayed) causes one fresh challenge and a re-measure;
the CLI never blindly retries. `benchmarkHash` is only ever what the server
returns — the CLI does not fabricate one.

## Machine-readable output and agents

* Every command accepts `--json`, before or after the subcommand.
* Errors go to stderr and exit non-zero. With `--json`, an error is a JSON
  object: `{"error": "...", "status": 422}`. Exit codes: `1` general, `3`
  auth/forbidden, `4` not found/conflict, `5` validation (422), `6` rate limit.
* `--verbose` logs HTTP requests with tokens redacted.
* Non-interactive usage: `--yes` for deletes and `recipe run`,
  `AIRIGS_TOKEN` / `RECIPES_TOKEN` for auth, `AIRIGS_YES=1` as the non-TTY
  confirmation gate, `--no-browser` for login.
* `airigs bench --json` is the Benchmark v1 payload; `--output` also writes it.
* `airigs --json whoami`, `rig list`, `rig show` return the server objects.
* `airigs --self-check --json` returns `{ok, uploadable, schemaProblems,
  missingRequired, benchmark}`. `ok` is false if the payload is malformed **or**
  if any server-required measurement is missing, so a green self-check means
  "this host can produce an uploadable rig".

Configuration lives in `$XDG_CONFIG_HOME/airigs/config.json`
(`~/.config/airigs/config.json`), mode `0600`, directory `0700`. Environment:
`AIRIGS_URL` (default `http://localhost:4567`), `RECIPES_URL` (default
`http://localhost:4568`), `AIRIGS_TOKEN`, `RECIPES_TOKEN`, and
`AIRIGS_CLIENT_ID` (the public client the administrator registered with
`bin/dev-client.js`; AIRigs generates its `aic_…` id, so the default
`airigs-cli` only matches an instance that happens to use that id).

## What is measured

`bench` produces the Benchmark v1 payload:

* system: OS/kernel, CPU model/cores/clocks (procfs + sysfs), RAM size/type/
  speed (dmidecode when readable), driver + CUDA/ROCm versions;
* CPU: single/multi-threaded SHA-256 MB/s, fp32/fp64 GFLOP/s (median of repeats);
* RAM: measured STREAM-like read/write/copy/triad GB/s, warmup + median;
* GPUs: VRAM, PCIe current and max gen/width, clocks, temperature, power
  limit, **measured** device-to-device bandwidth, TFLOPS for fp64/fp32/tf32/
  bf16/fp16/fp8/int8 (fp8 only on supported hardware/torch — otherwise null);
* interconnect: topology matrix from `nvidia-smi topo -m`, peer access,
  NVLink presence, measured peer-copy bandwidth per direction;
* power: per-GPU idle + sustained load from NVML/rocm-smi (`measured:true`),
  and whole-system idle/load from RAPL/battery/hwmon (`measurement:"sensor"`).
  Load power is the median of the load-window samples; individual readings
  above 1.5x the enforced limit (or negative/NaN) are discarded, and if more
  than half are implausible the value is null with a note instead.
  A PSU *rating* is not readable from any standard sensor, so `psuWatts` is null.
* `system.underLoad` is an **additive** extension (not in the required schema):
  CPU %, RAM peak, and per-GPU power/util/temp sampled during the load phase.

All timing uses `time.perf_counter`. Every optional dependency or sensor that is
absent yields `null` plus an entry in `notes[]`.

`recipe bench` drives `GET /v1/models` and streaming
`POST /v1/chat/completions` (`stream:true`, `stream_options.include_usage`),
measures TTFT to the first content chunk, and reports aggregate output/total
tokens per wall-clock second at the chosen concurrency. It fails if the
endpoint does not serve the requested model or serves a different one. VRAM
peak and GPU power come from NVML **on the host running the CLI**, so run it on
the inference host; a remote endpoint produces a note saying so.

## Troubleshooting

**`airigs login` says the token belongs to another origin.** Tokens are bound
to the issuer origin (`tokenOrigin`). Changing `--airigs-url`/`AIRIGS_URL` to a
different host requires a fresh login; reusing a token across origins is
refused. Plaintext HTTP issuers are refused outside localhost.

**Everything is `null` on Linux.** `pynvml`/NVML needs the NVIDIA driver;
TFLOPS and GPU bandwidth need `pip install 'airigs[gpu]'`. Power needs readable
RAPL (`/sys/class/powercap/intel-rapl:*`) or a battery/hwmon sensor. dmidecode
needs root. Each case produces a note saying which command/tool to install.

**No GPU detected.** Install the vendor driver. `nvidia-smi` must list the
device; for AMD, `/sys/class/drm/card*/device/vendor` is `0x1002` and/or
`rocm-smi` must work.

**macOS / Apple Silicon.** The harness reports the chip and unified memory and
measures MPS bandwidth/TFLOPS when torch has MPS. macOS exposes no power sensor
and an integrated GPU has no PCIe link, so those fields stay `null`; the server
requires both for a verified rig, so an Apple rig cannot be uploaded as
verified until the contract provides an alternative. This is reported as a note.

**Upload refused.** The CLI prints the exact missing fields with remediation;
run `airigs bench --output bench.json` and inspect. `--no-power` intentionally
leaves power null and therefore cannot produce an uploadable rig.

**Windows** is not supported by the harness (procfs/NVML paths and `sh` script
execution are POSIX). WSL2 works as Linux.

## Security model

* Tokens are stored `0600` in a `0700` directory. Access/refresh tokens are
  opaque values that must be replayed to the provider, so the client stores them
  as issued; the provider stores only hashes of its own token records. They are
  never printed, and `--verbose` redacts bearer tokens and form-encoded token
  parameters.
* Tokens are bound to their issuer origin. A redirect is **never** followed:
  `urllib` otherwise forwards `Authorization` across hosts, so the CLI installs
  a handler that turns any 3xx into an error. `join_url` accepts only relative
  paths, so a server response cannot point a request at another origin.
* Bearer tokens are refused over plaintext HTTP to anything but
  localhost/127.0.0.1/::1.
* `--token` only accepts `-` (read stdin); passing a token as an argument is
  rejected because it would appear in `ps` and shell history.
* `logout` revokes the refresh token at `/oauth/revoke` (best effort) and always
  removes local state, reporting whether revocation was confirmed.
* `recipe bench` uses the same no-redirect HTTP opener. `--api-key` is never
  sent over plaintext HTTP to a non-loopback host (a keyless plaintext remote is
  warned about), endpoint URLs with embedded credentials or non-http(s) schemes
  are rejected, and `/v1/models` plus each SSE event/stream are capped so a
  hostile endpoint cannot exhaust the CLI's memory. Every request also has a
  total wall-clock deadline (30 s for the models probe, `--timeout`-style
  `TIMEOUT_S` otherwise), so a slow-drip endpoint cannot hold the CLI open with
  per-read timeouts that never expire.
* `recipe run` executes author-controlled shell. It always displays the script
  and requires `--yes`/`AIRIGS_YES=1`; the child process environment is
  scrubbed of token/secret/password/API-key variables so a recipe cannot read
  `AIRIGS_TOKEN`, `RECIPES_TOKEN`, `AIRIGS_CLIENT_SECRET`, `HF_TOKEN`, etc.
* No telemetry, no analytics, no automatic uploads. The CLI talks only to
  `AIRIGS_URL`, `RECIPES_URL`, and the endpoint passed to `recipe bench`.

## Development

```sh
make install       # pip install -e .
make test          # python3 -m pytest tests -q  (no network, no GPU required)
make lint          # compileall over airigs/ and tests/
make self-check    # quick benchmark + server-required field validation
```

The machine-readable contract lives in [`docs/benchmark-v1.md`](docs/benchmark-v1.md);
[`AGENTS.md`](AGENTS.md) is the orientation for coding agents.

The tests include a local fake AIRigs/Recipes/OpenAI server, so the suite runs
offline. They cover the RFC 8628 polling state machine (including
`slow_down`/`expired_token`/`access_denied`), config permissions and redaction,
HTTP error mapping, payload shaping, recipe benchmark math from synthetic SSE
streams, benchmark fallbacks (missing torch/NVML), script safety, redirect and
origin-binding security, single-flight refresh, and a full
login → challenge → bench → rig submit → recipe draft → benchmark → publish →
script flow against the fake server.
