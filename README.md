# airigs

The AIRigs CLI: sign in, benchmark a machine, submit a rig, and publish
reproducible recipes. It talks to **AIRigs** (identity, rigs, benchmarks) and
the **Recipes platform** (recipes, benchmarks, scripts), and uploads only when
you submit or publish.

## Install (Python 3.11+, Linux/WSL2 or macOS)

```sh
git clone https://git.fedesito.me/fedes1to/airigs-cli.git
cd airigs-cli && pipx install .
```

`pipx install '.[gpu]'` adds PyTorch for TFLOPS, GPU memory bandwidth and P2P;
without it those fields are `null` and a rig upload is refused with the exact
missing measurements.

## Log in

Run `airigs login`. The OAuth device flow (RFC 8628) prints a URL and a short
code, opens your browser, and stores the token once approved. On a headless host
use `--no-browser`, or pipe a token on stdin (argv leaks through `ps`):

```sh
printf '%s' "$AIRIGS_TOKEN" | airigs login --token -
```

Local defaults are `http://localhost:4567` (AIRigs) and `http://localhost:4568`
(Recipes). For another deployment, set both and log in again — tokens are bound
to their issuer origin, so a new host always needs a fresh login:

```sh
export AIRIGS_URL=https://airigs.example.com
export RECIPES_URL=https://recipes.example.com
airigs login
```

## Benchmark the machine

```sh
airigs bench --output bench.json
airigs bench --quick          # fewer iterations for a smoke test
```

CPU, RAM, GPUs, PCIe, interconnect and power are measured; a missing sensor or
optional library yields `null` plus a note, never a made-up number. `--json`
prints the Benchmark v1 payload.

## Submit a rig

```sh
airigs rig submit --name "workstation"
```

This requests a single-use challenge, measures, uploads, and fails unless the
server returns `verified: true`. Reuse a saved run with `--benchmark bench.json`.
Also: `rig list`, `rig show <id|name>`, `rig delete <id> --yes`.

## Work with recipes

```sh
airigs recipe init --name "70B on 2x4090"   # then edit recipe.json
airigs recipe validate
airigs recipe submit --publish
```

Benchmark an already-running OpenAI-compatible endpoint, then publish:

```sh
airigs recipe bench my-recipe \
  --endpoint http://localhost:8000/v1 \
  --engine vllm --model meta-llama/Llama-3.1-8B \
  --context 8192 --concurrency 4 --requests 16 --publish
```

The launch script is always shown first; `run` needs `--yes` (or
`AIRIGS_YES=1`) when stdin is not a TTY.

```sh
airigs recipe script my-recipe --out run.sh
airigs recipe run my-recipe --yes -- --port 8000
```

## Troubleshooting

`airigs doctor` is a read-only checklist: Python, libraries, NVIDIA/AMD tools,
config permissions, token state, and service reachability. `airigs --self-check`
reports whether this host can produce an uploadable rig.

## Docs

- [`docs/installing.md`](docs/installing.md) — install, login, config, env vars, logout.
- [`docs/benchmark-v1.md`](docs/benchmark-v1.md) — the frozen Benchmark v1 contract.
- Recipes deployments publish this walkthrough at `/docs/cli`.
- [`AGENTS.md`](AGENTS.md) — layout and conventions for coding agents.

Every command accepts `--json`; exit codes: `0` ok, `1` general, `3` auth,
`4` not found/conflict, `5` validation, `6` rate limit.
