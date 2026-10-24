# Installing and upgrading the `airigs` CLI

The CLI is the measurement and submission tool for AIRigs and Recipes. It is a normal Python
package with one executable, `airigs`. It has no service component, no daemon and no database of
its own.

- Python **3.11 or newer**;
- POSIX host: Linux (including WSL2) or macOS. The benchmark harness reads `/proc`, `/sys` and
  NVML, and `recipe run` executes `sh`; native Windows is not supported;
- network access to the AIRigs instance, and optionally the Recipes instance.

Everything the CLI measures is local. It uploads only when you run a submit/publish command, and
never writes a measurement it did not take — a missing sensor yields `null` plus a note.

## Install with pipx (recommended)

Run from the repository root (the directory containing `pyproject.toml`):

```sh
pipx install .
```

Upgrade an existing pipx install after updating the checkout:

```sh
cd /path/to/airigs-cli
git pull
pipx install --force .
```

`pipx upgrade airigs` only works when the package came from an index; for a local path,
`pipx install --force .` is the reliable form. Uninstall with:

```sh
pipx uninstall airigs
```

Uninstalling does **not** delete your token store (it lives in your XDG config directory). See
"Where the config lives" below and run `airigs logout` first if you want the session revoked.

## Development install

```sh
cd airigs-cli
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
pip install -e '.[gpu,dev]'      # optional measurements + pytest
```

`make install` runs `pip install -e .`; `make test` runs the offline suite.

## Optional `[gpu]` extra

The base install pulls the three required dependencies:

| Package | What it is for |
|---|---|
| `numpy` | CPU matmul benchmarks, RAM bandwidth calculations |
| `psutil` | RAM size, system identity |
| `pynvml` | NVIDIA VRAM, power, PCIe link — imported but only *used* when a driver/device is present |

`pip install 'airigs[gpu]'` (or `pipx install '.[gpu]'`) adds `torch>=2.1`, which enables:

- **TFLOPS** for fp64, fp32, tf32, bf16, fp16, fp8 (where the hardware/torch supports it) and
  int8;
- **measured GPU memory bandwidth**;
- **P2P / NVLink peer-copy bandwidth** per direction and the topology matrix;
- **MPS** measurements on Apple Silicon.

Without torch, all of those fields are `null` and `notes[]` says which extra would fill them.
That is a supported state: you can inspect and benchmark a CPU-only host, but a **rig upload is
refused** with the exact list of missing server-required measurements. Install `[gpu]` and
re-run `airigs bench` on a machine that is meant to become a verified rig.

Use `pipx inject airigs torch` only if your platform needs a specific torch build (for example
ROCm rather than CUDA).

## Log in

```sh
airigs login
```

This starts the OAuth 2.0 **device flow** (RFC 8628): the CLI requests a device code from
AIRigs, prints a verification URL and a short user code, and tries to open your browser. Sign in
to AIRigs, enter/approve the code, and the CLI polls until the flow completes. It then fetches
your profile to prove the token works before storing it.

On a headless machine (SSH, a benchmark box with no browser):

```sh
airigs login --no-browser
```

and open the printed URL on another device. For automation, a token can be supplied on stdin —
never as an argument, because argv is visible in `ps` and shell history:

```sh
printf '%s' "$AIRIGS_TOKEN" | airigs login --token -
```

No environment override is needed for a standard deployment: the CLI authenticates as the public
client id `airigs-cli` and requests
`user:read rig:read rig:write recipe:read recipe:write`. The AIRigs administrator must have
registered that exact client (public, device flow, loopback redirect
`http://127.0.0.1:8765/callback`). If your instance deliberately registers a different id, set
`AIRIGS_CLIENT_ID` for the CLI process.

Check the session at any time with:

```sh
airigs whoami
```

## Pointing at a non-local deployment

Defaults are the local development pair:

| Variable | Default | Meaning |
|---|---|---|
| `AIRIGS_URL` | `http://localhost:4567` | identity provider, rigs, benchmarks |
| `RECIPES_URL` | `http://localhost:4568` | Recipes platform |
| `AIRIGS_TOKEN` | — | bearer token used directly (no refresh chain; for CI) |
| `RECIPES_TOKEN` | falls back to `AIRIGS_TOKEN` | token sent to Recipes |
| `AIRIGS_CLIENT_ID` | `airigs-cli` | public client id; only needed if the instance uses a different one |

For a public instance:

```sh
export AIRIGS_URL=https://airigs.example.com
export RECIPES_URL=https://recipes.example.com
airigs login
```

Rules that the CLI enforces rather than warns about:

- **Tokens are bound to the issuer origin** (scheme, host and port). Change `AIRIGS_URL` to a
  different host and the old token is refused; log in again. This prevents sending an AIRIGS
  token to whatever host a variable points at.
- **Plaintext HTTP is refused for remote hosts.** Only loopback may use `http://`. Use HTTPS for
  anything public; there is no `--insecure` escape hatch.
- Redirects are never followed, so a misconfigured URL fails instead of forwarding your
  `Authorization` header somewhere else.

Environment variables override the stored config, and explicit flags override both. If you set
`AIRIGS_URL` only for one command, prefix it (`AIRIGS_URL=https://… airigs whoami`) rather than
exporting it permanently.

## Check the host

```sh
airigs doctor
```

`doctor` is read-only: it checks the Python version, required imports, optional measurement
libraries, the NVIDIA/AMD tools, the config file's permissions, the stored token's state and
origin, and whether AIRigs and Recipes are reachable. It does not upload anything.

Typical output on a host with a GPU driver but no torch and no login yet:

```text
state  check              detail                                                    action
ok     python             3.14.8
ok     numpy              2.5.4
ok     psutil             7.2.2
ok     pynvml             installed
warn   torch              not installed                                             -> install `airigs[gpu]` for TFLOPS / GPU bandwidth / P2P measurements
ok     nvidia-smi         /usr/bin/nvidia-smi
ok     nvml               1 GPU(s) visible
skip   rocm               not present
warn   config file        not created yet                                           -> run `airigs login`
warn   token              missing (source: none, issuer: unknown, expires in n/as)  -> run `airigs login`
ok     issuer transport   http://localhost:4567
ok     AIRigs reachable   http://localhost:4567 (HTTP 200)
ok     Recipes reachable  http://localhost:4568 (HTTP 200)
skip   mongodb            not used by the CLI; the forum owns its own database
```

`warn` rows are normal before login and without `[gpu]`; `fail` rows (for example a config file
other users can read) need fixing. `airigs doctor --json` gives the same checks as objects.

`airigs --self-check` is the deeper check: it runs a quick benchmark and validates the payload
against the server-required field list. It exits non-zero when this host cannot produce an
uploadable rig (no torch, no readable power sensor, and so on), which is the point of the check.

## Where the config lives

```text
${XDG_CONFIG_HOME:-$HOME/.config}/airigs/config.json
```

- directory mode `0700`, file mode `0600`;
- it contains the access token, refresh token, expiry, the issuer URL, `tokenOrigin` and the
  Recipes URL;
- tokens are stored as issued because they are opaque values that must be replayed; they are
  never printed, and `--verbose` redacts them from request logs;
- it is **not** encrypted. Do not copy it between machines, sync it, or include it in a backup
  you would not trust with a password. On a shared host, keep your home directory private and
  check the mode with `airigs doctor`;
- deleting the file signs you out locally but leaves the refresh token valid at the server.
  Prefer `airigs logout`.

## Log out / revoke

```sh
airigs logout
```

This posts the refresh token to AIRigs' `/oauth/revoke` endpoint (best effort, and only to the
issuer it was minted by), which revokes the token family, then removes the local token fields.
It prints `Logged out; local tokens removed.` and adds `Refresh token revoked.` when the server
confirmed it. If revocation could not be confirmed — offline, or the token belonged to another
origin — it warns on stderr and still removes local state, so the CLI is never left in a
half-signed-in state.

To sign out on the server side without the CLI (lost laptop), an AIRigs administrator can revoke
the client-wide tokens from **Admin → Plugins → airigs SSO** on the client card, or the account
can change its password/revoke sessions in AIRigs.

## Troubleshooting

**`airigs login` says the token belongs to another origin.** The stored token was minted for a
different `AIRIGS_URL`. Log in once against the new origin; reusing a token across origins is
refused by design.

**Login fails with `invalid_client`.** The instance has no client with id `airigs-cli`. Ask the
administrator to register it (see `airigs/docs/upgrading.md`, section 5), or set
`AIRIGS_CLIENT_ID` to the id they registered.

**`airigs doctor` says config mode is wrong.** `chmod 600 "$XDG_CONFIG_HOME/airigs/config.json"`
and `chmod 700 "$XDG_CONFIG_HOME/airigs"`.

**GPU fields are `null`.** Install `airigs[gpu]`, confirm `nvidia-smi` works, and re-run
`airigs bench`. Power additionally needs readable RAPL/battery/hwmon; dmidecode needs root.
Every missing value produces a note naming the requirement.

**`rig submit` refuses the payload.** It prints the missing server-required fields. Run
`airigs bench --output bench.json` and inspect; do not hand-edit numbers into the payload.

## Verified here

On 2026-10-24, in a throwaway virtualenv on the development host:

- `python3 -m venv /tmp/airigs-cli-venv && pip install -e .` — succeeded;
- `airigs --version` — `airigs 0.1.0`;
- `airigs doctor` — the table above (`exit=0`).

`pipx` was not installed on this host, and no browser login was driven end to end here. The
`pipx install .`, `airigs login` and revocation behaviour is documented from the packaging and
code, and from the device-flow self-test in `airigs/bin/sso-selftest.js`, which exercises the
same server endpoints.
