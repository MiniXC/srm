# `srm.tpu` — TPU helper library for ML experiments

A small, opinionated library that takes a TRC TPU grant and a YAML inventory
of pools, and gives you: parallel provisioning with retry-until-capacity,
per-VM bootstrap, an in-process `torchax` shim, checkpoint I/O against GCS,
and clean teardown. Designed to be reusable across projects (no project-specific
strings in code; everything that varies lives in one inventory file).

The design is informed by — and a deliberate refactor of — the TPU machinery in
the sibling `sdm/` repo. That codebase works well in practice but has grown
three independent implementations of the create-retry loop, the runtime-version
table, and the launch-on-VM flow (see [§2.1](#21-what-already-exists-in-sdm-and-what-we-keep-or-discard)).
This spec collapses all of them into one Python library plus a single ~30-line
bootstrap shell script.

---

## Goals

1. **One source of truth.** The retry regex, the accel→runtime mapping, the
   "which secrets get shipped to the VM", the experiment list — each lives in
   exactly one place, and most of them live in a YAML inventory rather than in
   code.
2. **Transparency.** Every gcloud invocation is logged verbatim with a
   timestamp before it runs. Every retry prints which class of error matched
   and what the next sleep is. `--dry-run` everywhere prints the command tree
   without executing. A `--json-logs` mode emits one structured record per
   attempt.
3. **Fail loud, never silently degrade.** When the user asked for a TPU run,
   never silently fall back to CPU. When a dependency is missing, raise with a
   message that says how to fix it. (Inherited from `sdm.train.xla_utils`.)
4. **DRY.** No copy-pasted regexes, no duplicated `case "$ACCEL" in` blocks,
   no parallel hardcoded `NAMES=(...)` arrays in teardown vs provision.
5. **Composable.** Each step (poll-create, wait-ready, push-env, launch) is a
   public function. The CLI is a thin wrapper. Library users can compose their
   own flows.
6. **Project-agnostic.** Nothing in the library hardcodes `sdm`, `ml-edinburgh`,
   `europe-west4-a`, model names, teacher deps, etc. All come from inventory or
   env.

## Non-goals

- A general GCP-resource manager. We only manage `tpus tpu-vm`.
- A scheduler. Fanout is "spawn N background workers"; no priority queue, no
  cross-experiment dependency graph beyond "wait for all of group A before
  group B".
- A Kubernetes/XPK competitor. If you need that, use XPK.
- A replacement for `torchax` itself. We thinly wrap it for the ~8 calls a
  training loop actually makes.

---

## 1. Top-level shape

```
srm/tpu/
├── ARCHITECTURE.md                  # this file
├── pyproject.toml                   # `srm-tpu` distribution
├── srm_tpu/
│   ├── __init__.py
│   ├── inventory.py                 # YAML inventory loader (Pool, Project)
│   ├── pools.py                     # accel→runtime, pool helpers
│   ├── gcloud.py                    # typed wrappers around `gcloud compute tpus ...`
│   ├── retry.py                     # one retry loop, one regex, one log format
│   ├── secrets.py                   # .env loader; whitelist of keys shipped to VM
│   ├── provision.py                 # poll_create, wait_ready, push_env, launch_run, provision
│   ├── daemon.py                    # background workers, pidfiles, log dir
│   ├── bootstrap.py                 # on-VM: detect accel, install wheels, smoke-test
│   ├── preempt.py                   # SIGTERM handler -> StopState
│   ├── io.py                        # local + gs:// checkpoint I/O via gsutil
│   ├── log.py                       # human + JSON line logger (single format)
│   └── cli.py                       # `srm-tpu` Typer app
├── scripts/
│   └── bootstrap.sh                 # ~30 lines; only runs before Python is installed
└── tests/
    ├── test_inventory.py
    ├── test_retry.py
    ├── test_gcloud_dry_run.py
    ├── test_provision_flow.py       # mocks gcloud, asserts on call sequence
    └── test_io_gcs.py               # uses a fake gsutil on PATH
```

### Distribution

`srm_tpu/` lives inside this repository alongside the project's models, configs,
and training scripts. No separate install — the launch script clones the whole
repo onto the VM, and the CLI is invoked via `uv run srm-tpu` or
`python -m srm_tpu.cli`. Pinned deps: `typer`, `pyyaml`, `rich` (for
tables/logs). No `torch` or `torchax` dependency — the library only manages
TPU VMs and ships project code onto them; it never imports torch itself.

- Console entry point `srm-tpu = srm_tpu.cli:app`.

### 1.1 What already exists in `sdm/` and what we keep or discard

| sdm artefact | Status in `srm.tpu` | Why |
|---|---|---|
| `sdm/train/xla_utils.py` | **Drop** — replaced by torchax used directly | torchax (`enable_globally()`, `device='jax'`, `jax.jit`) replaces torch_xla entirely. The user writes torchax code themselves; we don't wrap it. |
| `sdm/train/preempt.py` | **Port verbatim → `srm_tpu/preempt.py`** | 30 lines, correct. |
| `sdm/train/io.py` | **Port → `srm_tpu/io.py`** with a `Backend` ABC | Keep the `gsutil`-shellout default; allow projects to swap in a `gcsfs` backend without forking. |
| `sdm/.tpu/_lib.sh` | **Replace** with `srm_tpu/provision.py` | Bash is the wrong language for retry/state machines. Python gives us testability. |
| `sdm/.tpu/provision_<exp>.sh` ×9 | **Replace** with `srm-tpu launch --pool NAME --config CFG --command CMD` | Nine 6-line scripts that differ only in `NAME=` and `CONFIG=` are an anti-pattern. |
| `sdm/.tpu/teardown.sh` | **Replace** with `srm-tpu teardown` | Stops the parallel `NAMES=(...)` list from drifting from `provision_all.sh`. |
| `sdm/scripts/_tpu_common.sh` | **Replace** with `srm_tpu/bootstrap.py` + `scripts/bootstrap.sh` | Makes the on-VM bootstrap reusable across projects; extra deps go into project's `pyproject.toml`. The torch_xla wheel dance is replaced with CPU torch + jax[tpu] + torchax. |
| `sdm/scripts/run_finetune.sh` | **Replace** with `srm-tpu run --command CMD --retry-on-preempt` | Generic; project supplies its own entry-point command. |
| `sdm/scripts/tpu/sdm_tpu.py` | **Subsume** into `srm_tpu/cli.py` | The good ideas (Pool dataclass, runtime auto-derivation, daemonised workers, pidfiles) become the spine of the new CLI. |
| `sdm/scripts/tpu/bake_vm.sh` | **Subsume** into `srm-tpu bake` | Same accel→runtime detection; now uses torchax install chain. |
| `sdm/scripts/tpu/request_until_available.sh` | **Subsume** into `srm-tpu request` | Only client-side polling (`tpu-vm create` loop); queued-resources is not supported. |
| `sdm/scripts/request_tpus_wristband.sh` | **Replace** with `srm-tpu request --pool ... --parallel N` | The `declare -f | nohup bash -c` trick is fragile; `daemon.py` does the equivalent in a debuggable way. |

### Critique of the sdm implementation we are *not* keeping

- **Retry-regex copy-paste.** The same regex appears in `_lib.sh`, `sdm_tpu.py`,
  `request_tpus_wristband.sh`, and `request_until_available.sh`. Drift is
  guaranteed and has already started (the wristband script omits `already exists`).
  → centralised in `srm_tpu/retry.py`.
- **Three accel→runtime mappings.** `bake_vm.sh`, `request_until_available.sh`,
  `request_tpus_wristband.sh`, and `sdm_tpu.Pool.runtime` all encode the same
  table independently. → centralised in `srm_tpu/pools.py`.
- **Per-experiment provision shell scripts.** `provision_pitch.sh` etc. are
  literally `NAME=...; CONFIG=...; source _lib.sh; provision`. Nine files.
  Adding an experiment requires editing the script, `provision_all.sh`, and
  `teardown.sh`. → one row in YAML.
- **Hardcoded teacher install in `_tpu_common.sh`.** `phonemizer`, `pyworld`,
  `wespeaker-unofficial`, `funasr`, `allosaurus`, `masked_prosody_model`, plus
  `apt-get install espeak-ng` — all hardcoded, all installed on every run even
  when only one teacher is needed. → declared in the project's `pyproject.toml`
  extras and read by `bootstrap.py`.
- **torch_xla is superseded by torchax.** The old install chain (`torch ~2.7`,
  `torch_xla[tpu] ~2.7`, `libtpu` wheels) is replaced with CPU torch +
  `jax[tpu]` + `torchax`. This eliminates the fragile ABI-pin dance and the
  separate `--index-url` / `--find-links` wheel repositories. torchax lets
  users write normal PyTorch that runs on JAX/XLA with `device='jax'` and
  `jax.jit`, no `xm.mark_step()`/`xm.optimizer_step()`/`MpDeviceLoader`
  boilerplate. The library itself does not wrap torchax — projects call
  `torchax.enable_globally()` directly in their trainer entrypoint.
- **Inconsistent log/pid dirs.** `.tpu/logs/`, `.tpu-logs/`, `/tmp/sdm_tpu/`,
  `~/sdm-run.log` all coexist. → one configurable location, default
  `.srm-tpu/{logs,pids}/`.
- **`is_xla()` re-imports `torch_xla` and re-reads env on every call.**
  Cheap but pointless. → removed; torchax is called directly by user code.
- **Function shipping via `declare -f | nohup bash -c "..."`** in
  `request_tpus_wristband.sh`. Clever, but a quoting bug there is silent and
  leaves orphan processes. → `daemon.py` spawns `python -m srm_tpu.daemon worker`
  which is just argv.

---

## 2. The inventory file

The library is driven by a single YAML file per project (default location:
`./srm-tpu.yaml`, override with `--inventory PATH` or `SRM_TPU_INVENTORY`).
This is the **only** place that names projects, zones, pools, or wheel
versions. Experiment-level details (config file, launch command, env vars)
are passed on the command line at `srm-tpu launch` time, not stored here.

```yaml
# srm-tpu.yaml
project:
  gcp_project: ml-edinburgh           # required
  default_zone: europe-west4-a        # used when a pool doesn't pin a zone
  log_dir: .srm-tpu/logs              # local
  pid_dir: .srm-tpu/pids              # local
  remote_repo: https://github.com/me/srm.git
  remote_branch: main
  remote_workdir: ~/srm               # where the repo lives on the VM
  tmux_session: srm                   # name of the on-VM tmux session

# Which env vars get copied from the local .env into ~/.env on the VM.
# Anything not listed here is never sent.
secrets:
  - WANDB_API_KEY
  - HF_TOKEN

# TRC pool inventory. `runtime` is optional; if absent, derived from `accel`.
# `instances` is the maximum number of concurrent VMs the quota allows in that
# pool. The CLI uses it to cap `--parallel` and to report capacity.
pools:
  v6e-euw4:   { accel: v6e-8,       zone: europe-west4-a, spot: true,  instances: 64 }
  v6e-use1:   { accel: v6e-8,       zone: us-east1-d,     spot: true,  instances: 64 }
  v5e-euw4:   { accel: v5litepod-8, zone: europe-west4-b, spot: true,  instances: 64 }
  v5e-usc1:   { accel: v5litepod-8, zone: us-central1-a,  spot: true,  instances: 64 }
  v4-od:      { accel: v4-8,        zone: us-central2-b,  spot: false, instances: 32 }
  v4-spot:    { accel: v4-8,        zone: us-central2-b,  spot: true,  instances: 32 }

# Bootstrap recipe. Run on the VM by `srm-tpu bake` (and by `provision`
# right after the VM goes READY).
bootstrap:
  python: "3.11"                      # torchax wheels are built for 3.10+
  apt:                                # apt-get install -y
    - espeak-ng
    - libespeak-ng1
    - tmux
  # Install order: CPU torch → jax[tpu] → torchax. This replaces the old
  # torch_xla + libtpu wheel dance.
  torch:
    torch:        "2.7.1"              # CPU-only, pinned
    torchaudio:   "2.7.1"
    jax:          "tpu"                # -> jax[tpu], pulls libtpu transitively
    torchax:      "*"                  # latest compatible
  project_install: "uv sync --extra dev --extra tpu --extra tracking"
  extra_pip:                          # post-sync, idempotent
    - phonemizer
    - pyworld
    - "masked_prosody_model @ git+https://github.com/MiniXC/masked_prosody_model"
  smoke_test: "import torch, torchax; t = torch.randn(2, device='jax'); print(t.device, torchax.__version__)"
```

### Inventory invariants (validated on load)

- `pools.*.instances` is a positive integer.
- `pools.*.accel` resolves to a known `runtime` via `pools.runtime_for()`.
- `secrets` keys are present in the local `.env` only when about to be
  shipped (not at load time, so `srm-tpu pools` works without secrets).
- `bootstrap.python` is a string version (`"3.11"`, not `3.11`).

If validation fails the loader raises `InventoryError` with the offending
path-into-the-YAML included in the message. Tested in
`tests/test_inventory.py`.

---

## 3. The CLI

Single Typer app, exposed as `srm-tpu`. Subcommands:

| Command | Purpose |
|---|---|
| `srm-tpu pools` | Print the inventory's pools as a table (name, accel, zone, spot, instances). |
| `srm-tpu request [--pool NAME] [--parallel N] [--prefix NAME] [--detached]` | Bring up `N` VMs in the given pool (default: one in each pool, capped by `instances`). Polls until capacity. |
| `srm-tpu launch --pool NAME --config PATH --command CMD [--prefix NAME] [--parallel N] [--detached]` | For each VM: provision + push env + launch the command in tmux. |
| `srm-tpu status [--pool NAME]` | Show local poll loops + **all** TPMs in known zones (not just ours) in one table. Preempted/terminated VMs are highlighted. |
| `srm-tpu list [--pool NAME]` | List every TPU VM across all known zones for the GCP project, with state and age. |
| `srm-tpu ssh <name> [--command CMD] [--worker N]` | SSH into a VM (auto-detects zone). |
| `srm-tpu tail <name>` | `tail -F` the relevant local log file for a VM's daemon worker. |
| `srm-tpu logs <name> [--remote-path PATH]` | `gcloud ... ssh ... tail -f` the in-tmux log on the VM. |
| `srm-tpu stop [name]` | Kill local poll loop(s) for the given VM (no VM impact). |
| `srm-tpu delete <name> [--all]` | Delete a specific VM by name, or all VMs matching the prefix pattern. |
| `srm-tpu delete --filter 'state=PREEMPTED'` | Delete all VMs in a given state across known zones. |
| `srm-tpu delete --filter 'name~srm-.*-wristband'` | Delete all VMs whose names match a regex. |
| `srm-tpu teardown [--pool NAME]` | `delete --all` + `stop --all` in all zones. |
| `srm-tpu bake [--vm-name NAME]` | Run the bootstrap recipe on the VM (or locally on the VM). |
| `srm-tpu run --command CMD [--retry-on-preempt] [--config PATH]` | On-VM entry point used by `launch`'s tmux command. Sets `PJRT_DEVICE=TPU`, calls `torchax.enable_globally()`, runs `CMD`. |

Global flags (every subcommand):

- `--inventory PATH` — override the default `./srm-tpu.yaml`.
- `--dry-run` — print the gcloud commands that would run; don't execute.
- `--json-logs` — switch the human logger to one JSON record per line on stdout.
- `-v / --verbose` — DEBUG-level logging.

### Examples

```bash
# Spin up 8 VMs in the v6e-euw4 pool with a custom prefix (creates srm-ft-01 .. srm-ft-08).
srm-tpu request --pool v6e-euw4 --parallel 8 --prefix srm-ft

# Launch a training run on one VM from that batch.
srm-tpu launch --pool v6e-euw4 --prefix srm-ft-01 \
    --config configs/finetune_pitch.yaml \
    --command 'python -m srm.train.run_distill --config configs/finetune_pitch.yaml'

# See everything in the project, not just our VMs.
srm-tpu list
srm-tpu status

# Clean up preempted VMs that are sitting dead.
srm-tpu delete --filter 'state=PREEMPTED'

# Delete a whole family of VMs by name pattern.
srm-tpu delete --filter 'name~srm-ft-'

# Just allocate a single test VM, don't launch anything.
srm-tpu request --pool v4-od --prefix srm-test

# SSH into it.
srm-tpu ssh srm-test-v4-od

# Watch its daemon log.
srm-tpu tail srm-test-v4-od

# Watch its on-VM training log.
srm-tpu logs srm-test-v4-od

# Done with everything in every zone.
srm-tpu teardown
```

---

## 4. Module-by-module spec

### 4.1 `srm_tpu/inventory.py`

```python
@dataclass(frozen=True)
class ProjectConfig:
    gcp_project: str
    default_zone: str
    log_dir: Path
    pid_dir: Path
    remote_repo: str
    remote_branch: str
    remote_workdir: str
    tmux_session: str

@dataclass(frozen=True)
class Pool:
    name: str
    accel: str
    zone: str
    spot: bool
    instances: int            # max concurrent VMs allowed by TRC quota
    runtime: str              # auto-filled from accel if not in YAML

@dataclass(frozen=True)
class BootstrapRecipe:
    python: str
    apt: tuple[str, ...]
    torch: TorchPin           # nested dataclass: {torch, torchaudio, jax, torchax} pins
    project_install: str
    extra_pip: tuple[str, ...]
    smoke_test: str

@dataclass(frozen=True)
class Inventory:
    project: ProjectConfig
    pools: Mapping[str, Pool]
    secrets: tuple[str, ...]
    bootstrap: BootstrapRecipe

    @classmethod
    def load(cls, path: Path | None = None) -> "Inventory": ...
```

`load()`:
1. Parse YAML.
2. For every pool with no `runtime`, fill from `pools.runtime_for(accel)`.
3. Validate that every `instances` is ≥ 1 and that accels are recognised.
4. Raise `InventoryError(path_in_yaml, message)` on any mismatch.

### 4.2 `srm_tpu/pools.py`

Pure-function module. No I/O.

```python
RUNTIME_FOR_FAMILY: dict[str, str] = {
    "v4":         "tpu-ubuntu2204-base",
    "v5e":        "v2-alpha-tpuv5-lite",
    "v5litepod":  "v2-alpha-tpuv5-lite",
    "v6e":        "v2-alpha-tpuv6e",
}

def family_of(accel: str) -> str: ...
def runtime_for(accel: str) -> str: ...
```

This is the **one and only** mapping. `bake.py`, `inventory.py`, and the CLI
all import from here. Tested with a parametrised matrix
(`tests/test_pools.py::test_runtime_for[v4-8-tpu-ubuntu2204-base]` etc).

### 4.3 `srm_tpu/retry.py`

```python
RETRYABLE_PATTERNS = (
    "no more capacity", "Insufficient capacity", "RESOURCE_EXHAUSTED",
    "UNAVAILABLE", "resourceExhausted", "Stockout", "currently unavailable",
    "tenant project creation", '"code": 8', '"code": 10',
    "HttpError", "503", "504", "deadline exceeded", "Internal error",
    "already exists",        # treat as success-after-the-fact, see below
)
RETRYABLE_RE = re.compile("|".join(map(re.escape, RETRYABLE_PATTERNS)),
                          re.IGNORECASE)

@dataclass
class RetryPolicy:
    max_attempts: int = 0           # 0 = infinite
    base_sleep_s: float = 30.0
    max_sleep_s: float = 300.0
    backoff: float = 1.0            # 1.0 = constant, >1 = exponential
    attempt_timeout_s: float | None = None    # hard wall per attempt; None = off

@dataclass
class AttemptResult:
    ok: bool
    attempt: int
    returncode: int | None
    elapsed_s: float
    stdout: str
    stderr: str
    classification: Literal["success", "retryable", "unknown", "fatal", "timeout"]

def classify(returncode: int | None, stderr: str) -> str: ...
def run_with_retry(
    factory: Callable[[], list[str]],     # returns the argv each attempt
    policy: RetryPolicy,
    on_attempt: Callable[[AttemptResult], None] | None = None,
) -> AttemptResult: ...
```

`classify()` rules:
- `returncode == 0`                    → `"success"`
- `returncode is None` (timeout)        → `"timeout"`  (retried)
- `RETRYABLE_RE.search(stderr)`         → `"retryable"` (retried)
- otherwise                             → `"unknown"`   (retried with WARN)

There is no `"fatal"` returned by gcloud creates that we currently know how
to discriminate, so we deliberately default-to-retry and rely on the user to
Ctrl-C if the loop is genuinely stuck. (This matches `_lib.sh` behaviour.)
`"fatal"` is reserved as an extension point for callers (e.g. `delete`,
where 404 should not be retried).

### 4.4 `srm_tpu/gcloud.py`

Typed wrappers. Every public function takes `dry_run: bool = False`. When
`dry_run`, it calls `log.command(argv)` and returns a stub. Every real call
goes through `subprocess.run(..., capture_output=True, text=True,
errors="replace")`.

```python
@dataclass(frozen=True)
class VmStatus:
    name: str
    zone: str
    accel: str
    state: str
    create_time: str | None

def create_vm(pool: Pool, name: str, *, project: str,
              dry_run: bool = False) -> AttemptResult: ...

def describe_vm(name: str, zone: str, *, project: str) -> VmStatus | None: ...

def list_vms(zone: str, *, project: str) -> list[VmStatus]: ...

def list_vms_all_zones(zones: Iterable[str], *, project: str) -> list[VmStatus]:
    """List TPU VMs across all provided zones. Sorts by zone then name.""" ...

def delete_vm(name: str, zone: str, *, project: str,
              dry_run: bool = False) -> AttemptResult: ...

def delete_by_filter(zones: Iterable[str], *, project: str,
                     name_regex: str | None = None,
                     state: str | None = None,
                     dry_run: bool = False) -> list[VmStatus]:
    """Delete every VM in `zones` whose name matches `name_regex` (Python regex)
    and/or whose state equals `state` (case-insensitive comparison against the
    gcloud-reported state string). Returns the list of VMs that were deleted.""" ...

def ssh(name: str, zone: str, *, project: str, command: str | None = None,
        worker: str = "all", dry_run: bool = False) -> int: ...

def scp(local: Path, remote: str, name: str, zone: str, *, project: str,
        worker: str = "all", dry_run: bool = False) -> int: ...
```

These are the only places `subprocess.run([GCLOUD, ...])` is called. The CLI
and provisioner consume them.

Note: there is no `queued_create`. Google's `queued-resources` API for TPU is
documented in TRC welcome emails but not yet functional in practice. We use
client-side polling (`tpu-vm create` with a retry loop) exclusively.

### 4.5 `srm_tpu/secrets.py`

```python
def load_dotenv(path: Path = Path(".env")) -> dict[str, str]: ...
def select(env: Mapping[str, str], whitelist: Iterable[str]) -> dict[str, str]:
    """Return only the keys in `whitelist`. Raise if a required key is missing."""
def write_env_file(env: Mapping[str, str]) -> Path:
    """Write to a NamedTemporaryFile with mode 0600; return the path."""
```

`provision.push_env(vm, inventory)` does:
`scp(write_env_file(select(load_dotenv(), inventory.secrets)), "~/.env.srm", ...)`.

The whitelist is enforced both for safety (no accidental shipping of
`AWS_SECRET_ACCESS_KEY`) and for transparency (logged: "shipping
WANDB_API_KEY, HF_TOKEN to vm-name").

### 4.6 `srm_tpu/provision.py`

```python
@dataclass(frozen=True)
class LaunchSpec:
    """Everything that varies between invocations of `srm-tpu launch`."""
    vm_name: str               # TPU VM name (e.g. "srm-ft-01")
    pool: str                  # pool name (must resolve in inventory)
    config: str                # path to experiment config, relative to repo root
    command: str               # the command that runs in tmux; may use {config}
    retry_on_exit_codes: tuple[int, ...] = (130,)
    max_retries: int = 50
    env: Mapping[str, str] | None = None

@dataclass(frozen=True)
class ProvisionResult:
    vm_name: str
    pool: str
    state: Literal["LAUNCHED", "FAILED", "DRY_RUN"]
    attempts: int
    elapsed_s: float
    log_path: Path

def poll_create(pool: Pool, name: str, *, project: str,
                policy: RetryPolicy, log: Logger,
                dry_run: bool = False) -> AttemptResult: ...

def wait_ready(name: str, zone: str, *, project: str,
               poll_interval_s: float = 15.0,
               log: Logger) -> Literal["READY", "PREEMPTED", "TERMINATED", "FAILED"]: ...

def push_env(name: str, zone: str, env: Mapping[str, str], *,
             project: str, remote_path: str = "~/.env.srm",
             log: Logger, dry_run: bool = False) -> None: ...

def launch_run(name: str, zone: str, *, project: str,
               inventory: Inventory, spec: LaunchSpec,
               log: Logger, dry_run: bool = False) -> None:
    """SSH in, clone/pull repo, copy ~/.env.srm into place, start tmux session
    iff not already running. Idempotent."""

def provision(spec: LaunchSpec, *, inventory: Inventory, secrets: Mapping[str, str],
              policy: RetryPolicy, log: Logger,
              dry_run: bool = False) -> ProvisionResult:
    """End-to-end: poll_create → wait_ready → push_env → launch_run."""
```

`launch_run` builds the on-VM command from a single template:

```bash
set -euo pipefail
if [[ ! -d {workdir}/.git ]]; then
    git clone --branch {branch} {repo} {workdir}
fi
cd {workdir}
git fetch --quiet origin
git checkout {branch}
git pull --ff-only
install -m 600 ~/.env.srm {workdir}/.env
sudo apt-get install -y tmux >/dev/null 2>&1 || true
if tmux has-session -t {tmux} 2>/dev/null; then
    echo "tmux session {tmux} already running; not relaunching"
else
    tmux new-session -d -s {tmux} "{command} 2>&1 | tee -a {log}"
fi
tmux ls
```

The template lives in `provision.py` as a `LAUNCH_SCRIPT` constant; tested
verbatim in `tests/test_provision_flow.py::test_launch_script_renders`.

### 4.7 `srm_tpu/daemon.py`

Replaces the `nohup bash -c "$(declare -f ...)" &` trick. A "worker" is
just `python -m srm_tpu.daemon worker <vm-name> --inventory PATH`. The
parent CLI command:

1. Records pidfile: `inventory.project.pid_dir / f"{vm_name}.pid"`.
2. `start_new_session=True`, stdout/stderr → `inventory.project.log_dir / f"{vm_name}.log"`.
3. Prints a one-line summary per spawned worker.

`stop` just reads the pidfile and `os.killpg(os.getpgid(pid), SIGTERM)`.
`status` reads pidfile + `tail -n 200` of the log to extract the latest
attempt count and last status line, and also polls `list_vms_all_zones()`
to show **all** TPMs in the project, not just ours. Preempted/terminated
VMs get an extra column callout.

The same daemonisation is used for `request --detached` (no launch, just
allocate) and `launch --detached` (allocate + push + launch).

### 4.8 `srm_tpu/bootstrap.py`

Runs on the VM. Importable from `python -m srm_tpu.bootstrap`. No external
deps beyond stdlib + `pyyaml`. (Bash bootstrap installs `srm-tpu` first;
this module does the rest in Python so we get tracebacks instead of `set -e`
exits.)

Steps, each emitting a `[bootstrap] step=...` log line:

1. **Detect accel** from the metadata server
   (`http://metadata.google.internal/.../accelerator-type`).
   Fall back to env var `SRM_TPU_ACCEL` for offline tests.
2. **Pick the runtime version** from `pools.runtime_for(accel)`.
3. **apt-get install** the listed packages (no-op if already present).
4. **Force python3.11 venv** at `./.venv`. If a venv exists with the wrong
   python, log + recreate.
5. **Run `project_install`** (e.g. `uv sync ...`).
6. **Install torchax stack** in order: CPU torch at the pinned version,
   then `jax[tpu]`, then `torchax`. This replaces the old torch_xla +
   libtpu wheel dance with a simpler three-step pip chain.
7. **Extra pip deps** from `bootstrap.extra_pip`.
8. **Smoke test**: run the inventory's `smoke_test` string in a fresh
   `python -c`. Non-zero exit fails the bootstrap loudly.

`scripts/bootstrap.sh` is the only bash:

```bash
#!/usr/bin/env bash
set -euo pipefail
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
PY="${SRM_TPU_PYTHON:-python3.11}"
if ! command -v "$PY" >/dev/null; then
    sudo apt-get update -y && sudo apt-get install -y "$PY" "${PY}-venv"
fi
[[ -d .venv ]] || uv venv --python "$PY" .venv
uv sync --extra dev --extra tpu
uv run python -m srm_tpu.bootstrap "$@"
```

That's the entire bash surface area of the project.

### 4.9 `srm_tpu/preempt.py`

Direct port of `sdm/train/preempt.py`. ~30 lines. Installs SIGTERM / SIGINT
handlers that flip a `StopState.requested` flag so the training loop can flush a
checkpoint before the VM is reclaimed.

### 4.10 `srm_tpu/io.py`

Port of `sdm/train/io.py`, refactored to use a `Backend` ABC so projects can
plug in `gcsfs` later without changing call sites.

```python
class Backend(Protocol):
    def is_remote(self, path: str) -> bool: ...
    def save(self, state: Any, path: str) -> None: ...
    def load(self, path: str, map_location: str = "cpu") -> Any: ...
    def exists(self, path: str) -> bool: ...
    def list_glob(self, pattern: str) -> list[str]: ...
    def copy(self, src: str, dst: str) -> None: ...

class GsutilBackend(Backend): ...     # default
class LocalOnlyBackend(Backend): ...   # for tests / no-GCS environments

_DEFAULT: Backend = GsutilBackend()
def set_backend(b: Backend) -> None: ...
def save_state(state, path): _DEFAULT.save(state, path)
# ... etc

def latest_checkpoint(ckpt_dir: str) -> str | None:
    """latest.pt if present, else max step-NNNNN.pt in the directory."""
```

### 4.11 `srm_tpu/log.py`

One logger, two formats. Centralised so every retry attempt looks the same
and `--json-logs` works everywhere.

```python
class Logger:
    def __init__(self, *, json: bool, level: int, out: TextIO = sys.stderr): ...

    # Structured calls — emit one log record each.
    def event(self, kind: str, **fields) -> None: ...        # {"ts","kind",...fields}
    def command(self, argv: list[str]) -> None: ...           # kind="command"
    def attempt(self, result: AttemptResult, **fields) -> None: ...
    def error(self, msg: str, exc: BaseException | None = None) -> None: ...

    # Plain calls — used by Typer help and progress tables.
    def info(self, msg: str) -> None: ...
    def debug(self, msg: str) -> None: ...
```

Human format: `[2026-05-09T14:32:11+00:00] [pitch] attempt=3 cls=retryable rc=1 dt=42.1s last="...stockout..."`.

JSON format: one record per line, `{"ts": "...", "kind": "attempt", "experiment": "pitch", ...}`.

---

## 5. Cross-cutting: directory layout on disk

### Local (laptop / CI)

```
.srm-tpu/
├── logs/
│   ├── srm-ft-01.log           # daemon worker stdout/stderr, one per detached job
│   ├── srm-ft-01.attempts.jsonl # JSON-lines audit of every gcloud attempt
│   └── ...
└── pids/
    └── srm-ft-01.pid
```

(All under `inventory.project.{log_dir,pid_dir}`, configurable.)

### On the VM

```
~/.env.srm                     # whitelisted secrets (mode 0600)
{remote_workdir}/              # the project repo
{remote_workdir}/.env          # symlink-installed copy of ~/.env.srm
{remote_workdir}/.venv/        # python3.11 venv created by bootstrap
~/srm-run.log                  # tmux pane mirror; what `srm-tpu logs` tails
```

---

## 6. Failure modes + transparency contract

For each failure class, the library promises a specific user-visible
behaviour. This is the test plan for `tests/test_provision_flow.py`.

| Failure | Visible behaviour |
|---|---|
| `gcloud` not on PATH | `srm-tpu request` prints `gcloud not found; install Google Cloud SDK` and exits 2 immediately. |
| Inventory missing required field | `InventoryError: project.gcp_project is required (srm-tpu.yaml line N)` |
| Inventory `instances` is zero or negative | `InventoryError: pools.v6e-euw4.instances must be >= 1, got 0` |
| `.env` missing a whitelisted secret | `SecretError: WANDB_API_KEY listed in srm-tpu.yaml secrets but not in .env` |
| Pool out of capacity | One log line per attempt with `cls=retryable`, then exit code 0 once it succeeds; `--json-logs` users get a record per attempt to count. |
| Pool out of capacity after `max_attempts` | `RetryExhausted: pool=v6e-euw4 attempts=50 last_error="Stockout"`; non-zero exit. Daemon worker writes the same to its log and removes its pidfile. |
| Pool returns "already exists" | Treated as success: we describe the VM, log `state=...`, proceed to `wait_ready`. |
| VM goes `PREEMPTED` mid-launch | `wait_ready` returns `PREEMPTED`; `provision` deletes the VM, sleeps 10s, re-enters `poll_create`. Logged as `event=preempted recreating`. |
| `torchax` not installed but TPU env vars are present | The bootstrap smoke test fails loudly with the import error; no training is attempted. Users get a clear "install torchax/jax[tpu]" message. |
| `gsutil` not installed but `gs://` path used | `IOBackendError: gsutil not on PATH; install gcloud SDK or set backend with srm_tpu.io.set_backend(...)`. |
| SIGTERM (spot preemption) | `preempt.StopState.requested = True`; trainer is expected to checkpoint and exit 130; `srm-tpu run --retry-on-preempt` re-enters its retry loop. |
| Bootstrap smoke test fails | Bootstrap exits non-zero with the captured stderr; no training is attempted. |
| `--dry-run` | Every gcloud call is printed, none executed. Exit code 0 iff all calls would have been valid argv. |
| `delete --filter 'state=PREEMPTED'` | Lists, then deletes every VM in the `PREEMPTED` state across all known zones. `--dry-run` prints the names that would be deleted without touching them. |
| `delete --filter 'name~srm-.*'` | Lists, then deletes every VM whose name matches the regex. Both filter keys (`state=` and `name~`) can be combined with `,`. |

---

## 7. Public Python API (for project code that wants to integrate directly)

```python
from srm_tpu import (
    Inventory, Pool, LaunchSpec,
    provision,
    preempt, io,
)

# Single-VM ad-hoc:
inv = Inventory.load()
spec = LaunchSpec(
    vm_name="srm-ft-01", pool="v6e-euw4",
    config="configs/finetune_pitch.yaml",
    command="python -m srm.train.run_distill --config configs/finetune_pitch.yaml",
)
result = provision(spec, inventory=inv, secrets=secrets.load_dotenv(),
                   policy=RetryPolicy(max_attempts=50), log=Logger(json=False))

# Inside the trainer (torchax used directly, not through srm_tpu):
import torchax
torchax.enable_globally()
device = torch.device('jax')
model = MyModel().to(device)
stop = preempt.install()
for batch in loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    ...
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if stop.requested:
        io.save_state(state, "gs://srm-ckpts/pitch/latest.pt")
        sys.exit(130)
```

`srm_tpu` handles the provisioning/infra side. The trainer is a normal
PyTorch script with two extra lines at the top (`import torchax;
torchax.enable_globally()`) and `device='jax'` instead of `'cuda'`. No
wrapper, no abstraction — torchax is used directly.

---

## 8. Testing strategy

- `tests/test_pools.py` — pure-function table tests for `runtime_for`.
- `tests/test_inventory.py` — fixtures with malformed YAMLs (missing
  `instances`, unknown accels, etc.), asserts on exact `InventoryError` messages.
- `tests/test_retry.py` — feed `classify()` known stderr samples (one per
  retryable pattern); test `run_with_retry` with a fake factory that
  succeeds on attempt N.
- `tests/test_gcloud_dry_run.py` — every public function in `gcloud.py`
  with `dry_run=True` produces the expected argv. Includes `delete_by_filter`
  with name regex and state filters. Snapshot-tested.
- `tests/test_provision_flow.py` — `gcloud` is monkeypatched to a recorder;
  `provision()` for a happy path and for each failure mode in §6 produces
  the expected sequence of recorded calls and the expected log records.
- `tests/test_io_gcs.py` — install a fake `gsutil` shim on `PATH` that
  echoes args to a log; assert `save_state(..., "gs://x/y")` calls
  `gsutil cp` with the right args.
- `tests/test_daemon.py` — spawn a worker that sleeps 5s, assert pidfile
  appears, `stop` removes it, log file is non-empty.

CI: `pytest -q` on Linux + Mac; no actual GCP access required (every test
mocks `gcloud`/`gsutil`).

---

## 9. Migration path from sdm

A project moving off the sdm scripts onto `srm-tpu` does, in order:

1. The repo is already cloned on the VM by `launch_run` — no separate install.
2. Install CPU torch + `jax[tpu]` + `torchax` in the project environment
   (replaces the old `torch_xla[tpu]` + libtpu wheel dance).
3. Translate the pool list from `TPU_RESOURCES.md` into `srm-tpu.yaml`
   (mechanical; add `instances` from the TRC quota email).
4. Replace `from sdm.train import xla_utils, preempt, io` with direct
   torchax usage (`import torchax; torchax.enable_globally()` at the top of
   the trainer) plus `from srm_tpu import preempt, io`. Remove all
   `xm.mark_step()`, `xm.optimizer_step()`, and `xmp.spawn()` calls — they
   are not needed with torchax.
5. Replace `./.tpu/provision_X.sh` invocations with
   `srm-tpu launch --pool N --config PATH --command CMD`.
6. Delete `.tpu/`, `scripts/_tpu_common.sh`, `scripts/run_finetune.sh`,
   `scripts/request_tpus*.sh`, `scripts/tpu/`. (~1300 lines of bash and one
   Python file gone.)

Net change for an sdm-sized project: ~1300 lines of bash → ~50 lines of YAML.

---

## 10. Open questions for the implementer

- **torchax compatibility surface.** torchax is still maturing — the ops
  registry (`jaten.py`) may not cover every aten op a project uses. The
  smoke test should catch missing ops early. Document a
  `TORCHAX_OPS_ALLOWLIST` escape hatch for projects that need to run
  partial ops on CPU during bootstrap validation.
- **Should `bootstrap.py` be runnable on the laptop too** (via `srm-tpu
  bake --remote NAME` shipping itself over and exec'ing)? Probably yes; it
  would let us drop `scripts/bootstrap.sh` entirely on hosts that already
  have `python3.11` and `uv`. Out of scope for v1.
- **JAX process model on TPU VMs.** torchax currently runs as a
  single-process PyTorch program on the JAX device. Multi-device programs
  (e.g. `pmap` across 8 TPU cores) use JAX's internal SPMD — the CLI
  doesn't need to spawn workers. Verify this holds for v5e and v6e before
  removing the `--worker=all` from SSH calls.

---

## Appendix A — sdm artefact reference (reading list)

For the implementer, these are the sdm files to read before starting:

- `sdm/.tpu/_lib.sh` — current bash provisioner; the canonical retry/launch flow.
- `sdm/scripts/tpu/sdm_tpu.py` — current Python CLI; `Pool` + daemon ideas.
- `sdm/scripts/_tpu_common.sh` — current bootstrap; what `bootstrap.py` replaces.
- `sdm/scripts/request_tpus_wristband.sh` — the `--detached` fanout pattern,
  including the `declare -f | nohup bash -c` trick we're replacing.
- `sdm/sdm/train/xla_utils.py` — read for context on the old torch_xla shim
  that is being dropped entirely; torchax is used directly by project code.
- `sdm/sdm/train/{io,preempt}.py` — port verbatim into `srm_tpu/{io,preempt}.py`.
- `sdm/TPU_RESOURCES.md` — TRC inventory table; the seed for `srm-tpu.yaml`.
- `sdm/.tpu/README.md` — user-facing description of the current flow; the
  `srm-tpu` user docs should cover the same scenarios.
