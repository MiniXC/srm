# srm

Speech Representation Models.

## Structure

- `tpu/` — `srm-tpu` helper library for TPU VM provisioning and training orchestration.
  See [`tpu/ARCHITECTURE.md`](tpu/ARCHITECTURE.md) for the full spec.

## Quick start

```bash
cd tpu
uv sync                     # install CLI deps
uv run srm-tpu --help       # show commands
uv run srm-tpu pools        # list configured TPU pools
uv run srm-tpu status       # see all TPU VMs across zones
```
