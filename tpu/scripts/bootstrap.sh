#!/usr/bin/env bash
# Bootstrap a fresh TPU VM: install uv, python3.11, create venv, sync deps,
# then hand off to the Python bootstrap.
set -euo pipefail

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

PY="${SRM_TPU_PYTHON:-python3.11}"
if ! command -v "$PY" >/dev/null; then
    sudo apt-get update -y && sudo apt-get install -y "$PY" "${PY}-venv"
fi

cd "$(dirname "$0")/.."
[[ -d .venv ]] || uv venv --python "$PY" .venv
uv sync --extra dev --extra tpu
uv run python -m srm_tpu.bootstrap "$@"
