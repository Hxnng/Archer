#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

export ARCHER_LLADA_MODEL_PATH="${ARCHER_LLADA_MODEL_PATH:-${ROOT}/models/LLaDA-8B-Instruct}"

uv sync --locked
"${ROOT}/.venv/bin/python" scripts/check_benchmarks.py

if [[ ! -f "${ARCHER_LLADA_MODEL_PATH}/config.json" ]]; then
  "${ROOT}/scripts/download_models.sh" llada
fi
if [[ ! -f "${ARCHER_LLADA_MODEL_PATH}/config.json" ]]; then
  echo "Model checkout is incomplete: ${ARCHER_LLADA_MODEL_PATH}" >&2
  exit 1
fi

"${ROOT}/scripts/setup_external_repos.sh"
"${ROOT}/.venv/bin/python" scripts/prepare_livecodebench.py
"${ROOT}/.venv/bin/python" scripts/check_benchmarks.py --require-livecodebench

echo "Environment ready."
