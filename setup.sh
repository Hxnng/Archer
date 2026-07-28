#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

export ARCHER_LLADA_MODEL_PATH="${ARCHER_LLADA_MODEL_PATH:-${ROOT}/models/LLaDA-8B-Instruct}"
export ARCHER_DREAM_MODEL_PATH="${ARCHER_DREAM_MODEL_PATH:-${ROOT}/models/Dream-v0-Instruct-7B}"
export ARCHER_DIFFUCODER_MODEL_PATH="${ARCHER_DIFFUCODER_MODEL_PATH:-${ROOT}/models/DiffuCoder-7B-cpGRPO}"

uv sync --locked
"${ROOT}/.venv/bin/python" scripts/check_benchmarks.py

ensure_model() {
  local model="$1"
  local path="$2"
  if [[ ! -f "${path}/config.json" ]]; then
    "${ROOT}/scripts/download_models.sh" "${model}"
  fi
  if [[ ! -f "${path}/config.json" ]]; then
    echo "Model checkout is incomplete: ${path}" >&2
    exit 1
  fi
}

ensure_model llada "${ARCHER_LLADA_MODEL_PATH}"
ensure_model dream "${ARCHER_DREAM_MODEL_PATH}"
ensure_model diffucoder "${ARCHER_DIFFUCODER_MODEL_PATH}"

"${ROOT}/scripts/setup_external_repos.sh"
"${ROOT}/.venv/bin/python" scripts/prepare_livecodebench.py
"${ROOT}/.venv/bin/python" scripts/check_benchmarks.py --require-livecodebench

echo "Environment ready."
