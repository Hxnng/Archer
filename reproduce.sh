#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ARCHER_LLADA_MODEL_PATH="${ARCHER_LLADA_MODEL_PATH:-${ROOT}/models/LLaDA-8B-Instruct}"

if [[ ! -x "${ROOT}/.venv/bin/python" ||
      ! -f "${ARCHER_LLADA_MODEL_PATH}/config.json" ||
      ! -f "${ROOT}/benchmarks/livecodebench/test.jsonl" ||
      ! -d "${ROOT}/baseline/LiveCodeBench" ||
      ! -d "${ROOT}/baseline/dKV-Cache" ||
      ! -d "${ROOT}/baseline/dLLM-cache" ]]; then
  echo "Environment is not ready. Run ./setup.sh first." >&2
  exit 1
fi

"${ROOT}/.venv/bin/python" scripts/check_benchmarks.py --require-livecodebench
"${ROOT}/.venv/bin/python" scripts/run_experiment.py \
  configs/table1_main.json \
  --resume

echo "Main experiment complete. Results are under outputs/paper/table1_main/."
