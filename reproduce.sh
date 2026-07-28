#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-all}"

if [[ "$#" -gt 1 ]]; then
  echo "Usage: ./reproduce.sh [all|table1|table2|table3|table4]" >&2
  exit 2
fi

case "${TARGET}" in
  all|table1|table2|table3|table4) ;;
  *)
    echo "Usage: ./reproduce.sh [all|table1|table2|table3|table4]" >&2
    exit 2
    ;;
esac

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ARCHER_LLADA_MODEL_PATH="${ARCHER_LLADA_MODEL_PATH:-${ROOT}/models/LLaDA-8B-Instruct}"
export ARCHER_DREAM_MODEL_PATH="${ARCHER_DREAM_MODEL_PATH:-${ROOT}/models/Dream-v0-Instruct-7B}"
export ARCHER_DIFFUCODER_MODEL_PATH="${ARCHER_DIFFUCODER_MODEL_PATH:-${ROOT}/models/DiffuCoder-7B-cpGRPO}"

if [[ ! -x "${ROOT}/.venv/bin/python" ||
      ! -f "${ARCHER_LLADA_MODEL_PATH}/config.json" ||
      ! -f "${ARCHER_DREAM_MODEL_PATH}/config.json" ||
      ! -f "${ARCHER_DIFFUCODER_MODEL_PATH}/config.json" ||
      ! -f "${ROOT}/benchmarks/livecodebench/test.jsonl" ||
      ! -d "${ROOT}/baseline/LiveCodeBench" ||
      ! -d "${ROOT}/baseline/dKV-Cache" ||
      ! -d "${ROOT}/baseline/dLLM-cache" ]]; then
  echo "Environment is not ready. Run ./setup.sh first." >&2
  exit 1
fi

"${ROOT}/.venv/bin/python" scripts/check_benchmarks.py --require-livecodebench

run_config() {
  "${ROOT}/.venv/bin/python" scripts/run_experiment.py "$1" --resume
}

if [[ "${TARGET}" == "all" || "${TARGET}" == "table1" ]]; then
  run_config configs/table1_main.json
fi
if [[ "${TARGET}" == "all" || "${TARGET}" == "table2" ]]; then
  run_config configs/table2_backbones.json
fi
if [[ "${TARGET}" == "all" || "${TARGET}" == "table3" ]]; then
  run_config configs/table3_feedback_k11.json
fi
if [[ "${TARGET}" == "all" || "${TARGET}" == "table4" ]]; then
  run_config configs/table4_cache_validity_k11.json
fi

echo "Reproduction complete. Results are under outputs/paper/."
