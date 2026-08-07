#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="all"
TARGET_SET=0
REFRESH_RADIUS=""
LIMIT=0
DRY_RUN=0
RESUME=1

usage() {
  cat <<'EOF'
Usage: ./reproduce.sh [all|mbpp|livecodebench|humaneval] [options]

Reproduce the Archer rows in Table 1. Without arguments, all three datasets
use the paper defaults: MBPP K=11, LiveCodeBench K=10, HumanEval K=8.

Options:
  --k K          Override the refresh radius K for the selected dataset(s).
  --limit N      Run only the first N examples (useful for a smoke test).
  --dry-run      Print the exact command without loading the model.
  --no-resume    Start a fresh run instead of resuming matching output.
  -h, --help     Show this message.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    all|mbpp|livecodebench|humaneval)
      if [[ "${TARGET_SET}" -eq 1 ]]; then
        echo "Choose only one dataset target." >&2
        usage >&2
        exit 2
      fi
      TARGET="$1"
      TARGET_SET=1
      shift
      ;;
    --k)
      [[ "$#" -ge 2 ]] || { echo "--k requires an integer." >&2; exit 2; }
      REFRESH_RADIUS="$2"
      shift 2
      ;;
    --limit)
      [[ "$#" -ge 2 ]] || { echo "--limit requires an integer." >&2; exit 2; }
      LIMIT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-resume)
      RESUME=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ "${REFRESH_RADIUS}" =~ ^[0-9]+$ || -z "${REFRESH_RADIUS}" ]] || {
  echo "--k must be a non-negative integer." >&2
  exit 2
}
[[ "${LIMIT}" =~ ^[0-9]+$ ]] || {
  echo "--limit must be a non-negative integer." >&2
  exit 2
}

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ARCHER_LLADA_MODEL_PATH="${ARCHER_LLADA_MODEL_PATH:-${ROOT}/models/LLaDA-8B-Instruct}"

if [[ "${DRY_RUN}" -eq 0 ]]; then
  if [[ ! -x "${ROOT}/.venv/bin/python" ||
        ! -f "${ARCHER_LLADA_MODEL_PATH}/config.json" ]]; then
    echo "Environment is not ready. Run ./setup.sh first." >&2
    exit 1
  fi
  if [[ "${TARGET}" == "all" || "${TARGET}" == "livecodebench" ]]; then
    if [[ ! -f "${ROOT}/benchmarks/livecodebench/test.jsonl" ||
          ! -d "${ROOT}/baseline/LiveCodeBench" ]]; then
      echo "LiveCodeBench is not ready. Run ./setup.sh first." >&2
      exit 1
    fi
    "${ROOT}/.venv/bin/python" scripts/check_benchmarks.py --require-livecodebench
  else
    "${ROOT}/.venv/bin/python" scripts/check_benchmarks.py
  fi
fi

PYTHON="${ROOT}/.venv/bin/python"
if [[ "${DRY_RUN}" -eq 1 && ! -x "${PYTHON}" ]]; then
  PYTHON="${PYTHON_BIN:-python3}"
fi

COMMAND=(
  "${PYTHON}" scripts/run_experiment.py configs/table1_archer.json
  --output-root "${ROOT}/outputs/table1"
)

case "${TARGET}" in
  mbpp) COMMAND+=(--run table1_archer_mbpp) ;;
  livecodebench) COMMAND+=(--run table1_archer_livecodebench) ;;
  humaneval) COMMAND+=(--run table1_archer_humaneval) ;;
esac

if [[ -n "${REFRESH_RADIUS}" ]]; then
  COMMAND+=(--refresh-radius "${REFRESH_RADIUS}")
fi
if [[ "${LIMIT}" -gt 0 ]]; then
  COMMAND+=(--limit "${LIMIT}")
fi
if [[ "${RESUME}" -eq 1 ]]; then
  COMMAND+=(--resume)
fi
if [[ "${DRY_RUN}" -eq 1 ]]; then
  COMMAND+=(--dry-run)
fi

"${COMMAND[@]}"

if [[ "${DRY_RUN}" -eq 0 ]]; then
  echo "Archer Table 1 reproduction complete. Results are under outputs/table1/."
fi
