#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${ROOT}/models"

download_model() {
  local name="$1"
  local repository="$2"
  local revision="$3"
  local destination="${MODEL_ROOT}/${name}"

  mkdir -p "${destination}"
  uv run huggingface-cli download \
    "${repository}" \
    --revision "${revision}" \
    --local-dir "${destination}"
}

if [[ "$#" -gt 1 || ( "$#" -eq 1 && "$1" != "llada" ) ]]; then
  echo "Usage: scripts/download_models.sh [llada]" >&2
  exit 2
fi

download_model \
  "LLaDA-8B-Instruct" \
  "GSAI-ML/LLaDA-8B-Instruct" \
  "6059b30c9531767ce47e0ca6a8161a6bb0c67693"
