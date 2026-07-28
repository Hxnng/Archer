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

if [[ "$#" -eq 0 ]]; then
  set -- llada
fi

for model in "$@"; do
  case "${model}" in
    llada)
      download_model \
        "LLaDA-8B-Instruct" \
        "GSAI-ML/LLaDA-8B-Instruct" \
        "6059b30c9531767ce47e0ca6a8161a6bb0c67693"
      ;;
    dream)
      download_model \
        "Dream-v0-Instruct-7B" \
        "Dream-org/Dream-v0-Instruct-7B" \
        "05334cb9faaf763692dcf9d8737c642be2b2a6ae"
      ;;
    diffucoder)
      download_model \
        "DiffuCoder-7B-cpGRPO" \
        "apple/DiffuCoder-7B-cpGRPO" \
        "98bc42bfd871eeec92fd0a8439ab34f75b405204"
      ;;
    *)
      echo "Unknown model '${model}'. Choose: llada dream diffucoder" >&2
      exit 2
      ;;
  esac
done
