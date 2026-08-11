#!/usr/bin/env bash
set -euo pipefail

# Clone and pin the official LiveCodeBench evaluator.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ensure_repo() {
  local name="$1"
  local url="$2"
  local revision="$3"
  local destination="${ROOT}/baseline/${name}"

  if [[ ! -d "${destination}/.git" ]]; then
    git clone "${url}" "${destination}"
  fi

  if ! git -C "${destination}" diff --quiet ||
     ! git -C "${destination}" diff --cached --quiet; then
    echo "Refusing to update dirty checkout: ${destination}" >&2
    exit 1
  fi

  git -C "${destination}" fetch --depth 1 origin "${revision}"
  git -C "${destination}" checkout --detach "${revision}"
  echo "${name}: $(git -C "${destination}" rev-parse HEAD)"
}

mkdir -p "${ROOT}/baseline"

if [[ "$#" -gt 1 || ( "$#" -eq 1 && "$1" != "livecodebench" ) ]]; then
  echo "Usage: scripts/setup_external_repos.sh [livecodebench]" >&2
  exit 2
fi

ensure_repo \
  "LiveCodeBench" \
  "https://github.com/LiveCodeBench/LiveCodeBench.git" \
  "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"

echo "LiveCodeBench is ready under ${ROOT}/baseline."
