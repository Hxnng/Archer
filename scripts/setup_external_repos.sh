#!/usr/bin/env bash
set -euo pipefail

# Clone and pin the external implementations used by the unified benchmark.
# These checkouts are local dependencies and are intentionally git-ignored.

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

TARGET="${1:-all}"
case "${TARGET}" in
  all|livecodebench) ;;
  *)
    echo "Usage: scripts/setup_external_repos.sh [all|livecodebench]" >&2
    exit 2
    ;;
esac

ensure_repo \
  "LiveCodeBench" \
  "https://github.com/LiveCodeBench/LiveCodeBench.git" \
  "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"

if [[ "${TARGET}" == "all" ]]; then
  ensure_repo \
    "dKV-Cache" \
    "https://github.com/horseee/dKV-Cache.git" \
    "49a76fcc43b744ec2d960137f216e419317138b1"

  ensure_repo \
    "dLLM-cache" \
    "https://github.com/maomaocun/dLLM-cache.git" \
    "17235bffc8c5b587a2dc6f7dc76fcd01eab76e3a"
fi

echo "External repositories are ready under ${ROOT}/baseline."
