#!/usr/bin/env python3
"""Verify that benchmark snapshots match the paper runs byte for byte."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "benchmarks/humaneval/HumanEval.jsonl": (
        "f4c47d5090c07ba3e393c4ab4a22779b269fda6fcfc1e3f8fddb08911a8199c6"
    ),
    "benchmarks/humaneval/HumanEval_ET.jsonl": (
        "c9d9ccd7e299897a423a013ddd59e248b1befb68068d9e74811f1cfad49be0c5"
    ),
    "benchmarks/mbpp/mbpp_sanitized.jsonl": (
        "502db01bcc1e911d691d5212bd3d9c60ee52bbc2bd29457a520ace424bb2709d"
    ),
    "benchmarks/mbpp/MBPP_ET.jsonl": (
        "7c4b41c363f25a8b5a764478c8ca8e14fd1cf35f86b6f665a092f8bde14e41f9"
    ),
}
LIVECODEBENCH = {
    "benchmarks/livecodebench/test.jsonl": (
        "6ae6f79b27ea7274e76b88b53937bb32987f6725885426688948125182afdeba"
    )
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-livecodebench", action="store_true")
    args = parser.parse_args()

    expected = dict(EXPECTED)
    if args.require_livecodebench:
        expected.update(LIVECODEBENCH)

    failures: list[str] = []
    for relative_path, wanted in expected.items():
        path = ROOT / relative_path
        if not path.is_file():
            failures.append(f"missing: {relative_path}")
            continue
        actual = sha256(path)
        if actual != wanted:
            failures.append(
                f"hash mismatch: {relative_path}\n"
                f"  expected {wanted}\n"
                f"  actual   {actual}"
            )

    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Verified {len(expected)} benchmark snapshots.")


if __name__ == "__main__":
    main()
