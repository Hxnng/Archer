#!/usr/bin/env python3
"""Scan reviewer-facing files for common identity and machine-path leaks."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "models",
    "outputs",
}
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".jsonl",
    ".lock",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
PATTERNS = {
    "email address": re.compile(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE
    ),
    "macOS home path": re.compile(r"/Users/[^/\s]+/"),
    "Linux home path": re.compile(r"/home/[^/\s]+/"),
    "Windows user path": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+\\"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--term",
        action="append",
        default=[],
        help="Additional literal identity term to reject; repeat as needed.",
    )
    return parser.parse_args()


def candidate_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or SKIP_PARTS.intersection(path.parts):
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        relative = path.relative_to(root)
        # Benchmark snapshots contain synthetic email strings as test inputs.
        # Their integrity is checked separately; they are not author-written text.
        if "benchmarks" in relative.parts and path.suffix.lower() == ".jsonl":
            continue
        if path.suffix.lower() in TEXT_SUFFIXES:
            yield path


def main() -> None:
    args = parse_args()
    patterns = dict(PATTERNS)
    for index, term in enumerate(args.term, start=1):
        if not term:
            raise SystemExit("--term must not be empty")
        patterns[f"custom identity term {index}"] = re.compile(
            re.escape(term), re.IGNORECASE
        )

    findings: list[str] = []
    for path in candidate_files(REPO_ROOT):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(REPO_ROOT)
        for line_number, line in enumerate(text.splitlines(), start=1):
            for label, pattern in patterns.items():
                if pattern.search(line):
                    findings.append(f"{relative}:{line_number}: {label}")

    if findings:
        print("\n".join(findings))
        raise SystemExit(f"anonymity scan failed with {len(findings)} finding(s)")
    print("anonymity scan passed")


if __name__ == "__main__":
    main()
