#!/usr/bin/env python3
"""Materialize the 400-problem LiveCodeBench snapshot evaluated by Saber."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DATASET_REPOSITORY = "livecodebench/code_generation_lite"
DATASET_RELEASE = "release_v1"
DATASET_REVISION = "a16d03780493b939b3601fb9da2ac3ed2b23caa2"
EXPECTED_PROBLEMS = 400
EXPECTED_SHA256 = "6ae6f79b27ea7274e76b88b53937bb32987f6725885426688948125182afdeba"
REQUIRED_FIELDS = {
    "question_title",
    "question_content",
    "platform",
    "question_id",
    "contest_id",
    "contest_date",
    "starter_code",
    "difficulty",
    "public_test_cases",
    "private_test_cases",
    "metadata",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/livecodebench"),
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit("install the project environment with `uv sync` first") from exc

    dataset = load_dataset(
        DATASET_REPOSITORY,
        DATASET_RELEASE,
        split="test",
        revision=DATASET_REVISION,
    )
    if len(dataset) != EXPECTED_PROBLEMS:
        raise RuntimeError(
            f"expected {EXPECTED_PROBLEMS} LiveCodeBench problems, got {len(dataset)}"
        )
    missing = REQUIRED_FIELDS.difference(dataset.column_names)
    if missing:
        raise RuntimeError(f"LiveCodeBench snapshot is missing fields: {sorted(missing)}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test.jsonl"
    temporary_path = output_dir / "test.jsonl.incomplete"
    dataset.to_json(
        temporary_path,
        orient="records",
        lines=True,
        force_ascii=False,
    )
    actual_sha256 = sha256(temporary_path)
    if actual_sha256 != EXPECTED_SHA256:
        temporary_path.unlink()
        raise RuntimeError(
            "LiveCodeBench snapshot hash differs from the paper run: "
            f"expected {EXPECTED_SHA256}, got {actual_sha256}"
        )
    temporary_path.replace(output_path)

    metadata = {
        "dataset_repository": DATASET_REPOSITORY,
        "dataset_release": DATASET_RELEASE,
        "dataset_revision": DATASET_REVISION,
        "problems": len(dataset),
        "sha256": actual_sha256,
        "saber_source_revision": "a5433184cd627d9a0e31f26981736eca72802e47",
        "alignment_note": "Byte-identical to the snapshot used for the paper run.",
    }
    (output_dir / "snapshot.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
