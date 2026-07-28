"""Run MBPP functional-correctness evaluation outside the CUDA process.

The vendored MBPP evaluator creates worker processes.  Keeping that fork-heavy
workload in a fresh Python interpreter avoids forking after PyTorch has
initialized CUDA in the experiment runner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Completion JSONL file.")
    parser.add_argument("--problems", type=Path, required=True, help="MBPP problem JSONL file.")
    parser.add_argument("--result", type=Path, required=True, help="Output JSON result file.")
    parser.add_argument("--workers", type=int, default=8, help="Evaluator worker threads.")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


def main() -> None:
    args = parse_args()
    from ._vendor.evaluation.mbpp import evaluate_functional_correctness

    metrics, task_results = evaluate_functional_correctness(
        str(args.input),
        problem_file=str(args.problems),
        is_mbpp=True,
        k=[1],
        n_workers=args.workers,
        return_results=True,
    )
    compact_results = [
        {
            "task_id": result["task_id"],
            "completion_id": int(result["completion_id"]),
            "passed": bool(result["passed"]),
            "result": result["result"],
        }
        for result in sorted(
            task_results,
            key=lambda item: (str(item["task_id"]), int(item["completion_id"])),
        )
    ]
    args.result.write_text(
        json.dumps(
            {
                "pass_at_1": float(metrics["pass@1"]),
                "task_results": compact_results,
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
