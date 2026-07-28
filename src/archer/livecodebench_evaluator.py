"""Evaluate generated programs with a pinned official LiveCodeBench checkout."""

from __future__ import annotations

import argparse
import base64
import json
import pickle
import sys
import zlib
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def decode_tests(value: str) -> list[dict[str, Any]]:
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(value))))


def evaluation_sample(problem: dict[str, Any]) -> dict[str, str]:
    tests = decode_tests(problem["public_test_cases"]) + decode_tests(
        problem["private_test_cases"]
    )
    metadata = json.loads(problem["metadata"])
    return {
        "input_output": json.dumps(
            {
                "inputs": [test["input"] for test in tests],
                "outputs": [test["output"] for test in tests],
                "fn_name": metadata.get("func_name"),
            }
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=6)
    args = parser.parse_args()

    repo = args.repo.resolve()
    if not (repo / "lcb_runner" / "evaluation").is_dir():
        raise SystemExit(f"not a LiveCodeBench checkout: {repo}")
    sys.path.insert(0, str(repo))

    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

    rows = load_jsonl(args.input)
    problems = load_jsonl(args.problems)
    if not rows:
        raise SystemExit("no completion rows to evaluate")
    if len(rows) > len(problems):
        raise SystemExit(
            f"too many completion rows: {len(rows)} > {len(problems)}"
        )
    selected_problems = problems[: len(rows)]
    for index, row in enumerate(rows):
        if row.get("index") != index:
            raise SystemExit(f"completion rows are not contiguous at index {index}")
        expected_id = selected_problems[index].get("question_id")
        if row.get("question_id") != expected_id:
            raise SystemExit(
                f"question_id mismatch at index {index}: "
                f"{row.get('question_id')!r} != {expected_id!r}"
            )

    samples = [evaluation_sample(problem) for problem in selected_problems]
    generations = [[str(row["completion"])] for row in rows]
    metrics, raw_results, metadata = codegen_metrics(
        samples,
        generations,
        k_list=[1],
        num_process_evaluate=args.workers,
        timeout=args.timeout,
    )
    result = {
        "pass_at_1": float(metrics["pass@1"]),
        "detail": metrics.get("detail", {}),
        "raw_results": raw_results,
        "metadata": metadata,
        "workers": args.workers,
        "timeout": args.timeout,
    }
    args.result.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
