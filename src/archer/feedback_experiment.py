"""Run the paired prompt-feedback intervention on MBPP.

This is a diagnostic experiment, not a decoding baseline.  Each problem
produces two completions from one shared ARCHER checkpoint: immediate prompt
feedback and delayed prompt feedback.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

from .benchmark import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    append_jsonl,
    evaluate_mbpp_in_subprocess,
    get_dataset,
    json_dump,
    load_runtime,
    read_jsonl,
    require_path,
)
from .feedback_probe import generate_prompt_feedback_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--benchmark-root", type=Path, default=Path("benchmarks"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/feedback_probe"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--refresh-radius", type=int, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mbpp-eval-workers", type=int, default=8)
    args = parser.parse_args()
    if args.refresh_radius <= 0:
        parser.error("--refresh-radius must be positive")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        parser.error("--confidence-threshold must be in [0, 1]")
    if args.horizon <= 0 or args.limit < 0:
        parser.error("--horizon must be positive and --limit non-negative")
    return args


def _evaluation_rows(rows: List[Dict[str, Any]], branch: str) -> List[Dict[str, Any]]:
    return [
        {
            "task_id": row["task_id"],
            "sample_idx": 0,
            "completion": row[branch]["completion"],
        }
        for row in rows
    ]


def _evaluate_branch(
    rows: List[Dict[str, Any]],
    branch: str,
    problem_file: Path,
    et_problem_file: Path,
    run_dir: Path,
    workers: int,
) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".jsonl", delete=False, dir=run_dir
    ) as handle:
        path = Path(handle.name)
        for row in _evaluation_rows(rows, branch):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    try:
        standard = evaluate_mbpp_in_subprocess(path, problem_file, run_dir, workers)
        extended = evaluate_mbpp_in_subprocess(path, et_problem_file, run_dir, workers)
    finally:
        path.unlink(missing_ok=True)
    return {"standard": standard, "extended": extended}


def _passed_map(evaluation: Dict[str, Any]) -> Dict[str, bool]:
    return {
        str(item["task_id"]): bool(item["passed"])
        for item in evaluation["standard"]["task_results"]
    }


def _mean(values: List[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(
    rows: List[Dict[str, Any]],
    fresh_eval: Dict[str, Any],
    cached_eval: Dict[str, Any],
) -> Dict[str, Any]:
    eligible = [row for row in rows if row["probe"]["eligible"]]
    fresh_passed = _passed_map(fresh_eval)
    cached_passed = _passed_map(cached_eval)

    fresh_only = cached_only = both = neither = 0
    eligible_fresh_only = eligible_cached_only = 0
    eligible_ids = {str(row["task_id"]) for row in eligible}
    for row in rows:
        task_id = str(row["task_id"])
        fresh = fresh_passed[task_id]
        cached = cached_passed[task_id]
        if fresh and cached:
            both += 1
        elif fresh:
            fresh_only += 1
            if task_id in eligible_ids:
                eligible_fresh_only += 1
        elif cached:
            cached_only += 1
            if task_id in eligible_ids:
                eligible_cached_only += 1
        else:
            neither += 1

    def revision_rate(name: str) -> float | None:
        if not eligible:
            return None
        return sum(bool(row["probe"][name]) for row in eligible) / len(eligible)

    def revision_steps(name: str, flag: str) -> List[float]:
        return [
            float(row["probe"][name])
            for row in eligible
            if row["probe"][flag] and row["probe"][name] is not None
        ]

    margin_deltas = [
        float(row["probe"]["fresh_margin"])
        - float(row["probe"]["cached_margin"])
        for row in eligible
    ]
    probability_deltas = [
        float(row["probe"]["fresh_probability"])
        - float(row["probe"]["cached_probability"])
        for row in eligible
    ]
    return {
        "problems": len(rows),
        "eligible_events": len(eligible),
        "event_coverage": len(eligible) / len(rows),
        "fresh": {
            "pass_at_1": fresh_eval["standard"]["pass_at_1"],
            "et_pass_at_1": fresh_eval["extended"]["pass_at_1"],
            "revision_rate_within_horizon": revision_rate(
                "fresh_revised_within_horizon"
            ),
            "mean_revision_step": _mean(
                revision_steps(
                    "fresh_revision_step", "fresh_revised_within_horizon"
                )
            ),
        },
        "cached": {
            "pass_at_1": cached_eval["standard"]["pass_at_1"],
            "et_pass_at_1": cached_eval["extended"]["pass_at_1"],
            "revision_rate_within_horizon": revision_rate(
                "cached_revised_within_horizon"
            ),
            "mean_revision_step": _mean(
                revision_steps(
                    "cached_revision_step", "cached_revised_within_horizon"
                )
            ),
        },
        "mean_fresh_minus_cached_margin": _mean(margin_deltas),
        "mean_fresh_minus_cached_probability": _mean(probability_deltas),
        "paired_outcomes": {
            "both_pass": both,
            "both_fail": neither,
            "fresh_only_pass": fresh_only,
            "cached_only_pass": cached_only,
            "eligible_fresh_only_pass": eligible_fresh_only,
            "eligible_cached_only_pass": eligible_cached_only,
        },
        "mean_combined_wall_time_s": _mean(
            [float(row["wall_time_s"]) for row in rows]
        ),
    }


def main() -> None:
    args = parse_args()
    benchmark_root = args.benchmark_root.expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    run_dir = (args.output_dir.expanduser() / args.run_name).resolve()
    require_path(benchmark_root, "benchmark directory")
    require_path(model_path, "model directory")
    run_dir.mkdir(parents=True, exist_ok=True)

    requested_config = {
        "dataset": "mbpp",
        "model_path": str(model_path),
        "model_repository": MODEL_REPOSITORY,
        "model_revision": args.model_revision,
        "refresh_radius": args.refresh_radius,
        "confidence_threshold": args.confidence_threshold,
        "horizon": args.horizon,
        "gen_length": 256,
        "block_length": 256,
        "temperature": 0.0,
        "n": 2,
        "mu": 2,
        "limit": args.limit,
    }
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not args.resume:
            raise SystemExit(f"output exists: {run_dir}; pass --resume")
        if previous["requested_config"] != requested_config:
            raise SystemExit("--resume configuration does not match manifest")
    else:
        json_dump(
            manifest_path,
            {
                "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "experiment": "paired_prompt_feedback_intervention",
                "requested_config": requested_config,
            },
        )

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    runtime = load_runtime()
    torch = runtime["torch"]
    config = runtime["AutoConfig"].from_pretrained(
        model_path, trust_remote_code=True
    )
    if getattr(config, "model_type", None) != "llada":
        raise SystemExit("the paired feedback probe currently supports LLaDA only")
    model = runtime["LLaDAModelLM"].from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).cuda().eval()
    tokenizer = runtime["AutoTokenizer"].from_pretrained(
        model_path, trust_remote_code=True
    )

    docs, doc_to_text, problem_file, et_problem_file = get_dataset(
        "mbpp", benchmark_root, runtime
    )
    if args.limit:
        docs = docs[: args.limit]
    output_file = run_dir / "mbpp_feedback_pairs.jsonl"
    rows = read_jsonl(output_file)
    completed = {int(row["index"]) for row in rows}
    if len(completed) != len(rows):
        raise SystemExit("duplicate indices in existing output")

    for index, doc in enumerate(runtime["tqdm"].tqdm(docs, desc="mbpp:feedback")):
        if index in completed:
            continue
        context, _, trailing = doc_to_text(doc)
        prompt = (
            tokenizer.apply_chat_template(
                context, add_generation_prompt=True, tokenize=False
            )
            + trailing
        )
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        torch.cuda.synchronize()
        started = time.perf_counter()
        fresh, cached, probe = generate_prompt_feedback_probe(
            model,
            input_ids,
            n=2,
            mu=2,
            gen_length=256,
            block_length=256,
            temperature=0.0,
            mask_id=getattr(config, "mask_token_id", 126336),
            refresh_radius=args.refresh_radius,
            confidence_threshold=args.confidence_threshold,
            horizon=args.horizon,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        def decode(result: Dict[str, Any]) -> Dict[str, Any]:
            text = tokenizer.batch_decode(
                result["tokens"][:, input_ids.shape[1] :],
                skip_special_tokens=True,
            )[0]
            return {
                "full_response": text,
                "completion": runtime["extract_mbpp_answer"](
                    "```python\n" + text, doc
                ),
                "steps": int(result["steps"]),
                "nfe": int(result["nfe"]),
            }

        row = {
            "index": index,
            "task_id": doc["task_id"],
            "fresh": decode(fresh),
            "cached": decode(cached),
            "probe": probe,
            "wall_time_s": elapsed,
        }
        append_jsonl(output_file, row)
        rows.append(row)

    if len(rows) != len(docs):
        raise SystemExit(f"incomplete output: {len(rows)}/{len(docs)}")
    fresh_eval = _evaluate_branch(
        rows, "fresh", problem_file, et_problem_file, run_dir,
        args.mbpp_eval_workers,
    )
    cached_eval = _evaluate_branch(
        rows, "cached", problem_file, et_problem_file, run_dir,
        args.mbpp_eval_workers,
    )
    summary = {
        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
        "results": summarize(rows, fresh_eval, cached_eval),
    }
    json_dump(run_dir / "summary.json", summary)
    print(json.dumps(summary["results"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
