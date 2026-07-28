"""Measure whether ARCHER state distance predicts prompt-cache validity on MBPP.

This runner is intentionally separate from the benchmark timing harness:
fresh shadow forwards are diagnostic work and must not be reported as ARCHER
latency.  The generated completion itself always follows the configured
ARCHER trajectory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from .benchmark import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    REPO_ROOT,
    append_jsonl,
    evaluate_rows,
    get_dataset,
    git_revision,
    json_dump,
    load_runtime,
    read_jsonl,
    require_path,
)
from .cache_validity_probe import (
    generate_archer_with_validity_probe,
    summarize_validity_probes,
)

PROBE_MODULE_PATH = Path(__file__).with_name("cache_validity_probe.py")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--benchmark-root", type=Path, default=REPO_ROOT / "benchmarks")
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "outputs" / "cache_validity"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--refresh-radius", type=int, default=11,
        help="ARCHER refresh threshold K (default: 11).",
    )
    parser.add_argument(
        "--probe-every", type=int, default=4,
        help="Probe every Nth cached step within each problem (default: 4).",
    )
    parser.add_argument(
        "--max-probes", type=int, default=12,
        help="Maximum shadow full forwards per problem (default: 12).",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mbpp-eval-workers", type=int, default=8)
    args = parser.parse_args(argv)
    if args.refresh_radius <= 0:
        parser.error("--refresh-radius must be positive")
    if args.probe_every <= 0:
        parser.error("--probe-every must be positive")
    if args.max_probes < 0 or args.limit < 0:
        parser.error("--max-probes and --limit must be non-negative")
    if args.mbpp_eval_workers <= 0:
        parser.error("--mbpp-eval-workers must be positive")
    return args


def _requested_config(args: argparse.Namespace, model_path: Path, benchmark_root: Path) -> Dict[str, Any]:
    return {
        "dataset": "mbpp",
        "model_path": str(model_path),
        "model_repository": MODEL_REPOSITORY,
        "model_revision": args.model_revision,
        "benchmark_root": str(benchmark_root),
        "refresh_radius": args.refresh_radius,
        "probe_every": args.probe_every,
        "max_probes": args.max_probes,
        "gen_length": 256,
        "block_length": 256,
        "temperature": 0.0,
        "n": 2,
        "mu": 2,
        "limit": args.limit,
        "mbpp_eval_workers": args.mbpp_eval_workers,
    }


def _validate_resume(rows: List[Dict[str, Any]], count: int, output_file: Path) -> set[int]:
    completed = {row.get("index") for row in rows}
    if len(completed) != len(rows) or any(not isinstance(index, int) for index in completed):
        raise SystemExit(f"duplicate or invalid indices in {output_file}")
    if any(index < 0 or index >= count for index in completed):
        raise SystemExit(f"{output_file} does not match the selected dataset/--limit")
    if any("error" in row for row in rows):
        raise SystemExit(f"{output_file} contains generation errors")
    return completed


def summarize_run(
    rows: List[Dict[str, Any]],
    pass_at_1: float,
    et_pass_at_1: float,
    refresh_radius: int,
    runtime: Dict[str, Any],
) -> Dict[str, Any]:
    all_probes = [probe for row in rows for probe in row["validity_probes"]]
    return {
        "dataset": "mbpp",
        "problems": len(rows),
        "refresh_radius": refresh_radius,
        "pass_at_1": pass_at_1,
        "et_pass_at_1": et_pass_at_1,
        "mean_steps": sum(float(row["steps"]) for row in rows) / len(rows),
        "mean_main_nfe": sum(float(row["main_nfe"]) for row in rows) / len(rows),
        "shadow_nfe": sum(int(row["shadow_nfe"]) for row in rows),
        "mean_diagnostic_wall_time_s": (
            sum(float(row["diagnostic_wall_time_s"]) for row in rows) / len(rows)
        ),
        "cache_diagnostics": runtime.get(
            "summarize_cache_diagnostics", runtime.get("summarize_state_distance")
        )(
            [row["diagnostics"] for row in rows]
        ),
        "cache_validity": summarize_validity_probes(all_probes, refresh_radius),
        "timing_note": (
            "Diagnostic wall time includes fresh shadow forwards and is not an "
            "ARCHER latency or speedup measurement."
        ),
    }


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    benchmark_root = args.benchmark_root.expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    require_path(benchmark_root, "benchmark directory")
    require_path(model_path, "model directory")
    if not (model_path / "config.json").exists():
        raise SystemExit(f"model directory does not contain config.json: {model_path}")

    run_dir = (args.output_dir.expanduser() / args.run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    requested_config = _requested_config(args, model_path, benchmark_root)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not args.resume:
            raise SystemExit(f"output exists: {run_dir}; pass --resume")
        if previous.get("requested_config") != requested_config:
            raise SystemExit("--resume configuration does not match manifest")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    runtime = load_runtime()
    torch = runtime["torch"]
    if "AutoConfig" in runtime:
        model_type = getattr(
            runtime["AutoConfig"].from_pretrained(model_path, trust_remote_code=True),
            "model_type",
            None,
        )
    else:
        model_type = json.loads(
            (model_path / "config.json").read_text(encoding="utf-8")
        ).get("model_type")
    if model_type != "llada":
        raise SystemExit("the cache-validity probe currently supports LLaDA only")

    manifest = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment": "archer_cache_validity_shadow_forward",
        "requested_config": requested_config,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
            "archer_git_revision": git_revision(REPO_ROOT),
            "cache_validity_experiment_sha256": _sha256(Path(__file__)),
            "cache_validity_probe_sha256": _sha256(PROBE_MODULE_PATH),
        },
        "method_note": (
            "Fresh shadow forwards and shadow Saber transitions are observational: "
            f"they do not alter the K={args.refresh_radius} ARCHER trajectory "
            "and their NFE is separate."
        ),
    }
    if not manifest_path.exists():
        json_dump(manifest_path, manifest)
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    print(f"==> loading model from {model_path}", flush=True)
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
    if not docs:
        raise SystemExit("MBPP has no selected problems")

    stem = f"mbpp_archer_k{args.refresh_radius}_cache_validity"
    output_file = run_dir / f"{stem}.jsonl"
    rows = read_jsonl(output_file)
    if rows and not args.resume:
        raise SystemExit(f"output exists: {output_file}; pass --resume")
    completed = _validate_resume(rows, len(docs), output_file)

    for index in runtime["tqdm"].tqdm(range(len(docs)), desc="mbpp:cache-validity"):
        if index in completed:
            continue
        doc = docs[index]
        context, ground_truth, trailing = doc_to_text(doc)
        prompt = (
            tokenizer.apply_chat_template(
                context, add_generation_prompt=True, tokenize=False
            )
            + trailing
        )
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        torch.cuda.synchronize()
        started = time.perf_counter()
        tokens, steps, main_nfe, diagnostics, probes, shadow_nfe = (
            generate_archer_with_validity_probe(
                model,
                input_ids,
                n=2,
                mu=2,
                gen_length=256,
                block_length=256,
                temperature=0.0,
                mask_id=getattr(getattr(model, "config", None), "mask_token_id", 126336),
                refresh_radius=args.refresh_radius,
                probe_every=args.probe_every,
                max_probes=args.max_probes,
            )
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        generated_text = tokenizer.batch_decode(
            tokens[:, input_ids.shape[1]:], skip_special_tokens=True
        )[0]
        ground_truth = ground_truth if isinstance(ground_truth, dict) else {}
        row = {
            "index": index,
            "task_id": ground_truth.get("task_id", doc.get("task_id")),
            "sample_idx": 0,
            "completion": runtime["extract_mbpp_answer"](
                "```python\n" + generated_text, doc
            ),
            "full_response": generated_text,
            "steps": int(steps),
            "main_nfe": int(main_nfe),
            "shadow_nfe": int(shadow_nfe),
            "diagnostic_wall_time_s": elapsed,
            "diagnostics": diagnostics,
            "validity_probes": probes,
        }
        append_jsonl(output_file, row)
        rows.append(row)

    if len(rows) != len(docs):
        raise SystemExit(f"incomplete output: {len(rows)}/{len(docs)}")
    detail_path = run_dir / f"{stem}_evaluation.json"
    evaluation_kwargs = {"mbpp_eval_workers": args.mbpp_eval_workers}
    if "detail_output" in inspect.signature(evaluate_rows).parameters:
        evaluation_kwargs["detail_output"] = detail_path
    pass_at_1, et_pass_at_1 = evaluate_rows(
        "mbpp", rows, problem_file, et_problem_file, run_dir, runtime,
        **evaluation_kwargs,
    )
    if et_pass_at_1 is None:
        raise RuntimeError("MBPP-ET evaluation did not return pass@1")
    result = summarize_run(
        rows, pass_at_1, et_pass_at_1, args.refresh_radius, runtime
    )
    summary = {"manifest": manifest, "results": result}
    json_dump(run_dir / "summary.json", summary)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
