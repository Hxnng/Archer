"""Run unified DLM baselines on HumanEval, MBPP, and LiveCodeBench.

ARCHER is the prompt-prefix cache controller defined by an anchor-relative
Hamming-distance refresh radius. The runner writes one JSONL file per
dataset/variant, allowing interrupted runs to resume safely.

Example:
    CUDA_VISIBLE_DEVICES=0 archer-reproduce --dataset humaneval \
      --model-path models/LLaDA-8B-Instruct --run-name paper_he
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from . import SABER_SOURCE_REVISION
from .fast_dllm_v1 import FAST_DLLM_V1_SOURCE_REVISION
from .fast_dllm_saber import SABER_FAST_DLLM_SOURCE_REVISION
from .dkv_cache import DKV_CACHE_SOURCE_REVISION
from .dllm_cache import DLLM_CACHE_SOURCE_REVISION


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_REPOSITORY = "GSAI-ML/LLaDA-8B-Instruct"
MODEL_REVISION = "6059b30"
LIVECODEBENCH_SOURCE_REVISION = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
LIVECODEBENCH_DATASET_REVISION = "a16d03780493b939b3601fb9da2ac3ed2b23caa2"
PAPER_CONFIG = {
    "gen_length": 256,
    "block_length": 256,
    "mu": 2,
    "temperature": 0.0,
    "n": 2,
}
VARIANTS = (
    "llada",
    "fast_dllm",
    "fast_dllm_v1",
    "fast_dllm_saber",
    "dkv_cache_decode",
    "dkv_cache_greedy",
    "dllm_cache",
    "saber",
    "archer",
)
WORK_METRICS = {
    "llada": "nfe",
    "fast_dllm": "nfe",
    "fast_dllm_v1": "nfe",
    "fast_dllm_saber": "nfe",
    "dkv_cache_decode": "nfe",
    "dkv_cache_greedy": "nfe",
    "dllm_cache": "nfe",
    "saber": "nfe",
    "archer": "nfe",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DLM decoding and cache methods under one protocol."
    )
    parser.add_argument(
        "--dataset",
        choices=("humaneval", "mbpp", "livecodebench", "all"),
        default="all",
        help="Benchmark to run. Use separate run names when launching datasets in parallel.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=("saber", "archer"),
        help="Methods to run; default: saber archer.",
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("ARCHER_MODEL_PATH"),
        help=f"Local directory of {MODEL_REPOSITORY}. May also be set by ARCHER_MODEL_PATH.",
    )
    parser.add_argument(
        "--model-revision",
        default=os.environ.get("ARCHER_MODEL_REVISION", MODEL_REVISION),
        help=(
            "Hugging Face revision used to populate --model-path; recorded in the "
            f"manifest (default: {MODEL_REVISION})."
        ),
    )
    parser.add_argument(
        "--model-repository",
        default=os.environ.get("ARCHER_MODEL_REPOSITORY", MODEL_REPOSITORY),
        help="Source model repository recorded in the run manifest.",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=REPO_ROOT / "benchmarks",
        help="Directory containing fixed HumanEval, MBPP, and LiveCodeBench assets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "archer",
        help="Directory for raw completions, metadata, and metric summaries.",
    )
    parser.add_argument(
        "--run-name",
        default="paper_gen256_mu2",
        help="Name of this run below --output-dir.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Use the first N problems only (smoke test).")
    parser.add_argument("--resume", action="store_true", help="Continue from existing JSONL output in this run directory.")
    parser.add_argument("--continue-on-error", action="store_true", help="Record failed examples instead of stopping. Not suitable for a final reproduction.")
    parser.add_argument("--skip-warmup", action="store_true", help="Do not warm up each variant once before timing.")
    parser.add_argument(
        "--mbpp-eval-workers",
        type=int,
        default=8,
        help="Worker threads for the isolated MBPP evaluator (default: 8).",
    )
    parser.add_argument(
        "--livecodebench-repo",
        type=Path,
        default=Path(
            os.environ.get(
                "LIVECODEBENCH_REPO", REPO_ROOT / "baseline" / "LiveCodeBench"
            )
        ),
        help="Pinned checkout of the official LiveCodeBench evaluator.",
    )
    parser.add_argument(
        "--livecodebench-eval-workers",
        type=int,
        default=16,
        help="Worker processes for official LiveCodeBench execution (default: 16).",
    )
    parser.add_argument(
        "--livecodebench-timeout",
        type=int,
        default=6,
        help="Official LiveCodeBench per-test timeout in seconds (default: 6).",
    )

    # These options make deviations from the canonical evaluation configuration explicit.
    parser.add_argument("--gen-length", type=int, default=PAPER_CONFIG["gen_length"])
    parser.add_argument("--block-length", type=int, default=PAPER_CONFIG["block_length"])
    parser.add_argument("--mu", type=int, default=PAPER_CONFIG["mu"])
    parser.add_argument("--temperature", type=float, default=PAPER_CONFIG["temperature"])
    parser.add_argument("--n", type=int, default=PAPER_CONFIG["n"])
    parser.add_argument(
        "--fast-dllm-block-length",
        type=int,
        default=32,
        help="Fast-dLLM v1 cache block length (paper default: 32).",
    )
    parser.add_argument(
        "--fast-dllm-threshold",
        type=float,
        default=0.9,
        help="Fast-dLLM v1 parallel-decoding confidence threshold (paper default: 0.9).",
    )
    parser.add_argument(
        "--fast-dllm-saber-factor",
        type=float,
        default=1.0,
        help="Factor for the DualCache Fast-dLLM baseline used by Saber (default: 1).",
    )
    parser.add_argument(
        "--dkv-cache-repo",
        type=Path,
        default=Path(
            os.environ.get("DKV_CACHE_REPO", REPO_ROOT / "baseline" / "dKV-Cache")
        ),
        help="Pinned checkout of the official horseee/dKV-Cache repository.",
    )
    parser.add_argument(
        "--dkv-cache-reloading-step",
        type=int,
        default=8,
        help="dKV-Cache-Decode refresh interval (LLaDA paper default: 8).",
    )
    parser.add_argument(
        "--dkv-cache-greedy-reloading-step",
        type=int,
        default=2,
        help="dKV-Cache-Greedy refresh interval (LLaDA paper default: 2).",
    )
    parser.add_argument(
        "--dkv-cache-greedy-window-size",
        type=int,
        default=4,
        help="dKV-Cache-Greedy local cache window (LLaDA paper default: 4).",
    )
    parser.add_argument(
        "--dkv-cache-greedy-seed",
        type=int,
        default=42,
        help="Seed for the predefined random denoising order (default: 42).",
    )
    parser.add_argument(
        "--dllm-cache-repo",
        type=Path,
        default=Path(
            os.environ.get("DLLM_CACHE_REPO", REPO_ROOT / "baseline" / "dLLM-cache")
        ),
        help="Pinned checkout of the official maomaocun/dLLM-Cache repository.",
    )
    parser.add_argument(
        "--dllm-cache-prompt-interval",
        type=int,
        default=None,
        help="Prompt refresh interval; defaults to the paper's code-task setting.",
    )
    parser.add_argument(
        "--dllm-cache-response-interval",
        type=int,
        default=5,
        help="Response full-refresh interval (paper code-task setting: 5).",
    )
    parser.add_argument(
        "--dllm-cache-transfer-ratio",
        type=float,
        default=0.25,
        help="V-verify adaptive response update ratio (paper default: 0.25).",
    )
    parser.add_argument(
        "--dllm-cache-block-length",
        type=int,
        default=32,
        help="Semi-autoregressive block length used for code tasks in the paper.",
    )
    parser.add_argument(
        "--refresh-radius",
        type=int,
        default=5,
        help="Refresh ARCHER when D_t >= this anchor-relative Hamming distance.",
    )
    parser.add_argument(
        "--allow-nonstandard-config",
        action="store_true",
        help="Allow a parameter sweep instead of the canonical evaluation configuration.",
    )
    args = parser.parse_args()

    if not args.model_path:
        parser.error("--model-path (or ARCHER_MODEL_PATH) is required")
    if len(set(args.variants)) != len(args.variants):
        parser.error("--variants must not contain duplicates")
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.gen_length <= 0 or args.block_length <= 0 or args.gen_length % args.block_length:
        parser.error("--gen-length must be a positive multiple of --block-length")
    if args.refresh_radius < 0:
        parser.error("--refresh-radius must be non-negative")
    if args.fast_dllm_block_length <= 0 or args.gen_length % args.fast_dllm_block_length:
        parser.error("--fast-dllm-block-length must be positive and divide --gen-length")
    if not 0.0 <= args.fast_dllm_threshold <= 1.0:
        parser.error("--fast-dllm-threshold must lie in [0, 1]")
    if args.fast_dllm_saber_factor <= 0:
        parser.error("--fast-dllm-saber-factor must be positive")
    if args.dkv_cache_reloading_step <= 0:
        parser.error("--dkv-cache-reloading-step must be positive")
    if args.dkv_cache_greedy_reloading_step <= 0:
        parser.error("--dkv-cache-greedy-reloading-step must be positive")
    if args.dkv_cache_greedy_window_size < 0:
        parser.error("--dkv-cache-greedy-window-size must be non-negative")
    if {"dkv_cache_decode", "dkv_cache_greedy"}.intersection(args.variants) and len(args.variants) != 1:
        parser.error(
            "dKV-Cache variants use an official modified LLaDA model and must run "
            "as the only variant in its process"
        )
    if "dllm_cache" in args.variants and len(args.variants) != 1:
        parser.error(
            "dllm_cache installs official model hooks and must run as the only "
            "variant in its process"
        )
    if "dllm_cache" in args.variants and args.dataset == "all":
        parser.error("run dllm_cache separately for each dataset")
    if args.dllm_cache_prompt_interval is None:
        args.dllm_cache_prompt_interval = 25 if args.dataset == "humaneval" else 100
    if args.dllm_cache_prompt_interval <= 0 or args.dllm_cache_response_interval <= 0:
        parser.error("dLLM-Cache refresh intervals must be positive")
    if not 0.0 <= args.dllm_cache_transfer_ratio <= 1.0:
        parser.error("--dllm-cache-transfer-ratio must lie in [0, 1]")
    if args.dllm_cache_block_length <= 0 or args.gen_length % args.dllm_cache_block_length:
        parser.error("--dllm-cache-block-length must be positive and divide --gen-length")
    if args.mbpp_eval_workers <= 0:
        parser.error("--mbpp-eval-workers must be positive")
    if args.livecodebench_eval_workers <= 0:
        parser.error("--livecodebench-eval-workers must be positive")
    if args.livecodebench_timeout <= 0:
        parser.error("--livecodebench-timeout must be positive")
    if not args.allow_nonstandard_config:
        mismatches = [
            key for key, expected in PAPER_CONFIG.items() if getattr(args, key) != expected
        ]
        if mismatches:
            parser.error(
                "the canonical configuration requires "
                "gen_length=block_length=256, mu=2, temperature=0, n=2; "
                "use --allow-nonstandard-config for a different run"
            )
    return args


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        fail(f"{description} not found: {path}")


def git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def load_runtime() -> Dict[str, Any]:
    """Import GPU/runtime dependencies only after --help and path validation work."""
    try:
        import torch
        import tqdm
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from ._vendor.evaluation.humaneval import (
            evaluate_solution,
            evaluate_solution_et,
        )
        from ._vendor.evaluation.mbpp import evaluate_functional_correctness
        from ._vendor.llada import LLaDAModelLM
        from .datasets import (
            extract_humaneval_answer,
            extract_livecodebench_answer,
            extract_mbpp_answer,
            humaneval_doc_to_text,
            livecodebench_doc_to_text,
            load_jsonl,
            load_mbpp,
            mbpp_doc_to_text,
        )
        from .llada import generate_llada
        from .fast_dllm_v1 import generate_fast_dllm_v1
        from .fast_dllm_saber import (
            generate_fast_dllm_cache_only,
            generate_fast_dllm_saber,
        )
        from .dkv_cache import (
            generate_dkv_cache_decode,
            generate_dkv_cache_greedy,
            load_dkv_cache_greedy_model,
            load_dkv_cache_model,
        )
        from .dllm_cache import generate_dllm_cache, load_dllm_cache_model
        from .dream import (
            generate_dream_archer,
            generate_dream_saber,
            is_dream_family,
        )
        from .saber import generate_saber
        from .state_distance import (
            generate_archer,
            summarize_cache_diagnostics,
        )
    except ModuleNotFoundError as exc:
        fail(
            f"missing Python dependency {exc.name!r}. Activate the A100 environment and run "
            "uv sync --locked."
        )
    if not torch.cuda.is_available():
        fail("CUDA is unavailable. Run this on the A100 with a CUDA-enabled PyTorch build.")
    return {
        "torch": torch,
        "tqdm": tqdm,
        "AutoConfig": AutoConfig,
        "AutoModel": AutoModel,
        "AutoTokenizer": AutoTokenizer,
        "LLaDAModelLM": LLaDAModelLM,
        "load_jsonl": load_jsonl,
        "load_mbpp": load_mbpp,
        "humaneval_doc_to_text": humaneval_doc_to_text,
        "livecodebench_doc_to_text": livecodebench_doc_to_text,
        "mbpp_doc_to_text": mbpp_doc_to_text,
        "extract_humaneval_answer": extract_humaneval_answer,
        "extract_livecodebench_answer": extract_livecodebench_answer,
        "extract_mbpp_answer": extract_mbpp_answer,
        "evaluate_solution": evaluate_solution,
        "evaluate_solution_et": evaluate_solution_et,
        "evaluate_functional_correctness": evaluate_functional_correctness,
        "generate_llada": generate_llada,
        "generate_fast_dllm_v1": generate_fast_dllm_v1,
        "generate_fast_dllm_cache_only": generate_fast_dllm_cache_only,
        "generate_fast_dllm_saber": generate_fast_dllm_saber,
        "generate_dkv_cache_decode": generate_dkv_cache_decode,
        "generate_dkv_cache_greedy": generate_dkv_cache_greedy,
        "load_dkv_cache_model": load_dkv_cache_model,
        "load_dkv_cache_greedy_model": load_dkv_cache_greedy_model,
        "generate_dllm_cache": generate_dllm_cache,
        "load_dllm_cache_model": load_dllm_cache_model,
        "generate_dream_saber": generate_dream_saber,
        "generate_dream_archer": generate_dream_archer,
        "is_dream_family": is_dream_family,
        "generate_saber": generate_saber,
        "generate_archer": generate_archer,
        "summarize_cache_diagnostics": summarize_cache_diagnostics,
    }


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    fail(f"invalid JSON in {path}:{line_number}: {exc}")
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def datasets_to_run(dataset: str) -> Iterable[str]:
    return ("humaneval", "mbpp", "livecodebench") if dataset == "all" else (dataset,)


def get_dataset(
    dataset_name: str, benchmark_root: Path, runtime: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Any, Path, Path]:
    if dataset_name == "humaneval":
        data_dir = benchmark_root / "humaneval"
        for name in ("HumanEval.jsonl", "HumanEval_ET.jsonl"):
            require_path(data_dir / name, f"HumanEval asset {name}")
        data = runtime["load_jsonl"](data_dir / "HumanEval.jsonl")
        return (
            data,
            runtime["humaneval_doc_to_text"],
            data_dir / "HumanEval.jsonl",
            data_dir / "HumanEval_ET.jsonl",
        )

    if dataset_name == "livecodebench":
        data_dir = benchmark_root / "livecodebench"
        problem_file = data_dir / "test.jsonl"
        require_path(problem_file, "LiveCodeBench release_v1 snapshot")
        data = runtime["load_jsonl"](problem_file)
        if len(data) != 400:
            fail(
                "Saber-aligned LiveCodeBench requires release_v1 with 400 problems; "
                f"found {len(data)} in {problem_file}"
            )
        return (
            data,
            runtime["livecodebench_doc_to_text"],
            problem_file,
            problem_file,
        )

    data_dir = benchmark_root / "mbpp"
    for name in ("mbpp_sanitized.jsonl", "MBPP_ET.jsonl"):
        require_path(data_dir / name, f"MBPP asset {name}")
    return (
        runtime["load_mbpp"](data_dir / "mbpp_sanitized.jsonl"),
        runtime["mbpp_doc_to_text"],
        data_dir / "mbpp_sanitized.jsonl",
        data_dir / "MBPP_ET.jsonl",
    )


def variant_call(
    variant: str,
    model: Any,
    input_ids: Any,
    args: argparse.Namespace,
    runtime: Dict[str, Any],
    sample_index: int = 0,
) -> Tuple[Any, int, int, Dict[str, Any]]:
    if variant == "llada":
        tokens, steps = runtime["generate_llada"](
            model,
            input_ids,
            steps=args.gen_length,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
        )
        return tokens, int(steps), int(steps), {}
    if variant == "fast_dllm_v1":
        tokens, nfe = runtime["generate_fast_dllm_v1"](
            model,
            input_ids,
            gen_length=args.gen_length,
            block_length=args.fast_dllm_block_length,
            threshold=args.fast_dllm_threshold,
        )
        return tokens, int(nfe), int(nfe), {}
    if variant == "fast_dllm":
        tokens, nfe = runtime["generate_fast_dllm_cache_only"](
            model,
            input_ids,
            steps=args.gen_length,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
        )
        return tokens, int(nfe), int(nfe), {}
    if variant == "fast_dllm_saber":
        tokens, nfe = runtime["generate_fast_dllm_saber"](
            model,
            input_ids,
            steps=args.gen_length,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            factor=args.fast_dllm_saber_factor,
        )
        return tokens, int(nfe), int(nfe), {}
    if variant == "dkv_cache_decode":
        tokens, nfe = runtime["generate_dkv_cache_decode"](
            model,
            input_ids,
            repo_path=args.dkv_cache_repo,
            steps=args.gen_length,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            cache_reloading_step=args.dkv_cache_reloading_step,
        )
        return tokens, int(nfe), int(nfe), {}
    if variant == "dkv_cache_greedy":
        tokens, nfe = runtime["generate_dkv_cache_greedy"](
            model,
            input_ids,
            repo_path=args.dkv_cache_repo,
            steps=args.gen_length,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            cache_reloading_step=args.dkv_cache_greedy_reloading_step,
            window_size=args.dkv_cache_greedy_window_size,
            seed=args.dkv_cache_greedy_seed,
        )
        return tokens, int(nfe), int(nfe), {}
    if variant == "dllm_cache":
        tokens, nfe = runtime["generate_dllm_cache"](
            model,
            input_ids,
            repo_path=args.dllm_cache_repo,
            steps=args.gen_length,
            gen_length=args.gen_length,
            block_length=args.dllm_cache_block_length,
            temperature=args.temperature,
        )
        return tokens, int(nfe), int(nfe), {}
    common = {
        "n": args.n,
        "mu": args.mu,
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "temperature": args.temperature,
    }
    mask_id = getattr(getattr(model, "config", None), "mask_token_id", 126336)
    is_dream = runtime.get("is_dream_family", lambda _model: False)(model)
    if variant == "saber":
        generator = (
            runtime["generate_dream_saber"]
            if is_dream
            else runtime["generate_saber"]
        )
        tokens, steps = generator(
            model,
            input_ids,
            mask_id=mask_id,
            **common,
        )
        return tokens, int(steps), int(steps), {}
    if variant == "archer":
        generator = (
            runtime["generate_dream_archer"]
            if is_dream
            else runtime["generate_archer"]
        )
        tokens, steps, nfe, diagnostics = generator(
            model,
            input_ids,
            mask_id=mask_id,
            refresh_radius=args.refresh_radius,
            **common,
        )
        return tokens, int(steps), int(nfe), dict(diagnostics)
    raise ValueError(f"unsupported variant: {variant}")


def evaluate_rows(
    dataset_name: str,
    rows: List[Dict[str, Any]],
    problem_file: Path,
    et_problem_file: Path,
    run_dir: Path,
    runtime: Dict[str, Any],
    mbpp_eval_workers: int = 8,
    livecodebench_repo: Path | None = None,
    livecodebench_eval_workers: int = 16,
    livecodebench_timeout: int = 6,
    detail_output: Path | None = None,
) -> Tuple[float, float | None]:
    # HumanEval standard evaluation precedes ET evaluation on an isolated copy.
    evaluation_rows = copy.deepcopy(rows)
    if dataset_name == "humaneval":
        standard = runtime["evaluate_solution"](
            evaluation_rows, problem_file=str(problem_file), k=[1]
        )
        extended = runtime["evaluate_solution_et"](
            evaluation_rows, problem_file=str(et_problem_file), k=[1]
        )
        return float(standard["pass@1"]), float(extended["pass@1"])
    if dataset_name == "livecodebench":
        if livecodebench_repo is None:
            raise ValueError("livecodebench_repo is required for LiveCodeBench")
        result = evaluate_livecodebench_in_subprocess(
            rows,
            problem_file,
            run_dir,
            livecodebench_repo,
            livecodebench_eval_workers,
            livecodebench_timeout,
        )
        if detail_output is not None:
            json_dump(detail_output, result)
        return float(result["pass_at_1"]), None

    else:
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", delete=False, suffix=".jsonl", dir=run_dir
            ) as handle:
                temp_path = Path(handle.name)
                for row in evaluation_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            standard = evaluate_mbpp_in_subprocess(
                temp_path, problem_file, run_dir, mbpp_eval_workers
            )
            extended = evaluate_mbpp_in_subprocess(
                temp_path, et_problem_file, run_dir, mbpp_eval_workers
            )
            if detail_output is not None:
                json_dump(
                    detail_output,
                    {"standard": standard, "extended": extended},
                )
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
    return float(standard["pass_at_1"]), float(extended["pass_at_1"])


def evaluate_livecodebench_in_subprocess(
    rows: List[Dict[str, Any]],
    problem_file: Path,
    run_dir: Path,
    repo: Path,
    workers: int,
    timeout: int,
) -> Dict[str, Any]:
    """Execute untrusted solutions outside the CUDA-bearing inference process."""
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", delete=False, suffix=".jsonl", dir=run_dir
    ) as handle:
        completions_path = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", delete=False, suffix=".json", dir=run_dir
    ) as handle:
        result_path = Path(handle.name)
    result_path.unlink()
    command = [
        sys.executable,
        "-m",
        "archer.livecodebench_evaluator",
        "--input",
        str(completions_path),
        "--problems",
        str(problem_file),
        "--repo",
        str(repo),
        "--result",
        str(result_path),
        "--workers",
        str(workers),
        "--timeout",
        str(timeout),
    ]
    try:
        subprocess.run(command, check=True)
        return json.loads(result_path.read_text(encoding="utf-8"))
    finally:
        completions_path.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)


def evaluate_mbpp_in_subprocess(
    completions_path: Path,
    problem_file: Path,
    run_dir: Path,
    workers: int,
) -> Dict[str, Any]:
    """Evaluate one MBPP suite in a fresh interpreter with no CUDA state."""
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", delete=False, suffix=".json", dir=run_dir
    ) as handle:
        result_path = Path(handle.name)
    result_path.unlink()
    command = [
        sys.executable,
        "-m",
        "archer.mbpp_evaluator",
        "--input",
        str(completions_path),
        "--problems",
        str(problem_file),
        "--result",
        str(result_path),
        "--workers",
        str(workers),
    ]
    try:
        subprocess.run(command, check=True)
        return json.loads(result_path.read_text(encoding="utf-8"))
    finally:
        result_path.unlink(missing_ok=True)


def run_dataset(
    dataset_name: str,
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    run_dir: Path,
    runtime: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    torch = runtime["torch"]
    tqdm = runtime["tqdm"]
    docs, doc_to_text, problem_file, et_problem_file = get_dataset(
        dataset_name, args.benchmark_root, runtime
    )
    if args.limit:
        docs = docs[: args.limit]
    if not docs:
        fail(f"{dataset_name} has no selected problems")
    print(f"==> {dataset_name}: {len(docs)} problems | variants={list(args.variants)}", flush=True)

    def prepare(doc: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
        context, ground_truth, trailing = doc_to_text(doc)
        prompt = tokenizer.apply_chat_template(context, add_generation_prompt=True, tokenize=False) + trailing
        return ground_truth, tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    if not args.skip_warmup:
        _, warmup_ids = prepare(docs[0])
        for variant in args.variants:
            print(f"==> warmup {dataset_name}:{variant}", flush=True)
            variant_call(variant, model, warmup_ids, args, runtime, sample_index=-1)
        torch.cuda.synchronize()

    summaries: Dict[str, Dict[str, Any]] = {}
    for variant in args.variants:
        output_file = run_dir / f"{dataset_name}_{variant}.jsonl"
        existing_rows = read_jsonl(output_file)
        if existing_rows and not args.resume:
            fail(f"output already exists: {output_file}; choose another --run-name or pass --resume")
        completed = {row.get("index") for row in existing_rows}
        if len(completed) != len(existing_rows) or any(not isinstance(index, int) for index in completed):
            fail(f"{output_file} has duplicate or invalid indices; use a new --run-name")
        if any(index < 0 or index >= len(docs) for index in completed):
            fail(f"{output_file} does not match the selected dataset/--limit")
        rows = existing_rows
        print(f"==> {dataset_name}:{variant} | completed={len(rows)}/{len(docs)}", flush=True)

        for index in tqdm.tqdm(range(len(docs)), desc=f"{dataset_name}:{variant}"):
            if index in completed:
                continue
            doc = docs[index]
            ground_truth, input_ids = prepare(doc)
            try:
                torch.cuda.synchronize()
                started = dt.datetime.now(dt.timezone.utc)
                start_time = time.perf_counter()
                generated_ids, steps, method_work, diagnostics = variant_call(
                    variant, model, input_ids, args, runtime, sample_index=index
                )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start_time
                generated_text = tokenizer.batch_decode(
                    generated_ids[:, input_ids.shape[1] :], skip_special_tokens=True
                )[0]
                row: Dict[str, Any] = {
                    "index": index,
                    "sample_idx": 0,
                    "full_response": generated_text,
                    "steps": steps,
                    "method_work": method_work,
                    "work_metric": WORK_METRICS[variant],
                    "wall_time_s": elapsed,
                    "started_at_utc": started.isoformat(),
                }
                if WORK_METRICS[variant] == "nfe":
                    row["nfe"] = method_work
                if diagnostics:
                    row["diagnostics"] = diagnostics
                if dataset_name == "humaneval":
                    row["completion"] = runtime["extract_humaneval_answer"](
                        generated_text, doc
                    )
                elif dataset_name == "mbpp":
                    row["completion"] = runtime["extract_mbpp_answer"](
                        "```python\n" + generated_text, doc
                    )
                else:
                    row["completion"] = runtime["extract_livecodebench_answer"](
                        generated_text, doc
                    )
                ground_truth = ground_truth if isinstance(ground_truth, dict) else {}
                if ground_truth.get("task_id") is not None:
                    row["task_id"] = ground_truth["task_id"]
                elif doc.get("task_id") is not None:
                    row["task_id"] = doc["task_id"]
                if ground_truth.get("question_id") is not None:
                    row["question_id"] = ground_truth["question_id"]
            except Exception as exc:  # pragma: no cover - requires a GPU model failure
                if not args.continue_on_error:
                    raise RuntimeError(
                        f"generation failed for {dataset_name}:{variant} problem {index}; "
                        f"resume with --resume after fixing the cause"
                    ) from exc
                row = {
                    "index": index,
                    "sample_idx": 0,
                    "completion": "",
                    "full_response": "",
                    "steps": 0,
                    "method_work": 0,
                    "work_metric": WORK_METRICS[variant],
                    "wall_time_s": 0.0,
                    "error": repr(exc),
                }
                if doc.get("task_id") is not None:
                    row["task_id"] = doc["task_id"]
            append_jsonl(output_file, row)
            rows.append(row)

        if len(rows) != len(docs):
            fail(f"incomplete output for {dataset_name}:{variant}")
        if any("error" in row for row in rows):
            fail(f"{dataset_name}:{variant} contains generation errors and is not a valid reproduction")
        detail_output = (
            run_dir / f"{dataset_name}_{variant}_evaluation.json"
            if dataset_name in {"mbpp", "livecodebench"}
            else None
        )
        pass_at_1, pass_at_1_et = evaluate_rows(
            dataset_name,
            rows,
            problem_file,
            et_problem_file,
            run_dir,
            runtime,
            args.mbpp_eval_workers,
            args.livecodebench_repo,
            args.livecodebench_eval_workers,
            args.livecodebench_timeout,
            detail_output,
        )
        mean_time = sum(float(row["wall_time_s"]) for row in rows) / len(rows)
        mean_steps = sum(float(row["steps"]) for row in rows) / len(rows)
        mean_method_work = sum(
            float(row.get("method_work", row.get("nfe", 0))) for row in rows
        ) / len(rows)
        summaries[variant] = {
            "variant": variant,
            "problems": len(rows),
            "pass_at_1": pass_at_1,
            "pass_at_1_et": pass_at_1_et,
            "mean_wall_time_s": mean_time,
            "mean_steps": mean_steps,
            "mean_method_work": mean_method_work,
            "work_metric": WORK_METRICS[variant],
            "raw_completions": output_file.name,
        }
        if variant.startswith("archer"):
            summaries[variant]["cache_diagnostics"] = runtime[
                "summarize_cache_diagnostics"
            ](
                [row.get("diagnostics", {}) for row in rows]
            )
        if WORK_METRICS[variant] == "nfe":
            summaries[variant]["mean_nfe"] = mean_method_work
        et_text = "n/a" if pass_at_1_et is None else f"{pass_at_1_et:.4f}"
        print(
            f"[RESULT][{dataset_name}][{variant}] pass@1={pass_at_1:.4f} "
            f"ET={et_text} time={mean_time:.2f}s "
            f"{WORK_METRICS[variant]}={mean_method_work:.1f} "
            f"steps={mean_steps:.1f} N={len(rows)}",
            flush=True,
        )

    if "saber" in summaries:
        for variant, summary in summaries.items():
            if variant != "saber":
                summary["speedup_vs_saber"] = (
                    summaries["saber"]["mean_wall_time_s"]
                    / summary["mean_wall_time_s"]
                )
    return summaries


def main() -> None:
    args = parse_args()
    args.benchmark_root = args.benchmark_root.resolve()
    args.dkv_cache_repo = args.dkv_cache_repo.expanduser().resolve()
    args.dllm_cache_repo = args.dllm_cache_repo.expanduser().resolve()
    args.livecodebench_repo = args.livecodebench_repo.expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    require_path(args.benchmark_root, "benchmark directory")
    require_path(model_path, "model directory")
    if {"dkv_cache_decode", "dkv_cache_greedy"}.intersection(args.variants):
        require_path(args.dkv_cache_repo, "official dKV-Cache checkout")
    if "dllm_cache" in args.variants:
        require_path(args.dllm_cache_repo, "official dLLM-Cache checkout")
    if "livecodebench" in datasets_to_run(args.dataset):
        require_path(args.livecodebench_repo, "official LiveCodeBench checkout")
        actual_lcb_revision = git_revision(args.livecodebench_repo)
        if actual_lcb_revision != LIVECODEBENCH_SOURCE_REVISION:
            fail(
                "LiveCodeBench evaluator revision mismatch: expected "
                f"{LIVECODEBENCH_SOURCE_REVISION}, found {actual_lcb_revision}"
            )
    if not (model_path / "config.json").exists():
        fail(f"model directory does not contain config.json: {model_path}")

    run_dir = (args.output_dir / args.run_name).resolve()
    if run_dir.exists() and not args.resume and any(run_dir.iterdir()):
        fail(f"run directory already exists and is non-empty: {run_dir}; use --resume or a new --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    runtime = load_runtime()
    torch = runtime["torch"]
    model_config = runtime["AutoConfig"].from_pretrained(
        model_path, trust_remote_code=True
    )
    manifest = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "paper_config": PAPER_CONFIG,
        "requested_config": {
            "datasets": list(datasets_to_run(args.dataset)),
            "variants": list(args.variants),
            "limit": args.limit,
            "gen_length": args.gen_length,
            "block_length": args.block_length,
            "mu": args.mu,
            "temperature": args.temperature,
            "n": args.n,
            "refresh_radius": args.refresh_radius,
            "fast_dllm_block_length": args.fast_dllm_block_length,
            "fast_dllm_threshold": args.fast_dllm_threshold,
            "fast_dllm_saber_factor": args.fast_dllm_saber_factor,
            "dkv_cache_repo": str(args.dkv_cache_repo),
            "dkv_cache_reloading_step": args.dkv_cache_reloading_step,
            "dkv_cache_greedy_reloading_step": args.dkv_cache_greedy_reloading_step,
            "dkv_cache_greedy_window_size": args.dkv_cache_greedy_window_size,
            "dkv_cache_greedy_seed": args.dkv_cache_greedy_seed,
            "dllm_cache_repo": str(args.dllm_cache_repo),
            "dllm_cache_prompt_interval": args.dllm_cache_prompt_interval,
            "dllm_cache_response_interval": args.dllm_cache_response_interval,
            "dllm_cache_transfer_ratio": args.dllm_cache_transfer_ratio,
            "dllm_cache_block_length": args.dllm_cache_block_length,
            "model_path": str(model_path),
            "model_repository": args.model_repository,
            "model_revision": args.model_revision,
            "model_type": getattr(model_config, "model_type", "unknown"),
            "benchmark_root": str(args.benchmark_root),
            "mbpp_eval_workers": args.mbpp_eval_workers,
            "livecodebench_repo": str(args.livecodebench_repo),
            "livecodebench_release": "release_v1",
            "livecodebench_dataset_revision": LIVECODEBENCH_DATASET_REVISION,
            "livecodebench_eval_workers": args.livecodebench_eval_workers,
            "livecodebench_timeout": args.livecodebench_timeout,
        },
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
            "saber_source_revision": SABER_SOURCE_REVISION,
            "fast_dllm_v1_source_revision": FAST_DLLM_V1_SOURCE_REVISION,
            "saber_fast_dllm_source_revision": SABER_FAST_DLLM_SOURCE_REVISION,
            "dkv_cache_source_revision": DKV_CACHE_SOURCE_REVISION,
            "dllm_cache_source_revision": DLLM_CACHE_SOURCE_REVISION,
            "livecodebench_source_revision": git_revision(args.livecodebench_repo),
            "archer_git_revision": git_revision(REPO_ROOT),
        },
        "evaluation_note": (
            "HumanEval standard evaluation precedes ET evaluation on an isolated "
            "in-memory copy. LiveCodeBench uses the 400-problem release_v1 code-"
            "generation snapshot, Saber's zero-shot prompt, and the pinned official evaluator."
        ),
    }
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists() and args.resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("requested_config") != manifest["requested_config"]:
            fail("--resume configuration does not match the existing manifest; use a new --run-name")
    else:
        json_dump(manifest_path, manifest)

    print(f"==> loading model from {model_path}", flush=True)
    if "dkv_cache_decode" in args.variants:
        model = runtime["load_dkv_cache_model"](
            args.dkv_cache_repo, model_path, torch
        )
    elif "dkv_cache_greedy" in args.variants:
        model = runtime["load_dkv_cache_greedy_model"](
            args.dkv_cache_repo, model_path, torch
        )
    elif "dllm_cache" in args.variants:
        model = runtime["load_dllm_cache_model"](
            args.dllm_cache_repo,
            model_path,
            torch,
            prompt_interval_steps=args.dllm_cache_prompt_interval,
            gen_interval_steps=args.dllm_cache_response_interval,
            transfer_ratio=args.dllm_cache_transfer_ratio,
        )
    else:
        if getattr(model_config, "model_type", None) == "llada":
            model = runtime["LLaDAModelLM"].from_pretrained(
                model_path, torch_dtype=torch.bfloat16
            ).cuda().eval()
        else:
            model = runtime["AutoModel"].from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            ).cuda().eval()
    tokenizer = runtime["AutoTokenizer"].from_pretrained(model_path, trust_remote_code=True)

    all_summaries: Dict[str, Dict[str, Any]] = {}
    for dataset_name in datasets_to_run(args.dataset):
        all_summaries[dataset_name] = run_dataset(
            dataset_name, args, model, tokenizer, run_dir, runtime
        )
        json_dump(run_dir / "summary.json", {"manifest": manifest, "results": all_summaries})

    print(f"==> complete. Outputs: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
