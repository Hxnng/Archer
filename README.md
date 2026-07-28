# Archer

**Adaptive Reuse of Cached Hidden States for Efficient Rollback in Diffusion
Language Models**

This anonymous artifact contains the implementation and experiment
configurations for Archer. Archer reuses prompt K/V states while recomputing
the complete, revisable generation region at every decoding step. Cache
refreshes are triggered by the Hamming distance between the current generation
and the state at which the prompt cache was built.

## Artifact map

| Paper result | Configuration | Entry point |
|---|---|---|
| Table 1, main comparison | `configs/table1_main.json` | `archer.benchmark` |
| Table 2, three backbones | `configs/table2_backbones.json` | `archer.benchmark` |
| Table 3, feedback intervention, K=11 | `configs/table3_feedback_k11.json` | `archer.feedback_experiment` |
| Table 4, cache-validity probe, K=11 | `configs/table4_cache_validity_k11.json` | `archer.cache_validity_experiment` |
| Figure 2, K sensitivity | `configs/figure2_k_sweep.json` | `archer.benchmark` |

Machine-readable target values are in `results/reference_results.json` and
`results/figure2_reference.csv`.

## Requirements

- Linux with an NVIDIA CUDA GPU
- Python 3.10 or 3.11
- `git` and [`uv`](https://docs.astral.sh/uv/)
- One NVIDIA A100 40GB was used for the reported timing measurements

Install the locked environment and pinned external evaluators/baselines:

```bash
uv sync --locked
./scripts/setup_external_repos.sh
```

Download the exact model revisions. LLaDA is sufficient for Tables 1, 3, 4,
and Figure 2; all three models are needed for Table 2.

```bash
./scripts/download_models.sh llada
./scripts/download_models.sh dream diffucoder
```

Equivalent model checkouts or institutional mirrors may be used, but the
revision hashes in the configuration files must be preserved.

## Preflight checks

```bash
uv run pytest -q
uv run python scripts/check_anonymity.py
uv run python scripts/run_experiment.py configs/table1_main.json --dry-run
```

The anonymity scanner also accepts private, uncommitted terms:

```bash
uv run python scripts/check_anonymity.py \
  --term '<author-name>' \
  --term '<institution>'
```

## Smoke test

Run the first two MBPP problems for the Table 1 core methods:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_experiment.py \
  configs/table1_main.json \
  --run table1_mbpp_core \
  --model-path ARCHER_LLADA_MODEL_PATH=models/LLaDA-8B-Instruct \
  --limit 2
```

## Full reproduction

Set the model paths once:

```bash
export ARCHER_LLADA_MODEL_PATH="$PWD/models/LLaDA-8B-Instruct"
export ARCHER_DREAM_MODEL_PATH="$PWD/models/Dream-v0-Instruct-7B"
export ARCHER_DIFFUCODER_MODEL_PATH="$PWD/models/DiffuCoder-7B-cpGRPO"
```

Then run a complete paper configuration:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_experiment.py \
  configs/table1_main.json

CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_experiment.py \
  configs/table2_backbones.json

CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_experiment.py \
  configs/table3_feedback_k11.json

CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_experiment.py \
  configs/table4_cache_validity_k11.json
```

Use `--run <id>` to select one run and `--resume` to continue an interrupted
run. dKV-Cache and dLLM-Cache are intentionally launched as independent
processes because their official implementations install model-specific
modules or hooks.

## Outputs

Every run writes incrementally below `outputs/paper/<config-name>/<run-id>/`:

```text
manifest.json
<dataset>_<variant>.jsonl
<dataset>_<variant>_evaluation.json
summary.json
```

The manifest records the model repository and revision, method parameters,
runtime versions, CUDA device, upstream source revisions, and the artifact Git
revision. Generation time is measured after CUDA synchronization and excludes
model loading and execution-based evaluation.

## Reproducibility pins

| Component | Revision |
|---|---|
| LLaDA-8B-Instruct | `6059b30c9531767ce47e0ca6a8161a6bb0c67693` |
| Dream-v0-Instruct-7B | `05334cb9faaf763692dcf9d8737c642be2b2a6ae` |
| DiffuCoder-7B-cpGRPO | `98bc42bfd871eeec92fd0a8439ab34f75b405204` |
| Saber extraction | `a5433184cd627d9a0e31f26981736eca72802e47` |
| Fast-dLLM v1 reference | `a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b` |
| dKV-Cache | `49a76fcc43b744ec2d960137f216e419317138b1` |
| dLLM-Cache | `17235bffc8c5b587a2dc6f7dc76fcd01eab76e3a` |
| LiveCodeBench evaluator | `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24` |
| LiveCodeBench dataset | `a16d03780493b939b3601fb9da2ac3ed2b23caa2` |

HumanEval and MBPP snapshots are versioned in `benchmarks/`. LiveCodeBench
`release_v1` is materialized locally with
`scripts/prepare_livecodebench.py`.

## Safety note

HumanEval, MBPP, and LiveCodeBench evaluation executes model-generated Python
programs. Run correctness evaluation in an isolated, non-privileged
environment without credentials or sensitive files.
