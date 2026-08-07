# Archer

### Adaptive Reuse of Cached Hidden States for Efficient Rollback in Diffusion Language Models

<p align="center">
  <img src="assets/figure1.png" width="100%" alt="Archer reuses prompt key-value states while recomputing the revisable response." />
</p>

Archer is a training-free prompt KV-caching method for rollback-capable
diffusion language models (DLMs). It reuses the fixed prompt states, recomputes
the response states that remain subject to revision, and refreshes the prompt
cache when the response moves sufficiently far from its cached anchor. This
repository provides the minimal path needed to reproduce the **Archer rows in
Table 1**.

## Table 1 targets

The released configuration uses LLaDA-8B-Instruct and the exact model revision
from the paper. `Base` and `ET` are Pass@1 (%) on the original and Extended
Test Cases versions of each benchmark; time is mean generation time per
problem on one NVIDIA A100 40 GB.

| Benchmark | Base | ET | Time (s) | Default $K$ |
|---|---:|---:|---:|---:|
| MBPP | 44.50 | 32.55 | 6.20 | 11 |
| LiveCodeBench | 10.00 | — | 9.81 | 10 |
| HumanEval | 43.29 | 37.80 | 8.07 | 8 |

The reference values are also stored in
[`results/reference_results.json`](results/reference_results.json). Wall-clock
time is hardware-sensitive; accuracy should be evaluated with the versioned
benchmark snapshots and evaluators in this repository.

## Quick start

### 1. Requirements

- Linux and `git`;
- Python 3.10 or 3.11;
- [`uv`](https://docs.astral.sh/uv/);
- one NVIDIA GPU with at least 40 GB of memory.

### 2. Install the environment

```bash
git clone https://github.com/Hxnng/Archer.git
cd Archer
./setup.sh
```

`setup.sh` creates the locked Python environment, downloads the pinned
LLaDA-8B-Instruct checkpoint, verifies the bundled MBPP and HumanEval
snapshots, and prepares the pinned LiveCodeBench evaluator and dataset.

If Hugging Face is unavailable from your network, set a compatible endpoint
before setup, for example:

```bash
HF_ENDPOINT=https://hf-mirror.com ./setup.sh
```

An existing model checkout can be reused:

```bash
ARCHER_LLADA_MODEL_PATH=/path/to/LLaDA-8B-Instruct ./setup.sh
```

### 3. Reproduce the Archer rows

Run all three Table 1 benchmarks with their paper-default refresh radii:

```bash
CUDA_VISIBLE_DEVICES=0 ./reproduce.sh
```

Run a single benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 ./reproduce.sh mbpp
CUDA_VISIBLE_DEVICES=0 ./reproduce.sh livecodebench
CUDA_VISIBLE_DEVICES=0 ./reproduce.sh humaneval
```

The default values are read from
[`configs/table1_archer.json`](configs/table1_archer.json): MBPP uses $K=11$,
LiveCodeBench uses $K=10$, and HumanEval uses $K=8$.

## Choose another refresh radius

Pass `--k` to override the paper default without editing a configuration file:

```bash
CUDA_VISIBLE_DEVICES=0 ./reproduce.sh mbpp --k 16
```

Custom-radius outputs use a distinct run name such as
`table1_archer_mbpp_k16`, so they cannot overwrite the default Table 1 run.
The same override can be applied to all three benchmarks:

```bash
CUDA_VISIBLE_DEVICES=0 ./reproduce.sh all --k 16
```

## Verify before a full run

Print the exact commands without downloading or loading a model:

```bash
./reproduce.sh --dry-run
./reproduce.sh humaneval --k 16 --dry-run
```

After setup, run two examples as an end-to-end smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 ./reproduce.sh mbpp --limit 2
```

Interrupted runs resume by default. Add `--no-resume` to request a fresh run.

## Reproduction configuration

| Item | Value |
|---|---|
| Model | `GSAI-ML/LLaDA-8B-Instruct` |
| Revision | `6059b30c9531767ce47e0ca6a8161a6bb0c67693` |
| Generation / block length | 256 / 256 |
| Temperature | 0 |
| Batch size | 1 |
| Rollback parameters | $n=2$, $\mu=2$ |
| Evaluation | bundled MBPP/HumanEval evaluators; pinned LiveCodeBench evaluator |

Every run records its arguments, model revision, GPU, per-problem generations,
timing, and aggregate evaluation. Outputs are written below:

```text
outputs/table1/table1_archer/
├── table1_archer_mbpp/
├── table1_archer_livecodebench/
└── table1_archer_humaneval/
```

Each run directory contains the raw JSONL generations, evaluation details,
manifest, and `summary.json`.

## Repository layout

```text
configs/table1_archer.json       # Table 1 Archer defaults
reproduce.sh                     # user-facing reproduction entry point
scripts/run_experiment.py        # versioned config runner
src/archer/state_distance/       # Archer cache controller and sampler
src/archer/benchmark.py          # unified generation and evaluation pipeline
benchmarks/                      # versioned benchmark snapshots and checksums
results/reference_results.json  # paper reference values
```

Evaluation executes model-generated Python programs. Run the artifact in an
isolated environment without credentials or access to sensitive files.
