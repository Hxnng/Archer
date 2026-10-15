# Reproduce the Archer rows in Table 1

The public runner evaluates **Archer with LLaDA-8B-Instruct** on MBPP, LiveCodeBench, and HumanEval. Baseline numbers and other paper experiments are documented in the [results guide](RESULTS.md) and [paper](https://arxiv.org/abs/2608.08086).

## Environment and setup

Use Linux, `git`, Python 3.10 or 3.11, [`uv`](https://docs.astral.sh/uv/), and one NVIDIA GPU with at least 40 GB of memory. The paper uses an A100 40 GB.

```bash
git clone https://github.com/Hxnng/Archer.git
cd Archer
./setup.sh
```

Setup performs `uv sync --locked`, verifies the bundled benchmark inputs, downloads the pinned model, checks out the pinned official LiveCodeBench evaluator, and materializes the 400-task LiveCodeBench snapshot.

### Reuse an existing model

Keep the environment variable set for both setup and subsequent runs. Use a complete checkout of the pinned model revision:

```bash
export ARCHER_LLADA_MODEL_PATH=/path/to/LLaDA-8B-Instruct
./setup.sh
CUDA_VISIBLE_DEVICES=0 ./reproduce.sh mbpp
```

### Use a download mirror

If needed, set a compatible Hugging Face endpoint before setup:

```bash
HF_ENDPOINT=https://hf-mirror.com ./setup.sh
```

## Inspect commands without a GPU

From a fresh checkout, these commands need only Python 3.10 or 3.11. They do not download or load model weights:

```bash
./reproduce.sh --dry-run
./reproduce.sh humaneval --k 16 --dry-run
./reproduce.sh all --k 16 --dry-run
```

## Run a smoke test separately

After setup, run the first two MBPP examples in an independent output directory. Replace the model path below with the checkout prepared by setup, or with your existing pinned checkout:

```bash
.venv/bin/python scripts/run_experiment.py configs/table1_archer.json \
  --run table1_archer_mbpp \
  --model-path ARCHER_LLADA_MODEL_PATH=models/LLaDA-8B-Instruct \
  --output-root outputs/smoke \
  --limit 2 \
  --resume
```

The smoke test writes under `outputs/smoke/table1_archer/`. Use a separate root because partial and full runs have different manifests and cannot resume into each other.

## Run the paper settings

```bash
CUDA_VISIBLE_DEVICES=0 ./reproduce.sh
```

Or run a selected benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 ./reproduce.sh mbpp
CUDA_VISIBLE_DEVICES=0 ./reproduce.sh livecodebench
CUDA_VISIBLE_DEVICES=0 ./reproduce.sh humaneval
```

The [versioned configuration](../configs/table1_archer.json) uses:

| Setting | Value |
| :--- | :--- |
| Model | `GSAI-ML/LLaDA-8B-Instruct` |
| Model revision | `6059b30c9531767ce47e0ca6a8161a6bb0c67693` |
| Output length / block length | 256 / 256 |
| Temperature | 0 |
| Saber parameters | `n=2`, `mu=2` |
| Refresh radius | MBPP 11, LiveCodeBench 10, HumanEval 8 |

The cache radius measures the token-wise Hamming distance between the current response and the response snapshot saved at the most recent refresh. It is not a fixed refresh interval or a count of edits in only the latest step.

### Override the refresh radius

```bash
CUDA_VISIBLE_DEVICES=0 ./reproduce.sh mbpp --k 16
CUDA_VISIBLE_DEVICES=0 ./reproduce.sh all --k 16
```

The override adds `_k16` to the run name, keeping it separate from the paper-default run. Larger radii permit more prompt reuse; the paper reports that excessive staleness can reduce quality.

## Outputs and resume behavior

```text
outputs/table1/table1_archer/
├── table1_archer_mbpp/
├── table1_archer_livecodebench/
└── table1_archer_humaneval/
```

Each run records its requested configuration, model revision, GPU, per-problem generations and timing, number of model evaluations, cache diagnostics, and evaluation results. Start with `summary.json` for aggregates and `manifest.json` for the run configuration; raw generations and evaluation details are saved alongside them.

Matching runs resume by default. `--no-resume` disables resume and **refuses to overwrite an existing nonempty run**. To start a fresh experiment, choose a new output root with the configuration runner:

```bash
.venv/bin/python scripts/run_experiment.py configs/table1_archer.json \
  --run table1_archer_mbpp \
  --model-path ARCHER_LLADA_MODEL_PATH=models/LLaDA-8B-Instruct \
  --output-root outputs/new-run
```

When using a custom model checkout, replace the `--model-path` value in these examples. Measured latency depends on the GPU and runtime; compare your outputs with the [reported reference values](../results/table1_reference.json) using the same setting.

## Data and code provenance

- HumanEval: 164 tasks, with standard and extended tests bundled.
- MBPP: 427 sanitized tasks, with standard and extended tests bundled.
- LiveCodeBench: the pinned 400-task `release_v1` snapshot, prepared during setup.
- Saber source for vendored model and evaluators: `a5433184cd627d9a0e31f26981736eca72802e47`.
- Official LiveCodeBench evaluator: `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`.

See the [benchmark notes](../benchmarks/README.md), [vendored dependency notes](../src/archer/_vendor/README.md), and preparation scripts for the dataset revision and integrity checks. Evaluating a completion executes generated Python; run in an isolated environment without credentials or sensitive files.

[Back to README](../README.md) · [Reported results](RESULTS.md)
