# Archer

Anonymous code for *Adaptive Reuse of Cached Hidden States for Efficient
Rollback in Diffusion Language Models*.

Requirements: Linux, `git`, [`uv`](https://docs.astral.sh/uv/), and one NVIDIA
GPU with at least 40 GB of memory. The paper uses one A100 40 GB.

Set up the exact environment, models, datasets, and evaluators:

```bash
./setup.sh
```

Reproduce the main experiment:

```bash
./reproduce.sh
```

Results are saved to `outputs/paper/table1_main/`. The values reported in the
paper are in `results/reference_results.json`.

Set `CUDA_VISIBLE_DEVICES` to select another GPU. Evaluation executes generated
Python programs, so use an isolated environment without credentials.
