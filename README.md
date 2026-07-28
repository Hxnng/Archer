# Archer

Anonymous code for *Adaptive Reuse of Cached Hidden States for Efficient
Rollback in Diffusion Language Models*.

Requirements: Linux, `git`, [`uv`](https://docs.astral.sh/uv/), and one NVIDIA
GPU with at least 40 GB of memory. The paper uses one A100 40 GB.

Set up the exact environment, models, datasets, and evaluators:

```bash
./setup.sh
```

Reproduce Tables 1–4:

```bash
./reproduce.sh
```

To reproduce one table:

```bash
./reproduce.sh table1
./reproduce.sh table2
./reproduce.sh table3
./reproduce.sh table4
```

Results are saved under `outputs/paper/`. The values reported in the paper are
in `results/reference_results.json`.

Set `CUDA_VISIBLE_DEVICES` to select another GPU. Evaluation executes generated
Python programs, so use an isolated environment without credentials.
