# Paper configurations

Configuration files use schema version 1. Each run declares:

- a stable run id;
- the Python module to execute;
- the model repository, immutable revision, and path environment variable;
- every method-specific argument needed to reproduce the paper setting.

`scripts/run_experiment.py` injects the model path, model metadata, benchmark
root, output directory, and run name. It rejects duplicated injected flags so
that the effective command is unambiguous.

List commands without running a model:

```bash
uv run python scripts/run_experiment.py configs/table1_main.json --dry-run
```

Select one run:

```bash
uv run python scripts/run_experiment.py configs/table2_backbones.json \
  --run table2_dream_humaneval_k9
```

The `K=∞` Figure 2 setting is represented by `--refresh-radius 257`. The
generation region has 256 positions, so an anchor-relative Hamming distance of
257 is unreachable and no intermediate prompt-cache refresh can occur.
