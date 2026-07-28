# Experiment configurations

The JSON files contain the exact immutable settings used by `reproduce.sh`.
They are validated before execution and should not require manual editing.

To inspect the commands without loading a model:

```bash
uv run python scripts/run_experiment.py configs/table1_main.json --dry-run
```
