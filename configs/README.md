# Experiment configurations

`table1_archer.json` contains the three Archer runs reported in Table 1. The
default refresh radii are MBPP $K=11$, LiveCodeBench $K=10$, and HumanEval
$K=8$. `reproduce.sh` uses these values unless `--k` is supplied.

The remaining JSON files preserve the broader paper experiments. They are
versioned for provenance but are not required for the default reproduction.

To inspect the commands without loading a model:

```bash
./reproduce.sh --dry-run
```
