# Vendored reproduction dependencies

These files are the minimal runtime subset of
[`zhaoyMa/Saber`](https://github.com/zhaoyMa/Saber) at commit
`a5433184cd627d9a0e31f26981736eca72802e47`.

- `llada/`: KV-cache-enabled LLaDA model and configuration. Their original
  Apache-2.0 headers are preserved; the model import is package-relative.
- `evaluation/`: HumanEval/MBPP standard and ET evaluators plus execution
  helpers. Unused dataset-library imports were removed and the missing
  `collections.Counter` import in the MBPP evaluator was restored. Three
  regular-expression literals use raw strings to avoid Python syntax warnings;
  their matching behavior is unchanged. Trailing whitespace was normalized.

The Saber transition used by Archer, together with a small full-forward
reference implementation for regression testing, is audited in
`archer/saber.py`. Benchmark data provenance is documented in
`benchmarks/README.md`.
