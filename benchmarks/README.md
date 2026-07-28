# Benchmark snapshots

The repository versions the exact HumanEval and MBPP inputs used by the
artifact:

- `humaneval/HumanEval.jsonl`: 164 standard HumanEval tasks;
- `humaneval/HumanEval_ET.jsonl`: the corresponding extended tests;
- `mbpp/mbpp_sanitized.jsonl`: 427 sanitized MBPP tasks;
- `mbpp/MBPP_ET.jsonl`: the corresponding extended tests.

These snapshots and the minimal evaluation code under
`src/archer/_vendor/evaluation/` were extracted from the pinned Saber source
revision listed in the root README.

LiveCodeBench is not committed because of its size. Materialize the 400-task
`release_v1` snapshot at its pinned dataset revision:

```bash
uv run python scripts/prepare_livecodebench.py \
  --output-dir benchmarks/livecodebench
```

The preparation script records the dataset revision and SHA-256 digest in
`benchmarks/livecodebench/snapshot.json`.
