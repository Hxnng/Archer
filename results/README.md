# Reference results

`reference_results.json` and `figure2_reference.csv` contain the point
estimates used as reproduction targets. Timing is hardware-sensitive; the
reported measurements used one NVIDIA A100 40GB GPU, batch size one, and CUDA
synchronization around generation.

These files are compact reference tables, not substitutes for raw run output.
Fresh runs produce manifests, per-problem JSONL records, evaluation details,
and aggregate summaries under `outputs/`.
