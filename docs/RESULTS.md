# Reported results

These are the values reported in [Table 1 of the Archer paper](https://arxiv.org/html/2608.08086v2#S6.T1). They are reference values, rather than outputs of a new evaluation. The public release runs the Archer configurations on all three benchmarks; other methods are included here to make the paper comparison readable.

The sections below also include the current single-column manuscript's **Table 4** (rollback activity) and **Table 5** (non-code evaluation), supplied with the project as additional reference data. Table numbers refer to that manuscript; older paper versions use different numbering.

## Main comparison

Model: **LLaDA-8B-Instruct**. Hardware: **one NVIDIA A100 40 GB**. Quality is Pass@1 in percent.

| Method | MBPP | MBPP-ET | LiveCodeBench | HumanEval | HumanEval-ET | Mean quality ↑ | Mean speedup ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LLaDA-Confidence | 43.56 | 30.44 | 8.75 | 43.29 | 35.37 | 32.28 | 1.00× |
| Fast-dLLM + Cache | 44.50 | 31.62 | 7.75 | 41.46 | 36.59 | 32.38 | 1.71× |
| Fast-dLLM + Parallel | 40.05 | 29.27 | 8.25 | 41.46 | 36.59 | 31.12 | **4.84×** |
| dKV-Cache-Decode | **45.67** | 32.32 | **10.00** | 42.07 | 35.98 | 33.21 | 1.38× |
| dKV-Cache-Greedy | 23.19 | 18.27 | 2.00 | 15.24 | 14.63 | 14.67 | 1.26× |
| dLLM-Cache | 44.03 | 32.32 | **10.00** | 40.24 | 34.15 | 32.15 | 2.00× |
| Saber | 44.03 | 31.38 | 7.25 | **44.51** | 37.20 | 32.87 | 1.65× |
| **Archer** | 44.50 | **32.55** | **10.00** | 43.29 | **37.80** | **33.63** | 2.57× |

Archer has the highest mean quality and the second-highest mean speedup among these methods. Fast-dLLM + Parallel is faster, with lower mean quality. These results describe the evaluated setting; they do not imply quality improves on every individual benchmark.

## Latency

Mean generation time in **seconds per problem**, with speedup relative to **LLaDA-Confidence**.

| Method | MBPP ↓ | Speedup | LiveCodeBench ↓ | Speedup | HumanEval ↓ | Speedup |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| LLaDA-Confidence | 14.31 | 1.00× | 28.98 | 1.00× | 19.86 | 1.00× |
| Fast-dLLM + Cache | 11.85 | 1.21× | 12.75 | 2.27× | 11.95 | 1.66× |
| Fast-dLLM + Parallel | **3.40** | **4.21×** | **4.85** | **5.98×** | **4.60** | **4.32×** |
| dKV-Cache-Decode | 12.62 | 1.13× | 17.66 | 1.64× | 14.54 | 1.37× |
| dKV-Cache-Greedy | 12.36 | 1.16× | 21.25 | 1.36× | 15.82 | 1.26× |
| dLLM-Cache | 9.28 | 1.54× | 11.53 | 2.51× | 10.16 | 1.96× |
| Saber | 8.45 | 1.69× | 17.75 | 1.63× | 12.13 | 1.64× |
| **Archer** | 6.20 | 2.31× | 9.81 | 2.95× | 8.07 | 2.46× |

## Reading the numbers

- **Mean quality** is the arithmetic mean of five Pass@1 scores: MBPP, MBPP-ET, LiveCodeBench, HumanEval, and HumanEval-ET. For Archer this is 33.63% after rounding.
- **Mean speedup** is the arithmetic mean of the three benchmark speedup ratios, each measured relative to LLaDA-Confidence. For Archer this is 2.57×. It is not the ratio of pooled runtimes.
- **ET** adds extended test cases to the same MBPP or HumanEval tasks. The five quality columns are not five independent task collections.
- **Comparison with Saber** uses a different denominator: 8.45 / 6.20 = 1.36× on MBPP, 17.75 / 9.81 = 1.81× on LiveCodeBench, and 12.13 / 8.07 = 1.50× on HumanEval.
- **Percentage points** describe absolute score changes. For example, MBPP-ET improves by 1.17 points over Saber; HumanEval Base decreases by 1.22 points.
- **Reproduction scope:** this release provides the Archer rows, with 256 output tokens, deterministic decoding, and $K=11/10/8$ for MBPP/LiveCodeBench/HumanEval. Broader paper results have their own settings and are outside this runner's scope.

The reference data is available as [`results/table1_reference.json`](../results/table1_reference.json). Regenerate the README's transparent branding without additional packages:

```bash
python3 scripts/render_readme_assets.py
```

## Rollback activity

**Current manuscript Table 4 · fixed $K=8$ · LLaDA-8B-Instruct.** Revision is the percentage of response positions changed after their first non-mask prediction. Updates is the mean number of response updates.

| Method | MBPP revision | MBPP updates | LiveCodeBench revision | LiveCodeBench updates | HumanEval revision | HumanEval updates |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| No-Rollback | 0.00% | 65.97 | 0.00% | 74.73 | 0.00% | 66.26 |
| Saber | 29.18% | 146.98 | 28.21% | 158.83 | 27.87% | 157.55 |
| Archer | 30.03% | 147.51 | 26.93% | 163.25 | 30.24% | 158.49 |

Archer retains substantial revision activity and a similar number of response updates to Saber. This diagnostic supports the intended cache boundary: acceleration does not require freezing response tokens or eliminating rollback. It does not establish identical trajectories. Its fixed $K=8$ setting is separate from the dataset-specific radii in Table 1.

Reference values: [`results/rollback_reference.json`](../results/rollback_reference.json).

## Beyond code

**Current manuscript Table 5 · LLaDA-8B-Instruct.** Accuracy is in percent and time is seconds per example. All speedups in this section use **LLaDA-Confidence** as the denominator.

| Benchmark | LLaDA-Confidence accuracy | Saber accuracy | Archer accuracy | Archer latency | Archer speedup |
| :--- | ---: | ---: | ---: | ---: | ---: |
| MATH-500 | 24.00 | 25.60 | 25.00 | 6.39 s | 1.97× |
| FinQA | 28.51 | 29.56 | 30.17 | 9.86 s | 4.33× |
| PubMedQA | 67.40 | 69.40 | 69.50 | 9.58 s | 2.22× |
| ARC-Challenge | 86.09 | 86.18 | 85.75 | 9.12 s | 1.38× |
| MMLU-Pro (Engineering + Physics) | 30.60 | 30.42 | 31.44 | 7.58 s | 2.43× |

| Method | Macro-average accuracy | Mean speedup |
| :--- | ---: | ---: |
| LLaDA-Confidence | 47.32% | 1.00× |
| Saber | 48.23% | 1.47× |
| Archer | 48.37% | 2.47× |

Each benchmark contributes once to both means. MMLU-Pro pools the Engineering (969 questions) and Physics (1,299 questions) subsets into one 2,268-question result, weighted by question count. It does not cover the full MMLU-Pro suite.

MATH-500 uses canonical answer normalization. FinQA compares an extracted numerical answer with the gold `exe_ans`, reporting numerical-answer accuracy. PubMedQA uses exact normalized yes/no/maybe labels. ARC-Challenge and MMLU-Pro use multiple-choice answer accuracy. Archer uses $K=16$ on MATH-500, FinQA, ARC-Challenge, and MMLU-Pro, and $K=4$ on PubMedQA.

Relative to Saber, accuracy increases on FinQA, PubMedQA, and MMLU-Pro and decreases by 0.60 and 0.43 points on MATH-500 and ARC-Challenge. Relative to LLaDA-Confidence, Archer reduces latency on all five benchmarks, with accuracy gains on four of five. The chart preserves the ARC-Challenge decrease.

Reference values and all three methods' latencies: [`results/noncode_reference.json`](../results/noncode_reference.json). These results are provided for presentation and comparison; the public runner currently reproduces Table 1's Archer code-generation rows.

The homepage combines the eight benchmarks in one transparent figure, including the code benchmarks' ET scores. It shows the three methods evaluated across the full suite; all other baseline comparisons remain in this document. Table 4's fixed-radius rollback measurements remain separate because they measure response revision rather than benchmark quality or speedup.

To regenerate the combined result figure, use an environment with the optional plotting dependency `matplotlib`:

```bash
python3 scripts/plot_extended_results.py
```

[Back to README](../README.md) · [Reproduction guide](REPRODUCIBILITY.md)
