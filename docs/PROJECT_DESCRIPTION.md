# GitHub project description

Copy for the repository's About panel and project introductions.

## About

Training-free KV caching for rollback-capable diffusion language models. State-aware prompt reuse keeps responses revisable, with up to 4.33× speedup across code and reasoning benchmarks.

## Website

https://arxiv.org/abs/2608.08086

## Topics

`diffusion-language-models` · `kv-cache` · `efficient-inference` · `rollback` · `llada` · `code-generation` · `reasoning` · `training-free` · `pytorch`

## Short introduction

Archer makes rollback-capable diffusion language models more efficient through adaptive prompt KV caching. Bidirectional revisions can invalidate previously computed states, making conventional KV reuse difficult. Archer places the persistent cache at the fixed prompt, recomputes the mutable response at every step, and refreshes when the response drifts sufficiently far from its cached anchor. This preserves token revision while amortizing prompt computation; the paper also studies how delayed prompt feedback affects correction.

On the main LLaDA-8B-Instruct suite, Archer reaches 33.63% mean Pass@1 and 2.57× mean speedup over LLaDA-Confidence. Five additional reasoning benchmarks yield 48.37% macro-average accuracy and 2.47× mean speedup, reaching 4.33× on FinQA. The repository combines these results in one overview and provides a versioned reproduction path for MBPP, LiveCodeBench, and HumanEval, together with reference data and detailed evaluation notes.

## Visual assets

The bow-and-arrow mark, wordmark, and social card are transparent SVG files in [`assets/`](../assets/), with light and dark text variants. Regenerate them with `python3 scripts/render_readme_assets.py`.

The same script generates four flat navigation badges under [`assets/badges/`](../assets/badges/), linking to the paper, results, quick start, and reproduction guide.

The single eight-benchmark result graphic combines quality scores and speedups from the main and non-code reference tables. Regenerate it with `python3 scripts/plot_extended_results.py`, using the optional `matplotlib` dependency. It has a transparent canvas.

The method diagram uses the original Figure 3 in both themes. Its vector source is saved as [`assets/paper-method-source.pdf`](../assets/paper-method-source.pdf), with a direct PNG rendering at [`assets/figure3.png`](../assets/figure3.png). `scripts/export_paper_figure.py` renders the source without changing its colors, background, or content. This optional export requires `pypdfium2`.

[`assets/social-preview.png`](../assets/social-preview.png) is a 1280 × 640 PNG ready for GitHub's repository social preview setting. It is rasterized from [`assets/social-preview-light.svg`](../assets/social-preview-light.svg).
