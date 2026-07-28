"""State-anchored prompt-cache inference for revisable DLM decoding.

The sampler saves the generation state whenever it builds the prompt cache.
Before each later decoding step it computes
``D_t = d_H(g_t, g_r)`` and refreshes exactly when ``D_t >= K``. Generation
K/V states are recomputed on every cached step.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import torch

from ..saber import _saber_step
from .core import (
    build_prefix_cache,
    cached_forward_logits,
    generation_hamming_distance,
)


@dataclass
class ArcherDiagnostics:
    """Per-problem counters for state-anchored cache control."""

    cache_builds: int = 0
    extra_refreshes: int = 0
    cached_steps: int = 0
    distance_checks: int = 0
    distance_sum: int = 0
    max_distance: int = 0
    refresh_distance_sum: int = 0
    max_refresh_distance: int = 0

    def record_distance(self, distance: int) -> None:
        self.distance_checks += 1
        self.distance_sum += distance
        self.max_distance = max(self.max_distance, distance)

    def record_refresh(self, distance: int) -> None:
        self.extra_refreshes += 1
        self.refresh_distance_sum += distance
        self.max_refresh_distance = max(self.max_refresh_distance, distance)

    def as_dict(self) -> Dict[str, int]:
        return asdict(self)


def summarize_cache_diagnostics(
    reports: Iterable[Mapping[str, Any]],
) -> Dict[str, float | int]:
    """Aggregate per-problem counters without averaging ratios incorrectly."""
    reports = list(reports)
    if not reports:
        return {}

    def total(field: str) -> int:
        return sum(int(report.get(field, 0)) for report in reports)

    distance_checks = total("distance_checks")
    extra_refreshes = total("extra_refreshes")
    return {
        "mean_cache_builds": total("cache_builds") / len(reports),
        "mean_extra_refreshes": extra_refreshes / len(reports),
        "mean_cached_steps": total("cached_steps") / len(reports),
        "mean_snapshot_distance": (
            total("distance_sum") / distance_checks if distance_checks else 0.0
        ),
        "max_snapshot_distance": max(
            int(report.get("max_distance", 0)) for report in reports
        ),
        "mean_refresh_distance": (
            total("refresh_distance_sum") / extra_refreshes
            if extra_refreshes
            else 0.0
        ),
        "max_refresh_distance": max(
            int(report.get("max_refresh_distance", 0)) for report in reports
        ),
    }


@torch.no_grad()
def generate_archer(
    model,
    prompt: torch.Tensor,
    n: int = 2,
    mu: int = 2,
    gen_length: int = 256,
    block_length: int = 256,
    temperature: float = 0.0,
    mask_id: int = 126336,
    refresh_radius: int = 5,
):
    """Generate with cache refreshes triggered by ``D_t >= K``.

    The returned tuple is ``(tokens, decoding_steps, nfe, diagnostics)``.
    ``nfe`` counts every model call, including initial cache builds, cache
    refreshes, and cached generation forwards.
    """
    if gen_length % block_length:
        raise ValueError("gen_length must be divisible by block_length")
    if refresh_radius < 0:
        raise ValueError("refresh_radius must be non-negative")

    step = 0
    nfe = 0
    diagnostics = ArcherDiagnostics()
    prompt_length = prompt.shape[1]
    x = torch.full(
        (1, prompt_length + gen_length),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, :prompt_length] = prompt.clone()
    prompt_index = x != mask_id
    mask_index = x == mask_id
    global_transfer_index = torch.zeros_like(x, dtype=torch.bool)
    initial_confidence = torch.zeros_like(x, dtype=torch.float32)
    final_confidence = torch.full_like(x, -np.inf, dtype=torch.float32)
    last_confidence = torch.zeros_like(x, dtype=torch.float32)

    for block_number in range(gen_length // block_length):
        block_start = prompt_length + block_number * block_length
        block_end = block_start + block_length
        output, prefix_kv = build_prefix_cache(model, x, block_start)
        nfe += 1
        diagnostics.cache_builds += 1
        generation_snapshot = x[:, prompt_length:].clone()
        has_sampled_since_initial_build = False

        for _ in range(1024):
            step += 1
            distance = generation_hamming_distance(
                x[:, prompt_length:], generation_snapshot
            )
            diagnostics.record_distance(distance)

            if has_sampled_since_initial_build and distance >= refresh_radius:
                output, prefix_kv = build_prefix_cache(model, x, block_start)
                nfe += 1
                diagnostics.cache_builds += 1
                diagnostics.record_refresh(distance)
                generation_snapshot = x[:, prompt_length:].clone()
                logits = output.logits
            else:
                logits = cached_forward_logits(
                    model,
                    x,
                    prefix_kv,
                    output,
                    block_start,
                    block_end,
                )
                nfe += 1
                diagnostics.cached_steps += 1

            state = _saber_step(
                logits,
                x,
                mask_index,
                global_transfer_index,
                final_confidence,
                last_confidence,
                initial_confidence,
                block_start,
                block_end,
                prompt_index,
                n,
                mu,
                temperature,
                mask_id,
            )
            (
                x,
                global_transfer_index,
                initial_confidence,
                final_confidence,
                last_confidence,
                all_done,
                block_done,
                _,
                _,
            ) = state
            if all_done:
                return x, step, nfe, diagnostics.as_dict()
            if block_done:
                break
            has_sampled_since_initial_build = True

    return x, step, nfe, diagnostics.as_dict()
