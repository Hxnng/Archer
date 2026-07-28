"""Shadow-forward probe for measuring prompt-cache validity.

The main trajectory is ordinary ARCHER.  On a bounded subset of cached
steps, the probe additionally evaluates the same state with a full forward
pass and compares the two logits and the next Saber transition.  Shadow
forwards never feed back into decoding and are counted separately from the
main trajectory's NFE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import math
import numpy as np
import torch

try:
    from .saber import _saber_step
except ImportError:  # Compatibility with the pre-refactor V6 tree.
    from .sampler import _saber_step

try:
    from .state_distance.core import (
        build_prefix_cache,
        cached_forward_logits,
        generation_hamming_distance,
    )
    from .state_distance.sampler import ArcherDiagnostics
except ImportError:  # Compatibility with the pre-refactor server checkout.
    from .state_distance.sampler import (
        StateDistanceDiagnostics as ArcherDiagnostics,
        generation_hamming_distance,
    )

    def build_prefix_cache(model, x: torch.Tensor, block_start: int):
        output = model(x, use_cache=True)
        prefix_kv = [
            tuple(tensor[:, :, :block_start] for tensor in layer)
            for layer in output.past_key_values
        ]
        return output, prefix_kv

    def cached_forward_logits(
        model, x, prefix_kv, prefix_output, block_start: int, block_end: int
    ):
        block_logits = model(
            x[:, block_start:], past_key_values=prefix_kv, use_cache=True
        ).logits
        logits = torch.zeros(
            x.shape[0],
            x.shape[1],
            block_logits.shape[-1],
            device=x.device,
            dtype=block_logits.dtype,
        )
        logits[:, :block_start] = prefix_output.logits[:, :block_start]
        logits[:, block_start:block_end] = block_logits[:, : block_end - block_start]
        return logits


@dataclass
class _DecoderState:
    x: torch.Tensor
    global_transfer_index: torch.Tensor
    initial_confidence: torch.Tensor
    final_confidence: torch.Tensor
    last_confidence: torch.Tensor

    def clone(self) -> "_DecoderState":
        return _DecoderState(
            x=self.x.clone(),
            global_transfer_index=self.global_transfer_index.clone(),
            initial_confidence=self.initial_confidence.clone(),
            final_confidence=self.final_confidence.clone(),
            last_confidence=self.last_confidence.clone(),
        )


def _run_shadow_step(
    logits: torch.Tensor,
    state: _DecoderState,
    *,
    mask_index: torch.Tensor,
    block_start: int,
    block_end: int,
    prompt_index: torch.Tensor,
    n: int,
    mu: int,
    temperature: float,
    mask_id: int,
) -> _DecoderState:
    shadow = state.clone()
    result = _saber_step(
        logits,
        shadow.x,
        mask_index,
        shadow.global_transfer_index,
        shadow.final_confidence,
        shadow.last_confidence,
        shadow.initial_confidence,
        block_start,
        block_end,
        prompt_index,
        n,
        mu,
        temperature,
        mask_id,
    )
    return _DecoderState(*result[:5])


def _rng_state(device: torch.device) -> tuple[torch.Tensor, List[torch.Tensor] | None]:
    cpu = torch.random.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    return cpu, cuda


def _restore_rng(state: tuple[torch.Tensor, List[torch.Tensor] | None]) -> None:
    cpu, cuda = state
    torch.random.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)


def compare_fresh_and_cached(
    cached_logits: torch.Tensor,
    fresh_logits: torch.Tensor,
    state: _DecoderState,
    *,
    mask_index: torch.Tensor,
    prompt_index: torch.Tensor,
    prompt_length: int,
    block_start: int,
    block_end: int,
    n: int,
    mu: int,
    temperature: float,
    mask_id: int,
) -> Dict[str, Any]:
    """Compare logits and one decoder action without mutating ``state`` or RNG."""
    active = torch.zeros_like(mask_index)
    active[:, block_start:block_end] = True
    eligible = active & ~state.global_transfer_index
    positions = int(eligible.sum().item())

    if positions:
        cached_active = cached_logits[eligible]
        fresh_active = fresh_logits[eligible]
        absolute_error = (cached_active - fresh_active).float().abs()
        logit_mae = float(absolute_error.mean().item())
        logit_max_error = float(absolute_error.max().item())
        top1_disagreement = float(
            (cached_active.argmax(-1) != fresh_active.argmax(-1)).float().mean().item()
        )
    else:
        logit_mae = logit_max_error = top1_disagreement = 0.0

    before = state.global_transfer_index
    before_unmasked = state.x != mask_id
    saved_rng = _rng_state(state.x.device)
    try:
        cached_next = _run_shadow_step(
            cached_logits,
            state,
            mask_index=mask_index,
            block_start=block_start,
            block_end=block_end,
            prompt_index=prompt_index,
            n=n,
            mu=mu,
            temperature=temperature,
            mask_id=mask_id,
        )
        _restore_rng(saved_rng)
        fresh_next = _run_shadow_step(
            fresh_logits,
            state,
            mask_index=mask_index,
            block_start=block_start,
            block_end=block_end,
            prompt_index=prompt_index,
            n=n,
            mu=mu,
            temperature=temperature,
            mask_id=mask_id,
        )
    finally:
        # The real ARCHER transition must observe exactly the RNG state it
        # would have seen if this diagnostic had not run.
        _restore_rng(saved_rng)

    cached_selected = cached_next.global_transfer_index & ~before
    fresh_selected = fresh_next.global_transfer_index & ~before
    cached_removed = before_unmasked & (cached_next.x == mask_id)
    fresh_removed = before_unmasked & (fresh_next.x == mask_id)
    generation = slice(prompt_length, None)
    state_hamming = int(
        torch.count_nonzero(
            cached_next.x[:, generation] != fresh_next.x[:, generation]
        ).item()
    )
    return {
        "compared_positions": positions,
        "logit_mae": logit_mae,
        "logit_max_error": logit_max_error,
        "top1_disagreement_rate": top1_disagreement,
        "next_state_hamming": state_hamming,
        "next_action_disagrees": bool(
            state_hamming
            or not torch.equal(cached_selected, fresh_selected)
            or not torch.equal(cached_removed, fresh_removed)
        ),
        "selected_set_disagrees": not torch.equal(cached_selected, fresh_selected),
        "rollback_set_disagrees": not torch.equal(cached_removed, fresh_removed),
    }


@torch.no_grad()
def generate_archer_with_validity_probe(
    model,
    prompt: torch.Tensor,
    n: int = 2,
    mu: int = 2,
    gen_length: int = 256,
    block_length: int = 256,
    temperature: float = 0.0,
    mask_id: int = 126336,
    refresh_radius: int = 11,
    probe_every: int = 4,
    max_probes: int = 12,
):
    """Run ARCHER and attach non-interventional fresh-forward observations.

    Returns ``(tokens, steps, main_nfe, diagnostics, probes, shadow_nfe)``.
    ``main_nfe`` remains directly comparable to :func:`generate_archer`.
    """
    if gen_length % block_length:
        raise ValueError("gen_length must be divisible by block_length")
    if refresh_radius <= 0:
        raise ValueError("refresh_radius must be positive")
    if probe_every <= 0 or max_probes < 0:
        raise ValueError("probe_every must be positive and max_probes non-negative")

    step = main_nfe = shadow_nfe = cached_step_number = 0
    probes: List[Dict[str, Any]] = []
    diagnostics = ArcherDiagnostics()
    prompt_length = prompt.shape[1]
    x = torch.full(
        (1, prompt_length + gen_length), mask_id, dtype=torch.long, device=model.device
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
        main_nfe += 1
        diagnostics.cache_builds += 1
        generation_snapshot = x[:, prompt_length:].clone()
        has_sampled_since_build = False
        cache_age = 0

        for _ in range(1024):
            step += 1
            distance = generation_hamming_distance(
                x[:, prompt_length:], generation_snapshot
            )
            diagnostics.record_distance(distance)
            refreshed = has_sampled_since_build and distance >= refresh_radius
            if refreshed:
                output, prefix_kv = build_prefix_cache(model, x, block_start)
                main_nfe += 1
                diagnostics.cache_builds += 1
                diagnostics.record_refresh(distance)
                generation_snapshot = x[:, prompt_length:].clone()
                logits = output.logits
                cache_age = 0
            else:
                logits = cached_forward_logits(
                    model, x, prefix_kv, output, block_start, block_end
                )
                main_nfe += 1
                diagnostics.cached_steps += 1
                should_probe = (
                    len(probes) < max_probes
                    and cached_step_number % probe_every == 0
                )
                cached_step_number += 1
                if should_probe:
                    fresh_logits = model(x).logits
                    shadow_nfe += 1
                    decoder_state = _DecoderState(
                        x,
                        global_transfer_index,
                        initial_confidence,
                        final_confidence,
                        last_confidence,
                    )
                    comparison = compare_fresh_and_cached(
                        logits,
                        fresh_logits,
                        decoder_state,
                        mask_index=mask_index,
                        prompt_index=prompt_index,
                        prompt_length=prompt_length,
                        block_start=block_start,
                        block_end=block_end,
                        n=n,
                        mu=mu,
                        temperature=temperature,
                        mask_id=mask_id,
                    )
                    probes.append(
                        {
                            "step": step,
                            "block": block_number,
                            "cache_age": cache_age,
                            "anchor_distance": distance,
                            **comparison,
                        }
                    )
                    del fresh_logits

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
            cache_age += 1
            if all_done:
                return (
                    x,
                    step,
                    main_nfe,
                    diagnostics.as_dict(),
                    probes,
                    shadow_nfe,
                )
            if block_done:
                break
            has_sampled_since_build = True

    return x, step, main_nfe, diagnostics.as_dict(), probes, shadow_nfe


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        for index in order[start:end]:
            ranks[index] = average_rank
        start = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean, right_mean = sum(left) / len(left), sum(right) / len(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_delta)
        * sum(value * value for value in right_delta)
    )
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(left_delta, right_delta)) / denominator


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    return _pearson(_ranks(left), _ranks(right))


def _partial(first: float | None, second: float | None, controls: float | None) -> float | None:
    if first is None or second is None or controls is None:
        return None
    denominator = math.sqrt((1 - second * second) * (1 - controls * controls))
    return (first - second * controls) / denominator if denominator > 0 else None


def _bucket_summary(probes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "count": len(probes),
        "mean_logit_mae": _mean([float(item["logit_mae"]) for item in probes]),
        "mean_top1_disagreement_rate": _mean(
            [float(item["top1_disagreement_rate"]) for item in probes]
        ),
        "next_action_disagreement_rate": _mean(
            [float(bool(item["next_action_disagrees"])) for item in probes]
        ),
        "mean_next_state_hamming": _mean(
            [float(item["next_state_hamming"]) for item in probes]
        ),
    }


def summarize_validity_probes(
    probes: Iterable[Mapping[str, Any]], refresh_radius: int
) -> Dict[str, Any]:
    """Aggregate probe rows and compare distance against cache age."""
    probes = list(probes)
    if not probes:
        return {"probe_count": 0}
    distance = [float(item["anchor_distance"]) for item in probes]
    age = [float(item["cache_age"]) for item in probes]
    distance_age = _spearman(distance, age)

    correlations: Dict[str, Any] = {}
    for metric in (
        "logit_mae",
        "top1_disagreement_rate",
        "next_state_hamming",
        "next_action_disagrees",
    ):
        outcome = [float(item[metric]) for item in probes]
        distance_outcome = _spearman(distance, outcome)
        age_outcome = _spearman(age, outcome)
        correlations[metric] = {
            "distance_spearman": distance_outcome,
            "age_spearman": age_outcome,
            "distance_partial_controlling_age": _partial(
                distance_outcome, distance_age, age_outcome
            ),
            "age_partial_controlling_distance": _partial(
                age_outcome, distance_age, distance_outcome
            ),
        }

    width = max(1, math.ceil(refresh_radius / 4))
    by_distance: Dict[str, Any] = {}
    for lower in range(0, refresh_radius, width):
        upper = min(refresh_radius - 1, lower + width - 1)
        selected = [
            item for item in probes
            if lower <= int(item["anchor_distance"]) <= upper
        ]
        by_distance[f"{lower}-{upper}"] = _bucket_summary(selected)

    normalized_edges = (0.0, 0.25, 0.5, 0.75, 1.0)
    by_normalized_distance: Dict[str, Any] = {}
    for lower, upper in zip(normalized_edges, normalized_edges[1:]):
        selected = [
            item
            for item in probes
            if lower
            <= float(item["anchor_distance"]) / refresh_radius
            < upper
        ]
        by_normalized_distance[f"{lower:.2f}-{upper:.2f}"] = _bucket_summary(
            selected
        )

    max_age = max(int(item["cache_age"]) for item in probes)
    age_edges = (0, 2, 5, 9)
    by_age: Dict[str, Any] = {}
    for index, lower in enumerate(age_edges):
        upper = age_edges[index + 1] - 1 if index + 1 < len(age_edges) else max_age
        if upper < lower:
            continue
        selected = [item for item in probes if lower <= int(item["cache_age"]) <= upper]
        by_age[f"{lower}-{upper}"] = _bucket_summary(selected)

    return {
        "probe_count": len(probes),
        "overall": _bucket_summary(probes),
        "distance_age_spearman": distance_age,
        "correlations": correlations,
        "by_anchor_distance": by_distance,
        "by_normalized_anchor_distance": by_normalized_distance,
        "by_cache_age": by_age,
    }
