"""Paired intervention for measuring prompt-feedback effects in ARCHER.

The probe follows ordinary ARCHER decoding until the first eligible
high-confidence commit.  It then forks from exactly the same post-commit
generation state:

* ``fresh`` immediately rebuilds the prompt prefix cache;
* ``cached`` keeps the pre-commit prompt prefix cache.

Both branches subsequently use the same state-distance refresh rule.  This
isolates whether feeding a newly committed token back through the prompt
representation changes its next-step support and later revisability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch

from .saber import _saber_step
from .state_distance.core import (
    build_prefix_cache,
    cached_forward_logits,
    generation_hamming_distance,
)


@dataclass
class _Branch:
    x: torch.Tensor
    global_transfer_index: torch.Tensor
    initial_confidence: torch.Tensor
    final_confidence: torch.Tensor
    last_confidence: torch.Tensor
    prefix_output: Any
    prefix_kv: Any
    generation_snapshot: torch.Tensor
    steps: int
    nfe: int


def _clone_branch(
    x: torch.Tensor,
    global_transfer_index: torch.Tensor,
    initial_confidence: torch.Tensor,
    final_confidence: torch.Tensor,
    last_confidence: torch.Tensor,
    prefix_output: Any,
    prefix_kv: Any,
    generation_snapshot: torch.Tensor,
    steps: int,
    nfe: int,
) -> _Branch:
    return _Branch(
        x=x.clone(),
        global_transfer_index=global_transfer_index.clone(),
        initial_confidence=initial_confidence.clone(),
        final_confidence=final_confidence.clone(),
        last_confidence=last_confidence.clone(),
        prefix_output=prefix_output,
        prefix_kv=prefix_kv,
        generation_snapshot=generation_snapshot.clone(),
        steps=steps,
        nfe=nfe,
    )


def _token_margin(logits: torch.Tensor, position: int, token_id: int) -> float:
    row = logits[0, position]
    target = row[token_id]
    competitor = torch.cat((row[:token_id], row[token_id + 1 :])).max()
    return float((target - competitor).float().item())


def _token_probability(logits: torch.Tensor, position: int, token_id: int) -> float:
    return float(torch.softmax(logits[0, position].float(), dim=-1)[token_id].item())


def _advance_branch(
    model,
    branch: _Branch,
    *,
    logits: torch.Tensor | None,
    prompt_length: int,
    block_start: int,
    block_end: int,
    mask_index: torch.Tensor,
    prompt_index: torch.Tensor,
    refresh_radius: int,
    n: int,
    mu: int,
    temperature: float,
    mask_id: int,
):
    if logits is None:
        distance = generation_hamming_distance(
            branch.x[:, prompt_length:], branch.generation_snapshot
        )
        if distance >= refresh_radius:
            branch.prefix_output, branch.prefix_kv = build_prefix_cache(
                model, branch.x, block_start
            )
            branch.generation_snapshot = branch.x[:, prompt_length:].clone()
            logits = branch.prefix_output.logits
        else:
            logits = cached_forward_logits(
                model,
                branch.x,
                branch.prefix_kv,
                branch.prefix_output,
                block_start,
                block_end,
            )
        branch.nfe += 1

    state = _saber_step(
        logits,
        branch.x,
        mask_index,
        branch.global_transfer_index,
        branch.final_confidence,
        branch.last_confidence,
        branch.initial_confidence,
        block_start,
        block_end,
        prompt_index,
        n,
        mu,
        temperature,
        mask_id,
    )
    (
        branch.x,
        branch.global_transfer_index,
        branch.initial_confidence,
        branch.final_confidence,
        branch.last_confidence,
        all_done,
        block_done,
        _,
        _,
    ) = state
    branch.steps += 1
    return bool(all_done), bool(block_done)


def _finish_branch(
    model,
    branch: _Branch,
    *,
    first_logits: torch.Tensor,
    target_position: int,
    target_token: int,
    horizon: int,
    prompt_length: int,
    block_start: int,
    block_end: int,
    mask_index: torch.Tensor,
    prompt_index: torch.Tensor,
    refresh_radius: int,
    n: int,
    mu: int,
    temperature: float,
    mask_id: int,
) -> Dict[str, Any]:
    revised_at = None
    continuation_step = 0
    logits = first_logits
    all_done = block_done = False
    while not (all_done or block_done):
        continuation_step += 1
        all_done, block_done = _advance_branch(
            model,
            branch,
            logits=logits,
            prompt_length=prompt_length,
            block_start=block_start,
            block_end=block_end,
            mask_index=mask_index,
            prompt_index=prompt_index,
            refresh_radius=refresh_radius,
            n=n,
            mu=mu,
            temperature=temperature,
            mask_id=mask_id,
        )
        logits = None
        if revised_at is None and int(branch.x[0, target_position]) != target_token:
            revised_at = continuation_step
        if branch.steps >= 1024:
            break

    return {
        "tokens": branch.x,
        "steps": branch.steps,
        "nfe": branch.nfe,
        "revised_within_horizon": revised_at is not None and revised_at <= horizon,
        "revision_step": revised_at,
    }


@torch.no_grad()
def generate_prompt_feedback_probe(
    model,
    prompt: torch.Tensor,
    n: int = 2,
    mu: int = 2,
    gen_length: int = 256,
    block_length: int = 256,
    temperature: float = 0.0,
    mask_id: int = 126336,
    refresh_radius: int = 11,
    confidence_threshold: float = 0.9,
    horizon: int = 5,
):
    """Generate paired fresh/cached continuations from one controlled event.

    One event is selected per problem: the first retained commit whose model
    confidence reaches ``confidence_threshold`` and whose post-commit state is
    still inside the current ARCHER refresh radius.  Problems without such an
    event return the ordinary ARCHER completion in both branches and set
    ``eligible`` to ``False``.
    """
    if gen_length != block_length:
        raise ValueError("the feedback probe currently requires one generation block")
    if refresh_radius <= 0:
        raise ValueError("refresh_radius must be positive")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    prompt_length = prompt.shape[1]
    block_start = prompt_length
    block_end = prompt_length + gen_length
    x = torch.full(
        (1, block_end), mask_id, dtype=torch.long, device=model.device
    )
    x[:, :prompt_length] = prompt.clone()
    prompt_index = x != mask_id
    mask_index = x == mask_id
    global_transfer_index = torch.zeros_like(x, dtype=torch.bool)
    initial_confidence = torch.zeros_like(x, dtype=torch.float32)
    final_confidence = torch.full_like(x, float("-inf"), dtype=torch.float32)
    last_confidence = torch.zeros_like(x, dtype=torch.float32)

    prefix_output, prefix_kv = build_prefix_cache(model, x, block_start)
    generation_snapshot = x[:, prompt_length:].clone()
    steps = 0
    nfe = 1

    while steps < 1024:
        distance = generation_hamming_distance(
            x[:, prompt_length:], generation_snapshot
        )
        if steps > 0 and distance >= refresh_radius:
            prefix_output, prefix_kv = build_prefix_cache(model, x, block_start)
            generation_snapshot = x[:, prompt_length:].clone()
            logits = prefix_output.logits
        else:
            logits = cached_forward_logits(
                model, x, prefix_kv, prefix_output, block_start, block_end
            )
        nfe += 1

        before_x = x.clone()
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
        steps += 1

        newly_committed = (
            (before_x == mask_id)
            & (x != mask_id)
            & (~prompt_index)
        )
        post_distance = generation_hamming_distance(
            x[:, prompt_length:], generation_snapshot
        )
        eligible = newly_committed & (final_confidence >= confidence_threshold)
        can_fork = bool(eligible.any()) and post_distance < refresh_radius
        if can_fork and not (all_done or block_done):
            positions = torch.where(eligible[0])[0]
            scores = final_confidence[0, positions]
            position = int(positions[torch.argmax(scores)].item())
            token_id = int(x[0, position].item())
            commit_confidence = float(final_confidence[0, position].item())

            cached = _clone_branch(
                x,
                global_transfer_index,
                initial_confidence,
                final_confidence,
                last_confidence,
                prefix_output,
                prefix_kv,
                generation_snapshot,
                steps,
                nfe,
            )
            fresh = _clone_branch(
                x,
                global_transfer_index,
                initial_confidence,
                final_confidence,
                last_confidence,
                prefix_output,
                prefix_kv,
                generation_snapshot,
                steps,
                nfe,
            )

            cached_logits = cached_forward_logits(
                model, cached.x, cached.prefix_kv, cached.prefix_output,
                block_start, block_end,
            )
            cached.nfe += 1
            fresh.prefix_output, fresh.prefix_kv = build_prefix_cache(
                model, fresh.x, block_start
            )
            fresh.generation_snapshot = fresh.x[:, prompt_length:].clone()
            fresh_logits = fresh.prefix_output.logits
            fresh.nfe += 1

            common = {
                "target_position": position,
                "target_token": token_id,
                "horizon": horizon,
                "prompt_length": prompt_length,
                "block_start": block_start,
                "block_end": block_end,
                "mask_index": mask_index,
                "prompt_index": prompt_index,
                "refresh_radius": refresh_radius,
                "n": n,
                "mu": mu,
                "temperature": temperature,
                "mask_id": mask_id,
            }
            cached_result = _finish_branch(
                model, cached, first_logits=cached_logits, **common
            )
            fresh_result = _finish_branch(
                model, fresh, first_logits=fresh_logits, **common
            )
            diagnostics = {
                "eligible": True,
                "event_step": steps,
                "target_position": position - prompt_length,
                "target_token": token_id,
                "commit_confidence": commit_confidence,
                "anchor_distance": post_distance,
                "fresh_margin": _token_margin(fresh_logits, position, token_id),
                "cached_margin": _token_margin(cached_logits, position, token_id),
                "fresh_probability": _token_probability(
                    fresh_logits, position, token_id
                ),
                "cached_probability": _token_probability(
                    cached_logits, position, token_id
                ),
                "fresh_revised_within_horizon": fresh_result[
                    "revised_within_horizon"
                ],
                "cached_revised_within_horizon": cached_result[
                    "revised_within_horizon"
                ],
                "fresh_revision_step": fresh_result["revision_step"],
                "cached_revision_step": cached_result["revision_step"],
            }
            return fresh_result, cached_result, diagnostics

        if all_done or block_done:
            break

    fallback = {"tokens": x, "steps": steps, "nfe": nfe,
                "revised_within_horizon": False, "revision_step": None}
    return fallback, {**fallback, "tokens": x.clone()}, {"eligible": False}
