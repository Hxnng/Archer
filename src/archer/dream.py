"""Dream-family Saber and ARCHER samplers.

Dream predicts token ``i`` at logit position ``i - 1`` and uses a
model-specific Saber transition.  Reusing the LLaDA transition without this
alignment makes the model generate EOS/control tokens almost exclusively.

The transition below follows ``zhaoyMa/Saber/Dream/generation_utils.py`` at
revision ``a5433184cd627d9a0e31f26981736eca72802e47``.  ARCHER changes only
the forward path: it reuses an anchored prompt cache while preserving the
same token-selection and backtracking rule.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .state_distance.core import (
    build_prefix_cache,
    cached_forward_logits,
    generation_hamming_distance,
)
from .state_distance.sampler import ArcherDiagnostics


def is_dream_family(model) -> bool:
    return getattr(getattr(model, "config", None), "model_type", None) == "Dream"


def align_dream_logits(logits: torch.Tensor) -> torch.Tensor:
    """Align next-token Dream logits with the positions being denoised."""
    return torch.cat((logits[:, :1], logits[:, :-1]), dim=1)


def _dream_saber_step(
    logits: torch.Tensor,
    x: torch.Tensor,
    prompt_index: torch.Tensor,
    first_confidence: torch.Tensor,
    mask_id: int,
    min_unmask: int = 2,
    remask_divisor: int = 8,
):
    """Apply one official Dream-specific Saber unmask/backtrack transition."""
    mask_index = x == mask_id
    probabilities = F.softmax(logits, dim=-1)
    confidence, candidates = probabilities.max(dim=-1)
    confidence = torch.where(mask_index, confidence, -torch.inf)
    candidates = torch.where(mask_index, candidates, x)

    selected = torch.zeros_like(mask_index)
    for batch_index in range(x.shape[0]):
        masked_positions = torch.where(
            mask_index[batch_index] & ~prompt_index[batch_index]
        )[0]
        if masked_positions.numel() == 0:
            continue
        previous = first_confidence[batch_index]
        previous = previous[~torch.isnan(previous)]
        threshold = previous.mean() if previous.numel() else 1.0
        position_confidence = confidence[batch_index, masked_positions]
        chosen = masked_positions[position_confidence > threshold]
        if chosen.numel() < min_unmask:
            count = min(min_unmask, masked_positions.numel())
            top_indices = torch.topk(position_confidence, k=count).indices
            chosen = masked_positions[top_indices]
        selected[batch_index, chosen] = True

    newly_unmasked = selected & mask_index
    x[newly_unmasked] = candidates[newly_unmasked]
    first_time = torch.isnan(first_confidence) & newly_unmasked
    first_confidence[first_time] = confidence[first_time].float()

    if not (x == mask_id).any():
        return x, first_confidence, True, int(newly_unmasked.sum().item()), 0

    current_probabilities = F.softmax(logits, dim=-1)
    token_probabilities = current_probabilities.gather(
        -1, x.unsqueeze(-1)
    ).squeeze(-1)
    remasked = 0
    for batch_index in range(x.shape[0]):
        generated_positions = torch.where(
            (x[batch_index] != mask_id) & ~prompt_index[batch_index]
        )[0]
        if generated_positions.numel() <= 1:
            continue
        new_count = int(newly_unmasked[batch_index].sum().item())
        count = math.ceil(new_count / remask_divisor)
        count = max(1, min(count, new_count - 1))
        count = min(count, generated_positions.numel() - 1)
        current_confidence = token_probabilities[
            batch_index, generated_positions
        ]
        lowest = torch.topk(-current_confidence, k=count).indices
        positions = generated_positions[lowest]
        x[batch_index, positions] = mask_id
        remasked += int(positions.numel())

    return (
        x,
        first_confidence,
        False,
        int(newly_unmasked.sum().item()),
        remasked,
    )


def _initial_state(model, prompt: torch.Tensor, gen_length: int, mask_id: int):
    prompt_length = prompt.shape[1]
    x = torch.full(
        (1, prompt_length + gen_length),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, :prompt_length] = prompt
    prompt_index = torch.zeros_like(x, dtype=torch.bool)
    prompt_index[:, :prompt_length] = True
    first_confidence = torch.full(
        x.shape, float("nan"), dtype=torch.float32, device=x.device
    )
    return x, prompt_index, first_confidence


@torch.no_grad()
def generate_dream_saber(
    model,
    prompt: torch.Tensor,
    gen_length: int = 256,
    block_length: int = 256,
    temperature: float = 0.0,
    mask_id: int = 151666,
    **_: object,
):
    """Run the Saber authors' Dream-family transition."""
    if gen_length != block_length:
        raise ValueError("the official Dream Saber path requires one full block")
    if temperature != 0:
        raise ValueError("the paper-aligned Dream Saber path uses temperature=0")
    x, prompt_index, first_confidence = _initial_state(
        model, prompt, gen_length, mask_id
    )
    reported_steps = 0
    for _ in range(1024):
        logits = align_dream_logits(model(x, attention_mask="full").logits)
        x, first_confidence, done, _, _ = _dream_saber_step(
            logits, x, prompt_index, first_confidence, mask_id
        )
        if done:
            return x, reported_steps
        reported_steps += 1
    raise RuntimeError("Dream Saber did not finish within 1024 iterations")


@torch.no_grad()
def generate_dream_archer(
    model,
    prompt: torch.Tensor,
    gen_length: int = 256,
    block_length: int = 256,
    temperature: float = 0.0,
    mask_id: int = 151666,
    refresh_radius: int = 5,
    **_: object,
):
    """Apply ARCHER's anchored prompt cache to the official Dream transition."""
    if gen_length != block_length:
        raise ValueError("the Dream ARCHER path requires one full block")
    if temperature != 0:
        raise ValueError("the paper-aligned Dream path uses temperature=0")
    if refresh_radius < 0:
        raise ValueError("refresh_radius must be non-negative")

    x, prompt_index, first_confidence = _initial_state(
        model, prompt, gen_length, mask_id
    )
    prompt_length = prompt.shape[1]
    block_end = prompt_length + block_length
    output, prefix_kv = build_prefix_cache(model, x, prompt_length)
    snapshot = x[:, prompt_length:].clone()
    diagnostics = ArcherDiagnostics(cache_builds=1)
    nfe = 1
    reported_steps = 0
    has_sampled = False

    for _ in range(1024):
        distance = generation_hamming_distance(
            x[:, prompt_length:], snapshot
        )
        diagnostics.record_distance(distance)
        if has_sampled and distance >= refresh_radius:
            output, prefix_kv = build_prefix_cache(model, x, prompt_length)
            nfe += 1
            diagnostics.cache_builds += 1
            diagnostics.record_refresh(distance)
            snapshot = x[:, prompt_length:].clone()
            logits = output.logits
        else:
            logits = cached_forward_logits(
                model, x, prefix_kv, output, prompt_length, block_end
            )
            nfe += 1
            diagnostics.cached_steps += 1

        logits = align_dream_logits(logits)
        x, first_confidence, done, _, _ = _dream_saber_step(
            logits, x, prompt_index, first_confidence, mask_id
        )
        if done:
            return x, reported_steps, nfe, diagnostics.as_dict()
        reported_steps += 1
        has_sampled = True
    raise RuntimeError("Dream ARCHER did not finish within 1024 iterations")
