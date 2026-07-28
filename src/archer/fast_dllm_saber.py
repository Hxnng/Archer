# Copyright 2025 Saber authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Fast-dLLM baseline exactly as configured by the Saber main experiment.

Source: ``zhaoyMa/Saber`` commit
``a5433184cd627d9a0e31f26981736eca72802e47``.  Saber evaluates the
DualCache variant with factor-based parallel decoding (factor=1), a 256-token
generation block, and temperature zero.  This module intentionally keeps that
configuration separate from the original Fast-dLLM PrefixCache/threshold
configuration in :mod:`archer.fast_dllm_v1`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .saber import add_gumbel_noise


SABER_FAST_DLLM_SOURCE_REVISION = "a5433184cd627d9a0e31f26981736eca72802e47"


def select_by_factor(
    logits: torch.Tensor,
    mask_index: torch.Tensor,
    current_tokens: torch.Tensor,
    *,
    factor: float = 1.0,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Saber's vendored Fast-dLLM factor-based transfer rule."""
    if factor <= 0:
        raise ValueError("factor must be positive")

    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    predictions = torch.argmax(logits_with_noise, dim=-1)
    if remasking == "low_confidence":
        probabilities = F.softmax(logits.to(torch.float64), dim=-1)
        confidence = torch.gather(
            probabilities, dim=-1, index=predictions.unsqueeze(-1)
        ).squeeze(-1)
    elif remasking == "random":
        confidence = torch.rand(
            predictions.shape, device=predictions.device, dtype=torch.float64
        )
    else:
        raise NotImplementedError(remasking)

    predictions = torch.where(mask_index, predictions, current_tokens)
    confidence = torch.where(mask_index, confidence, -torch.inf)
    transfer_index = torch.zeros_like(mask_index)

    for batch_index in range(confidence.shape[0]):
        masked_count = int(mask_index[batch_index].sum().item())
        if masked_count == 0:
            continue
        thresholds = [1.0 - factor / (rank + 1) for rank in range(1, masked_count + 1)]
        thresholds[0] = -1.0  # always transfer at least one token
        sorted_confidence = torch.sort(
            confidence[batch_index][mask_index[batch_index]], descending=True
        ).values

        top_index = 0
        for top_index, threshold in enumerate(thresholds):
            if sorted_confidence[top_index] < threshold:
                break
        if top_index == 0 or top_index == len(thresholds) - 1:
            top_index += 1

        selected = torch.topk(confidence[batch_index], k=top_index).indices
        transfer_index[batch_index, selected] = True

    return predictions, transfer_index


def select_one_by_confidence(
    logits: torch.Tensor,
    mask_index: torch.Tensor,
    current_tokens: torch.Tensor,
    *,
    temperature: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Saber's cache-only Fast-dLLM rule: commit one token per NFE."""
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    predictions = torch.argmax(logits_with_noise, dim=-1)
    probabilities = F.softmax(logits.to(torch.float64), dim=-1)
    confidence = torch.gather(
        probabilities, dim=-1, index=predictions.unsqueeze(-1)
    ).squeeze(-1)
    predictions = torch.where(mask_index, predictions, current_tokens)
    confidence = torch.where(mask_index, confidence, -torch.inf)
    transfer_index = torch.zeros_like(mask_index)
    for batch_index in range(confidence.shape[0]):
        if mask_index[batch_index].any():
            selected = torch.argmax(confidence[batch_index])
            transfer_index[batch_index, selected] = True
    return predictions, transfer_index


@torch.no_grad()
def generate_fast_dllm_saber(
    model,
    prompt: torch.Tensor,
    *,
    steps: int = 256,
    gen_length: int = 256,
    block_length: int = 256,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    factor: float = 1.0,
) -> tuple[torch.Tensor, int]:
    """Run the DualCache + factor-decoding Fast-dLLM used by Saber."""
    return _generate_fast_dllm_saber(
        model,
        prompt,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        remasking=remasking,
        mask_id=mask_id,
        factor=factor,
    )


@torch.no_grad()
def generate_fast_dllm_cache_only(
    model,
    prompt: torch.Tensor,
    *,
    steps: int = 256,
    gen_length: int = 256,
    block_length: int = 256,
    temperature: float = 0.0,
    mask_id: int = 126336,
) -> tuple[torch.Tensor, int]:
    """Run the cache-only Fast-dLLM row in Saber's main table."""
    return _generate_fast_dllm_saber(
        model,
        prompt,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        remasking="low_confidence",
        mask_id=mask_id,
        factor=None,
    )


def _generate_fast_dllm_saber(
    model,
    prompt: torch.Tensor,
    *,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float,
    remasking: str,
    mask_id: int,
    factor: float | None,
) -> tuple[torch.Tensor, int]:
    if prompt.ndim != 2:
        raise ValueError("prompt must have shape [batch, sequence]")
    if min(steps, gen_length, block_length) <= 0:
        raise ValueError("steps, gen_length, and block_length must be positive")
    if gen_length % block_length:
        raise ValueError("gen_length must be divisible by block_length")
    num_blocks = gen_length // block_length
    if steps % num_blocks:
        raise ValueError("steps must be divisible by the number of blocks")

    tokens = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=prompt.device,
    )
    tokens[:, : prompt.shape[1]] = prompt.clone()
    nfe = 0

    for block_number in range(num_blocks):
        block_start = prompt.shape[1] + block_number * block_length
        block_end = block_start + block_length

        output = model(tokens, use_cache=True)
        past_key_values = output.past_key_values
        if past_key_values is None:
            raise RuntimeError("model did not return past_key_values")

        mask_index = tokens == mask_id
        mask_index[:, block_end:] = False
        if factor is None:
            predictions, transfer_index = select_one_by_confidence(
                output.logits, mask_index, tokens, temperature=temperature
            )
        else:
            predictions, transfer_index = select_by_factor(
                output.logits,
                mask_index,
                tokens,
                factor=factor,
                temperature=temperature,
                remasking=remasking,
            )
        tokens[transfer_index] = predictions[transfer_index]
        nfe += 1

        replace_position = torch.zeros_like(tokens, dtype=torch.bool)
        replace_position[:, block_start:block_end] = True
        while (tokens[:, block_start:block_end] == mask_id).any():
            local_tokens = tokens[:, block_start:block_end]
            local_mask = local_tokens == mask_id
            logits = model(
                local_tokens,
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=replace_position,
            ).logits
            if factor is None:
                predictions, transfer_index = select_one_by_confidence(
                    logits, local_mask, local_tokens, temperature=temperature
                )
            else:
                predictions, transfer_index = select_by_factor(
                    logits,
                    local_mask,
                    local_tokens,
                    factor=factor,
                    temperature=temperature,
                    remasking=remasking,
                )
            local_tokens[transfer_index] = predictions[transfer_index]
            nfe += 1

    return tokens, nfe
