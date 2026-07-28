"""Vanilla LLaDA confidence-based decoding baseline.

The implementation follows the official ``ML-GSAI/LLaDA`` generation
algorithm: a linear transfer schedule, low-confidence remasking, and a full
bidirectional model forward at every denoising step.  It intentionally uses
neither rollback nor KV caching.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .saber import add_gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Distribute initially masked tokens over a linear denoising schedule."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    mask_count = mask_index.sum(dim=1, keepdim=True)
    base = mask_count // steps
    remainder = mask_count % steps
    transfers = torch.zeros(
        mask_count.shape[0],
        steps,
        dtype=torch.int64,
        device=mask_index.device,
    )
    transfers += base
    for batch_index in range(mask_count.shape[0]):
        transfers[batch_index, : int(remainder[batch_index].item())] += 1
    return transfers


@torch.no_grad()
def generate_llada(
    model,
    prompt: torch.Tensor,
    *,
    steps: int = 256,
    gen_length: int = 256,
    block_length: int = 256,
    temperature: float = 0.0,
    mask_id: int = 126336,
):
    """Generate with the official vanilla low-confidence LLaDA sampler."""
    if gen_length <= 0 or block_length <= 0:
        raise ValueError("gen_length and block_length must be positive")
    if gen_length % block_length:
        raise ValueError("gen_length must be divisible by block_length")

    num_blocks = gen_length // block_length
    if steps <= 0 or steps % num_blocks:
        raise ValueError("steps must be positive and divisible by the number of blocks")
    steps_per_block = steps // num_blocks

    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, : prompt.shape[1]] = prompt.clone()
    prompt_length = prompt.shape[1]
    completed_steps = 0

    for block_number in range(num_blocks):
        block_start = prompt_length + block_number * block_length
        block_end = block_start + block_length
        initial_block_mask = x[:, block_start:block_end] == mask_id
        transfer_schedule = get_num_transfer_tokens(
            initial_block_mask, steps_per_block
        )

        for step_index in range(steps_per_block):
            completed_steps += 1
            mask_index = x == mask_id
            logits = model(x).logits
            noisy_logits = add_gumbel_noise(logits, temperature)
            predictions = torch.argmax(noisy_logits, dim=-1)

            probabilities = F.softmax(logits, dim=-1)
            confidence = torch.gather(
                probabilities, -1, predictions.unsqueeze(-1)
            ).squeeze(-1)
            confidence = torch.where(mask_index, confidence, -np.inf)
            confidence[:, :block_start] = -np.inf
            if block_end < confidence.shape[1]:
                confidence[:, block_end:] = -np.inf

            transfer_index = torch.zeros_like(x, dtype=torch.bool)
            for batch_index in range(x.shape[0]):
                count = int(transfer_schedule[batch_index, step_index].item())
                if count:
                    _, selected = torch.topk(confidence[batch_index], k=count)
                    transfer_index[batch_index, selected] = True
            x[transfer_index] = predictions[transfer_index]

    return x, completed_steps
