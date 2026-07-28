"""Minimal Saber baseline used by the ARCHER main experiment.

Extracted from ``zhaoyMa/Saber`` at revision
``a5433184cd627d9a0e31f26981736eca72802e47``. Only the published Saber
sampler and its Gumbel helper are retained.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _saber_step(
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
):
    """Apply one unmodified Saber transition to logits from any forward path."""
    x0 = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)
    probabilities = F.softmax(logits, dim=-1)
    x0_p = torch.gather(probabilities, -1, x0.unsqueeze(-1)).squeeze(-1)
    x0_p[:, block_end:] = -np.inf

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)
    confidence_delta = confidence - last_confidence
    confidence_delta = torch.where(
        torch.isfinite(confidence_delta),
        confidence_delta,
        torch.full_like(confidence_delta, float("inf")),
    )
    new_last_confidence = confidence.clone()
    confidence = torch.where(global_transfer_index, -np.inf, confidence)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
    fix_list = []
    for batch_index in range(confidence.shape[0]):
        generated = final_confidence[batch_index, :block_end]
        generated = generated[generated > -np.inf]
        threshold = generated.mean() if generated.numel() else 1.0
        selected = torch.where(confidence[batch_index] > threshold)[0]
        if selected.numel() < n:
            _, selected = torch.topk(confidence[batch_index], k=n)
        fix_list.append(max(n, selected.numel()))
        transfer_index[batch_index, selected] = True
        global_transfer_index[batch_index, selected] = True

    n_unmasked = transfer_index.sum().item()
    x[transfer_index] = x0[transfer_index]
    newly_generated = transfer_index & (initial_confidence == 0.0)
    if newly_generated.any():
        initial_confidence = torch.where(
            newly_generated, x0_p, initial_confidence
        )
    final_confidence = torch.where(transfer_index, x0_p, final_confidence)

    all_done = torch.all(global_transfer_index[mask_index]).item()
    block_done = torch.all(x[:, block_start:block_end] != mask_id)
    n_rolled_back = 0

    if not all_done and not block_done:
        removal_delta = torch.where(
            mask_index,
            confidence_delta.clone(),
            torch.full_like(confidence_delta, float("inf")),
        )
        removal_delta[:, :block_start] = float("inf")
        if block_end < removal_delta.shape[1]:
            removal_delta[:, block_end:] = float("inf")
        removal_delta[x == mask_id] = float("inf")
        removal_delta[prompt_index] = float("inf")

        for batch_index in range(removal_delta.shape[0]):
            count = max(int(n / 2), (fix_list[batch_index] + mu - 1) // mu)
            count = min(count, fix_list[batch_index] - 1)
            _, removed = torch.topk(
                removal_delta[batch_index], k=count, largest=False
            )
            valid = removed[(removed >= block_start) & (removed < block_end)]
            if valid.numel():
                x[batch_index, valid] = mask_id
                initial_confidence[batch_index, valid] = 0.0
                global_transfer_index[batch_index, valid] = False
                n_rolled_back += valid.numel()

    return (
        x,
        global_transfer_index,
        initial_confidence,
        final_confidence,
        new_last_confidence,
        all_done,
        block_done,
        n_unmasked,
        n_rolled_back,
    )


@torch.no_grad()
def generate_saber(
    model,
    prompt: torch.Tensor,
    n: int = 2,
    mu: int = 2,
    gen_length: int = 256,
    block_length: int = 256,
    temperature: float = 0.0,
    mask_id: int = 126336,
):
    """Run the upstream full-forward Saber sampler."""
    step = 0
    x = torch.full(
        (1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()
    prompt_index = x != mask_id
    mask_index = x == mask_id
    if gen_length % block_length:
        raise ValueError("gen_length must be divisible by block_length")

    global_transfer_index = torch.zeros_like(x, dtype=torch.bool)
    initial_confidence = torch.zeros_like(x, dtype=torch.float32)
    final_confidence = torch.full_like(x, -np.inf, dtype=torch.float32)
    last_confidence = torch.zeros_like(x, dtype=torch.float32)

    for block_number in range(gen_length // block_length):
        block_start = prompt.shape[1] + block_number * block_length
        block_end = block_start + block_length
        for _ in range(1024):
            step += 1
            logits = model(x).logits
            x0 = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)
            probabilities = F.softmax(logits, dim=-1)
            x0_p = torch.gather(probabilities, -1, x0.unsqueeze(-1)).squeeze(-1)
            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            confidence_delta = confidence - last_confidence
            confidence_delta = torch.where(
                torch.isfinite(confidence_delta),
                confidence_delta,
                torch.full_like(confidence_delta, float("inf")),
            )
            last_confidence = confidence
            confidence = torch.where(global_transfer_index, -np.inf, confidence)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            fix_list = []
            for batch_index in range(confidence.shape[0]):
                generated = final_confidence[batch_index, :block_end]
                generated = generated[generated > -np.inf]
                threshold = generated.mean() if generated.numel() else 1
                selected = torch.where(confidence[batch_index] > threshold)[0]
                if selected.numel() < n:
                    _, selected = torch.topk(confidence[batch_index], k=n)
                fix_list.append(max(n, selected.numel()))
                transfer_index[batch_index, selected] = True
                global_transfer_index[batch_index, selected] = True

            x[transfer_index] = x0[transfer_index]
            newly_generated = transfer_index & (initial_confidence == 0.0)
            if newly_generated.any():
                initial_confidence = torch.where(
                    newly_generated, x0_p, initial_confidence
                )
            final_confidence = torch.where(transfer_index, x0_p, final_confidence)

            if torch.all(global_transfer_index[mask_index]).item():
                return x, step
            if torch.all(x[:, :block_end] != mask_id):
                break

            removal_delta = torch.where(
                mask_index,
                confidence_delta.clone(),
                torch.full_like(confidence_delta, float("inf")),
            )
            if block_end < removal_delta.shape[1]:
                removal_delta[:, block_end:] = float("inf")
            removal_delta[x == mask_id] = float("inf")
            removal_delta[prompt_index] = float("inf")

            for batch_index in range(removal_delta.shape[0]):
                count = max(int(n / 2), (fix_list[batch_index] + mu - 1) // mu)
                count = min(count, fix_list[batch_index] - 1)
                _, removed = torch.topk(
                    removal_delta[batch_index], k=count, largest=False
                )
                x[batch_index, removed] = mask_id
                initial_confidence[batch_index, removed] = 0.0
                global_transfer_index[batch_index, removed] = False
    return x, step
