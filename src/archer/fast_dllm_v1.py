# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Thin wrapper around the official Fast-dLLM v1 LLaDA sampler.

The decoding logic below follows ``v1/llada/generate.py`` from NVLabs/Fast-dLLM
at commit ``a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b``.  Only unused decoding
variants were omitted; the Prefix Cache + Parallel path is intentionally kept
structurally aligned with the upstream implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


FAST_DLLM_V1_SOURCE_REVISION = "a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b"


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Official Fast-dLLM/LLaDA Gumbel-max transform."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(block_mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Distribute the current block's masked tokens over diffusion steps."""
    total = block_mask_index.sum(dim=1)
    base = torch.div(total, steps, rounding_mode="floor")
    remainder = total - base * steps
    num_transfer_tokens = base.unsqueeze(1).expand(-1, steps).to(torch.long)
    columns = torch.arange(steps, device=block_mask_index.device).unsqueeze(0)
    return num_transfer_tokens + (columns < remainder.unsqueeze(1)).to(torch.long)


def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    num_transfer_tokens: torch.Tensor | None,
    threshold: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official confidence-aware token transfer rule."""
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == "low_confidence":
        probabilities = F.softmax(logits.to(torch.float64), dim=-1)
        x0_probability = torch.gather(
            probabilities, dim=-1, index=x0.unsqueeze(-1)
        ).squeeze(-1)
    elif remasking == "random":
        x0_probability = torch.rand(
            x0.shape, device=x0.device, dtype=torch.float64
        )
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    negative_infinity = torch.tensor(
        torch.finfo(x0_probability.dtype).min,
        device=x0_probability.device,
        dtype=x0_probability.dtype,
    )
    confidence = torch.where(mask_index, x0_probability, negative_infinity)

    if threshold is not None:
        transfer_index = mask_index & (confidence >= threshold)
        max_confidence_index = torch.argmax(confidence, dim=1, keepdim=True)
        force_mask = torch.zeros_like(transfer_index).scatter_(
            1, max_confidence_index, True
        )
        transfer_index = (transfer_index | force_mask) & mask_index
        return x0, transfer_index

    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens is required when threshold is None")
    if num_transfer_tokens.ndim == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(
        dtype=torch.long, device=confidence.device
    ).clamp(min=0)

    order = torch.argsort(confidence, dim=1, descending=True)
    ranks = torch.argsort(order, dim=1)
    transfer_index = ranks < num_transfer_tokens.unsqueeze(1)
    return x0, transfer_index & mask_index


@torch.no_grad()
def generate_with_prefix_cache(
    model,
    prompt: torch.Tensor,
    *,
    steps: int = 8,
    gen_length: int = 256,
    block_length: int = 32,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    threshold: float | None = 0.9,
) -> tuple[torch.Tensor, int]:
    """Fast-dLLM v1 Prefix Cache + Confidence-aware Parallel Decoding."""
    if prompt.ndim != 2:
        raise ValueError("prompt must have shape [batch, sequence]")
    if gen_length <= 0 or block_length <= 0 or steps <= 0:
        raise ValueError("steps, gen_length, and block_length must be positive")
    if gen_length % block_length != 0:
        raise ValueError("gen_length must be divisible by block_length")
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise ValueError("steps must be divisible by the number of blocks")
    steps_per_block = steps // num_blocks

    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=prompt.device,
    )
    x[:, : prompt.shape[1]] = prompt.clone()
    nfe = 0

    for block_number in range(num_blocks):
        current_block_start = prompt.shape[1] + block_number * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = x[:, current_block_start:current_block_end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(
            block_mask_index, steps_per_block
        )

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        if past_key_values is None:
            raise RuntimeError("model did not return past_key_values")

        mask_index = x == mask_id
        mask_index[:, current_block_end:] = False
        x0, transfer_index = get_transfer_index(
            output.logits,
            temperature,
            remasking,
            mask_index,
            x,
            num_transfer_tokens[:, 0] if threshold is None else None,
            threshold,
        )
        x[transfer_index] = x0[transfer_index]

        prefix_cache = []
        for layer_cache in past_key_values:
            prefix_cache.append(
                tuple(
                    state[:, :, :current_block_start] for state in layer_cache
                )
            )
        nfe += 1

        step_index = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            local_x = x[:, current_block_start:]
            local_mask_index = local_x == mask_id
            local_mask_index[:, block_length:] = False

            logits = model(
                local_x,
                past_key_values=prefix_cache,
                use_cache=True,
            ).logits
            quota = None
            if threshold is None:
                if step_index >= steps_per_block:
                    raise RuntimeError("fixed-step decoding exhausted its block quota")
                quota = num_transfer_tokens[:, step_index]
            x0, transfer_index = get_transfer_index(
                logits,
                temperature,
                remasking,
                local_mask_index,
                local_x,
                quota,
                threshold,
            )
            local_x[transfer_index] = x0[transfer_index]
            step_index += 1

    return x, nfe


def select_confident_tokens(
    logits: torch.Tensor,
    mask_index: torch.Tensor,
    current_tokens: torch.Tensor,
    *,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility alias exposing the official threshold transfer rule."""
    return get_transfer_index(
        logits,
        temperature=0.0,
        remasking="low_confidence",
        mask_index=mask_index,
        x=current_tokens,
        num_transfer_tokens=None,
        threshold=threshold,
    )


@torch.no_grad()
def generate_fast_dllm_v1(
    model,
    prompt: torch.Tensor,
    *,
    gen_length: int = 256,
    block_length: int = 32,
    threshold: float = 0.9,
    mask_id: int = 126336,
) -> tuple[torch.Tensor, int]:
    """Paper configuration used by the unified ARCHER benchmark runner."""
    return generate_with_prefix_cache(
        model,
        prompt,
        steps=gen_length // block_length,
        gen_length=gen_length,
        block_length=block_length,
        temperature=0.0,
        remasking="low_confidence",
        mask_id=mask_id,
        threshold=threshold,
    )
