"""Prompt-cache primitives for state-anchored ARCHER."""

from __future__ import annotations

from types import SimpleNamespace

import torch


def _forward_with_kv(model, input_ids: torch.Tensor, past_key_values=None):
    """Run a cache-capable forward pass under the project's KV interface.

    Most DLM implementations, including the vendored LLaDA model, expose a
    conventional ``(logits, past_key_values)`` API.  Dream-v0 instead mutates
    HuggingFace's :class:`DynamicCache` internally but drops that cache from
    its public ``MaskedLMOutput``.  This adapter restores the same observable
    contract without changing either model implementation.
    """
    if getattr(getattr(model, "config", None), "model_type", None) != "Dream":
        return model(
            input_ids, past_key_values=past_key_values, use_cache=True
        )

    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    if past_key_values is not None:
        cache.key_cache = [key for key, _ in past_key_values]
        cache.value_cache = [value for _, value in past_key_values]
    outputs = model.model(
        input_ids=input_ids, past_key_values=cache, use_cache=True
    )
    return SimpleNamespace(
        logits=model.lm_head(outputs.last_hidden_state),
        past_key_values=list(zip(cache.key_cache, cache.value_cache)),
    )


def generation_hamming_distance(
    current: torch.Tensor, snapshot: torch.Tensor
) -> int:
    """Return token-wise Hamming distance between two generation states."""
    if current.shape != snapshot.shape:
        raise ValueError(
            "current generation state and cache snapshot must have the same shape"
        )
    return int(torch.count_nonzero(current != snapshot).item())


def build_prefix_cache(model, x: torch.Tensor, block_start: int):
    """Run a full forward and retain only prefix K/V for every layer."""
    output = _forward_with_kv(model, x)
    prefix_kv = [
        tuple(tensor[:, :, :block_start] for tensor in layer)
        for layer in output.past_key_values
    ]
    return output, prefix_kv


def cached_forward_logits(
    model,
    x: torch.Tensor,
    prefix_kv,
    prefix_output,
    block_start: int,
    block_end: int,
):
    """Recompute the current generation suffix and stitch full-length logits."""
    block_logits = _forward_with_kv(
        model, x[:, block_start:], past_key_values=prefix_kv
    ).logits
    logits = torch.zeros(
        x.shape[0],
        x.shape[1],
        block_logits.shape[-1],
        device=x.device,
        dtype=block_logits.dtype,
    )
    logits[:, :block_start] = prefix_output.logits[:, :block_start]
    logits[:, block_start:block_end] = block_logits[
        :, : block_end - block_start
    ]
    return logits
