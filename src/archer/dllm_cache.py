"""Thin adapter for the official dLLM-Cache implementation.

The feature-cache hooks and sampler are imported from a pinned checkout of the
authors' repository.  ARCHER supplies only the common benchmark protocol.

Official source: https://github.com/maomaocun/dLLM-Cache
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Tuple


DLLM_CACHE_SOURCE_REVISION = "17235bffc8c5b587a2dc6f7dc76fcd01eab76e3a"


@lru_cache(maxsize=1)
def _load_official_components(
    repo_path: str,
) -> Tuple[type, Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    root = Path(repo_path).resolve()
    required = (
        root / "dllm_cache" / "cache" / "Cache.py",
        root / "dllm_cache" / "hooks" / "cache_hook_LLaDA.py",
        root / "utils" / "generate_function.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "official dLLM-Cache checkout is incomplete; missing: "
            + ", ".join(missing)
        )

    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    cache_module = importlib.import_module("dllm_cache.cache")
    hook_module = importlib.import_module("dllm_cache.hooks")
    generation_module = importlib.import_module("utils.generate_function")
    return (
        cache_module.dLLMCache,
        cache_module.dLLMCacheConfig,
        hook_module.register_cache_LLaDA,
        generation_module.generate,
    )


def load_dllm_cache_model(
    repo_path: str | Path,
    model_path: str | Path,
    torch: Any,
    *,
    prompt_interval_steps: int,
    gen_interval_steps: int,
    transfer_ratio: float,
) -> Any:
    """Load the stock LLaDA model and attach the authors' feature-cache hooks."""
    from transformers import AutoModel

    cache_class, config_class, register_cache, _ = _load_official_components(
        str(Path(repo_path).resolve())
    )
    cache_class.new_instance(
        **asdict(
            config_class(
                prompt_interval_steps=prompt_interval_steps,
                gen_interval_steps=gen_interval_steps,
                transfer_ratio=transfer_ratio,
            )
        )
    )
    model = (
        AutoModel.from_pretrained(
            str(Path(model_path).resolve()),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        .cuda()
        .eval()
    )
    register_cache(model, "model.transformer.blocks")
    return model


def generate_dllm_cache(
    model: Any,
    input_ids: Any,
    *,
    repo_path: str | Path,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float = 0.0,
) -> Tuple[Any, int]:
    """Run the official sampler and restore the prompt prefix for the runner."""
    if steps <= 0 or gen_length <= 0 or block_length <= 0:
        raise ValueError("steps, gen_length, and block_length must be positive")
    if gen_length % block_length:
        raise ValueError("gen_length must be divisible by block_length")
    if steps % (gen_length // block_length):
        raise ValueError("steps must be divisible by the number of blocks")

    _, _, _, official_generate = _load_official_components(
        str(Path(repo_path).resolve())
    )
    attention_mask = input_ids.new_ones(input_ids.shape)
    response = official_generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        model=model,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        cfg_scale=0.0,
        remasking="low_confidence",
    )
    return __import__("torch").cat([input_ids, response], dim=1), int(steps)
