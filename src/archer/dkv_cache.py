"""Thin adapters for the official dKV-Cache LLaDA implementations.

The caching algorithm and modified LLaDA model are loaded directly from the
official repository instead of being reimplemented here.  This keeps the
baseline auditable while allowing ARCHER's dataset and evaluation protocol to
be reused unchanged.

Official source: https://github.com/horseee/dKV-Cache
"""

from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Tuple


DKV_CACHE_SOURCE_REVISION = "49a76fcc43b744ec2d960137f216e419317138b1"


@lru_cache(maxsize=1)
def _load_official_components(repo_path: str) -> Tuple[type, Callable[..., Any]]:
    """Load the model and decoder from a pinned checkout of the official repo."""
    root = Path(repo_path).resolve()
    required = (
        root / "models" / "modeling_llada_dkv_cache_decode.py",
        root / "generation_utils" / "llada_dkv_cache_decode.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "official dKV-Cache checkout is incomplete; missing: " + ", ".join(missing)
        )

    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    model_module = importlib.import_module("models.modeling_llada_dkv_cache_decode")
    generation_module = importlib.import_module(
        "generation_utils.llada_dkv_cache_decode"
    )
    return model_module.LLaDAModelLM, generation_module.generate


@lru_cache(maxsize=1)
def _load_official_greedy_components(
    repo_path: str,
) -> Tuple[type, Callable[..., Any], Callable[[int], None]]:
    """Load the official dKV-Cache-Greedy model, decoder, and seed helper."""
    root = Path(repo_path).resolve()
    required = (
        root / "models" / "modeling_llada_dkv_cache_greedy.py",
        root / "generation_utils" / "llada_dkv_cache_greedy.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "official dKV-Cache checkout is incomplete; missing: " + ", ".join(missing)
        )

    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    model_module = importlib.import_module("models.modeling_llada_dkv_cache_greedy")
    generation_module = importlib.import_module(
        "generation_utils.llada_dkv_cache_greedy"
    )
    return (
        model_module.LLaDAModelLM,
        generation_module.generate,
        generation_module.set_random_seed,
    )


def load_dkv_cache_model(repo_path: str | Path, model_path: str | Path, torch: Any) -> Any:
    """Load LLaDA weights into the official cache-aware model implementation."""
    model_class, _ = _load_official_components(str(Path(repo_path).resolve()))
    return (
        model_class.from_pretrained(
            str(Path(model_path).resolve()),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        .cuda()
        .eval()
    )


def load_dkv_cache_greedy_model(
    repo_path: str | Path, model_path: str | Path, torch: Any
) -> Any:
    """Load LLaDA weights into the official Greedy cache model."""
    model_class, _, _ = _load_official_greedy_components(
        str(Path(repo_path).resolve())
    )
    return (
        model_class.from_pretrained(
            str(Path(model_path).resolve()),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        .cuda()
        .eval()
    )


def generate_dkv_cache_decode(
    model: Any,
    input_ids: Any,
    *,
    repo_path: str | Path,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float = 0.0,
    cache_reloading_step: int = 8,
) -> Tuple[Any, int]:
    """Run official dKV-Cache-Decode and return tokens plus network evaluations."""
    if cache_reloading_step <= 0:
        raise ValueError("cache_reloading_step must be positive")
    if steps <= 0 or gen_length <= 0 or block_length <= 0:
        raise ValueError("steps, gen_length, and block_length must be positive")
    if gen_length % block_length:
        raise ValueError("gen_length must be divisible by block_length")

    _, official_generate = _load_official_components(
        str(Path(repo_path).resolve())
    )
    tokens = official_generate(
        model,
        None,
        input_ids,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        cfg_scale=0.0,
        remasking="low_confidence",
        enable_cache=True,
        cache_reloading_step=cache_reloading_step,
    )
    return tokens, int(steps)


def generate_dkv_cache_greedy(
    model: Any,
    input_ids: Any,
    *,
    repo_path: str | Path,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float = 0.0,
    cache_reloading_step: int = 2,
    window_size: int = 4,
    seed: int = 42,
) -> Tuple[Any, int]:
    """Run the official aggressive dKV-Cache-Greedy LLaDA configuration."""
    if cache_reloading_step <= 0:
        raise ValueError("cache_reloading_step must be positive")
    if window_size < 0:
        raise ValueError("window_size must be non-negative")
    if steps <= 0 or gen_length <= 0 or block_length <= 0:
        raise ValueError("steps, gen_length, and block_length must be positive")
    if gen_length % block_length:
        raise ValueError("gen_length must be divisible by block_length")

    _, official_generate, set_random_seed = _load_official_greedy_components(
        str(Path(repo_path).resolve())
    )
    # Greedy uses a predefined random denoising order. Resetting it per sample
    # makes full and resumed benchmark runs exactly reproducible.
    set_random_seed(seed)
    tokens = official_generate(
        model,
        None,
        input_ids,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        cfg_scale=0.0,
        remasking="random",
        enable_cache=True,
        cache_reloading_step=cache_reloading_step,
        window_size=window_size,
    )
    return tokens, int(steps)
