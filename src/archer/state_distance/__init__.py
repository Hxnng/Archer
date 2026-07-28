"""State-anchored prompt-cache controller for ARCHER."""

from .core import generation_hamming_distance
from .sampler import (
    ArcherDiagnostics,
    generate_archer,
    summarize_cache_diagnostics,
)

__all__ = [
    "ArcherDiagnostics",
    "generate_archer",
    "generation_hamming_distance",
    "summarize_cache_diagnostics",
]
